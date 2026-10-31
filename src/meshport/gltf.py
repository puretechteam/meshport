"""glTF 2.0 (.gltf and .glb) <-> Scene.

Reads the core spec: node hierarchy, triangle/strip/fan primitives, normals, UV sets,
vertex colors, sparse accessors, PBR materials and their textures, skins, morph targets, and
animations (translation, rotation, scale and morph weights, in all three interpolations). Meshes compressed with
Draco or meshopt are refused with a clear error. Writes GLB with everything in one file,
or .gltf with the buffer and images embedded as data URIs.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from . import __version__
from .geometry import decompose_trs, quaternion, scaling, translation, unique_rows
from .scene import (
    DAMAGED,
    INTERPOLATIONS,
    PATHS,
    WEIGHTS,
    Animation,
    Channel,
    Image,
    Material,
    Mesh,
    MeshportError,
    Morph,
    Node,
    Scene,
    Skin,
    node_budget,
    read_image_file,
)

_GLB_MAGIC = b"glTF"
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
MAX_ACCESSOR_VALUES = 1 << 28  # values a file's accessors may hold in all, plus 4 per byte of its buffers
_UNSUPPORTED = {"KHR_draco_mesh_compression": "Draco", "EXT_meshopt_compression": "meshopt",
                "KHR_meshopt_compression": "meshopt"}

# ============================================================================ reading


def read(data: bytes, base_dir: str | os.PathLike | None = None) -> Scene:
    try:
        if data[:4] == _GLB_MAGIC:
            doc, binary = _split_glb(data)
        else:
            try:
                doc = json.loads(data.decode("utf-8-sig"))
            except ValueError:  # bad UTF-8 or bad JSON
                raise MeshportError("This isn't a glTF file") from None
            binary = None
        if not isinstance(doc, dict):
            raise MeshportError("This isn't a glTF file")
        return _Reader(doc, binary, Path(base_dir) if base_dir else None).scene()
    except (*DAMAGED, struct.error) as e:
        raise MeshportError(f"This glTF file is damaged ({type(e).__name__}: {e})") from None


def read_file(path: str | os.PathLike) -> Scene:
    path = Path(path)
    return read(path.read_bytes(), path.parent)


def _split_glb(data: bytes) -> tuple[dict, bytes | None]:
    if len(data) < 20:
        raise MeshportError("This GLB file is cut short")
    _, version, length = struct.unpack_from("<4sII", data)
    if version != 2:
        raise MeshportError(f"Only glTF 2.0 is supported (this file is version {version})")
    pos, doc, binary = 12, None, None
    end = min(length, len(data))
    while pos + 8 <= end:
        size, kind = struct.unpack_from("<II", data, pos)
        chunk = data[pos + 8:pos + 8 + size]
        if kind == _JSON_CHUNK:
            doc = json.loads(chunk.decode("utf-8"))
        elif kind == _BIN_CHUNK and binary is None:
            binary = chunk
        pos += 8 + size + (-size % 4)
    if doc is None:
        raise MeshportError("This GLB file has no JSON chunk")
    return doc, binary


class _Reader:
    def __init__(self, doc: dict, binary: bytes | None, base_dir: Path | None):
        self.doc, self.binary, self.base_dir = doc, binary, base_dir
        version = str(doc.get("asset", {}).get("version", "2.0"))
        if not version.startswith("2"):
            raise MeshportError(f"Only glTF 2.0 is supported (this file is version {version})")
        for ext in doc.get("extensionsRequired", []):
            if ext in _UNSUPPORTED:
                raise MeshportError(f"This file uses {_UNSUPPORTED[ext]} mesh compression, which meshport can't decode")
        self._buffers: dict[int, bytes] = {}
        self._images: dict[int, Image | None] = {}
        self._materials: dict[int, Material] = {}
        self._meshes: dict[int, tuple[Mesh, list[Material]] | None] = {}
        self._influences: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # mesh -> its vertices' joints and weights
        self._nodes: dict[int, Node] = {}
        self._skinned: list[tuple[Node, int, int]] = []  # (node, skin, mesh)
        self._values = 0  # accessor values read so far: an accessor without data is zeros, of any count

    def _list(self, key):
        return self.doc.get(key, [])

    def _uri(self, uri: str, image: bool = False) -> bytes:
        """A buffer or image the file points to. Only relative paths are followed: buffers must
        be in the model's folder, and images anywhere relative to it but actually images. So a
        hostile file can't copy other files on the computer into the output."""
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        if self.base_dir is None:
            raise MeshportError(f"This glTF file needs {uri}, which wasn't given")
        rel = unquote(uri)
        if ":" in rel or rel.replace("\\", "/").startswith("/"):
            raise MeshportError(f"This glTF file points to {rel}; meshport only follows paths relative to it")
        path = self.base_dir / rel
        if not path.is_file():
            raise MeshportError(f"This glTF file needs {rel} next to it, and it's missing")
        if image:
            data = read_image_file(path)
            if data is None:
                raise MeshportError(f"{rel} isn't an image")
            return data
        if not path.resolve().is_relative_to(self.base_dir.resolve()):
            raise MeshportError(f"This glTF file's buffer {rel} is outside its folder")
        return path.read_bytes()

    def _buffer(self, i: int) -> bytes:
        if i not in self._buffers:
            buf = self._list("buffers")[i]
            if "uri" in buf:
                self._buffers[i] = self._uri(buf["uri"])
            elif self.binary is not None:
                self._buffers[i] = self.binary
            else:
                raise MeshportError("This glTF file's binary buffer is missing")
        return self._buffers[i]

    def _view(self, i: int) -> tuple[bytes, int, int, int | None]:
        view = self._list("bufferViews")[i]
        return self._buffer(view["buffer"]), view.get("byteOffset", 0), view["byteLength"], view.get("byteStride")

    def accessor(self, i: int) -> np.ndarray:
        acc = self._list("accessors")[i]
        dtype = np.dtype(_COMPONENT[acc["componentType"]])
        width, count = _WIDTH[acc["type"]], acc["count"]
        if not isinstance(count, int) or count < 0:
            raise MeshportError("This glTF file is damaged (an accessor has a bad count)")
        if "bufferView" in acc:
            buf, offset, _, stride = self._view(acc["bufferView"])
            self._spend(count * width)
            offset += acc.get("byteOffset", 0)
            item = dtype.itemsize * width
            if stride and stride != item:
                raw = np.frombuffer(buf, np.uint8, count=stride * (count - 1) + item if count else 0, offset=offset)
                rows = np.lib.stride_tricks.as_strided(raw, (count, item), (stride, 1))
                out = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
            else:
                out = np.frombuffer(buf, dtype, count=count * width, offset=offset).reshape(count, width)
            out = out.astype(np.float64) if dtype.kind == "f" else out.copy()
        else:
            self._spend(count * width)
            out = np.zeros((count, width), np.float64 if dtype.kind == "f" else dtype)
        sparse = acc.get("sparse")
        if sparse:
            idx_info, val_info = sparse["indices"], sparse["values"]
            buf, off, _, _ = self._view(idx_info["bufferView"])
            idx = np.frombuffer(buf, _COMPONENT[idx_info["componentType"]], count=sparse["count"],
                                offset=off + idx_info.get("byteOffset", 0))
            buf, off, _, _ = self._view(val_info["bufferView"])
            vals = np.frombuffer(buf, dtype, count=sparse["count"] * width,
                                 offset=off + val_info.get("byteOffset", 0)).reshape(-1, width)
            out = out.astype(np.float64) if dtype.kind == "f" else out
            out[idx.astype(np.int64)] = vals
        if acc.get("normalized") and dtype.kind in "iu":
            info = np.iinfo(dtype)
            out = np.maximum(out.astype(np.float64) / info.max, -1.0)
        return out

    def _spend(self, values: int) -> None:
        self._values += values
        limit = MAX_ACCESSOR_VALUES + 4 * sum(len(b) for b in self._buffers.values())
        if self._values > limit:
            raise MeshportError(f"This glTF file's accessors hold too much data (over {limit:,} values); it may be "
                                "damaged or made to fill the memory")

    # ---- scene

    def scene(self) -> Scene:
        scenes = self._list("scenes")
        nodes = self._list("nodes")
        if scenes:
            idx = self.doc.get("scene", 0)
            roots = scenes[idx if idx < len(scenes) else 0].get("nodes", [])
        else:  # no scene list: every node nobody points to is a root
            used = {c for n in nodes for c in n.get("children", [])}
            roots = [i for i in range(len(nodes)) if i not in used]
        self._check_size(roots)
        built: dict[int, Node] = {}
        scene = Scene([self._node(i, built, set()) for i in roots])
        if not scene.meshes:
            raise MeshportError("No 3D geometry found in this file")
        for node, skin, mesh in self._skinned:  # once every joint is built
            node.skin = self._skin(skin, mesh, node.mesh)
        scene.animations = [a for a in (self._animation(k, a) for k, a in enumerate(self._list("animations"))) if a]
        return scene

    def _skin(self, i: int, mesh_index: int, mesh: Mesh) -> Skin | None:
        """A node's skin, or None when it's damaged or its mesh has no joints and weights (the mesh
        then stays where its node puts it)."""
        try:
            data = self._list("skins")[i]
            joints = [self._nodes.get(j) for j in data.get("joints", [])]
            if not joints or None in joints or mesh_index not in self._influences:
                return None
            if "inverseBindMatrices" in data:
                binds = self.accessor(data["inverseBindMatrices"]).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
            else:
                binds = np.tile(np.eye(4), (len(joints), 1, 1))
            indices, weights = self._influences[mesh_index]
            weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
            indices = np.where(weights > 0, indices, 0)
            total = weights.sum(1, keepdims=True)
            weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
            weights[total[:, 0] <= 0, 0] = 1.0  # a vertex nothing moves: give it to the first joint
            skin = Skin(joints, binds, indices, weights, data.get("name", f"skin{i}"))
            skin.validate(mesh)
            return skin if np.isfinite(binds).all() else None
        except (*DAMAGED, struct.error, MeshportError):
            return None

    def _check_size(self, roots: list) -> None:
        """Refuse a hierarchy that places more than scene.MAX_NODES nodes, before building any. A node
        listed under several parents (glTF forbids it) is built once per parent, so a chain of them
        doubles at every step; this counts the placements without making them."""
        nodes = self._list("nodes")

        def kids(i: int) -> list[int]:
            data = nodes[i] if isinstance(nodes[i], dict) else {}
            got = data.get("children", [])
            return [c for c in got if type(c) is int and 0 <= c < len(nodes)] if isinstance(got, list) else []

        budget = node_budget()
        cap = budget.left + 1
        size: dict[int, int] = {}
        open_: set[int] = set()
        stack = [(r, False) for r in roots if type(r) is int and 0 <= r < len(nodes)]
        while stack:
            i, done = stack.pop()
            if done:
                size[i] = min(1 + sum(size[c] for c in kids(i)), cap)
                open_.discard(i)
            elif i not in size:
                if i in open_:
                    raise MeshportError("This glTF file's node hierarchy loops back on itself")
                open_.add(i)
                stack.append((i, True))
                stack += [(c, False) for c in kids(i)]
        budget.spend(min(sum(size.get(r, 0) for r in roots if type(r) is int), cap))

    def _node(self, i: int, built: dict, path: set) -> Node:
        if i in path:
            raise MeshportError("This glTF file's node hierarchy loops back on itself")
        data = self._list("nodes")[i]
        if "matrix" in data:
            matrix = np.array(data["matrix"], np.float64).reshape(4, 4).T
        else:
            matrix = translation(data.get("translation", (0, 0, 0))) @ quaternion(data.get("rotation", (0, 0, 0, 1))) \
                @ scaling(data.get("scale", (1, 1, 1)))
        node = Node(data.get("name", f"node{i}"), matrix)
        self._nodes.setdefault(i, node)
        if "mesh" in data:
            got = self._mesh(data["mesh"])
            if got is not None:
                node.mesh, node.materials = got
                weights = data.get("weights")  # a node's own weights for its mesh's targets
                if isinstance(weights, list) and len(weights) == len(node.mesh.targets) and weights:
                    node.mesh.target_weights = np.array(weights, np.float64)
                if "skin" in data:
                    self._skinned.append((node, data["skin"], data["mesh"]))
        node.children = [self._node(c, built, path | {i}) for c in data.get("children", [])]
        return node

    def _animation(self, k: int, data: dict) -> Animation | None:
        """An animation's channels. A channel that's damaged, or moves something else (a node that
        isn't in the scene, weights of a mesh without targets), is left out rather than losing the model."""
        channels = []
        samplers = data.get("samplers", [])
        for ch in data.get("channels", []):
            try:
                target = ch["target"]
                node, path = self._nodes.get(target.get("node")), target.get("path")
                if node is None or (path not in PATHS and path != WEIGHTS):
                    continue
                width = PATHS.get(path) or (len(node.mesh.targets) if node.mesh is not None else 0)
                sampler = samplers[ch["sampler"]]
                interpolation = sampler.get("interpolation", "LINEAR")
                if interpolation not in INTERPOLATIONS or not width:
                    continue
                times = self.accessor(sampler["input"]).astype(np.float64).ravel()
                values = self.accessor(sampler["output"]).astype(np.float64)
                if path == WEIGHTS:  # one scalar per target per key
                    if values.size % width:
                        continue
                    values = values.reshape(-1, width)
                cubic = interpolation == "CUBICSPLINE"
                if values.shape[1] != width or len(values) != len(times) * (3 if cubic else 1):
                    continue
                if cubic:
                    values = values.reshape(len(times), 3, width)
                keep = np.isfinite(times) & np.isfinite(values.reshape(len(times), -1)).all(1)
                times, values = times[keep], values[keep]
                if not len(times):
                    continue
                order = np.argsort(times, kind="stable")  # keys out of order: put them in order
                channels.append(Channel(node, path, times[order], values[order], interpolation))
            except (*DAMAGED, struct.error):
                continue
        return Animation(data.get("name", f"animation{k}"), channels) if channels else None

    def _mesh(self, i: int):
        if i in self._meshes:
            return self._meshes[i]
        data = self._list("meshes")[i]
        parts, materials, slot_of = [], [], {}
        for prim in data.get("primitives", []):
            mode = prim.get("mode", 4)
            if mode not in (4, 5, 6) or "POSITION" not in prim.get("attributes", {}):
                continue  # points and lines carry no surface
            attrs = prim["attributes"]
            pos = self.accessor(attrs["POSITION"]).astype(np.float64)
            pos_key = attrs["POSITION"]
            idx = self.accessor(prim["indices"]).ravel().astype(np.int64) if "indices" in prim \
                else np.arange(len(pos), dtype=np.int64)
            tris = _to_triangles(idx, mode)
            if not len(tris):
                continue
            mat_index = prim.get("material", -1)
            if mat_index not in slot_of:
                slot_of[mat_index] = len(materials)
                materials.append(self._material(mat_index))
            corner_vertex = tris.ravel()
            normals = self.accessor(attrs["NORMAL"])[corner_vertex] if "NORMAL" in attrs else None
            uvs, k = [], 0
            while f"TEXCOORD_{k}" in attrs:
                uv = self.accessor(attrs[f"TEXCOORD_{k}"])[corner_vertex].astype(np.float64)
                uv[:, 1] = 1.0 - uv[:, 1]  # glTF counts v from the top of the image
                uvs.append(uv)
                k += 1
            colors = None
            if "COLOR_0" in attrs:
                c = self.accessor(attrs["COLOR_0"]).astype(np.float64)
                if c.shape[1] == 3:
                    c = np.hstack([c, np.ones((len(c), 1))])
                colors = c[corner_vertex]
            targets = []
            for t in prim.get("targets", []):
                offsets = self.accessor(t["POSITION"]).astype(np.float64) if "POSITION" in t else np.zeros_like(pos)
                shift = self.accessor(t["NORMAL"]).astype(np.float64) if "NORMAL" in t else None
                if offsets.shape != pos.shape or (shift is not None and shift.shape != pos.shape):
                    targets = None  # damaged: the mesh loads without them
                    break
                targets.append((offsets, shift[corner_vertex] if shift is not None else None))
            parts.append((pos_key, pos, tris, normals, uvs, colors, slot_of[mat_index], targets,
                          self._influence(attrs, len(pos))))
        if not parts:
            self._meshes[i] = None
            return None
        mesh, influence = _merge_parts(data.get("name", f"mesh{i}"), parts)
        names = (data.get("extras") or {}).get("targetNames") if isinstance(data.get("extras"), dict) else None
        for k, m in enumerate(mesh.targets):
            m.name = str(names[k]) if isinstance(names, list) and k < len(names) else f"target{k}"
        weights = data.get("weights")
        if mesh.targets and isinstance(weights, list) and len(weights) == len(mesh.targets):
            mesh.target_weights = np.array(weights, np.float64)
        if influence is not None:
            self._influences[i] = influence
        self._meshes[i] = (mesh, materials)
        return self._meshes[i]

    def _influence(self, attrs: dict, count: int) -> tuple[np.ndarray, np.ndarray] | None:
        """A primitive's joints and weights per vertex, every JOINTS_n/WEIGHTS_n set side by side."""
        joints, weights, k = [], [], 0
        while f"JOINTS_{k}" in attrs and f"WEIGHTS_{k}" in attrs:
            j, w = self.accessor(attrs[f"JOINTS_{k}"]), self.accessor(attrs[f"WEIGHTS_{k}"])
            if len(j) != count or j.shape != w.shape:
                return None
            joints.append(j.astype(np.int64))
            weights.append(w.astype(np.float64))
            k += 1
        return (np.hstack(joints), np.hstack(weights)) if joints else None

    def _image(self, tex_info: dict | None) -> Image | None:
        if not tex_info or "index" not in tex_info:
            return None
        textures = self._list("textures")
        if tex_info["index"] >= len(textures):
            return None
        tex = textures[tex_info["index"]]
        source = tex.get("source")
        for ext in tex.get("extensions", {}).values():  # KHR_texture_basisu, EXT_texture_webp, ...
            source = ext.get("source", source) if source is None else source
        if source is None:
            return None
        if source not in self._images:
            img = self._list("images")[source]
            name = img.get("name") or Path(unquote(img.get("uri", ""))).stem or f"image{source}"
            try:
                if "bufferView" in img:
                    buf, off, length, _ = self._view(img["bufferView"])
                    data = bytes(buf[off:off + length])
                    path = None
                else:
                    data = self._uri(img["uri"], image=True)
                    path = None if img["uri"].startswith("data:") else unquote(img["uri"])
                self._images[source] = Image(name, data, path)
            except MeshportError:
                self._images[source] = Image(name, None, unquote(img.get("uri", "")) or None)
        return self._images[source]

    def _material(self, i: int) -> Material:
        if i in self._materials:
            return self._materials[i]
        if i < 0 or i >= len(self._list("materials")):
            mat = Material("default")
        else:
            data = self._list("materials")[i]
            pbr = data.get("pbrMetallicRoughness", {})
            mat = Material(data.get("name", f"material{i}"),
                           tuple(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))),
                           self._image(pbr.get("baseColorTexture")),
                           float(pbr.get("metallicFactor", 1.0)), float(pbr.get("roughnessFactor", 1.0)),
                           tuple(data.get("emissiveFactor", (0.0, 0.0, 0.0))),
                           self._image(data.get("emissiveTexture")), self._image(data.get("normalTexture")),
                           bool(data.get("doubleSided", False)), data.get("alphaMode", "OPAQUE"))
        self._materials[i] = mat
        return mat


def _to_triangles(idx: np.ndarray, mode: int) -> np.ndarray:
    if mode == 4:
        return idx[: len(idx) // 3 * 3].reshape(-1, 3)
    if len(idx) < 3:
        return np.zeros((0, 3), np.int64)
    k = np.arange(len(idx) - 2)
    if mode == 5:  # strip: every other triangle flips to keep the winding
        a, b, c = idx[k], idx[k + 1], idx[k + 2]
        odd = k % 2 == 1
        a, b = np.where(odd, b, a), np.where(odd, a, b)
        return np.stack([a, b, c], 1)
    return np.stack([np.full(len(k), idx[0]), idx[k + 1], idx[k + 2]], 1)  # fan


def _merge_parts(name: str, parts: list) -> tuple[Mesh, tuple[np.ndarray, np.ndarray] | None]:
    """The primitives as one mesh, and its vertices' joints and weights when every primitive has them."""
    positions, corners, normals, colors, mats = [], [], [], [], []
    n_uv = max(len(p[4]) for p in parts)
    uvs = [[] for _ in range(n_uv)]
    has_n = all(p[3] is not None for p in parts)
    has_c = any(p[5] is not None for p in parts)
    counts = {len(p[7]) if p[7] is not None else -1 for p in parts}
    n_targets = counts.pop() if len(counts) == 1 and counts != {-1} else 0  # glTF: every primitive has as many
    target_pos = [[] for _ in range(n_targets)]
    target_nrm = [[] for _ in range(n_targets)]
    has_tn = has_n and n_targets > 0 and all(t[1] is not None for p in parts for t in p[7])
    skinned = all(p[8] is not None for p in parts)
    width = max(p[8][0].shape[1] for p in parts) if skinned else 0
    joints, weights = [], []
    offsets: dict[int, int] = {}  # primitives often share one vertex buffer: store it once
    total = 0
    for key, pos, tris, nrm, uv, col, slot, targets, influence in parts:
        if key not in offsets:
            offsets[key] = total
            positions.append(pos)
            total += len(pos)
            for m in range(n_targets):
                target_pos[m].append(targets[m][0])
            if skinned:
                pad = width - influence[0].shape[1]
                joints.append(np.pad(influence[0], ((0, 0), (0, pad))))
                weights.append(np.pad(influence[1], ((0, 0), (0, pad))))
        for m in range(n_targets if has_tn else 0):
            target_nrm[m].append(targets[m][1])
        corners.append(tris.ravel() + offsets[key])
        c = tris.size
        if has_n:
            normals.append(nrm)
        for k in range(n_uv):
            uvs[k].append(uv[k] if k < len(uv) else np.zeros((c, 2)))
        if has_c:
            colors.append(col if col is not None else np.ones((c, 4)))
        mats.append(np.full(len(tris), slot, np.int64))
    corner_arr = np.concatenate(corners)
    mesh = Mesh(name, np.concatenate(positions), np.full(len(corner_arr) // 3, 3, np.int64), corner_arr,
                np.concatenate(normals).astype(np.float64) if has_n else None,
                [np.concatenate(u) for u in uvs],
                np.concatenate(colors).astype(np.float64) if has_c else None,
                np.concatenate(mats))
    mesh.targets = [Morph(f"target{m}", np.concatenate(target_pos[m]),
                          np.concatenate(target_nrm[m]) if has_tn else None) for m in range(n_targets)]
    mesh.validate()
    return mesh, ((np.concatenate(joints), np.concatenate(weights)) if skinned else None)


# ============================================================================ writing


def write(scene: Scene, binary: bool = True) -> bytes:
    """GLB bytes (binary=True) or a self-contained .gltf JSON document."""
    doc, blob = _Writer(scene).build()
    if not binary:
        if blob:
            doc["buffers"] = [{"byteLength": len(blob),
                               "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}]
        return json.dumps(doc, separators=(",", ":")).encode()
    if blob:
        doc["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    chunks = struct.pack("<II", len(js), _JSON_CHUNK) + js
    if blob:
        chunks += struct.pack("<II", len(blob), _BIN_CHUNK) + blob
    return struct.pack("<4sII", _GLB_MAGIC, 2, 12 + len(chunks)) + chunks


def write_file(scene: Scene, path: str | os.PathLike) -> None:
    path = Path(path)
    path.write_bytes(write(scene, binary=path.suffix.lower() != ".gltf"))


_GLTF_MIME = {"image/png", "image/jpeg"}


class _Writer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.doc = {"asset": {"version": "2.0", "generator": f"meshport {__version__}"},
                    "scene": 0, "scenes": [{"nodes": []}], "nodes": []}
        self.blob = bytearray()
        self.meshes: dict[tuple, int] = {}
        self.materials: dict[int, int] = {}
        self.images: dict[int, int | None] = {}
        self.nodes: dict[int, int] = {}
        self.animated = {id(c.node) for a in scene.animations for c in a.channels if c.path in PATHS}
        in_scene = {id(n) for n, _ in scene.walk()}
        # a skin whose joints are all in the scene (glTF's joints are nodes); the others are left out
        self.skins = {id(n): n.skin for n, _ in scene.walk() if n.skin is not None and n.mesh is not None
                      and all(id(j) in in_scene for j in n.skin.joints)}
        self.skin_of: dict[int, int] = {}
        self.pending: list[tuple[dict, Skin]] = []  # nodes waiting for their skin's joints to be written
        # a skinned mesh's transforms don't move it (glTF ignores them), so one that hangs under another
        # node is written as a root of the scene without one: what glTF recommends
        roots = {id(n) for n in scene.roots}
        self.lifted: dict[int, np.ndarray] = {}  # node -> its matrix as a root
        for n, _ in scene.walk():
            if id(n) in self.skins and id(n) not in roots and not n.children and id(n) not in self.animated:
                self.lifted[id(n)] = np.eye(4)
        self.lifted_nodes = [n for n, _ in scene.walk() if id(n) in self.lifted]

    def _arr(self, key: str) -> list:
        return self.doc.setdefault(key, [])

    def build(self) -> tuple[dict, bytes]:
        roots = [self._node(n) for n in self.scene.roots] + [self._node(n, as_root=True) for n in self.lifted_nodes]
        self.doc["scenes"][0]["nodes"] = [r for r in roots if r is not None]
        for out, skin in self.pending:
            out["skin"] = self._skin(skin)
        for animation in self.scene.animations:
            self._animation(animation)
        return self.doc, bytes(self.blob)

    def _skin(self, skin: Skin) -> int:
        if id(skin) not in self.skin_of:
            binds = np.array(skin.inverse_binds, np.float64)
            binds[:, 3] = (0, 0, 0, 1)  # exactly, as glTF requires (arithmetic on the way can leave 1e-17s)
            binds = binds.transpose(0, 2, 1).reshape(-1, 16)  # column major
            self._arr("skins").append({"name": skin.name, "joints": [self.nodes[id(j)] for j in skin.joints],
                                       "inverseBindMatrices": self._accessor(binds.astype(np.float32), "MAT4", None)})
            self.skin_of[id(skin)] = len(self.doc["skins"]) - 1
        return self.skin_of[id(skin)]

    def _animation(self, animation: Animation) -> None:
        samplers, channels, inputs = [], [], {}
        for c in animation.channels:
            if id(c.node) not in self.nodes or not len(c.times):
                continue  # its node isn't in the scene
            # glTF's times are 32-bit: keys closer than that (a key next to a frame sample) are one key
            times = np.asarray(c.times, np.float32)
            keep = np.concatenate(([True], ~((np.diff(c.times) > 0) & (np.diff(times) <= 0))))
            times, c = times[keep], Channel(c.node, c.path, c.times[keep], c.values[keep], c.interpolation)
            if c.path == WEIGHTS:
                out = self.doc["nodes"][self.nodes[id(c.node)]]
                targets = len(c.node.mesh.targets) if c.node.mesh is not None and "mesh" in out else 0
                if not targets or c.values.shape[-1] != targets:
                    continue  # no targets for it to move
                values, kind = np.asarray(c.values, np.float32).reshape(-1, 1), "SCALAR"  # each key's weights in turn
            elif c.path in PATHS:
                values = np.asarray(c.values, np.float32).reshape(-1, PATHS[c.path])
                kind = "VEC4" if c.path == "rotation" else "VEC3"
            else:
                continue
            key = times.tobytes()
            if key not in inputs:  # channels keyed at the same times share one accessor
                inputs[key] = self._accessor(times, "SCALAR", None, bounds=True)
            samplers.append({"input": inputs[key], "output": self._accessor(values, kind, None),
                             "interpolation": c.interpolation})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": self.nodes[id(c.node)], "path": c.path}})
        if channels:
            self._arr("animations").append({"name": animation.name, "samplers": samplers, "channels": channels})

    def _view(self, data: bytes, target: int | None = None) -> int:
        self.blob += b"\0" * (-len(self.blob) % 4)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.blob += data
        self._arr("bufferViews").append(view)
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, a: np.ndarray, kind: str, target: int | None, bounds: bool = False) -> int:
        component = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125, np.dtype(np.uint16): 5123,
                     np.dtype(np.uint8): 5121}[a.dtype]
        acc = {"bufferView": self._view(np.ascontiguousarray(a).tobytes(), target),
               "componentType": component, "count": len(a), "type": kind}
        if bounds:
            acc["min"] = np.atleast_1d(a.min(0)).tolist()  # always a list, a scalar's too
            acc["max"] = np.atleast_1d(a.max(0)).tolist()
        self._arr("accessors").append(acc)
        return len(self.doc["accessors"]) - 1

    def _node(self, node: Node, as_root: bool = False) -> int | None:
        if id(node) in self.lifted and not as_root:
            return None  # written as a root instead
        idx = len(self.doc["nodes"])
        out = {"name": node.name}
        self.doc["nodes"].append(out)
        self.nodes.setdefault(id(node), idx)
        matrix = self.lifted[id(node)] if as_root else node.matrix
        if id(node) in self.animated:  # glTF animates translation, rotation and scale, never a matrix
            t, q, s = decompose_trs(node.matrix)
            for name, value, rest in (("translation", t, (0, 0, 0)), ("rotation", q, (0, 0, 0, 1)), ("scale", s, (1, 1, 1))):
                if not np.allclose(value, rest):
                    out[name] = [float(v) for v in value]
        elif not np.allclose(matrix, np.eye(4)):
            out["matrix"] = np.asarray(matrix).T.ravel().tolist()
        if node.mesh is not None:
            skin = self.skins.get(id(node))
            m = self._mesh(node.mesh, node.materials, skin)
            if m is not None:
                out["mesh"] = m
                if skin is not None:
                    self.pending.append((out, skin))
        kids = [k for k in (self._node(c) for c in node.children) if k is not None]
        if kids:
            out["children"] = kids
        return idx

    def _mesh(self, mesh: Mesh, materials: list[Material], skin: Skin | None = None) -> int | None:
        key = (id(mesh), tuple(id(m) for m in materials), id(skin))
        if key in self.meshes:
            return self.meshes[key]
        tris = mesh.triangles()
        if not len(tris):
            return None
        tri_face = mesh.triangle_faces()
        corner = tris.ravel()
        # One glTF vertex per distinct (position, normal, uv..., color) combination.
        columns = [mesh.positions[mesh.corners[corner]].astype(np.float32)]
        if mesh.normals is not None:
            columns.append(mesh.normals[corner].astype(np.float32))
        for uv in mesh.uvs:
            columns.append(uv[corner].astype(np.float32))
        if mesh.colors is not None:
            columns.append(mesh.colors[corner].astype(np.float32))
        if mesh.targets or skin is not None:  # vertices in the same place can move apart: keep them apart
            columns.append(mesh.corners[corner].astype(np.int32).view(np.float32)[:, None])
        normal_shift = mesh.normals is not None and all(m.normals is not None for m in mesh.targets)
        if mesh.targets and normal_shift:
            columns += [m.normals[corner].astype(np.float32) for m in mesh.targets]
        table = np.hstack(columns)
        first, inverse = unique_rows(table)
        vertices = table[first]
        source = mesh.corners[corner[first]]  # each glTF vertex's position in the mesh
        index = inverse.reshape(-1, 3)

        col = 0
        attrs = {"POSITION": self._accessor(np.ascontiguousarray(vertices[:, 0:3]), "VEC3", 34962, bounds=True)}
        col = 3
        if mesh.normals is not None:
            n = vertices[:, col:col + 3].astype(np.float64)
            length = np.linalg.norm(n, axis=1, keepdims=True)
            bad = length[:, 0] < 1e-12
            n = np.divide(n, length, out=np.zeros_like(n), where=~bad[:, None])
            n[bad] = (0.0, 1.0, 0.0)
            attrs["NORMAL"] = self._accessor(n.astype(np.float32), "VEC3", 34962)
            col += 3
        for k in range(len(mesh.uvs)):
            uv = vertices[:, col:col + 2].copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            attrs[f"TEXCOORD_{k}"] = self._accessor(uv, "VEC2", 34962)
            col += 2
        if mesh.colors is not None:
            attrs["COLOR_0"] = self._accessor(np.clip(vertices[:, col:col + 4], 0, 1), "VEC4", 34962)
        if skin is not None:
            self._influences(attrs, skin, source)
        targets = []
        for k, m in enumerate(mesh.targets):
            target = {"POSITION": self._accessor(m.positions[source].astype(np.float32), "VEC3", 34962, bounds=True)}
            if normal_shift:
                at = vertices.shape[1] - 3 * (len(mesh.targets) - k)
                target["NORMAL"] = self._accessor(np.ascontiguousarray(vertices[:, at:at + 3]), "VEC3", 34962)
            targets.append(target)

        slots = mesh.face_materials[tri_face] if mesh.face_materials is not None else np.zeros(len(tris), np.int64)
        if materials:
            slots = np.clip(slots, 0, len(materials) - 1)
        primitives = []
        for slot in np.unique(slots):
            part = index[slots == slot].ravel()
            dtype = np.uint16 if len(vertices) < 65535 else np.uint32
            prim = {"attributes": attrs, "indices": self._accessor(part.astype(dtype), "SCALAR", 34963), "mode": 4}
            if materials:
                prim["material"] = self._material(materials[int(slot)])
            if targets:
                prim["targets"] = targets
            primitives.append(prim)
        out = {"name": mesh.name, "primitives": primitives}
        if mesh.targets:
            weights = mesh.target_weights if mesh.target_weights is not None else np.zeros(len(mesh.targets))
            out["weights"] = [float(w) for w in weights]
            out["extras"] = {"targetNames": [m.name for m in mesh.targets]}  # Blender's and three.js's names
        self._arr("meshes").append(out)
        self.meshes[key] = len(self.doc["meshes"]) - 1
        return self.meshes[key]

    def _influences(self, attrs: dict, skin: Skin, source: np.ndarray) -> None:
        """JOINTS_n and WEIGHTS_n, four influences a set, the strongest first."""
        weights = np.asarray(skin.weights, np.float64)[source]
        joints = np.asarray(skin.indices)[source]
        order = np.argsort(-weights, axis=1, kind="stable")
        weights, joints = np.take_along_axis(weights, order, 1), np.take_along_axis(joints, order, 1)
        used = max(1, int((weights > 0).sum(1).max()))
        width = -(-used // 4) * 4
        weights = np.pad(weights[:, :used], ((0, 0), (0, width - used)))
        joints = np.where(weights > 0, np.pad(joints[:, :used], ((0, 0), (0, width - used))), 0)
        weights = (weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)).astype(np.float32)
        dtype = np.uint8 if len(skin.joints) <= 256 else np.uint16
        for k in range(width // 4):
            attrs[f"JOINTS_{k}"] = self._accessor(np.ascontiguousarray(joints[:, 4 * k:4 * k + 4].astype(dtype)), "VEC4",
                                                  34962)
            attrs[f"WEIGHTS_{k}"] = self._accessor(np.ascontiguousarray(weights[:, 4 * k:4 * k + 4]), "VEC4", 34962)

    def _material(self, mat: Material) -> int:
        if id(mat) in self.materials:
            return self.materials[id(mat)]
        pbr = {"baseColorFactor": [float(np.clip(c, 0, 1)) for c in mat.base_color],
               "metallicFactor": float(mat.metallic), "roughnessFactor": float(mat.roughness)}
        out = {"name": mat.name, "pbrMetallicRoughness": pbr}
        tex = self._texture(mat.base_color_texture)
        if tex is not None:
            pbr["baseColorTexture"] = {"index": tex}
        if any(mat.emissive):
            out["emissiveFactor"] = [float(np.clip(c, 0, 1)) for c in mat.emissive]
        tex = self._texture(mat.emissive_texture)
        if tex is not None:
            out["emissiveTexture"] = {"index": tex}
        tex = self._texture(mat.normal_texture)
        if tex is not None:
            out["normalTexture"] = {"index": tex}
        if mat.alpha_mode != "OPAQUE":
            out["alphaMode"] = mat.alpha_mode
        if mat.double_sided:
            out["doubleSided"] = True
        self._arr("materials").append(out)
        self.materials[id(mat)] = len(self.doc["materials"]) - 1
        return self.materials[id(mat)]

    def _texture(self, image: Image | None) -> int | None:
        if image is None or not image.data:
            return None
        if id(image) not in self.images:
            data, mime = image.data, image.mime
            if mime not in _GLTF_MIME:
                data, mime = _to_png(data)
            if data is None:
                self.images[id(image)] = None
            else:
                self._arr("images").append({"name": image.name, "mimeType": mime, "bufferView": self._view(data)})
                if not self.doc.get("samplers"):
                    self.doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
                self._arr("textures").append({"sampler": 0, "source": len(self.doc["images"]) - 1})
                self.images[id(image)] = len(self.doc["textures"]) - 1
        return self.images[id(image)]


def _to_png(data: bytes) -> tuple[bytes | None, str | None]:
    """glTF only allows PNG and JPEG. Re-encode anything else when Pillow is around."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None, None
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            im.load()
            if im.mode not in ("RGB", "RGBA", "L", "LA"):
                im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
            out = io.BytesIO()
            im.save(out, "PNG")
            return out.getvalue(), "image/png"
    except Exception:  # noqa: BLE001 - an unreadable texture shouldn't sink the whole model
        return None, None
