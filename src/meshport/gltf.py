"""glTF 2.0 (.gltf and .glb) <-> Scene.

Reads the core spec: node hierarchy, triangle/strip/fan primitives, normals, UV sets,
vertex colors, sparse accessors, PBR materials and their textures. Meshes compressed with
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
from .geometry import scaling, translation, unique_rows
from .scene import DAMAGED, Image, Material, Mesh, MeshportError, Node, Scene

_GLB_MAGIC = b"glTF"
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
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

    def _list(self, key):
        return self.doc.get(key, [])

    def _uri(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        if self.base_dir is None:
            raise MeshportError(f"This glTF file needs {uri}, which wasn't given")
        path = self.base_dir / unquote(uri)
        if not path.is_file():
            raise MeshportError(f"This glTF file needs {unquote(uri)} next to it, and it's missing")
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
        if "bufferView" in acc:
            buf, offset, length, stride = self._view(acc["bufferView"])
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
        built: dict[int, Node] = {}
        scene = Scene([self._node(i, built, set()) for i in roots])
        if not scene.meshes:
            raise MeshportError("No 3D geometry found in this file")
        return scene

    def _node(self, i: int, built: dict, path: set) -> Node:
        if i in path:
            raise MeshportError("This glTF file's node hierarchy loops back on itself")
        data = self._list("nodes")[i]
        if "matrix" in data:
            matrix = np.array(data["matrix"], np.float64).reshape(4, 4).T
        else:
            matrix = translation(data.get("translation", (0, 0, 0))) @ _quat(data.get("rotation", (0, 0, 0, 1))) \
                @ scaling(data.get("scale", (1, 1, 1)))
        node = Node(data.get("name", f"node{i}"), matrix)
        if "mesh" in data:
            got = self._mesh(data["mesh"])
            if got is not None:
                node.mesh, node.materials = got
        node.children = [self._node(c, built, path | {i}) for c in data.get("children", [])]
        return node

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
            parts.append((pos_key, pos, tris, normals, uvs, colors, slot_of[mat_index]))
        if not parts:
            self._meshes[i] = None
            return None
        mesh = _merge_parts(data.get("name", f"mesh{i}"), parts)
        self._meshes[i] = (mesh, materials)
        return self._meshes[i]

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
                    data = self._uri(img["uri"])
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


def _quat(q) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return m


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


def _merge_parts(name: str, parts: list) -> Mesh:
    positions, corners, normals, colors, mats = [], [], [], [], []
    n_uv = max(len(p[4]) for p in parts)
    uvs = [[] for _ in range(n_uv)]
    has_n = all(p[3] is not None for p in parts)
    has_c = any(p[5] is not None for p in parts)
    offsets: dict[int, int] = {}  # primitives often share one vertex buffer: store it once
    total = 0
    for key, pos, tris, nrm, uv, col, slot in parts:
        if key not in offsets:
            offsets[key] = total
            positions.append(pos)
            total += len(pos)
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
    mesh.validate()
    return mesh


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

    def _arr(self, key: str) -> list:
        return self.doc.setdefault(key, [])

    def build(self) -> tuple[dict, bytes]:
        self.doc["scenes"][0]["nodes"] = [self._node(n) for n in self.scene.roots]
        return self.doc, bytes(self.blob)

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
            acc["min"] = a.min(0).tolist()
            acc["max"] = a.max(0).tolist()
        self._arr("accessors").append(acc)
        return len(self.doc["accessors"]) - 1

    def _node(self, node: Node) -> int:
        idx = len(self.doc["nodes"])
        out = {"name": node.name}
        self.doc["nodes"].append(out)
        if not np.allclose(node.matrix, np.eye(4)):
            out["matrix"] = node.matrix.T.ravel().tolist()
        if node.mesh is not None:
            m = self._mesh(node.mesh, node.materials)
            if m is not None:
                out["mesh"] = m
        kids = [self._node(c) for c in node.children]
        if kids:
            out["children"] = kids
        return idx

    def _mesh(self, mesh: Mesh, materials: list[Material]) -> int | None:
        key = (id(mesh), tuple(id(m) for m in materials))
        if key in self.meshes:
            return self.meshes[key]
        tris = mesh.triangles()
        if not len(tris):
            return None
        tri_face = mesh.triangle_faces()
        corner = tris.ravel()
        # One glTF vertex per distinct (position, normal, uv..., color) combination.
        columns = [mesh.positions[mesh.corners[corner]]]
        if mesh.normals is not None:
            columns.append(mesh.normals[corner])
        for uv in mesh.uvs:
            columns.append(uv[corner])
        if mesh.colors is not None:
            columns.append(mesh.colors[corner])
        table = np.hstack(columns).astype(np.float32)
        first, inverse = unique_rows(table)
        vertices = table[first]
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
            primitives.append(prim)
        self._arr("meshes").append({"name": mesh.name, "primitives": primitives})
        self.meshes[key] = len(self.doc["meshes"]) - 1
        return self.meshes[key]

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
