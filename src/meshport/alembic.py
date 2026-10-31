"""Alembic (.abc, the Ogawa kind written since 2013) <-> Scene.

An Ogawa file is a tree of groups and data blocks addressed by 64-bit offsets. Alembic builds
objects on it: each object's group holds its properties, its child objects, and a block of
child headers (names and metadata). Properties are compounds (more properties) or scalar and
array properties whose samples are data blocks, each led by a 16-byte key.

Reads Xform objects (every op kind, and transforms that don't inherit their parent's), and
PolyMesh and SubD objects (the cage, as a mesh) with normals, UV sets and colors in any scope,
indexed or not, and face sets as material slots. Animated transforms become node channels at
their samples' times (from the archive's uniform, cyclic or acyclic time samplings); animated
points open at their first sample. Cameras, curves, points, NURBS and lights are skipped. Alembic has no units: one unit is one
meter, Y up. Polygons are stored clockwise, so their corners are reversed on the way in and out.

Writes Xform and PolyMesh objects with normals, every UV set, colors and face sets (Alembic has
no materials, so they become face sets named after them), which Blender, Maya, Houdini and
Alembic's own tools read. Animated nodes get a transform sample every frame, on one uniform time
sampling (Alembic has one time line here too, so the first animation is written).

Written from Alembic's open-source code (BSD-3) and checked against files Blender writes.
"""

from __future__ import annotations

import hashlib
import os
import struct

import numpy as np

from .geometry import compose_trs, decompose_many, decompose_trs, scaling, translation
from .scene import (
    DAMAGED,
    MAX_FRAMES,
    WEIGHTS,
    Animation,
    Channel,
    Material,
    Mesh,
    MeshportError,
    Morph,
    Node,
    Scene,
    frame_rate,
    node_budget,
    too_many_frames,
)

MAGIC = b"Ogawa"
_DATA = 1 << 63
_POD = {0: "u1", 1: "u1", 2: "i1", 3: "<u2", 4: "<i2", 5: "<u4", 6: "<i4", 7: "<u8", 8: "<i8", 9: "<f2", 10: "<f4",
        11: "<f8"}  # 12: string, 13: wide string
_STRING = 12
_COMPOUND, _SCALAR, _ARRAY = 0, 1, 2
_CHANNELS = {0: 3, 1: 3, 2: 4, 3: 16, 4: 1, 5: 1, 6: 1}  # scale, translate, rotate, matrix, rotateX/Y/Z


# ============================================================================ reading

class _File:
    def __init__(self, data: bytes):
        if data[:5] != MAGIC:
            if data[:8] == b"\x89HDF\r\n\x1a\n":
                raise MeshportError("This Alembic file is in the old HDF5 form (from before 2013), "
                                    "which meshport can't read")
            raise MeshportError("This isn't an Alembic file")
        self.data = data
        self.visits = 0

    def _u64(self, at: int) -> int:
        if at < 0 or at + 8 > len(self.data):
            raise MeshportError("This Alembic file is damaged (a pointer leads outside it)")
        return struct.unpack_from("<Q", self.data, at)[0]

    def group(self, ref: int) -> list[int]:
        if ref & _DATA:
            raise MeshportError("This Alembic file is damaged (data where a group should be)")
        if ref == 0:
            return []
        self.visits += 1
        if self.visits > 10_000_000:
            raise MeshportError("This Alembic file is damaged (its tree loops)")
        n = self._u64(ref)
        if n > (len(self.data) - ref - 8) // 8:
            raise MeshportError("This Alembic file is damaged (a group is cut short)")
        return list(np.frombuffer(self.data, "<u8", n, ref + 8).tolist())

    def block(self, ref: int) -> memoryview:
        if not ref & _DATA:
            raise MeshportError("This Alembic file is damaged (a group where data should be)")
        at = ref & ~_DATA
        if at == 0:
            return memoryview(b"")
        n = self._u64(at)
        if at + 8 + n > len(self.data):
            raise MeshportError("This Alembic file is damaged (a data block is cut short)")
        return memoryview(self.data)[at + 8:at + 8 + n]


def _metadata(text: str) -> dict[str, str]:
    out = {}
    for part in text.split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = v
    return out


class _Prop:
    __slots__ = ("extent", "kind", "last", "meta", "name", "pod", "ref", "samples", "sampling")

    def __init__(self, name, kind, pod, extent, meta, ref, samples=1, last=0, sampling=0):
        self.name, self.kind, self.pod, self.extent, self.meta, self.ref = name, kind, pod, extent, meta, ref
        # samples written, the last one stored (later ones repeat it) and the time sampling's index
        self.samples, self.last, self.sampling = samples, last, sampling


class _Reader:
    def __init__(self, data: bytes):
        self.f = _File(data)
        archive = self.f.group(self.f._u64(8))
        if len(archive) < 6:
            raise MeshportError("This Alembic file is damaged (its archive is incomplete)")
        self.metas = [{}]
        raw = bytes(self.f.block(archive[5]))
        pos = 0
        while pos < len(raw):
            n = raw[pos]
            self.metas.append(_metadata(raw[pos + 1:pos + 1 + n].decode("utf-8", "replace")))
            pos += 1 + n
        self.root = archive[2]
        self.samplings = _time_samplings(bytes(self.f.block(archive[4])))
        self.animated: list[tuple[Node, _Prop, np.ndarray | None]] = []  # (node, .xform, parent world if not inheriting)
        self.caches: dict[int, np.ndarray] = {}  # mesh -> the times of its point samples, read as blend shapes
        # groups several objects share (a file can hold one object under many) are read once
        self._compounds: dict[int, dict[str, _Prop]] = {}
        self._objects: dict[int, list[tuple[str, dict, int]]] = {}
        self._meshes: dict[tuple, tuple | None] = {}
        self.nodes = node_budget()

    @staticmethod
    def _uint(buf: bytes, pos: int, hint: int) -> tuple[int, int]:
        size = (1, 2, 4)[hint] if hint < 3 else 4
        if pos + size > len(buf):
            raise MeshportError("This Alembic file is damaged (a header is cut short)")
        return int.from_bytes(buf[pos:pos + size], "little"), pos + size

    def _meta(self, index: int, buf: bytes, pos: int, hint: int) -> tuple[dict, int]:
        if index == 0xFF:
            n, pos = self._uint(buf, pos, hint)
            return _metadata(buf[pos:pos + n].decode("utf-8", "replace")), pos + n
        if index >= len(self.metas):
            raise MeshportError("This Alembic file is damaged (unknown metadata)")
        return self.metas[index], pos

    def compound(self, ref: int) -> dict[str, _Prop]:
        if ref not in self._compounds:
            self._compounds[ref] = self._compound(ref)
        return self._compounds[ref]

    def _compound(self, ref: int) -> dict[str, _Prop]:
        kids = self.f.group(ref)
        if not kids:
            return {}
        buf = bytes(self.f.block(kids[-1]))
        props: dict[str, _Prop] = {}
        pos, i = 0, 0
        while pos < len(buf):
            info, pos = self._uint(buf, pos, 2)
            kind, hint, pod = info & 3, (info >> 2) & 3, (info >> 4) & 0xF
            samples, last, sampling = 1, 0, 0
            if kind != _COMPOUND:
                samples, pos = self._uint(buf, pos, hint)
                last = samples - 1
                if info & 0x200:  # first and last changed sample
                    _, pos = self._uint(buf, pos, hint)
                    last, pos = self._uint(buf, pos, hint)
                elif info & 0x800:  # all the same
                    last = 0
                if info & 0x100:
                    sampling, pos = self._uint(buf, pos, hint)
            n, pos = self._uint(buf, pos, hint)
            name = buf[pos:pos + n].decode("utf-8", "replace")
            pos += n
            meta, pos = self._meta((info >> 20) & 0xFF, buf, pos, hint)
            if i >= len(kids) - 1:
                raise MeshportError("This Alembic file is damaged (more properties than groups)")
            props[name] = _Prop(name, kind, pod, (info >> 12) & 0xFF, meta, kids[i], samples, last, sampling)
            i += 1
        return props

    def sample(self, prop: _Prop | None, index: int = 0) -> np.ndarray | list | None:
        """A sample (the first by default): an (n, extent) array (or (extent,) for a scalar), or a
        list of strings. Samples after the last one stored repeat it."""
        if prop is None or prop.kind == _COMPOUND:
            return None
        kids = self.f.group(prop.ref)
        step = 2 if prop.kind == _ARRAY else 1  # an array sample is its data and its dimensions
        at = min(index, prop.last, len(kids) // step - 1) * step
        if at < 0:
            return None
        raw = bytes(self.f.block(kids[at]))[16:]  # after the sample's key
        if prop.pod in (_STRING, 13):
            return raw.decode("utf-8" if prop.pod == _STRING else "utf-16-le", "replace").split("\0")[:-1]
        if prop.pod not in _POD:
            raise MeshportError("This Alembic file is damaged (unknown value type)")
        dtype = np.dtype(_POD[prop.pod])
        extent = max(prop.extent, 1)
        count = len(raw) // (dtype.itemsize * extent)
        arr = np.frombuffer(raw, dtype, count * extent).astype(dtype.newbyteorder("="))
        return arr.reshape(count, extent) if prop.kind == _ARRAY else arr

    def stored(self, prop: _Prop) -> int:
        """How many samples a property really has: those up to its last change (later ones repeat
        it), as far as it stores them."""
        step = 2 if prop.kind == _ARRAY else 1
        return max(0, min(prop.samples, prop.last + 1, len(self.f.group(prop.ref)) // step))

    def objects(self, ref: int) -> list[tuple[str, dict, int]]:
        """(name, metadata, group) for each child object."""
        if ref not in self._objects:
            self._objects[ref] = self._read_objects(ref)
        return self._objects[ref]

    def _read_objects(self, ref: int) -> list[tuple[str, dict, int]]:
        kids = self.f.group(ref)
        if len(kids) < 2:
            return []
        buf = bytes(self.f.block(kids[-1]))
        buf = buf[:max(len(buf) - 32, 0)]  # 32 bytes of hashes at the end
        out, pos, i = [], 0, 1
        while pos < len(buf):
            n, pos = self._uint(buf, pos, 2)
            name = buf[pos:pos + n].decode("utf-8", "replace")
            pos += n
            if pos >= len(buf):
                raise MeshportError("This Alembic file is damaged (an object header is cut short)")
            index = buf[pos]
            pos += 1
            meta, pos = self._meta(index, buf, pos, 2)
            if i >= len(kids) - 1:
                raise MeshportError("This Alembic file is damaged (more objects than groups)")
            out.append((name, meta, kids[i]))
            i += 1
        return out

    # -- scene

    def scene(self) -> Scene:
        roots = self._children(self.root, np.eye(4), 0)
        scene = Scene(roots)
        if not scene.meshes:
            raise MeshportError("No 3D geometry found in this file")
        in_scene = {id(n) for n, _ in scene.walk()}
        channels = []
        for node, prop, parent_world in self.animated:
            if id(node) in in_scene:
                try:
                    channels += self._channels(node, prop, parent_world)
                except (MeshportError, *DAMAGED):
                    continue  # a damaged sample loses this object's animation, not the file
        for node, _ in scene.walk():
            if node.mesh is not None and id(node.mesh) in self.caches:
                times = self.caches[id(node.mesh)]
                # sample k is target k - 1 at full weight, the others at none: linear between, as Alembic plays it
                weights = np.vstack([np.zeros(len(times) - 1), np.eye(len(times) - 1)])
                channels.append(Channel(node, WEIGHTS, times, weights, "LINEAR"))
        if channels:
            scene.animations = [Animation("animation", channels)]
        return scene

    def _channels(self, node: Node, prop: _Prop, parent_world: np.ndarray | None) -> list[Channel]:
        """An animated Xform's samples as translation, rotation and scale channels, at each
        sample's time."""
        props = self.compound(prop.ref)
        vals = props.get(".vals")
        ops = props.get(".ops")
        n = max(self.stored(p) for p in (vals, ops) if p is not None)  # the samples after them repeat them
        if n > MAX_FRAMES:
            raise too_many_frames(n)
        times = _sample_times(self.samplings, vals.sampling, n)
        if times is None or len(times) < 2:
            return []
        m = np.array([self._xform_matrix(props, i) for i in range(n)])
        if parent_world is not None:
            m = np.linalg.inv(parent_world) @ m
        order = np.argsort(times, kind="stable")
        out = []
        for name, values in zip(("translation", "rotation", "scale"), decompose_many(m[order])):
            if not np.allclose(values, values[0], atol=1e-9):
                out.append(Channel(node, name, times[order], values, "LINEAR"))
        return out

    def _children(self, ref: int, parent_world: np.ndarray, depth: int) -> list[Node]:
        if depth > 1000:
            raise MeshportError("This Alembic file is damaged (objects nested too deep)")
        nodes = []
        for name, meta, group in self.objects(ref):
            node = self._object(name, meta.get("schema", ""), group, parent_world, depth)
            if node is not None:
                nodes.append(node)
        return nodes

    def _object(self, name: str, schema: str, ref: int, parent_world: np.ndarray, depth: int) -> Node | None:
        if schema.startswith("AbcGeom_FaceSet"):  # read with its mesh
            return None
        kids = self.f.group(ref)
        props = self.compound(kids[0]) if kids else {}
        self.nodes.spend()
        node = Node(name)
        if schema.startswith("AbcGeom_Xform"):
            local, inherits = self._xform(props.get(".xform"))
            node.matrix = local if inherits else np.linalg.inv(parent_world) @ local
            vals = self.compound(props[".xform"].ref).get(".vals") if ".xform" in props else None
            if vals is not None and vals.samples > 1 and vals.last > 0:
                self.animated.append((node, props[".xform"], None if inherits else parent_world))
        elif schema.startswith(("AbcGeom_PolyMesh", "AbcGeom_SubD")):
            geom = props.get(".geom")
            key = (name, geom.ref if geom is not None else None, ref)
            if key not in self._meshes:  # an object shown in many places: one mesh, shared
                self._meshes[key] = self._mesh(name, geom, ref)
            got = self._meshes[key]
            if got is not None:
                node.mesh, node.materials = got
        world = parent_world @ node.matrix
        node.children = self._children(ref, world, depth + 1)
        if node.mesh is None and not node.children:
            return None
        return node

    def _point_cache(self, geom: dict, count: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """A mesh whose P changes over time (and whose faces don't) as (the samples' times, each
        later sample's offsets from the first), within CACHE_BUDGET values (its offsets and the
        weights that play them, one row a sample); else None (the mesh stands as its first sample)."""
        prop, faces = geom.get("P"), geom.get(".faceCounts")
        if prop is None or prop.samples < 2 or prop.last < 1 or (faces is not None and faces.last > 0):
            return None
        samples = self.stored(prop)  # the ones after the last stored repeat it
        if samples < 2 or samples * count * 3 + samples * (samples - 1) > CACHE_BUDGET:
            return None
        times = _sample_times(self.samplings, prop.sampling, samples)
        if times is None or (np.diff(times) <= 0).any():
            return None
        first = np.asarray(self.sample(prop, 0), np.float64).reshape(-1, 3)
        offsets = []
        for n in range(1, samples):
            pts = self.sample(prop, n)
            if pts is None or np.asarray(pts).size != first.size:
                return None
            offsets.append(np.asarray(pts, np.float64).reshape(-1, 3) - first)
        return times, offsets

    def _xform(self, prop: _Prop | None) -> tuple[np.ndarray, bool]:
        if prop is None:
            return np.eye(4), True
        props = self.compound(prop.ref)
        inherits = self.sample(props.get(".inherits"))
        inherits = True if inherits is None or not len(inherits) else bool(np.ravel(inherits)[0])
        return self._xform_matrix(props, 0), inherits

    def _xform_matrix(self, props: dict, index: int) -> np.ndarray:
        ops = self.sample(props.get(".ops"), index)
        vals = self.sample(props.get(".vals"), index)
        if ops is None or vals is None:
            return np.eye(4)
        ops = np.ravel(ops).astype(np.int64)
        vals = np.ravel(vals).astype(np.float64)
        m, at = np.eye(4), 0
        for op in ops:
            kind = int(op) >> 4
            n = _CHANNELS.get(kind)
            if n is None or at + n > len(vals):
                raise MeshportError("This Alembic file is damaged (a transform has too few values)")
            v = vals[at:at + n]
            at += n
            m = m @ _op_matrix(kind, v)
        return m

    def _param(self, geom: dict, name: str, counts, indices):
        """A geometry parameter (plain, or with .vals and .indices) expanded to one value per
        corner, in Alembic's corner order, or None."""
        prop = geom.get(name)
        if prop is None:
            return None
        if prop.kind == _COMPOUND:
            sub = self.compound(prop.ref)
            vals = self.sample(sub.get(".vals"))
            idx = self.sample(sub.get(".indices"))
            meta = sub[".vals"].meta if ".vals" in sub else prop.meta
            if vals is None:
                return None
            if idx is not None:
                idx = np.ravel(idx).astype(np.int64)
                if len(idx) and (idx.min() < 0 or idx.max() >= len(vals)):
                    raise MeshportError(f"This Alembic file is damaged ({name} index out of range)")
                vals = vals[idx]
            scope = prop.meta.get("geoScope") or meta.get("geoScope", "")
        else:
            vals = self.sample(prop)
            scope = prop.meta.get("geoScope", "")
        if vals is None or not len(vals) or isinstance(vals, list):
            return None
        vals = np.asarray(vals, np.float64)
        n = len(indices)
        if scope == "fvr" and len(vals) == n:
            return vals
        if scope in ("vtx", "var") and len(vals) > indices.max(initial=-1):
            return vals[indices]
        if scope == "uni" and len(vals) == len(counts):
            return np.repeat(vals, counts, 0)
        if scope == "con":
            return np.repeat(vals[:1], n, 0)
        if len(vals) == n:  # no scope given: guess from the length
            return vals
        return None

    def _mesh(self, name: str, geom_prop: _Prop | None, obj_ref: int):
        if geom_prop is None:
            return None
        geom = self.compound(geom_prop.ref)
        points = self.sample(geom.get("P"))
        indices = self.sample(geom.get(".faceIndices"))
        counts = self.sample(geom.get(".faceCounts"))
        if points is None or indices is None or counts is None or not len(points):
            return None
        points = np.asarray(points, np.float64).reshape(-1, 3)
        indices = np.ravel(indices).astype(np.int64)
        counts = np.ravel(counts).astype(np.int64)
        if counts.min(initial=0) < 0 or int(counts.sum()) != len(indices):
            raise MeshportError(f"Mesh {name!r}: its face sizes don't add up to its index list")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
            raise MeshportError(f"Mesh {name!r}: a face points past its vertex list")
        normals = self._param(geom, "N", counts, indices)
        uvs = []
        uv = self._param(geom, "uv", counts, indices)
        if uv is not None:
            uvs.append(uv[:, :2])
        colors = None
        arb = geom.get(".arbGeomParams")
        if arb is not None and arb.kind == _COMPOUND:
            extra = self.compound(arb.ref)
            for pname, p in extra.items():
                meta = p.meta if p.kind != _COMPOUND else {**self._vals_meta(p), **p.meta}
                interp = meta.get("interpretation", "")
                values = None
                if interp in ("vector", "") and meta.get("podExtent") == "2":
                    values = self._param(extra, pname, counts, indices)
                    if values is not None:
                        uvs.append(values[:, :2])
                elif interp in ("rgb", "rgba") and colors is None:
                    values = self._param(extra, pname, counts, indices)
                    if values is not None:
                        colors = np.ones((len(indices), 4))
                        colors[:, :values.shape[1]] = values[:, :4]
        # face sets become material slots
        face_slot = None
        materials: list[Material] = []
        for set_name, meta, ref in self.objects(obj_ref):
            if not meta.get("schema", "").startswith("AbcGeom_FaceSet"):
                continue
            kids = self.f.group(ref)
            props = self.compound(kids[0]) if kids else {}
            fs = props.get(".faceset")
            faces = self.sample(self.compound(fs.ref).get(".faces")) if fs is not None else None
            if faces is None:
                continue
            faces = np.ravel(faces).astype(np.int64)
            faces = faces[(faces >= 0) & (faces < len(counts))]
            if face_slot is None:
                face_slot = np.zeros(len(counts), np.int64)
                materials.append(Material("default"))
            materials.append(Material(set_name))
            face_slot[faces] = len(materials) - 1
        keep = counts >= 3
        corner_keep = np.repeat(keep, counts)
        sizes, corners = counts[keep], indices[corner_keep]
        if not len(sizes):
            return None
        flip = _reversed(sizes)  # clockwise -> counter-clockwise
        normals = normals[corner_keep][flip] if normals is not None else None
        uvs = [u[corner_keep][flip] for u in uvs]
        colors = colors[corner_keep][flip] if colors is not None else None
        cached = self._point_cache(geom, len(points))
        mesh = Mesh(name, points, sizes, corners[flip], normals, uvs, colors,
                    face_slot[keep] if face_slot is not None else None)
        if cached is not None:
            times, offsets = cached
            mesh.targets = [Morph(f"sample {n + 2}", o) for n, o in enumerate(offsets)]
            self.caches[id(mesh)] = times
        return mesh, materials

    def _vals_meta(self, prop: _Prop) -> dict:
        sub = self.compound(prop.ref)
        return sub[".vals"].meta if ".vals" in sub else {}


def _time_samplings(raw: bytes) -> list[tuple[float, list[float]]]:
    """The archive's time samplings: (time per cycle, the times in a cycle) each. Each entry is
    the largest sample count using it (u32), the time per cycle (f64), the count of times in a
    cycle (u32) and those times (f64)."""
    out, pos = [], 0
    while pos + 16 <= len(raw):
        _, per_cycle, n = struct.unpack_from("<IdI", raw, pos)
        pos += 16
        if n > (len(raw) - pos) // 8:
            break
        out.append((per_cycle, list(struct.unpack_from(f"<{n}d", raw, pos))))
        pos += 8 * n
    return out


def _sample_times(samplings: list, index: int, count: int) -> np.ndarray | None:
    """Seconds of each of count samples: uniform (one time a cycle), cyclic (several, repeating
    every time per cycle) or acyclic (every time listed; its time per cycle is huge)."""
    if not 0 <= index < len(samplings):
        return None if index else np.arange(count, dtype=np.float64)  # index 0 is one sample a second from 0
    per_cycle, times = samplings[index]
    if not times:
        return None
    i = np.arange(count)
    if not np.isfinite(per_cycle) or per_cycle > 1e300 or len(times) >= count:  # acyclic (DBL_MAX / 32)
        return np.array(times[:count], np.float64) if len(times) >= count else None
    cycle = np.array(times, np.float64)
    return cycle[i % len(cycle)] + (i // len(cycle)) * per_cycle


def _reversed(sizes: np.ndarray) -> np.ndarray:
    """Corner order that reverses every face."""
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    local = np.arange(int(sizes.sum())) - np.repeat(starts, sizes)
    return np.repeat(starts + sizes - 1, sizes) - local


def _op_matrix(kind: int, v: np.ndarray) -> np.ndarray:
    if kind == 0:
        return scaling(v)
    if kind == 1:
        return translation(v)
    if kind == 3:  # row-major, for row vectors
        return v.reshape(4, 4).T.copy()
    if kind == 2:
        return _axis_angle(v[:3], v[3])
    return _axis_angle(np.eye(3)[kind - 4], v[0])


def _axis_angle(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    n = np.linalg.norm(axis)
    m = np.eye(4)
    if n == 0:
        return m
    x, y, z = axis / n
    a = np.radians(degrees)
    c, s, t = np.cos(a), np.sin(a), 1 - np.cos(a)
    m[:3, :3] = [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                 [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                 [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    return m


def read(data: bytes, base_dir: str | os.PathLike | None = None) -> Scene:
    try:
        return _Reader(data).scene()
    except MeshportError:
        raise
    except (*DAMAGED, struct.error) as e:
        raise MeshportError(f"This Alembic file is damaged ({type(e).__name__}: {e})") from None


def read_file(path: str | os.PathLike) -> Scene:
    with open(path, "rb") as f:
        return read(f.read())


# ============================================================================ writing

# an object's metadata names its schema and where its properties are (schemaObjTitle); the
# property compound holding them names just the schema
_XFORM_PROP = "schema=AbcGeom_Xform_v3"
_XFORM = _XFORM_PROP + ";schemaObjTitle=AbcGeom_Xform_v3:.xform"
_MESH_PROP = "schema=AbcGeom_PolyMesh_v1;schemaBaseType=AbcGeom_GeomBase_v1"
_MESH = _MESH_PROP + ";schemaObjTitle=AbcGeom_PolyMesh_v1:.geom"
_FACESET_PROP = "schema=AbcGeom_FaceSet_v1;schemaBaseType=AbcGeom_GeomBase_v1"
_FACESET = _FACESET_PROP + ";schemaObjTitle=AbcGeom_FaceSet_v1:.faceset"
_POD_CODE = {"bool": 0, "u1": 1, "<u4": 5, "<i4": 6, "<f4": 10, "<f8": 11}
CACHE_BUDGET = 30_000_000  # point-cache values (samples x points x 3, and the samples' weights) read as blend shapes


class _Writer:
    def __init__(self):
        self.out = bytearray(MAGIC + b"\xff\x00\x01" + bytes(8))
        self.metas: dict[str, int] = {}  # the archive's metadata table; index 0 is "none"
        self.samples: dict[tuple, int] = {}
        self.moving: dict[int, list[Channel]] = {}  # node -> its channels, when the scene moves
        self.frames: np.ndarray | None = None  # the time of each sample, then
        self.fps = 24.0
        self.bent: dict[int, list[np.ndarray]] = {}  # mesh node -> its points at every frame, when they bend

    def meta_index(self, meta: str) -> int:
        """Metadata goes in the archive's table, as Alembic itself writes it (readers such as
        Blender's look there); 0xff means it follows inline, for the rare one that doesn't fit."""
        if not meta:
            return 0
        if meta not in self.metas and len(self.metas) < 254 and len(meta.encode("utf-8")) < 256:
            self.metas[meta] = len(self.metas) + 1
        return self.metas.get(meta, 0xFF)

    def data(self, payload: bytes) -> int:
        if not payload:
            return _DATA
        at = len(self.out)
        self.out += struct.pack("<Q", len(payload)) + payload
        return at | _DATA

    def group(self, refs: list[int]) -> int:
        if not refs:
            return 0
        at = len(self.out)
        self.out += struct.pack("<Q", len(refs)) + np.array(refs, "<u8").tobytes()
        return at

    # -- properties: (name, kind, pod, extent, metadata, ref)

    def scalar(self, name: str, values, pod: str, meta: str = "") -> tuple:
        arr = np.ascontiguousarray(np.asarray(values, "u1" if pod == "bool" else pod)).ravel()
        return (name, _SCALAR, _POD_CODE[pod], len(arr), meta, self.group([self._sample(arr.tobytes())]))

    def scalar_samples(self, name: str, samples: list, pod: str, sampling: int) -> tuple:
        """A scalar property with a sample each time of a time sampling."""
        rows = [np.ascontiguousarray(np.asarray(v, pod)).ravel() for v in samples]
        refs = [self._sample(r.tobytes()) for r in rows]
        return (name, _SCALAR, _POD_CODE[pod], len(rows[0]), "", self.group(refs), (len(rows), sampling))

    def array_samples(self, name: str, samples: list, pod: str, sampling: int, meta: str = "") -> tuple:
        """An array property with a sample each time of a time sampling."""
        rows = [np.ascontiguousarray(np.asarray(v, pod)) for v in samples]
        extent = rows[0].shape[1] if rows[0].ndim == 2 else 1
        refs = [ref for r in rows for ref in (self._sample(r.tobytes()), _DATA)]
        return (name, _ARRAY, _POD_CODE[pod], extent, meta, self.group(refs), (len(rows), sampling))

    def array(self, name: str, values, pod: str, meta: str = "") -> tuple:
        arr = np.ascontiguousarray(np.asarray(values, pod))
        extent = arr.shape[1] if arr.ndim == 2 else 1
        return (name, _ARRAY, _POD_CODE[pod], extent, meta, self.group([self._sample(arr.tobytes()), _DATA]))

    def _sample(self, raw: bytes) -> int:
        """A sample, led by its key. Identical samples are stored once, as Alembic does."""
        key = hashlib.md5(raw).digest()
        if (key, len(raw)) not in self.samples:
            self.samples[(key, len(raw))] = self.data(key + raw)
        return self.samples[(key, len(raw))]

    def compound(self, name: str, props: list[tuple], meta: str = "") -> tuple:
        return (name, _COMPOUND, 0, 0, meta, self.compound_group(props))

    def compound_group(self, props: list[tuple]) -> int:
        if not props:
            return 0
        headers = bytearray()
        for name, kind, pod, extent, meta, _, *timed in props:
            samples, sampling = timed[0] if timed else (1, 0)
            raw_name, raw_meta = name.encode("utf-8"), meta.encode("utf-8")
            biggest = max(len(raw_name), len(raw_meta), samples, 1)
            hint = 0 if biggest < 256 else (1 if biggest < 65536 else 2)
            size = (1, 2, 4)[hint]
            index = self.meta_index(meta)
            info = kind | hint << 2 | pod << 4 | index << 20
            if kind != _COMPOUND:
                # homogenous; one sample is constant, several change from the second to the last
                info |= 0x400 | min(extent, 255) << 12 | (0x800 if samples == 1 else 0) | (0x100 if sampling else 0)
            headers += struct.pack("<I", info)
            if kind != _COMPOUND:
                headers += samples.to_bytes(size, "little")
                if sampling:
                    headers += sampling.to_bytes(size, "little")
            headers += len(raw_name).to_bytes(size, "little") + raw_name
            if index == 0xFF:
                headers += len(raw_meta).to_bytes(size, "little") + raw_meta
        return self.group([p[5] for p in props] + [self.data(bytes(headers))])

    # -- objects: (name, metadata, group)

    def object(self, props: list[tuple], children: list[tuple]) -> int:
        headers = bytearray()
        for name, meta, _ in children:
            raw_name, raw_meta = name.encode("utf-8"), meta.encode("utf-8")
            index = self.meta_index(meta)
            headers += struct.pack("<I", len(raw_name)) + raw_name + bytes([index])
            if index == 0xFF:
                headers += struct.pack("<I", len(raw_meta)) + raw_meta
        headers += hashlib.md5(bytes(headers)).digest() * 2
        return self.group([self.compound_group(props), *(c[2] for c in children), self.data(bytes(headers))])

    def write(self, scene: Scene) -> bytes:
        """Skins and blend shapes are applied where they stand (Alembic has neither); a mesh they
        bend over time gets its points every frame instead."""
        flat = scene.deformed()
        animation = scene.animations[0] if scene.animations else None
        flat_animation = flat.animations[0] if flat.animations else None
        for c in flat_animation.channels if flat_animation is not None else []:
            if c.path in ("translation", "rotation", "scale") and len(c.times):
                self.moving.setdefault(id(c.node), []).append(c)
        bending = _bending(scene, animation)
        times = [c.times for cs in self.moving.values() for c in cs]
        times += [c.times for c in animation.channels if bending and len(c.times)] if animation else []
        if times:  # one uniform time sampling: a sample every frame, start to end
            times = np.concatenate(times)
            self.fps, start, end = frame_rate(times), float(times.min()), float(times.max())
            if (end - start) * self.fps >= MAX_FRAMES:
                raise too_many_frames((end - start) * self.fps)
            self.frames = start + np.arange(round((end - start) * self.fps) + 1) / self.fps
        if bending:
            copies = dict(zip((id(n) for n, _ in scene.walk()), (n for n, _ in flat.walk())))
            for t in self.frames:
                worlds = {id(n): w for n, w in scene.walk_at(float(t), animation)}
                for node in bending:
                    points = node.mesh.morphed(animation.weights(node, float(t)))
                    if node.skin is not None:
                        joints = np.array([worlds[id(j)] for j in node.skin.joints]).reshape(-1, 4, 4)
                        inverse = np.linalg.inv(worlds[id(node)])
                        points = node.skin.pose(points, joints) @ inverse[:3, :3].T + inverse[:3, 3]
                    self.bent.setdefault(id(copies[id(node)]), []).append(points)
        taken: set[str] = set()
        return self.finish([self._node(n, taken) for n in flat.roots])

    def finish(self, top_objects: list[tuple]) -> bytes:
        """The archive around the top objects: versions, metadata, the time samplings (index 0 a
        single sample at time 0; 1, when anything moves, a sample every frame) and the metadata table."""
        root = self.object([], top_objects)
        times = struct.pack("<IdI", 1, 1.0, 1) + struct.pack("<d", 0.0)
        if self.frames is not None:
            times += struct.pack("<IdI", len(self.frames), 1.0 / self.fps, 1) + struct.pack("<d", self.frames[0])
        table = b"".join(bytes([len(m.encode("utf-8"))]) + m.encode("utf-8") for m in self.metas)
        archive = self.group([self.data(struct.pack("<i", 0)), self.data(struct.pack("<i", 10709)), root,
                              self.data(b"_ai_Application=meshport"), self.data(times), self.data(table)])
        self.out[8:16] = struct.pack("<Q", archive)
        return bytes(self.out)

    def _node(self, top: Node, taken: set[str]) -> tuple:
        """The object for a node and everything under it, each object written after the ones it
        holds. A loop rather than recursion, so no depth of nesting can overflow the stack."""
        def start(node: Node, taken: set[str]) -> list:  # [node, name, xform, child names, children, next child]
            name = _unique(node.name or "node", taken)
            if id(node) in self.moving:
                vals = self.scalar_samples(".vals", [m.T.ravel() for m in self._poses(node)], "<f8", 1)
            else:
                vals = self.scalar(".vals", node.matrix.T.ravel(), "<f8")
            xform = self.compound(".xform", [self.scalar(".inherits", [1], "bool"),
                                             self.scalar(".ops", [0x30], "u1"), vals], _XFORM_PROP)
            kids: set[str] = set()
            children = []
            if node.mesh is not None and len(node.mesh.corners):
                children.append(self._mesh(_unique(f"{name}Shape", kids), node.mesh, node.materials,
                                           self.bent.get(id(node))))
            return [node, name, xform, kids, children, 0]
        stack = [start(top, taken)]
        while True:
            frame = stack[-1]
            node, name, xform, kids, children, i = frame
            if i < len(node.children):
                frame[5] = i + 1
                stack.append(start(node.children[i], kids))
                continue
            stack.pop()
            done = (name, _XFORM, self.object([xform], children))
            if not stack:
                return done
            stack[-1][4].append(done)

    def _poses(self, node: Node) -> list[np.ndarray]:
        """The node's local matrix at every frame."""
        t0, q0, s0 = decompose_trs(node.matrix)
        trs = {"translation": np.tile(t0, (len(self.frames), 1)), "rotation": np.tile(q0, (len(self.frames), 1)),
               "scale": np.tile(s0, (len(self.frames), 1))}
        for c in self.moving[id(node)]:  # as Animation.local_matrix: a later channel for the same property wins
            trs[c.path] = c.sample(self.frames)
        return [compose_trs(t, q, s) for t, q, s in zip(trs["translation"], trs["rotation"], trs["scale"])]

    def _mesh(self, name: str, mesh: Mesh, materials: list[Material], frames: list | None = None) -> tuple:
        flip = _reversed(mesh.face_sizes)  # counter-clockwise -> Alembic's clockwise
        pts = np.asarray(mesh.positions, np.float32)
        lo, hi = pts.min(0), pts.max(0)
        point_meta = "geoScope=vtx;interpretation=point"
        if frames:  # points (and bounds) every frame
            bounds = self.scalar_samples(".selfBnds", [np.concatenate([f.min(0), f.max(0)]) for f in frames], "<f8", 1)
            props = [bounds[:4] + ("interpretation=box",) + bounds[5:],
                     self.array_samples("P", [np.asarray(f, np.float32) for f in frames], "<f4", 1, point_meta)]
        else:
            props = [self.scalar(".selfBnds", np.concatenate([lo, hi]), "<f8", "interpretation=box"),
                     self.array("P", pts, "<f4", point_meta)]
        props += [
                  self.array(".faceIndices", np.asarray(mesh.corners, np.int32)[flip], "<i4"),
                 self.array(".faceCounts", np.asarray(mesh.face_sizes, np.int32), "<i4")]
        param = "arrayExtent=1;geoScope=fvr;isGeomParam=true;podName=float32_t;"
        if mesh.normals is not None:
            props.append(self.array("N", np.asarray(mesh.normals, np.float32)[flip], "<f4",
                                    param + "interpretation=normal;podExtent=3"))
        arb = []
        for i, uv in enumerate(mesh.uvs):
            vals, index = _indexed(np.asarray(uv, np.float32)[flip])
            prop = self.compound("uv" if i == 0 else f"uv{i}", [self.array(".vals", vals, "<f4"),
                                                               self.array(".indices", index, "<u4")],
                                 param + "interpretation=vector;podExtent=2")
            (props if i == 0 else arb).append(prop)
        if mesh.colors is not None:
            arb.append(self.array("Cd", np.asarray(mesh.colors, np.float32)[flip], "<f4",
                                  param + "interpretation=rgba;podExtent=4"))
        if arb:
            props.append(self.compound(".arbGeomParams", arb))
        sets = []
        if materials and mesh.face_materials is not None and len(materials) > 1:
            slots = np.clip(mesh.face_materials, 0, len(materials) - 1)
            taken: set[str] = set()
            for slot in np.unique(slots):
                faces = np.nonzero(slots == slot)[0].astype(np.int32)
                used = pts[mesh.corners[np.repeat(slots == slot, mesh.face_sizes)]]
                bounds = self.scalar(".selfBnds", np.concatenate([used.min(0), used.max(0)]), "<f8",
                                     "interpretation=box")  # Alembic's face-set schema needs these
                fs = self.compound(".faceset", [bounds, self.array(".faces", faces, "<i4")], _FACESET_PROP)
                sets.append((_unique(materials[int(slot)].name or "material", taken), _FACESET, self.object([fs], [])))
        return name, _MESH, self.object([self.compound(".geom", props, _MESH_PROP)], sets)


def _indexed(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .geometry import unique_rows
    first, index = unique_rows(np.ascontiguousarray(values))
    return values[first], index.astype(np.uint32)


def _unique(name: str, taken: set[str]) -> str:
    base = name.replace("/", "_") or "node"
    out, i = base, 1
    while out in taken:
        out = f"{base}_{i}"
        i += 1
    taken.add(out)
    return out


def _bending(scene: Scene, animation) -> list[Node]:
    """The mesh nodes whose shape the animation changes: their blend shape weights move, or a joint
    of their skin (or a node above one) does."""
    if animation is None:
        return []
    moving = {id(c.node) for c in animation.channels if c.path in ("translation", "rotation", "scale")}
    weighted = {id(c.node) for c in animation.channels if c.path == WEIGHTS}
    parent = {id(c): n for n, _ in scene.walk() for c in n.children}

    def moves(n) -> bool:
        while n is not None:
            if id(n) in moving:
                return True
            n = parent.get(id(n))
        return False
    return [n for n, _ in scene.walk() if n.mesh is not None and len(n.mesh.corners) and (
        (n.mesh.targets and id(n) in weighted) or (n.skin is not None and any(moves(j) for j in n.skin.joints)))]


def write(scene: Scene) -> bytes:
    """Alembic has no skins or blend shapes: they're applied where they stand, and the meshes'
    nodes keep their animation; a mesh they bend over time gets its points every frame."""
    return _Writer().write(scene)


def write_file(scene: Scene, path: str | os.PathLike) -> None:
    with open(path, "wb") as f:
        f.write(write(scene))
