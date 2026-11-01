"""Blender (.blend) -> Scene. Read only.

A .blend file is Blender's memory written out in blocks: each starts with a code, the index of
the struct it holds, the address it had in memory, and a length. The file also carries its own
catalogue of every struct's layout (SDNA), so fields are found by name in files from any
version. Blender 5 writes a longer file header and block header, and keeps addresses unique only
within each datablock, so a pointer is looked up among its own datablock's blocks first.
Compressed files (zstd since Blender 3.0, gzip before) need Python 3.14, the backports.zstd
package or zstandard for zstd.

What comes across, from the scene Blender had open:
- every mesh, curve, text and metaball object in the scene's collections (blendcurve.py turns
  the last three into meshes), with its parent, location, rotation (every mode) and scale (delta
  transforms included), and empties that instance a collection;
- the modifiers that are on in the viewport, in order (blendmods.py has which ones);
- meshes stored all three ways Blender has used: attribute storage (5.0+), named layers
  (3.6 to 4.x) and the older vertex, polygon and loop arrays (2.8 to 3.5; checked with files
  from 2.83, 2.93, 3.3 and 3.5); every UV map, color attributes, per-face materials, and flat or
  smooth shading;
- materials from their Principled BSDF (base color, metallic, roughness, alpha, emission) with an
  image texture on Base Color or a Normal Map, packed in the file or next to it;
- each object's action (blendanim.py) as node channels at the scene's frame rate: location,
  rotation in every mode, scale and the delta transforms;
- armatures, their bones as nodes posed by the action's pose bone curves, meshes they deform by
  vertex groups as skins, and shape keys as blend shapes, their values animated by the shape
  keys' own action (blendskin.py).

read(..., modifiers=False) leaves modifiers off, as Blender's glTF export does by default. NURBS
surface, grease pencil, hair, point cloud and volume objects are skipped. Blender's axes (Z up)
become Y up.
"""

from __future__ import annotations

import io
import math
import os
import re
import struct
import zlib
from bisect import bisect_right
from pathlib import Path

import numpy as np

from . import blendanim, blendcurve, blendmods, blendskin
from .geometry import decompose_many, euler, scaling, translation
from .scene import (
    DAMAGED,
    MAX_FRAMES,
    WEIGHTS,
    Animation,
    Budget,
    Channel,
    Image,
    Light,
    Material,
    Mesh,
    MeshportError,
    Node,
    Scene,
    Skin,
    node_budget,
    pack_influences,
)

_Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
_PRIMITIVE = {"char": "i1", "uchar": "u1", "int8_t": "i1", "uint8_t": "u1", "short": "i2", "ushort": "u2",
              "int16_t": "i2", "uint16_t": "u2", "int": "i4", "uint": "u4", "int32_t": "i4", "uint32_t": "u4",
              "float": "f4", "double": "f8", "int64_t": "i8", "uint64_t": "u8", "long": "i4", "ulong": "u4",
              "bool": "u1"}
_EULER = {1: "XYZ", 2: "XZY", 3: "YXZ", 4: "YZX", 5: "ZXY", 6: "ZYX"}
OB_MESH, OB_EMPTY, OB_CURVES_LEGACY, OB_FONT, OB_MBALL, OB_LAMP, OB_ARMATURE = 1, 0, 2, 4, 5, 10, 25
# Light.type and Light.mode; Blender's watts become candela (and W/m^2 lux) at 683 lm/W, as its glTF export has it
_LIGHT_KINDS = {0: "point", 1: "directional", 2: "spot"}
_LA_CUSTOM_DISTANCE = 1 << 20
_LUMENS_PER_WATT = 683.0
_OB_DUPLICOLLECTION = 1 << 8
# CustomData layer types (named layers, Blender 3.x and 4.x)
_CD = {10: ("f4", 1), 11: ("i4", 1), 17: ("u1", 4), 45: ("i1", 1), 46: ("i4", 2), 47: ("f4", 4), 48: ("f4", 3),
       49: ("f4", 2), 50: ("u1", 1), 16: ("f4", 2)}  # 16: MLoopUV (uv[2] then flags)
# Attribute storage data types (Blender 5)
_ATTR = {0: ("u1", 1), 1: ("i1", 1), 2: ("i2", 2), 3: ("i4", 1), 4: ("i4", 2), 5: ("f4", 1), 6: ("f4", 2),
         7: ("f4", 3), 8: ("f4", 16), 9: ("u1", 4), 10: ("f4", 4), 11: ("f4", 4)}
_POINT, _EDGE, _FACE, _CORNER = 0, 1, 2, 3
MAX_INFLATED = 4 << 30  # bytes a compressed .blend may inflate to (and at most 256 times its own size, or 1 GB)
_PIECE = 16 << 20


def _decompress(data: bytes, limit: int | None = None) -> bytes:
    """The file's bytes, inflated if it's compressed, piece by piece up to its limit. With a limit
    given, just that many bytes from the start (for telling what a file is)."""
    head = limit is not None
    if limit is None:
        limit = min(MAX_INFLATED, max(1 << 30, 256 * len(data)))
    try:
        return _inflate(data, limit, head)
    except MeshportError:
        raise
    except Exception as e:  # noqa: BLE001 - each decompressor has its own error types
        raise MeshportError(f"This .blend file is damaged (its compressed data is broken: {e})") from None


def _inflate(data: bytes, limit: int, head: bool) -> bytes:
    out = bytearray()
    step = min(_PIECE, limit) if head else _PIECE

    def take(piece: bytes) -> bool:  # True once there's enough
        out.extend(piece)
        if len(out) > limit and not head:
            raise MeshportError(f"This .blend file is too big once decompressed (over {limit >> 20:,} MB); it may be "
                                "damaged or made to fill the memory")
        return len(out) >= limit and head
    if data[:4] == b"\x28\xb5\x2f\xfd":
        zstd = None
        for name in ("compression.zstd", "backports.zstd"):
            try:
                zstd = __import__(name, fromlist=["ZstdDecompressor"])
                break
            except ImportError:
                continue
        if zstd is None:
            try:
                import zstandard
            except ImportError:
                raise MeshportError("This .blend file is compressed with zstd: install the backports.zstd "
                                    "package (or use Python 3.14) to read it") from None
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data), read_across_frames=True)
            while not take(piece := reader.read(step)) and piece:
                pass
            return bytes(out[:limit] if head else out)
        rest = data
        while rest:  # each frame (Blender writes several, and a table of them in a skippable one)
            d = zstd.ZstdDecompressor()
            piece = d.decompress(rest, step)
            while True:
                if take(piece):
                    return bytes(out[:limit])
                if d.eof:
                    break
                if d.needs_input:
                    raise MeshportError("This .blend file is cut short (its compressed data ends early)")
                piece = d.decompress(b"", step)
            rest = d.unused_data
        return bytes(out)
    if data[:2] == b"\x1f\x8b":
        rest = data
        while rest:  # each gzip member, as gzip.decompress reads them
            d = zlib.decompressobj(31)
            piece = d.decompress(rest, step)
            while True:
                if take(piece):
                    return bytes(out[:limit])
                if d.eof:
                    break
                tail = d.unconsumed_tail
                piece = d.decompress(tail, step)
                if not piece and not tail:
                    raise MeshportError("This .blend file is cut short (its compressed data ends early)")
            rest = d.unused_data.lstrip(b"\0")
        return bytes(out)
    return data


# ============================================================================ blocks and SDNA

class _Block:
    __slots__ = ("code", "count", "index", "length", "offset", "old", "owner", "sdna")

    def __init__(self, code, sdna, old, length, count, offset, owner, index):
        self.code, self.sdna, self.old, self.length, self.count, self.offset = code, sdna, old, length, count, offset
        self.owner, self.index = owner, index  # owner: the datablock (ID) block this one belongs to

    @property
    def scope(self) -> int:
        """The datablock whose address space pointers stored in this block use."""
        return self.owner if self.code == b"DATA" else self.index


class _File:
    def __init__(self, data: bytes):
        data = _decompress(data)
        if data[:7] != b"BLENDER":
            raise MeshportError("This isn't a Blender file")
        if data[7:9].isdigit():  # Blender 5: "BLENDER17-01v0501"
            head = int(data[7:9])
            ptr, endian, self.version = 8, chr(data[12]), int(data[13:17])
            large = True
        else:  # "BLENDER-v405"
            head, ptr, endian, self.version = 12, 8 if data[7:8] == b"-" else 4, chr(data[8]), int(data[9:12])
            large = False
        if endian != "v":
            raise MeshportError("This .blend file is from a big-endian computer (a PowerPC Mac), "
                                "which meshport can't read")
        self.data, self.ptr = data, ptr
        self.ptr_dtype = "<u8" if ptr == 8 else "<u4"
        if large:
            fmt, size = "<4siQqq", 32
        elif ptr == 8:
            fmt, size = "<4siQii", 24
        else:
            fmt, size = "<4siIii", 20
        self.blocks: list[_Block] = []
        pos, owner = head, -1
        while pos + size <= len(data):
            if large:
                code, sdna, old, length, count = struct.unpack_from(fmt, data, pos)
            else:
                code, length, old, sdna, count = struct.unpack_from(fmt, data, pos)
            if code == b"ENDB":
                break
            if length < 0 or pos + size + length > len(data):
                raise MeshportError("This .blend file is damaged or cut short")
            if code != b"DATA":
                owner = len(self.blocks)
            self.blocks.append(_Block(code, sdna, old, length, count, pos + size, owner, len(self.blocks)))
            pos += size + length
        dna = next((b for b in self.blocks if b.code == b"DNA1"), None)
        if dna is None:
            raise MeshportError("This .blend file is damaged (it has no struct catalogue)")
        self._sdna(dna.offset)
        # addresses: ID blocks are unique; data blocks are looked up within their datablock first
        self.ids: dict[int, _Block] = {}
        self.scoped: dict[tuple[int, int], _Block] = {}
        self.any: dict[int, _Block] = {}
        for b in self.blocks:
            if b.code == b"DATA":
                self.scoped.setdefault((b.owner, b.old), b)
            else:
                self.ids.setdefault(b.old, b)
            self.any.setdefault(b.old, b)
        self.sorted_old = sorted(self.any)

    def _sdna(self, at: int) -> None:
        d = self.data

        def tag(name: bytes, pos: int) -> int:
            if d[pos:pos + 4] != name:
                raise MeshportError("This .blend file is damaged (its struct catalogue is unreadable)")
            return pos + 4

        def align(pos: int) -> int:  # sections start on 4-byte boundaries from the catalogue's start
            return at + ((pos - at + 3) & ~3)

        def strings(pos: int) -> tuple[list[str], int]:
            n = struct.unpack_from("<i", d, pos)[0]
            pos += 4
            out = []
            for _ in range(n):
                end = d.index(b"\0", pos)
                out.append(d[pos:end].decode("latin-1"))
                pos = end + 1
            return out, align(pos)

        pos = tag(b"SDNA", at)
        names, pos = strings(tag(b"NAME", pos))
        types, pos = strings(tag(b"TYPE", pos))
        pos = tag(b"TLEN", pos)
        tlen = struct.unpack_from(f"<{len(types)}h", d, pos)
        pos = align(pos + 2 * len(types))
        pos = tag(b"STRC", pos)
        n = struct.unpack_from("<i", d, pos)[0]
        pos += 4
        self.structs: list[tuple[str, dict]] = []
        self.struct_index: dict[str, int] = {}
        self.struct_size: dict[str, int] = {}
        for _ in range(n):
            t, nf = struct.unpack_from("<hh", d, pos)
            pos += 4
            pairs = np.frombuffer(d, "<i2", nf * 2, pos).reshape(nf, 2)
            pos += nf * 4
            fields, off = {}, 0
            for ft, fn in pairs.tolist():
                raw = names[fn]
                is_ptr = raw.startswith(("*", "(*"))
                dims = [int(x) for x in re.findall(r"\[(\d+)\]", raw)]
                clean = re.sub(r"\[.*", "", raw).strip("*()")
                count = math.prod(dims) if dims else 1
                elem = self.ptr if is_ptr else tlen[ft]
                fields[clean] = (off, types[ft], is_ptr, dims, raw.startswith("**"))
                off += elem * count
            self.struct_index[types[t]] = len(self.structs)
            self.structs.append((types[t], fields))
            self.struct_size[types[t]] = tlen[t]

    # -- pointers

    def at(self, address: int, scope: int = -1) -> tuple[_Block, int] | None:
        """The block an address points into, and the offset within it (for pointers into the
        middle of an array)."""
        if not address:
            return None
        b = self.scoped.get((scope, address)) or self.ids.get(address) or self.any.get(address)
        if b is not None:
            return b, 0
        i = bisect_right(self.sorted_old, address) - 1
        if i >= 0:
            b = self.any[self.sorted_old[i]]
            if address - b.old < b.length:
                return b, address - b.old
        return None

    def struct_of(self, b: _Block) -> str:
        if b.sdna < 0 or b.sdna >= len(self.structs):
            raise MeshportError("This .blend file is damaged (a block names an unknown struct)")
        return self.structs[b.sdna][0]


class _S:
    """One struct in the file: fields by name, pointers followed on request."""
    __slots__ = ("f", "fields", "name", "offset", "owner")

    def __init__(self, f: _File, name: str, offset: int, owner: int):
        self.f, self.name, self.offset, self.owner = f, name, offset, owner
        idx = f.struct_index.get(name)
        if idx is None:
            raise MeshportError(f"This .blend file has no {name} struct")
        self.fields = f.structs[idx][1]

    def has(self, field: str) -> bool:
        return field in self.fields

    def get(self, field: str, default=None):
        spec = self.fields.get(field)
        if spec is None:
            return default
        off, t, is_ptr, dims, _ = spec
        at = self.offset + off
        d = self.f.data
        if is_ptr:
            if dims:
                return np.frombuffer(d, self.f.ptr_dtype, math.prod(dims), at).astype(np.uint64).tolist()
            return int(np.frombuffer(d, self.f.ptr_dtype, 1, at)[0])
        if t in _PRIMITIVE:
            dt = np.dtype("<" + _PRIMITIVE[t])
            if t == "char" and dims:
                raw = d[at:at + math.prod(dims)]
                return raw.split(b"\0", 1)[0].decode("utf-8", "replace")
            if dims:
                return np.frombuffer(d, dt, math.prod(dims), at).reshape(dims).astype(dt.newbyteorder("="))
            return np.frombuffer(d, dt, 1, at)[0].item()
        if t in self.f.struct_index and not dims:
            return _S(self.f, t, at, self.owner)
        return default

    def follow(self, field: str, struct_name: str | None = None) -> _S | None:
        """The struct a pointer field points to: as given, else the field's declared type, else
        what the block says it holds."""
        got = self.f.at(self.get(field, 0), self.owner)
        if got is None:
            return None
        b, off = got
        declared = self.fields[field][1] if field in self.fields else ""
        name = struct_name or (declared if declared in self.f.struct_index and declared not in ("ID", "void")
                               else self.f.struct_of(b))
        return _S(self.f, name, b.offset + off, b.scope)

    def raw(self, field: str, dtype: str, count: int) -> np.ndarray | None:
        """count values of dtype from the block a pointer field points to."""
        return self.f_raw(self.get(field, 0), dtype, count)

    def f_raw(self, address: int, dtype: str, count: int) -> np.ndarray | None:
        got = self.f.at(address, self.owner)
        if got is None or count <= 0:
            return None
        b, off = got
        dt = np.dtype(dtype)
        if off + dt.itemsize * count > b.length:
            raise MeshportError("This .blend file is damaged (an array is shorter than it says)")
        return np.frombuffer(self.f.data, dt, count, b.offset + off)

    def listbase(self, field: str, struct_name: str | None = None):
        lb = self.get(field)
        if lb is None:
            return
        address, seen = lb.get("first", 0), set()
        while address and address not in seen and len(seen) < 1_000_000:
            seen.add(address)
            got = self.f.at(address, self.owner)
            if got is None:
                return
            b, off = got
            item = _S(self.f, struct_name or self.f.struct_of(b), b.offset + off, b.scope)
            yield item
            # structs like modifiers keep their links in a header struct, their first field
            head = item if item.has("next") else item.get(next(iter(item.fields), ""))
            address = head.get("next", 0) if isinstance(head, _S) else 0

    def id_name(self) -> str:
        ident = self.get("id")
        name = ident.get("name", "") if ident is not None else ""
        return name[2:] if len(name) > 2 else name


# ============================================================================ scene

def read(data: bytes, base_dir: str | os.PathLike | None = None, modifiers: bool = True) -> Scene:
    """modifiers=False reads meshes as stored, without their modifiers."""
    try:
        return _Reader(_File(data), Path(base_dir) if base_dir else None, modifiers).scene()
    except MeshportError:
        raise
    except (*DAMAGED, struct.error, OSError, EOFError) as e:
        raise MeshportError(f"This .blend file is damaged ({type(e).__name__}: {e})") from None


def read_file(path: str | os.PathLike, modifiers: bool = True) -> Scene:
    path = Path(path)
    return read(path.read_bytes(), path.parent, modifiers)


class _Reader:
    def __init__(self, f: _File, base_dir: Path | None, modifiers: bool = True):
        self.f, self.base_dir, self.modifiers = f, base_dir, modifiers
        self.raw: dict[int, tuple[Mesh, np.ndarray] | None] = {}
        self.capping: set[int] = set()  # objects whose mesh is being made for an Array's cap
        self.meshes: dict[int, Mesh | None] = {}
        self.worlds: dict[int, np.ndarray] = {}
        self.materials: dict[int, Material] = {}
        self.images: dict[int, Image | None] = {}
        self.rigs: dict[int, blendskin.Rig] = {}  # armature object -> its bones
        self.nodes = node_budget()  # collection instances can show the same objects again and again
        self.generated = Budget(blendmods.MAX_GENERATED, "This .blend file is too big once its modifiers and curves "
                                f"are applied (over {blendmods.MAX_GENERATED:,} corners); it may be made to stall "
                                "the reader")

    def _id(self, address: int, struct_name: str) -> _S | None:
        b = self.f.ids.get(address)
        if b is None:
            return None
        return _S(self.f, struct_name, b.offset, b.index)

    def scene(self) -> Scene:
        glob = next((b for b in self.f.blocks if b.code == b"GLOB"), None)
        scene = None
        if glob is not None:
            scene = self._id(_S(self.f, "FileGlobal", glob.offset, -1).get("curscene", 0), "Scene")
        if scene is None:
            sc = next((b for b in self.f.blocks if b.code == b"SC\x00\x00"), None)
            if sc is None:
                raise MeshportError("This .blend file has no scene")
            scene = _S(self.f, "Scene", sc.offset, sc.index)
        objects: dict[int, _S] = {}
        master = scene.follow("master_collection", "Collection")
        if master is not None:
            self._collect(master, objects, set())
        # parents first, then children under them; an object whose parent isn't in the scene stays at the top
        self.objects = objects
        nodes: dict[int, Node] = {}
        worlds = self.worlds
        for address, ob in objects.items():
            self._world(address, ob, worlds)
        roots = []
        for address, ob in objects.items():
            node = self._object_node(ob, address)
            nodes[address] = node
        for address, ob in objects.items():
            parent = ob.get("parent", 0)
            node = nodes[address]
            if parent in nodes:
                node.matrix = np.linalg.inv(worlds[parent]) @ worlds[address]
                nodes[parent].children.append(node)
            else:
                node.matrix = worlds[address]
                roots.append(node)
        for node in roots:
            node.matrix = _Z_UP_TO_Y_UP @ node.matrix
        for address, ob in objects.items():
            if ob.get("type", -1) == OB_MESH and nodes[address].mesh is not None:
                try:
                    self._deform(address, ob, nodes[address])
                except (MeshportError, *DAMAGED):
                    nodes[address].skin = None  # damaged deform data: the mesh as it's stored
        bones = {id(n) for rig in self.rigs.values() for n in rig.nodes.values()}
        roots = [n for n in roots if _keep(n, bones)]
        result = Scene(roots)
        if not result.meshes:
            raise MeshportError("No 3D geometry found in this file (meshport reads mesh, curve, text and "
                                "metaball objects)")
        result.animations = self._animations(scene, objects, nodes)
        return result

    def _animations(self, scene: _S, objects: dict, nodes: dict) -> list[Animation]:
        """The objects' actions as one animation, at the scene's frame rate (frame f is f / fps
        seconds, as Blender's own exports have it)."""
        fps = _frame_rate(scene)
        channels = []
        for address, ob in objects.items():
            if not ob.get("adt", 0):
                continue
            try:
                curves = blendanim.fcurves(self.f, ob.follow("adt", "AnimData"))
                if not curves:
                    continue
                parent = ob.get("parent", 0)
                parentinv = np.asarray(ob.get("parentinv"), np.float64).T if parent else np.eye(4)
                if parent in nodes:  # its node's matrix is relative to its parent's
                    before = parentinv
                else:
                    before = _Z_UP_TO_Y_UP @ (self.worlds[parent] @ parentinv if parent in self.worlds else np.eye(4))
                channels += _channels(ob, nodes[address], curves, fps, before, scene)
                if address in self.rigs:
                    channels += _bone_channels(self.rigs[address], curves, fps, scene)
            except (MeshportError, *DAMAGED):
                continue  # a damaged action loses this object's animation, not the file
        for address, ob in objects.items():  # shape key values, from each mesh's keys' own action
            node = nodes[address]
            if node.mesh is None or not node.mesh.targets or ob.get("type", -1) != OB_MESH:
                continue
            try:
                me = self._id(ob.get("data", 0), "Mesh")
                key = self._id(me.get("key", 0), "Key") if me is not None else None
                if key is not None and key.get("adt", 0):
                    weights = _weight_channel(node, blendanim.fcurves(self.f, key.follow("adt", "AnimData")),
                                              fps, scene)
                    if weights is not None:
                        channels.append(weights)
            except (MeshportError, *DAMAGED):
                continue
        if not channels:
            return []
        return [Animation(scene.id_name() or "animation", channels)]

    def _collect(self, collection: _S, out: dict[int, _S], seen: set[int]) -> None:
        key = collection.offset
        if key in seen or len(seen) > 100_000:
            return
        seen.add(key)
        for co in collection.listbase("gobject", "CollectionObject"):
            address = co.get("ob", 0)
            ob = self._id(address, "Object")
            if ob is not None:
                out.setdefault(address, ob)
        for child in collection.listbase("children", "CollectionChild"):
            sub = self._id(child.get("collection", 0), "Collection")
            if sub is not None:
                self._collect(sub, out, seen)

    # -- transforms

    def _world(self, address: int, ob: _S, worlds: dict[int, np.ndarray], depth: int = 0) -> np.ndarray:
        if address in worlds:
            return worlds[address]
        if depth > 1000:
            raise MeshportError("This .blend file is damaged (its parents loop)")
        local = _local_matrix(ob)
        parent_address = ob.get("parent", 0)
        parent = self._id(parent_address, "Object") if parent_address else None
        if parent is not None:
            parentinv = np.asarray(ob.get("parentinv"), np.float64).T  # stored column by column
            world = self._world(parent_address, parent, worlds, depth + 1) @ parentinv @ local
        else:
            world = local
        worlds[address] = world
        return world

    def _object_node(self, ob: _S, address: int) -> Node:
        self.nodes.spend()
        node = Node(ob.id_name() or "object")
        kind = ob.get("type", -1)
        got = None
        if kind == OB_MESH:
            got = self._geometry(ob, address, "Mesh", _read_mesh)
        elif kind in (OB_CURVES_LEGACY, OB_FONT):
            got = self._geometry(ob, address, "Curve", lambda cu: blendcurve.curve_mesh(cu, self.base_dir))
        elif kind == OB_MBALL:
            got = self._geometry(ob, address, "MetaBall", lambda mb: self._metaball(ob, address, mb), per_object=True)
        if got is not None and got[0] is not None:
            node.mesh, node.materials = got
        elif kind == OB_EMPTY and ob.get("transflag", 0) & _OB_DUPLICOLLECTION:
            node.children += self._instances(ob, set())
        elif kind == OB_LAMP:
            node.light = self._light(ob)
        elif kind == OB_ARMATURE:
            arm = self._id(ob.get("data", 0), "bArmature")
            if arm is not None:
                try:
                    rig = blendskin.Rig(ob, arm)
                except (MeshportError, *DAMAGED):
                    rig = None
                if rig is not None:
                    self.rigs[address] = rig
                    node.children += rig.roots
        return node

    # -- armatures and shape keys

    def _deform(self, address: int, ob: _S, node: Node) -> None:
        """The mesh's shape keys and its armature's skin, when its modifiers left its vertices as
        they are (what they're keyed and weighted by still fits)."""
        me = self._id(ob.get("data", 0), "Mesh")
        count = int(me.get("totvert", me.get("verts_num", 0)) or 0) if me is not None else 0
        if me is None or len(node.mesh.positions) != count:
            return
        keys = blendskin.shape_keys(self.f, ob, me, count)
        if keys and not node.mesh.targets:
            node.mesh.targets, node.mesh.target_weights = blendskin.morphs(keys)
        md = blendskin.armature_modifier(ob)
        rig = self.rigs.get(md.get("object", 0)) if md is not None else None
        if rig is None or md.get("object", 0) not in self.worlds:
            return
        got = blendskin.vertex_weights(self.f, me, count)
        if got is None:
            return
        rows, groups, weights = got
        names = blendskin.group_names(ob, me)
        bones = [n for n in dict.fromkeys(names) if n in rig.nodes]
        joint_of = np.array([bones.index(n) if n in rig.nodes else -1 for n in names] + [-1], np.int64)
        joints = joint_of[np.where((groups >= 0) & (groups < len(names)), groups, len(names))]
        keep = (joints >= 0) & np.isfinite(weights) & (weights > 0)
        rows, joints, weights = rows[keep], joints[keep], weights[keep]
        total = np.bincount(rows, weights, count)
        weights = weights / total[rows]  # the modifier normalizes each vertex's weights
        to_mesh = np.linalg.inv(self.worlds[md.get("object", 0)]) @ self.worlds[address]
        binds = [np.linalg.inv(rig.rest[b]) @ to_mesh for b in bones]
        nodes = [rig.nodes[b] for b in bones]
        free = np.flatnonzero(total <= 0)
        if len(free):  # vertices no bone moves stay with the mesh object
            rows, weights = np.concatenate([rows, free]), np.concatenate([weights, np.ones(len(free))])
            joints = np.concatenate([joints, np.full(len(free), len(nodes))])
            nodes.append(node)
            binds.append(np.eye(4))
        if not bones:
            return
        indices, packed = pack_influences(rows, joints, weights, count)
        skin = Skin(nodes, np.array(binds), indices, packed, rig_name(self.objects.get(md.get("object", 0))))
        skin.validate(node.mesh)
        node.skin = skin

    def _instances(self, ob: _S, seen: set[int]) -> list[Node]:
        """An empty that shows a collection: that collection's objects, under the empty."""
        field = "instance_collection" if ob.has("instance_collection") else "dup_group"
        address = ob.get(field, 0)
        coll = self._id(address, "Collection")
        if coll is None or address in seen or len(seen) > 64:
            return []
        objects: dict[int, _S] = {}
        self._collect(coll, objects, set())
        offset = np.asarray(coll.get("instance_offset", coll.get("dupli_ofs", np.zeros(3))), np.float64)
        out = []
        for address_ob, member in objects.items():
            world = translation(-offset) @ self._world(address_ob, member, self.worlds)
            node = self._object_node(member, address_ob)
            node.matrix = world
            if member.get("type", -1) == OB_EMPTY and member.get("transflag", 0) & _OB_DUPLICOLLECTION:
                node.children = self._instances(member, seen | {address})
            if _keep(node, {id(n) for rig in self.rigs.values() for n in rig.nodes.values()}):
                out.append(node)
        return out

    # -- meshes

    def _geometry(self, ob: _S, ob_address: int, struct_name: str, build, per_object: bool = False):
        """(mesh, material slots) for an object whose data (a Mesh, Curve or MetaBall) build turns
        into a mesh and smooth flags, with the object's modifiers applied."""
        address = ob.get("data", 0)
        me = self._id(address, struct_name)
        if me is None:
            return None
        key = ("object", ob_address) if per_object else address
        if key not in self.raw:
            self.raw[key] = build(me)
            if self.raw[key] is not None and struct_name != "Mesh":  # curves and metaballs: meshes meshport makes
                self.generated.spend(len(self.raw[key][0].corners))
        raw = self.raw[key]
        if raw is None:
            return None
        slots = self._slots(ob, me)
        modifiers = blendmods.stack(ob) if self.modifiers else []
        if modifiers:  # this object's own copy
            ctx = blendmods.Context(self._world(ob_address, ob, self.worlds), self._world_of, len(slots),
                                    self.generated, lambda a: self._cap_mesh(a, ob_address))
            mesh, smooth = blendmods.apply(*raw, modifiers, ctx)
            self.generated.spend(len(mesh.corners))
            return (_finish(mesh, smooth) if len(mesh.face_sizes) else None), slots
        if key not in self.meshes:
            self.meshes[key] = _finish(*raw)
        return self.meshes[key], slots

    def _metaball(self, ob: _S, ob_address: int, mb: _S):
        """The surface of every metaball in this object's family, if it's the family's base
        object (the others show nothing)."""
        name = ob.id_name()
        fam = blendcurve.family(name)
        members = [(a, o) for a, o in getattr(self, "objects", {}).items()
                   if o.get("type", -1) == OB_MBALL and blendcurve.family(o.id_name()) == fam]
        if ob_address not in dict(members):
            members.append((ob_address, ob))
        base = next((a for a, o in members if o.id_name() == fam), min(members, key=lambda m: m[1].id_name())[0])
        if base != ob_address:
            return None
        inverse = np.linalg.inv(self._world(ob_address, ob, self.worlds))
        elements = []
        for address, member in members:
            data = self._id(member.get("data", 0), "MetaBall")
            if data is None:
                continue
            to_base = inverse @ self._world(address, member, self.worlds)
            for el in data.listbase("elems", "MetaElem"):
                local = translation([float(el.get(k, 0.0)) for k in "xyz"]) @ _quat(el.get("quat", np.array([1, 0, 0, 0.0])))
                elements.append((int(el.get("type", 0)), int(el.get("flag", 0)), to_base @ local,
                                 float(el.get("rad", 1.0)), float(el.get("s", 2.0)),
                                 np.array([float(el.get(k, 1.0)) for k in ("expx", "expy", "expz")])))
        return blendcurve.metaball_mesh(elements, float(mb.get("wiresize", 0.4)), float(mb.get("thresh", 0.6)),
                                        mb.id_name() or "metaball")

    def _cap_mesh(self, address: int, user: int):
        """A mesh object's (mesh, smooth flags) with its own modifiers applied, for an Array on another
        object (user) to cap with; None for other kinds of object, the user itself and loops."""
        ob = self._id(address, "Object")
        if ob is None or ob.get("type", -1) != OB_MESH or address == user or address in self.capping:
            return None
        key = ob.get("data", 0)
        me = self._id(key, "Mesh")
        if me is None:
            return None
        if key not in self.raw:
            self.raw[key] = _read_mesh(me)
        raw = self.raw[key]
        modifiers = blendmods.stack(ob) if self.modifiers else []
        if raw is None or not modifiers:
            return raw
        self.capping.add(address)
        try:
            ctx = blendmods.Context(self._world(address, ob, self.worlds), self._world_of, len(self._slots(ob, me)),
                                    self.generated, lambda a: self._cap_mesh(a, address))
            return blendmods.apply(*raw, modifiers, ctx)
        finally:
            self.capping.discard(address)

    def _light(self, ob: _S) -> Light | None:
        """A point, spot or sun light (area lights have no glTF kind and are left out), in glTF's
        units as Blender's glTF export gives them."""
        la = self._id(ob.get("data", 0), "Lamp" if "Lamp" in self.f.struct_index else "Light")
        if la is None:
            return None
        kind = _LIGHT_KINDS.get(int(la.get("type", -1)))
        if kind is None:
            return None
        watts = float(la.get("energy_new", la.get("energy", 10.0)) or 0.0) * 2.0 ** float(la.get("exposure", 0.0) or 0.0)
        per_area = 1.0 if kind == "directional" else 4 * math.pi  # a point's watts spread over the sphere
        light = Light(la.id_name() or "light", kind, tuple(float(la.get(c, 1.0)) for c in "rgb"),
                      watts * _LUMENS_PER_WATT / per_area)
        if int(la.get("mode", 0) or 0) & _LA_CUSTOM_DISTANCE and kind != "directional":
            light.range = float(la.get("att_dist", 0.0)) or None
        if kind == "spot":
            light.outer_cone = float(la.get("spotsize", math.pi / 4)) / 2
            light.inner_cone = light.outer_cone * (1 - min(max(float(la.get("spotblend", 0.15)), 0.0), 1.0))
        return light

    def _world_of(self, address: int) -> np.ndarray | None:
        ob = self._id(address, "Object")
        return None if ob is None else self._world(address, ob, self.worlds)

    def _slots(self, ob: _S, me: _S) -> list[Material]:
        n = max(int(ob.get("totcol", 0) or 0), int(me.get("totcol", 0) or 0))
        ob_mats = ob.raw("mat", self.f.ptr_dtype, int(ob.get("totcol", 0) or 0))
        me_mats = me.raw("mat", self.f.ptr_dtype, int(me.get("totcol", 0) or 0))
        bits = ob.raw("matbits", "u1", int(ob.get("totcol", 0) or 0))
        out = []
        for i in range(n):
            address = 0
            if bits is not None and i < len(bits) and bits[i] and ob_mats is not None:
                address = int(ob_mats[i])
            elif me_mats is not None and i < len(me_mats):
                address = int(me_mats[i])
            out.append(self._material(address))
        return out

    def _material(self, address: int) -> Material:
        if address in self.materials:
            return self.materials[address]
        ma = self._id(address, "Material")
        if ma is None:
            mat = Material("default")
        else:
            mat = Material(ma.id_name() or "material",
                           (float(ma.get("r", 0.8)), float(ma.get("g", 0.8)), float(ma.get("b", 0.8)),
                            float(ma.get("a", 1.0))),
                           metallic=float(ma.get("metallic", 0.0)), roughness=float(ma.get("roughness", 0.5)))
            tree = ma.follow("nodetree", "bNodeTree") if ma.get("nodetree", 0) else None
            if tree is not None:
                self._principled(tree, mat)
        self.materials[address] = mat
        return mat

    def _principled(self, tree: _S, mat: Material) -> None:
        nodes = list(tree.listbase("nodes", "bNode"))
        bsdf = next((n for n in nodes if n.get("idname") == "ShaderNodeBsdfPrincipled"), None)
        if bsdf is None:
            return
        links = {}
        for link in tree.listbase("links", "bNodeLink"):
            links[link.get("tosock", 0)] = link
        sockets = {s.get("identifier") or s.get("name"): (s, s_address)
                   for s, s_address in _sockets(bsdf, "inputs")}

        def value(name: str):
            got = sockets.get(name)
            if got is None:
                return None
            sock = got[0]
            dv = sock.follow("default_value") if sock.get("default_value", 0) else None
            if dv is None:
                return None
            v = dv.get("value")
            return np.ravel(v).astype(np.float64) if v is not None else None

        def texture(name: str, via_normal_map: bool = False) -> Image | None:
            got = sockets.get(name)
            if got is None or got[1] not in links:
                return None
            link = links[got[1]]
            src = link.follow("fromnode", "bNode")
            if src is None:
                return None
            if via_normal_map and src.get("idname") == "ShaderNodeNormalMap":
                inner = {s_addr: s for s, s_addr in _sockets(src, "inputs")}
                for addr in inner:
                    if addr in links:
                        src = links[addr].follow("fromnode", "bNode")
                        break
            if src is None or src.get("idname") != "ShaderNodeTexImage":
                return None
            return self._image(src.get("id", 0))

        base = value("Base Color")
        alpha = value("Alpha")
        a = float(alpha[0]) if alpha is not None and len(alpha) else 1.0
        if base is not None and len(base) >= 3:
            mat.base_color = (*map(float, base[:3]), a)
        else:
            mat.base_color = (*mat.base_color[:3], a)
        for key, attr in (("Metallic", "metallic"), ("Roughness", "roughness")):
            v = value(key)
            if v is not None and len(v):
                setattr(mat, attr, float(v[0]))
        emission = value("Emission Color") if "Emission Color" in sockets else value("Emission")
        strength = value("Emission Strength")
        if emission is not None and len(emission) >= 3:
            s = float(strength[0]) if strength is not None and len(strength) else 1.0
            mat.emissive = tuple(float(c) * s for c in emission[:3])
        tex = texture("Base Color")
        if tex is not None:
            mat.base_color_texture = tex
            mat.base_color = (1.0, 1.0, 1.0, a)
        normal = texture("Normal", via_normal_map=True)
        if normal is not None:
            mat.normal_texture = normal
        if a < 1:
            mat.alpha_mode = "BLEND"

    def _image(self, address: int) -> Image | None:
        if address in self.images:
            return self.images[address]
        im = self._id(address, "Image")
        image = None
        if im is not None:
            name = im.id_name() or "texture"
            path = im.get("name", "") or im.get("filepath", "")
            data = None
            if im.get("packedfile", 0):
                pf = im.follow("packedfile", "PackedFile")
                if pf is not None:
                    raw = pf.raw("data", "u1", int(pf.get("size", 0)))
                    data = raw.tobytes() if raw is not None else None
            if data is None and path:
                data = self._image_file(path)
            image = Image(name, data, path or None) if data or path else None
        self.images[address] = image
        return image

    def _image_file(self, path: str) -> bytes | None:
        from .fbx import TextureFinder
        rel = path.removeprefix("//")  # "//" means "next to the .blend"
        return TextureFinder(self.base_dir).find([rel])


def _keep(node: Node, bones: set[int] = frozenset()) -> bool:
    """Whether a node holds anything (a mesh, or a bone a mesh may follow), dropping the children
    that don't."""
    node.children = [c for c in node.children if _keep(c, bones)]
    return node.mesh is not None or node.light is not None or bool(node.children) or id(node) in bones


def _sockets(node: _S, field: str):
    """(socket, its address) for each socket in a node's input or output list."""
    lb = node.get(field)
    address, seen = (lb.get("first", 0) if lb is not None else 0), set()
    while address and address not in seen and len(seen) < 10_000:
        seen.add(address)
        got = node.f.at(address, node.owner)
        if got is None:
            return
        b, off = got
        sock = _S(node.f, "bNodeSocket", b.offset + off, b.scope)
        yield sock, address
        address = sock.get("next", 0)


def _local_matrix(ob: _S, values: dict | None = None) -> np.ndarray:
    """Blender's object matrix: location, then rotation (with its delta), then scale. values
    overrides the object's own fields (an animation's at some frame)."""
    def get(name, default=None):
        return values[name] if values and name in values else ob.get(name, default)
    loc = np.asarray(get("loc"), np.float64) + np.asarray(get("dloc", np.zeros(3)), np.float64)
    size = np.asarray(get("size"), np.float64) * np.asarray(get("dscale", np.ones(3)), np.float64)
    mode = int(ob.get("rotmode", 1))
    if mode > 0:
        order = _EULER.get(mode, "XYZ")
        rot = euler(np.degrees(get("rot")), order)
        drot = euler(np.degrees(get("drot", np.zeros(3))), order)
    elif mode == -1:  # axis and angle
        rot = _axis_angle(get("rotAxis"), float(get("rotAngle", 0.0)))
        drot = _axis_angle(get("drotAxis", np.array([0, 1, 0.0])), float(get("drotAngle", 0.0)))
    else:  # quaternion (w, x, y, z)
        rot = _quat(get("quat"))
        drot = _quat(get("dquat", np.array([1, 0, 0, 0.0])))
    return translation(loc) @ drot @ rot @ scaling(size)


def _frame_rate(scene: _S) -> float:
    r = scene.get("r")
    fps = float(r.get("frs_sec", 24) or 24) if isinstance(r, _S) else 24.0
    base = float(r.get("frs_sec_base", 1.0) or 1.0) if isinstance(r, _S) else 1.0
    return fps / base if fps > 0 and base > 0 else 24.0


def _frames(used: list, scene: _S) -> np.ndarray | None:
    """Every key of the curves and every frame between, or None when that's under two frames."""
    keys = np.concatenate([c.keys[:, 1, 0] for c in used])
    lo, hi = float(keys.min()), float(keys.max())
    if any(c.carries_on for c in used):  # a curve that carries on: as far as the scene plays
        r = scene.get("r")
        if isinstance(r, _S):
            lo, hi = min(lo, float(r.get("sfra", lo))), max(hi, float(r.get("efra", hi)))
    if hi - lo >= MAX_FRAMES:
        return None
    frames = np.unique(np.concatenate([keys, np.arange(math.ceil(lo), math.floor(hi) + 1)]))
    frames = frames[(frames >= lo) & (frames <= hi)]
    return frames if len(frames) >= 2 else None


def _bone_channels(rig, curves: list, fps: float, scene: _S) -> list[Channel]:
    """Pose bone F-curves as channels of the bones' nodes: each bone's rest matrix (relative
    to its parent's) then its channel's animated transform, posed at every key and frame between."""
    by_bone: dict[str, list] = {}
    for c in curves:
        got = blendskin.bone_curve(c.path)
        if got is not None and got[0] in rig.nodes and got[0] in rig.channels and (
                got[1] in blendskin.BONE_FIELDS or got[1] == "rotation_axis_angle"):
            by_bone.setdefault(got[0], []).append((got[1], c))
    out = []
    if not by_bone:
        return out
    # bones that leave out their parent's rotation or scale move when it does: pose the whole rig
    together = bool(rig.options)
    groups = [list(by_bone)] if together else [[b] for b in by_bone]
    for group in groups:
        frames = _frames([c for b in group for _, c in by_bone[b]], scene)
        if frames is None:
            continue
        bases = {b: [] for b in (rig.channels if together else group) if b in rig.nodes}
        for bone in bases:
            ch = rig.channels[bone]
            values: dict[str, np.ndarray] = {}
            for prop, c in by_bone.get(bone, []):
                field, index = blendskin.BONE_FIELDS.get(prop), c.index
                if prop == "rotation_axis_angle":  # the angle, then the axis
                    field, index = ("rotAngle", 0) if c.index == 0 else ("rotAxis", c.index - 1)
                if field not in values:
                    values[field] = np.tile(np.atleast_1d(np.asarray(ch.get(field), np.float64)), (len(frames), 1))
                if index < values[field].shape[1]:
                    values[field][:, index] = c.evaluate(frames)
            bases[bone] = [blendskin.basis(ch, {k: (v[i, 0] if k == "rotAngle" else v[i]) for k, v in values.items()})
                           for i in range(len(frames))]
        if together:
            per_frame = [rig.locals({b: m[i] for b, m in bases.items()}) for i in range(len(frames))]
            locals_ = {b: np.array([f[b] for f in per_frame]) for b in rig.order}
        else:
            locals_ = {b: np.array([rig.relative[b] @ m for m in bases[b]]) for b in group}
        for bone, m in locals_.items():
            for name, vals in zip(("translation", "rotation", "scale"), decompose_many(m)):
                if not np.allclose(vals, vals[0], atol=1e-9):
                    out.append(Channel(rig.nodes[bone], name, frames / fps, vals, "LINEAR"))
    return out


def _weight_channel(node: Node, curves: list, fps: float, scene: _S) -> Channel | None:
    """The mesh's shape key values over time, the keys nothing animates held at their own."""
    names = [m.name for m in node.mesh.targets]
    used = {}
    for c in curves:
        key = blendskin.key_curve(c.path)
        if key in names and key not in used:
            used[key] = c
    if not used:
        return None
    frames = _frames(list(used.values()), scene)
    if frames is None:
        return None
    rest = node.mesh.target_weights
    values = np.stack([used[n].evaluate(frames) if n in used else np.full(len(frames), rest[k])
                       for k, n in enumerate(names)], 1)
    return Channel(node, WEIGHTS, frames / fps, values, "LINEAR") if np.isfinite(values).all() else None


def rig_name(ob: _S | None) -> str:
    return (ob.id_name() if ob is not None else "") or "armature"


def _channels(ob: _S, node: Node, curves: list, fps: float, before: np.ndarray, scene: _S) -> list[Channel]:
    """An object's F-curves as channels: its matrix (before, then the animated local matrix)
    posed at every key and every frame between, then split into translation, rotation and scale."""
    used = [c for c in curves if c.path in blendanim.PATHS or c.path == "rotation_axis_angle"]
    if not used:
        return []
    frames = _frames(used, scene)
    if frames is None:
        return []
    values: dict[str, np.ndarray] = {}
    for c in used:
        field, index = blendanim.PATHS.get(c.path), c.index
        if c.path == "rotation_axis_angle":  # the angle, then the axis
            field, index = ("rotAngle", 0) if c.index == 0 else ("rotAxis", c.index - 1)
        rest = ob.get(field)
        if rest is None:
            continue
        if field not in values:
            values[field] = np.tile(np.atleast_1d(np.asarray(rest, np.float64)), (len(frames), 1))
        if index < values[field].shape[1]:
            values[field][:, index] = c.evaluate(frames)
    if not values:
        return []
    m = np.array([before @ _local_matrix(ob, {k: (v[i, 0] if k == "rotAngle" else v[i]) for k, v in values.items()})
                  for i in range(len(frames))])
    out = []
    for name, vals in zip(("translation", "rotation", "scale"), decompose_many(m)):
        if not np.allclose(vals, vals[0], atol=1e-9):
            out.append(Channel(node, name, frames / fps, vals, "LINEAR"))
    return out


def _quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.ravel(q)[:4])
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return m


def _axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(4)
    half = angle / 2
    return _quat([math.cos(half), *(a / n * math.sin(half))])


# ---- mesh layouts

def _read_mesh(me: _S) -> tuple[Mesh, np.ndarray] | None:
    """The mesh as stored (without normals) and whether each face is smooth."""
    name = me.id_name() or "mesh"
    nv = int(me.get("totvert", me.get("verts_num", 0)) or 0)
    nf = int(me.get("totpoly", me.get("faces_num", 0)) or 0)
    nc = int(me.get("totloop", me.get("corners_num", 0)) or 0)
    if not nv or not nf or not nc:
        return None
    attrs = _attributes(me, nv, nf, nc)
    positions = attrs.get(("position", _POINT))
    offsets = None
    if me.get("poly_offset_indices", 0) or me.get("face_offset_indices", 0):
        field = "poly_offset_indices" if me.has("poly_offset_indices") else "face_offset_indices"
        offsets = me.raw(field, "<i4", nf + 1)
    corner_vert = attrs.get((".corner_vert", _CORNER))
    material_index = attrs.get(("material_index", _FACE))
    smooth = None
    if corner_vert is None or offsets is None:  # Blender 2.8 to 3.5
        legacy = _legacy(me, nv, nf, nc, material_index)
        if legacy is None:
            return None
        # Blender 3.5 itself saves the old arrays, MVert included (checked with 3.5.1), but a file with
        # a "position" layer next to the old polygon and loop arrays reads too
        old_positions, offsets, corner_vert, material_index, smooth = legacy
        positions = positions if positions is not None else old_positions
    if positions is None:
        return None
    positions = np.asarray(positions, np.float64).reshape(-1, 3)
    corner_vert = np.asarray(corner_vert, np.int64).ravel()
    offsets = np.asarray(offsets, np.int64).ravel()
    sizes = np.diff(offsets)
    if (len(offsets) != nf + 1 or offsets[0] != 0 or offsets[-1] != nc or (sizes < 0).any()
            or len(corner_vert) != nc or (len(corner_vert) and (corner_vert.min() < 0 or corner_vert.max() >= nv))):
        raise MeshportError(f"Mesh {name!r} is damaged")
    if smooth is None:
        sharp = attrs.get(("sharp_face", _FACE))
        smooth = np.ones(nf, bool) if sharp is None else ~np.asarray(sharp, bool).ravel()
    uvs = [np.asarray(v, np.float64)[:, :2] for (n, dom), v in attrs.items()
           if dom == _CORNER and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2
           and v.dtype.kind == "f" and not n.startswith(".")]
    colors = None
    for (n, dom), v in attrs.items():
        if n.startswith(".") or not isinstance(v, np.ndarray) or v.ndim != 2 or v.shape[1] != 4:
            continue
        c = v.astype(np.float64) / (255.0 if v.dtype == np.uint8 else 1.0)
        if v.dtype == np.uint8:  # byte colors are sRGB
            c[:, :3] = np.where(c[:, :3] <= 0.04045, c[:, :3] / 12.92, ((c[:, :3] + 0.055) / 1.055) ** 2.4)
        if dom == _POINT:
            colors = c[corner_vert]
        elif dom == _CORNER:
            colors = c
        if colors is not None:
            break
    keep = sizes >= 3
    smooth = np.asarray(smooth, bool).ravel()
    if len(smooth) != nf:
        smooth = np.ones(nf, bool)
    mesh = Mesh(name, positions, sizes, corner_vert, None, uvs, colors,
                np.asarray(material_index, np.int64).ravel() if material_index is not None else None)
    if not keep.all():
        mesh, smooth = _drop_faces(mesh, keep), smooth[keep]
    pairs, crease = attrs.get((".edge_verts", _EDGE)), attrs.get(("crease_edge", _EDGE))
    vert_crease = attrs.get(("crease_vert", _POINT))
    if pairs is not None and (crease is not None or vert_crease is not None):  # Blender 4.0 on
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        crease = np.zeros(len(pairs)) if crease is None else np.asarray(crease, np.float64).ravel()
        if len(crease) == len(pairs) and (vert_crease is None or len(vert_crease) == nv):
            mesh.creases = blendmods.Creases.from_blender(pairs, crease, vert_crease)
    return mesh, smooth


def _finish(mesh: Mesh, smooth: np.ndarray) -> Mesh:
    mesh.normals = _normals(mesh, smooth)
    return mesh


def _attributes(me: _S, nv: int, nf: int, nc: int) -> dict:
    """{(name, domain): values} from attribute storage (5.0+) or named CustomData layers."""
    out: dict = {}
    counts = {_POINT: nv, _EDGE: int(me.get("totedge", 0) or 0), _FACE: nf, _CORNER: nc}
    storage = me.get("attribute_storage")
    if storage is not None and storage.get("dna_attributes_num", 0):
        f = me.f
        num = int(storage.get("dna_attributes_num"))
        got = f.at(storage.get("dna_attributes", 0), me.owner)
        if got is not None:
            b, off = got
            size = f.struct_size["Attribute"]
            for i in range(min(num, (b.length - off) // max(size, 1))):  # as many as its block holds
                a = _S(f, "Attribute", b.offset + off + i * size, b.scope)
                name_got = f.at(a.get("name", 0), me.owner)
                if name_got is None:
                    continue
                nb, noff = name_got
                name = f.data[nb.offset + noff:nb.offset + nb.length].split(b"\0", 1)[0].decode("utf-8", "replace")
                kind, domain = _ATTR.get(int(a.get("data_type", -1))), int(a.get("domain", -1))
                if kind is None or domain not in counts:
                    continue
                n = counts[domain]
                holder = a.follow("data", "AttributeArray" if a.get("storage_type", 0) == 0 else "AttributeSingle")
                if holder is None:
                    continue
                single = a.get("storage_type", 0) == 1 or bool(holder.get("is_single", 0))
                dtype, comps = kind
                vals = holder.raw("data", "<" + dtype, comps * (1 if single else n))
                if vals is None:
                    continue
                vals = vals.reshape(-1, comps) if comps > 1 else vals
                if single:
                    vals = np.repeat(vals[:1], n, 0)
                out[(name, domain)] = vals
        if out:
            return out
    for field, domain in (("vdata", _POINT), ("edata", _EDGE), ("pdata", _FACE), ("ldata", _CORNER)):
        cd = me.get(field)
        if cd is None:
            continue
        n_layers = int(cd.get("totlayer", 0) or 0)
        got = me.f.at(cd.get("layers", 0), me.owner)
        if got is None or not n_layers:
            continue
        b, off = got
        size = me.f.struct_size["CustomDataLayer"]
        for i in range(min(n_layers, (b.length - off) // max(size, 1))):  # as many as its block holds
            layer = _S(me.f, "CustomDataLayer", b.offset + off + i * size, b.scope)
            kind = _CD.get(int(layer.get("type", -1)))
            if kind is None:
                continue
            dtype, comps = kind
            n = counts[domain]
            if layer.get("type") == 16:  # MLoopUV (to 3.4): uv[2] and an int of flags, as the file's DNA lays them out
                index = me.f.struct_index.get("MLoopUV")
                uv = me.f.structs[index][1].get("uv") if index is not None else None
                if uv is None:
                    continue
                stride = me.f.struct_size["MLoopUV"]
                raw = layer.raw("data", "u1", stride * n)
                vals = np.ndarray((n, 2), "<f4", raw.tobytes(), uv[0], (stride, 4)) if raw is not None else None
            else:
                vals = layer.raw("data", "<" + dtype, comps * n)
                vals = vals.reshape(-1, comps) if vals is not None and comps > 1 else vals
            if vals is None:
                continue
            name = layer.get("name", "") or ("UVMap" if layer.get("type") == 16 else f"layer{i}")
            out.setdefault((name, domain), vals)
    return out


def _legacy(me: _S, nv: int, nf: int, nc: int, material: np.ndarray | None = None):
    """Blender 2.8 to 3.5: MPoly, MLoop and MVert arrays (positions come back None without
    MVert). material is a "material_index" layer, used instead of mat_nr when the file has one."""
    f = me.f
    if not all(n in f.struct_index for n in ("MPoly", "MLoop")) or not me.get("mpoly", 0):
        return None

    def array(field: str, struct_name: str, count: int, member: str, dtype: str, comps: int = 1):
        size = f.struct_size[struct_name]
        spec = f.structs[f.struct_index[struct_name]][1][member]
        raw = me.raw(field, "u1", size * count)
        if raw is None:
            return None
        shape = (count, comps) if comps > 1 else (count,)
        strides = (size, np.dtype(dtype).itemsize) if comps > 1 else (size,)
        return np.ndarray(shape, dtype, raw.tobytes(), spec[0], strides).copy()

    has_mvert = "MVert" in f.struct_index and me.get("mvert", 0)
    positions = array("mvert", "MVert", nv, "co", "<f4", 3) if has_mvert else None
    loopstart = array("mpoly", "MPoly", nf, "loopstart", "<i4")
    totloop = array("mpoly", "MPoly", nf, "totloop", "<i4")
    mat_nr = array("mpoly", "MPoly", nf, "mat_nr", "<i2") if material is None else np.asarray(material).ravel()
    flag = array("mpoly", "MPoly", nf, "flag", "u1")
    corner_vert = array("mloop", "MLoop", nc, "v", "<i4")
    if any(x is None for x in (loopstart, totloop, corner_vert)):
        return None
    order = np.argsort(loopstart, kind="stable")
    if not (order == np.arange(nf)).all():  # faces listed out of order: renumber the corners
        corner_vert = np.concatenate([corner_vert[s:s + t] for s, t in zip(loopstart[order], totloop[order])])
        totloop, mat_nr, flag = totloop[order], mat_nr[order] if mat_nr is not None else None, flag[order]
    offsets = np.concatenate(([0], np.cumsum(totloop)))
    smooth = (flag & 1).astype(bool) if flag is not None else None  # ME_SMOOTH
    return positions, offsets, corner_vert, mat_nr, smooth


def _normals(mesh: Mesh, smooth: np.ndarray) -> np.ndarray:
    """Per-corner normals as Blender shades: flat faces get their own normal, smooth faces the
    average of the smooth faces around each vertex."""
    sizes = mesh.face_sizes
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    face = np.repeat(np.arange(len(sizes)), sizes)
    c = np.arange(len(mesh.corners))
    nxt = c + 1
    nxt[starts + sizes - 1] = starts
    p, pn = mesh.positions[mesh.corners], mesh.positions[mesh.corners[nxt]]
    cross = np.cross(p, pn)
    fn = np.zeros((len(sizes), 3))
    for k in range(3):
        fn[:, k] = np.bincount(face, cross[:, k], len(sizes))
    smooth = np.asarray(smooth, bool)
    if len(smooth) != len(sizes):
        smooth = np.ones(len(sizes), bool)
    vn = np.zeros((len(mesh.positions), 3))
    sm = smooth[face]
    for k in range(3):
        vn[:, k] = np.bincount(mesh.corners[sm], fn[face[sm], k], len(mesh.positions))
    out = np.where(sm[:, None], vn[mesh.corners], fn[face])
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, length, out=np.zeros_like(out), where=length > 0)


def _drop_faces(mesh: Mesh, keep: np.ndarray) -> Mesh:
    corner_keep = np.repeat(keep, mesh.face_sizes)
    return Mesh(mesh.name, mesh.positions, mesh.face_sizes[keep], mesh.corners[corner_keep],
                mesh.normals[corner_keep] if mesh.normals is not None else None,
                [u[corner_keep] for u in mesh.uvs],
                mesh.colors[corner_keep] if mesh.colors is not None else None,
                mesh.face_materials[keep] if mesh.face_materials is not None else None)
