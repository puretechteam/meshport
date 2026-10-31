"""A minimal .blend writer for tests: declare just the structs and fields a test needs (meshport
finds fields by name through the file's own struct catalogue), then add blocks."""

import re
import struct

import numpy as np

_PRIMITIVE = {"char": 1, "uchar": 1, "short": 2, "int": 4, "float": 4, "double": 8, "int64_t": 8, "void": 0}
_PACK = {"char": "b", "uchar": "B", "short": "h", "int": "i", "float": "f", "double": "d", "int64_t": "q"}

# the structs a mesh object in a scene needs, as Blender 2.8 to 3.5 lay them out (only the fields read)
STRUCTS = {
    "ListBase": [("void", "*first"), ("void", "*last")],
    "ID": [("void", "*next"), ("void", "*prev"), ("char", "name[66]")],
    "FileGlobal": [("Scene", "*curscene")],
    "Scene": [("ID", "id"), ("Collection", "*master_collection")],
    "Collection": [("ID", "id"), ("ListBase", "gobject"), ("ListBase", "children"), ("float", "instance_offset[3]")],
    "CollectionObject": [("CollectionObject", "*next"), ("CollectionObject", "*prev"), ("Object", "*ob")],
    "CollectionChild": [("CollectionChild", "*next"), ("CollectionChild", "*prev"), ("Collection", "*collection")],
    "Object": [("ID", "id"), ("short", "type"), ("short", "rotmode"), ("short", "transflag"), ("short", "totcol"),
               ("void", "*data"), ("Object", "*parent"), ("Collection", "*instance_collection"),
               ("float", "loc[3]"), ("float", "rot[3]"), ("float", "size[3]"), ("float", "parentinv[4][4]"),
               ("Material", "**mat"), ("char", "*matbits"), ("ListBase", "modifiers"), ("bPose", "*pose")],
    "Mesh": [("ID", "id"), ("int", "totvert"), ("int", "totpoly"), ("int", "totloop"), ("short", "totcol"),
             ("MVert", "*mvert"), ("MPoly", "*mpoly"), ("MLoop", "*mloop"), ("Material", "**mat"), ("Key", "*key"),
             ("MDeformVert", "*dvert"), ("ListBase", "vertex_group_names")],
    "MVert": [("float", "co[3]")],
    "MPoly": [("int", "loopstart"), ("int", "totloop"), ("short", "mat_nr"), ("char", "flag"), ("char", "pad")],
    "MLoop": [("int", "v"), ("int", "e")],
    "Key": [("ID", "id"), ("ListBase", "block"), ("KeyBlock", "*refkey"), ("short", "type"), ("short", "pad")],
    "KeyBlock": [("KeyBlock", "*next"), ("KeyBlock", "*prev"), ("float", "curval"), ("short", "relative"),
                 ("short", "flag"), ("int", "totelem"), ("void", "*data"), ("char", "name[64]"),
                 ("char", "vgroup[64]")],
    "MDeformVert": [("MDeformWeight", "*dw"), ("int", "totweight"), ("int", "flag")],
    "MDeformWeight": [("int", "def_nr"), ("float", "weight")],
    "bDeformGroup": [("bDeformGroup", "*next"), ("bDeformGroup", "*prev"), ("char", "name[64]")],
    "ModifierData": [("ModifierData", "*next"), ("ModifierData", "*prev"), ("int", "type"), ("int", "mode"),
                     ("char", "name[64]")],
    "ArrayModifierData": [("ModifierData", "modifier"), ("float", "offset[3]"), ("float", "scale[3]"),
                          ("int", "fit_type"), ("int", "offset_type"), ("int", "flags"), ("int", "count")],
    "SubsurfModifierData": [("ModifierData", "modifier"), ("short", "subdivType"), ("short", "levels"),
                            ("short", "flags"), ("short", "pad")],
    "ArmatureModifierData": [("ModifierData", "modifier"), ("short", "deformflag"), ("short", "pad"),
                             ("Object", "*object")],
    "bArmature": [("ID", "id"), ("ListBase", "bonebase")],
    "Bone": [("Bone", "*next"), ("Bone", "*prev"), ("Bone", "*parent"), ("ListBase", "childbase"),
             ("char", "name[64]"), ("float", "arm_mat[4][4]"), ("int", "flag"), ("int", "pad")],
    "bPose": [("ListBase", "chanbase")],
    "bPoseChannel": [("bPoseChannel", "*next"), ("bPoseChannel", "*prev"), ("char", "name[64]"),
                     ("float", "loc[3]"), ("float", "size[3]"), ("float", "quat[4]"), ("short", "rotmode"),
                     ("short", "pad")],
}
IDENTITY = np.eye(4, dtype=np.float32).T.ravel().tolist()


class BlendWriter:
    def __init__(self, structs: dict = STRUCTS):
        self.structs = structs
        named = [t for fields in structs.values() for t, _ in fields if t not in _PRIMITIVE and t not in structs]
        self.types = list(_PRIMITIVE) + list(structs) + list(dict.fromkeys(named))  # the rest only pointed to
        self.size = {t: s for t, s in _PRIMITIVE.items()} | {t: 0 for t in named}
        self.layout: dict[str, dict] = {}
        for name in structs:
            self._lay(name)
        self.blocks: list[bytes] = []
        self.next_address = 0x10000

    def _lay(self, name: str) -> int:
        if name in self.layout:
            return self.size[name]
        fields, off = {}, 0
        for t, raw in self.structs[name]:
            is_ptr = raw.startswith("*")
            dims = [int(x) for x in re.findall(r"\[(\d+)\]", raw)]
            clean = re.sub(r"\[.*", "", raw).strip("*")
            count = int(np.prod(dims)) if dims else 1
            elem = 8 if is_ptr else (self.size[t] if t in _PRIMITIVE else self._lay(t))
            fields[clean] = (off, t, is_ptr, count)
            off += elem * count
        self.layout[name], self.size[name] = fields, off
        return off

    def pack(self, name: str, values: dict) -> bytes:
        out = bytearray(self.size[name])
        for key, value in values.items():
            off, t, is_ptr, count = self.layout[name][key]
            if is_ptr:
                vals = np.ravel(value).tolist() if count > 1 else [value]
                struct.pack_into(f"<{count}Q", out, off, *[int(v) for v in vals])
            elif t in _PACK:
                if t == "char" and isinstance(value, (str, bytes)):
                    raw = value.encode() if isinstance(value, str) else value
                    out[off:off + len(raw)] = raw[:count]
                else:
                    vals = np.ravel(value).tolist() if count > 1 else [value]
                    struct.pack_into(f"<{len(vals)}{_PACK[t]}", out, off, *vals)
            else:
                raw = self.pack(t, value)
                out[off:off + len(raw)] = raw
        return bytes(out)

    def sdna(self, name: str) -> int:
        """A block's struct: its index among the structs (0 for plain data)."""
        return list(self.structs).index(name) if name in self.structs else 0

    def address(self) -> int:
        self.next_address += 0x1000
        return self.next_address

    def block(self, code: bytes, name: str, items, address: int | None = None) -> int:
        """A block of one struct (items a dict) or several (a list of dicts). Returns its address."""
        items = [items] if isinstance(items, dict) else items
        address = address or self.address()
        body = b"".join(self.pack(name, v) for v in items)
        self.blocks.append(struct.pack("<4siQii", code, len(body), address, self.sdna(name), len(items)) + body)
        return address

    def raw(self, data: bytes, name: str = "char", count: int = 1, address: int | None = None) -> int:
        address = address or self.address()
        self.blocks.append(struct.pack("<4siQii", b"DATA", len(data), address, self.sdna(name), count) + data)
        return address

    def _dna(self) -> bytes:
        names = [raw for name in self.structs for _, raw in self.structs[name]]
        name_index = {n: i for i, n in enumerate(dict.fromkeys(names))}

        def strings(tag: bytes, items) -> bytes:
            body = tag + struct.pack("<i", len(items)) + b"".join(s.encode() + b"\0" for s in items)
            return body + bytes(-len(body) % 4)
        out = b"SDNA" + strings(b"NAME", list(name_index)) + strings(b"TYPE", self.types)
        tlen = b"TLEN" + struct.pack(f"<{len(self.types)}h", *[self.size[t] for t in self.types])
        out += tlen + bytes(-len(tlen) % 4)
        out += b"STRC" + struct.pack("<i", len(self.structs))
        for name, fields in self.structs.items():
            out += struct.pack("<hh", self.types.index(name), len(fields))
            for t, raw in fields:
                out += struct.pack("<hh", self.types.index(t), name_index[raw])
        return out

    def file(self) -> bytes:
        dna = self._dna()
        dna_block = struct.pack("<4siQii", b"DNA1", len(dna), 1, 0, 1) + dna
        return b"BLENDER-v280" + b"".join(self.blocks) + dna_block + struct.pack("<4siQii", b"ENDB", 0, 0, 0, 0)


def cube_mesh(w: BlendWriter, name: str = "cube", **extra) -> int:
    """A cube Mesh datablock (the 2.8 to 3.5 layout); returns its address."""
    p = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    verts = w.block(b"DATA", "MVert", [{"co": v} for v in p])
    polys = w.block(b"DATA", "MPoly", [{"loopstart": 4 * i, "totloop": 4} for i in range(6)])
    loops = w.block(b"DATA", "MLoop", [{"v": v} for q in quads for v in q])
    return w.block(b"ME\0\0", "Mesh", {"id": {"name": "ME" + name}, "totvert": 8, "totpoly": 6, "totloop": 24,
                                       "mvert": verts, "mpoly": polys, "mloop": loops, **extra})


def mesh_object(w: BlendWriter, name: str, data: int, **extra) -> int:
    return w.block(b"OB\0\0", "Object", {"id": {"name": "OB" + name}, "type": 1, "rotmode": 1, "data": data,
                                         "size": [1, 1, 1], "parentinv": IDENTITY, **extra})


def collection(w: BlendWriter, objects: list[int], children: list[int] = (), code: bytes = b"GR\0\0",
               name: str = "coll") -> int:
    """A Collection holding these objects and child collections."""
    def links(struct_name: str, field: str, targets) -> dict:
        addrs = [w.address() for _ in targets]
        for i, (a, t) in enumerate(zip(addrs, targets)):
            w.block(b"DATA", struct_name, {"next": addrs[i + 1] if i + 1 < len(addrs) else 0,
                                           "prev": addrs[i - 1] if i else 0, field: t}, a)
        return {"first": addrs[0], "last": addrs[-1]} if addrs else {}
    return w.block(code, "Collection", {"id": {"name": "GR" + name}, "gobject": links("CollectionObject", "ob", objects),
                                        "children": links("CollectionChild", "collection", children)})


def scene_file(w: BlendWriter, objects: list[int], children: list[int] = ()) -> bytes:
    """The file with a scene whose master collection holds these objects (and child collections)."""
    master = collection(w, objects, children, b"DATA", "master")
    sc = w.block(b"SC\0\0", "Scene", {"id": {"name": "SCScene"}, "master_collection": master})
    w.block(b"GLOB", "FileGlobal", {"curscene": sc})
    return w.file()
