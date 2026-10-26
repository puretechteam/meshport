"""Autodesk 3DS (.3ds) <-> Scene.

3DS is a tree of chunks: a 2-byte id and a 4-byte length (header included), then the
payload and any sub-chunks. Reading takes the mesh objects (points, triangles, the first
UV set, smoothing groups, per-face materials, the object's matrix) and the materials
(diffuse, opacity, shininess, two-sided, self-illumination and the texture, bump and
self-illumination maps). When the file has a keyframer section, it gives the hierarchy,
pivots, dummy groups and instances, and animated objects are posed at their first frame,
as Blender and Assimp show them; without one, every object is a root in the pose it was
saved in. Lights, cameras and the animation itself are skipped.

The format's limits shape the writer: a mesh is triangles with at most 65,535 vertices and
65,535 faces, one UV per vertex, and normals only as smoothing groups. Bigger meshes are
split into several objects, vertices are split wherever a UV or normal changes, object
names are cut to 10 characters and material names to 16, and textures are referenced by
8.3 file names (write_file puts the image files next to the model).

3DS is Z-up, and 1 unit is taken as 1 meter.
"""

from __future__ import annotations

import dataclasses
import os
import struct
from pathlib import Path

import numpy as np

from .fbx import _EXT, TextureFinder
from .geometry import scaling, transform_points, translation, unique_rows
from .scene import DAMAGED, Image, Material, Mesh, MeshportError, Node, Scene

MAIN, VERSION, EDIT, MESH_VERSION, MASTER_SCALE = 0x4D4D, 0x0002, 0x3D3D, 0x3D3E, 0x0100
OBJECT, TRIMESH = 0x4000, 0x4100
POINT_ARRAY, FACE_ARRAY, MSH_MAT_GROUP, TEX_VERTS, SMOOTH_GROUP, MESH_MATRIX = \
    0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160
MATERIAL, MAT_NAME, MAT_AMBIENT, MAT_DIFFUSE, MAT_SPECULAR = 0xAFFF, 0xA000, 0xA010, 0xA020, 0xA030
MAT_SHININESS, MAT_SHIN2PCT, MAT_TRANSPARENCY, MAT_TWO_SIDE, MAT_SELF_ILPCT = 0xA040, 0xA041, 0xA050, 0xA081, 0xA084
MAT_TEXMAP, MAT_BUMPMAP, MAT_SELFIMAP, MAT_MAPNAME = 0xA200, 0xA230, 0xA33D, 0xA300
COLOR_F, COLOR_24, LIN_COLOR_24, LIN_COLOR_F = 0x0010, 0x0011, 0x0012, 0x0013
INT_PERCENTAGE, FLOAT_PERCENTAGE = 0x0030, 0x0031
CAMERA_OBJECT = 0x4700
KFDATA, AMBIENT_NODE, OBJECT_NODE, CAMERA_NODE, TARGET_NODE, SPOT_NODE = 0xB000, 0xB001, 0xB002, 0xB003, 0xB004, 0xB007
KF_SEGMENT, KF_CURRENT = 0xB008, 0xB009
NODE_HDR, INSTANCE_NAME, PIVOT, NODE_ID = 0xB010, 0xB011, 0xB013, 0xB030
POS_TRACK, ROT_TRACK, SCL_TRACK, ROLL_TRACK = 0xB020, 0xB021, 0xB022, 0xB024
DUMMY = "$$$DUMMY"  # the object name of a keyframer node that only groups others

LIMIT = 65535  # vertices and faces per object: 3DS counts and indexes them with 16 bits
_OBJECT_NAME, _MATERIAL_NAME = 10, 16

_HEAD = struct.Struct("<HI")
# 3DS (x, y, z), Z up -> the scene's (x, z, -y), Y up
_Z_UP = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# ============================================================================ reading


def read(data: bytes, base_dir: str | os.PathLike | None = None) -> Scene:
    """Parse 3DS bytes into a Scene. base_dir is where to look for the texture files."""
    if len(data) < 6 or _HEAD.unpack_from(data)[0] != MAIN:
        raise MeshportError("This isn't a 3DS file")
    try:
        return _Reader(data, Path(base_dir) if base_dir else None).scene()
    except (*DAMAGED, struct.error) as e:
        raise MeshportError(f"This 3DS file is damaged ({type(e).__name__}: {e})") from None


def read_file(path: str | os.PathLike) -> Scene:
    path = Path(path)
    return read(path.read_bytes(), path.parent)


def _chunks(data: bytes, start: int, end: int):
    """(id, payload start, end) for each chunk between start and end."""
    pos = start
    while pos + 6 <= end:
        cid, length = _HEAD.unpack_from(data, pos)
        if length < 6 or pos + length > end:
            raise MeshportError("This 3DS file is damaged or cut short")
        yield cid, pos + 6, pos + length
        pos += length


def _cstring(data: bytes, start: int, end: int) -> tuple[str, int]:
    """A zero-terminated string and the position after it."""
    stop = data.find(b"\0", start, end)
    if stop < 0:
        raise MeshportError("This 3DS file is damaged (a name runs past its chunk)")
    raw = data[start:stop]
    try:
        return raw.decode("utf-8"), stop + 1
    except UnicodeDecodeError:
        return raw.decode("latin-1"), stop + 1


def _counted(data: bytes, start: int, end: int, dtype, width: int) -> tuple[np.ndarray, int]:
    """A uint16 count followed by that many rows; the count is checked against the chunk."""
    if start + 2 > end:
        raise MeshportError("This 3DS file is damaged (a list is cut short)")
    (n,) = struct.unpack_from("<H", data, start)
    size = n * width * np.dtype(dtype).itemsize
    if start + 2 + size > end:
        raise MeshportError("This 3DS file is damaged (a list is longer than its chunk)")
    return np.frombuffer(data, dtype, n * width, start + 2).reshape(n, width), start + 2 + size


class _Reader:
    def __init__(self, data: bytes, base_dir: Path | None):
        self.data = data
        self.finder = TextureFinder(base_dir)
        self.materials: dict[str, Material] = {}
        self.images: dict[str, Image | None] = {}
        self.cameras: dict[str, tuple] = {}  # name -> (position, target, roll in degrees)
        self.segment_start: int | None = None
        self.saved_frame: int | None = None  # the frame the file was saved on, in ticks

    def scene(self) -> Scene:
        data = self.data
        end = min(_HEAD.unpack_from(data)[1], len(data))  # some writers get the outer length wrong
        edits = [(s, e) for cid, s, e in _chunks(data, 6, end) if cid == EDIT]
        for s, e in edits:  # materials first: objects refer to them by name, wherever they are
            for cid, cs, ce in _chunks(data, s, e):
                if cid == MATERIAL:
                    mat = self._material(cs, ce)
                    self.materials.setdefault(mat.name, mat)
        objects: dict[str, list[Node]] = {}
        loose = []
        for s, e in edits:
            for cid, cs, ce in _chunks(data, s, e):
                if cid == OBJECT:
                    name, nodes = self._object(cs, ce)
                    objects.setdefault(name, nodes)
                    loose += nodes
        keys = [k for cid, s, e in _chunks(data, 6, end) if cid == KFDATA for k in self._keyframer(s, e)]
        # Saved on the first frame, the stored geometry already is the first-frame pose, exactly.
        exact = self.saved_frame is not None and self.saved_frame in (self.segment_start, 160 * self.segment_start)
        scene = Scene(self._hierarchy(keys, objects, loose, exact) if keys else loose)
        if not scene.meshes:
            raise MeshportError("No 3D geometry found in this file")
        return scene

    # ---- objects

    def _object(self, start: int, end: int) -> tuple[str, list[Node]]:
        name, pos = _cstring(self.data, start, end)
        nodes = []
        for cid, s, e in _chunks(self.data, pos, end):
            if cid == TRIMESH:
                node = self._trimesh(name, s, e)
                if node is not None:
                    nodes.append(node)
            elif cid == CAMERA_OBJECT and e - s >= 28:
                v = struct.unpack_from("<7f", self.data, s)
                if all(np.isfinite(v)):
                    self.cameras.setdefault(name, (v[:3], v[3:6], v[6]))
        for i, node in enumerate(nodes[1:], 1):
            node.name = f"{name}.{i}"
        return name, nodes

    # ---- keyframer

    def _keyframer(self, start: int, end: int) -> list[dict]:
        """Each node: its kind (object, camera, target or other), id, name, parent id,
        instance name, pivot and the first key of each of its tracks."""
        data = self.data
        out = []
        for cid, s, e in _chunks(data, start, end):
            if cid == KF_SEGMENT and e - s >= 8:
                self.segment_start = struct.unpack_from("<I", data, s)[0]
            elif cid == KF_CURRENT and e - s >= 4:
                self.saved_frame = struct.unpack_from("<I", data, s)[0]  # 160 ticks a frame
            if not AMBIENT_NODE <= cid <= SPOT_NODE:  # ambient, object, camera, target, lights
                continue
            kind = {OBJECT_NODE: "object", CAMERA_NODE: "camera", TARGET_NODE: "target"}.get(cid, "other")
            node = {"kind": kind, "id": len(out), "name": "", "parent": -1, "instance": "",
                    "pivot": (0.0, 0.0, 0.0), POS_TRACK: None, ROT_TRACK: None, SCL_TRACK: None, ROLL_TRACK: None}
            for sub, ss, se in _chunks(data, s, e):
                if sub == NODE_ID and se - ss >= 2:
                    node["id"] = struct.unpack_from("<h", data, ss)[0]
                elif sub == NODE_HDR:
                    node["name"], p = _cstring(data, ss, se)
                    if p + 6 <= se:
                        node["parent"] = struct.unpack_from("<h", data, p + 4)[0]  # after two flag words
                elif sub == INSTANCE_NAME:
                    node["instance"] = _cstring(data, ss, se)[0]
                elif sub == PIVOT and se - ss >= 12:
                    node["pivot"] = struct.unpack_from("<3f", data, ss)
                elif sub in (POS_TRACK, ROT_TRACK, SCL_TRACK, ROLL_TRACK):
                    node[sub] = self._first_key(ss, se, {ROT_TRACK: 4, ROLL_TRACK: 1}.get(sub, 3))
            out.append(node)
        return out

    def _first_key(self, start: int, end: int, width: int) -> tuple | None:
        """A track's first key. A track is flags, 8 unused bytes and a key count; each key
        is a frame number, flags saying how many spline terms follow, those terms and the value."""
        data = self.data
        if start + 14 > end:
            raise MeshportError("This 3DS file is damaged (an animation track is cut short)")
        if struct.unpack_from("<I", data, start + 10)[0] == 0:
            return None
        pos = start + 14
        if pos + 6 > end:
            raise MeshportError("This 3DS file is damaged (an animation key is cut short)")
        flags = struct.unpack_from("<H", data, pos + 4)[0]
        pos += 6 + 4 * (flags & 0x1F).bit_count()  # one float per spline term present
        if pos + 4 * width > end:
            raise MeshportError("This 3DS file is damaged (an animation key is cut short)")
        value = struct.unpack_from(f"<{width}f", data, pos)
        return value if all(np.isfinite(value)) else None

    def _hierarchy(self, keys: list[dict], objects: dict[str, list[Node]], loose: list[Node],
                   exact: bool) -> list[Node]:
        """Nodes placed by the keyframer at its first frame, parented, plus any object it
        doesn't mention. With exact, or when a node hangs from something meshport can't
        place (a light, a missing node), an object keeps the pose it was saved in."""
        saved = {id(n): n.matrix.copy() for n in loose}  # world, Y up
        keep: dict[int, np.ndarray] = {}  # node -> the world matrix it keeps
        targets = {k["name"]: k[POS_TRACK] for k in keys if k["kind"] == "target" and k[POS_TRACK]}
        placeable = {k["id"] for k in keys if k["kind"] in ("object", "camera")}
        placed: list[tuple[Node, int]] = []
        by_id: dict[int, Node] = {}
        used: set[int] = set()
        cameras: set[int] = set()
        pivoted: dict[tuple, Mesh] = {}
        for key in keys:
            if key["kind"] == "camera":
                node = Node(key["name"] or "camera", _camera_matrix(key, targets, self.cameras))
                cameras.add(id(node))
                placed.append((node, key["parent"]))
                by_id.setdefault(key["id"], node)
                continue
            if key["kind"] != "object":
                continue
            sources = [] if key["name"] == DUMMY else objects.get(key["name"], [])
            if not sources:  # a dummy, or a node whose object isn't there: keep it as a group
                node = Node(key["instance"] or key["name"] or "group")
                parts = []
            elif id(sources[0]) in used:  # another instance of an object already placed
                src = sources[0]
                node = Node(f"{key['name']}.{key['instance'] or len(placed)}", mesh=src.mesh, materials=src.materials)
                parts = [Node(n.name, mesh=n.mesh, materials=n.materials) for n in sources[1:]]
            else:
                node, parts = sources[0], sources[1:]
                used.update(id(n) for n in sources)
                for part in parts:
                    part.matrix = np.eye(4)
            parent_id = key["parent"]
            if parent_id >= 0 and parent_id not in placeable:
                parent_id = -1
                if sources:
                    keep[id(node)] = saved[id(sources[0])]
            elif exact and id(node) in saved:
                keep[id(node)] = saved[id(node)]
            if id(node) in keep:
                node.matrix = _key_matrix(key)  # for now: replaced below, once its parent is placed
            else:
                # The mesh is kept relative to its own matrix; the node moves it from its pivot.
                if any(key["pivot"]):
                    for n in [node, *parts]:
                        if n.mesh is not None:
                            k = (id(n.mesh), key["pivot"])
                            if k not in pivoted:
                                pivoted[k] = _shifted(n.mesh, -np.array(key["pivot"]))
                            n.mesh = pivoted[k]
                node.matrix = _key_matrix(key)
            node.children = parts
            placed.append((node, parent_id))
            by_id.setdefault(key["id"], node)
        parent_of: dict[int, Node] = {}
        roots = []
        for node, parent_id in placed:
            parent = by_id.get(parent_id) if parent_id >= 0 else None
            up = parent
            while up is not None and up is not node:  # refuse links that would make a loop
                up = parent_of.get(id(up))
            if parent is None or up is node:
                roots.append(node)
            else:
                parent.children.append(node)
                parent_of[id(node)] = parent
        for node in roots:
            node.matrix = _Z_UP @ node.matrix

        def settle(node: Node, parent_world: np.ndarray):
            if id(node) in keep and abs(np.linalg.det(parent_world)) > 1e-12:
                node.matrix = np.linalg.inv(parent_world) @ keep[id(node)]
            for child in node.children:
                settle(child, parent_world @ node.matrix)

        for node in roots:
            settle(node, np.eye(4))
        roots = [n for n in roots if id(n) not in cameras or n.children]  # cameras only stay as parents
        for node, _ in Scene(roots).walk():
            node.children = [c for c in node.children if id(c) not in cameras or c.children]
        return roots + [n for n in loose if id(n) not in used]

    def _trimesh(self, name: str, start: int, end: int) -> Node | None:
        data = self.data
        points = uvs = matrix = faces = None
        groups: list[tuple[str, np.ndarray]] = []
        smooth = None
        for cid, s, e in _chunks(data, start, end):
            if cid == POINT_ARRAY:
                points = _counted(data, s, e, "<f4", 3)[0].astype(np.float64)
            elif cid == TEX_VERTS:
                uvs = _counted(data, s, e, "<f4", 2)[0].astype(np.float64)
            elif cid == MESH_MATRIX and e - s >= 48:
                rows = np.frombuffer(data, "<f4", 12, s).astype(np.float64).reshape(4, 3)
                matrix = np.eye(4)
                matrix[:3, :] = rows.T  # x axis, y axis, z axis, origin
            elif cid == FACE_ARRAY:
                faces, pos = _counted(data, s, e, "<u2", 4)
                for sub, ss, se in _chunks(data, pos, e):
                    if sub == MSH_MAT_GROUP:
                        mat_name, p = _cstring(data, ss, se)
                        groups.append((mat_name, _counted(data, p, se, "<u2", 1)[0].ravel()))
                    elif sub == SMOOTH_GROUP:
                        n = min(len(faces), (se - ss) // 4)
                        smooth = np.zeros(len(faces), np.int64)
                        smooth[:n] = np.frombuffer(data, "<u4", n, ss)
        if points is None or faces is None or not len(faces) or not len(points):
            return None
        tris = faces[:, :3].astype(np.int64)
        if tris.max() >= len(points):
            raise MeshportError(f"Object {name!r} has a face that points past its vertex list")
        # The points are stored in world space; the matrix is where the object's own axes are.
        if matrix is None or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
            matrix = np.eye(4)
        positions = transform_points(np.linalg.inv(matrix), points)
        if np.linalg.det(matrix) < 0:  # a mirrored matrix flips the faces: keep them facing out
            tris = tris[:, ::-1]
        corners = tris.ravel()
        mesh = Mesh(name, positions, np.full(len(tris), 3, np.int64), corners)
        if uvs is not None and len(uvs) >= len(points):
            mesh.uvs = [uvs[corners]]
        if smooth is not None:
            mesh.normals = _smooth_normals(positions, tris, smooth)
        node = Node(name, _Z_UP @ matrix, mesh)
        if groups:
            slots = np.full(len(tris), -1, np.int64)
            for mat_name, faces_in in groups:
                faces_in = faces_in[faces_in < len(tris)].astype(np.int64)
                mat = self._named_material(mat_name)
                if mat not in node.materials:
                    node.materials.append(mat)
                slots[faces_in] = node.materials.index(mat)
            if (slots < 0).any():
                node.materials.append(self._named_material(""))
                slots[slots < 0] = len(node.materials) - 1
            mesh.face_materials = slots
        mesh.validate()
        return node

    # ---- materials

    def _named_material(self, name: str) -> Material:
        if name not in self.materials:
            self.materials[name] = Material(name or "default")
        return self.materials[name]

    def _material(self, start: int, end: int) -> Material:
        data = self.data
        name, diffuse, alpha, shininess, self_illum = "", (0.8, 0.8, 0.8), 1.0, None, 0.0
        two_sided = False
        maps: dict[int, Image | None] = {}
        for cid, s, e in _chunks(data, start, end):
            if cid == MAT_NAME:
                name = _cstring(data, s, e)[0]
            elif cid == MAT_DIFFUSE:
                diffuse = self._color(s, e) or diffuse
            elif cid == MAT_TRANSPARENCY:
                alpha = 1.0 - self._percent(s, e, 0.0)
            elif cid == MAT_SHININESS:
                shininess = self._percent(s, e, 0.0)
            elif cid == MAT_SELF_ILPCT:
                self_illum = self._percent(s, e, 0.0)
            elif cid == MAT_TWO_SIDE:
                two_sided = True
            elif cid in (MAT_TEXMAP, MAT_BUMPMAP, MAT_SELFIMAP):
                maps[cid] = self._map(s, e)
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha < 0.01:  # a fully invisible material is an exporter mix-up, not intent
            alpha = 1.0
        mat = Material(name or "material", (*diffuse, alpha), double_sided=two_sided)
        if shininess is not None:
            mat.roughness = float(np.clip(1.0 - shininess, 0.0, 1.0))
        if alpha < 1.0:
            mat.alpha_mode = "BLEND"
        if self_illum > 0:
            mat.emissive = tuple(float(min(c * self_illum, 1.0)) for c in diffuse)
        # A texture that was found replaces the color under it; a missing one leaves the color.
        if maps.get(MAT_TEXMAP):
            mat.base_color_texture = maps[MAT_TEXMAP]
            if mat.base_color_texture.data:
                mat.base_color = (1.0, 1.0, 1.0, alpha)
        mat.normal_texture = maps.get(MAT_BUMPMAP)
        if maps.get(MAT_SELFIMAP):
            mat.emissive_texture = maps[MAT_SELFIMAP]
            if not any(mat.emissive) and mat.emissive_texture.data:
                mat.emissive = (1.0, 1.0, 1.0)
        return mat

    def _color(self, start: int, end: int) -> tuple[float, float, float] | None:
        found = {}
        for cid, s, e in _chunks(self.data, start, end):
            if cid in (COLOR_F, LIN_COLOR_F) and e - s >= 12:
                found[cid] = struct.unpack_from("<3f", self.data, s)
            elif cid in (COLOR_24, LIN_COLOR_24) and e - s >= 3:
                found[cid] = tuple(c / 255.0 for c in self.data[s:s + 3])
        for cid in (LIN_COLOR_F, LIN_COLOR_24, COLOR_F, COLOR_24):  # linear values first
            if cid in found and all(np.isfinite(found[cid])):
                return tuple(float(np.clip(c, 0.0, 1.0)) for c in found[cid])
        return None

    def _percent(self, start: int, end: int, default: float) -> float:
        for cid, s, e in _chunks(self.data, start, end):
            if cid == INT_PERCENTAGE and e - s >= 2:
                return struct.unpack_from("<h", self.data, s)[0] / 100.0
            if cid == FLOAT_PERCENTAGE and e - s >= 4:
                v = struct.unpack_from("<f", self.data, s)[0]
                if np.isfinite(v):
                    return v / 100.0 if v > 1.0 else v
        return default

    def _map(self, start: int, end: int) -> Image | None:
        for cid, s, e in _chunks(self.data, start, end):
            if cid == MAT_MAPNAME:
                file = _cstring(self.data, s, e)[0].strip()
                if not file:
                    return None
                if file.lower() not in self.images:
                    data = self.finder.find([file])
                    self.images[file.lower()] = Image(Path(file.replace("\\", "/")).stem, data, file)
                return self.images[file.lower()]
        return None


def _key_matrix(key: dict) -> np.ndarray:
    """A keyframer node's local matrix at its first frame: translate, rotate, scale."""
    m = translation(key[POS_TRACK] or (0.0, 0.0, 0.0))
    if key[ROT_TRACK]:
        angle, *axis = key[ROT_TRACK]
        axis = np.array(axis, np.float64)
        length = np.linalg.norm(axis)
        if length > 0:
            x, y, z = axis / length
            k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            r = np.eye(4)
            # 3DS turns the other way round its axis
            r[:3, :3] = np.eye(3) - np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
            m = m @ r
    return m @ scaling(key[SCL_TRACK] or (1.0, 1.0, 1.0))


def _camera_matrix(key: dict, targets: dict, cameras: dict) -> np.ndarray:
    """A camera node's matrix at its first frame. 3DS cameras aim at a target point; as in
    3ds Max, the camera's local -Z looks at it, +Y is up and the roll turns it about -Z."""
    position, target, roll = cameras.get(key["name"], ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0))
    position = np.array(key[POS_TRACK] or position, np.float64)
    target = np.array(targets.get(key["name"]) or target, np.float64)
    roll = key[ROLL_TRACK][0] if key[ROLL_TRACK] else roll
    forward = target - position
    if not np.linalg.norm(forward):
        return translation(position)
    forward /= np.linalg.norm(forward)
    up = (0.0, 0.0, 1.0) if np.linalg.norm(forward[:2]) > 1e-9 else (-1.0, 0.0, 0.0)  # looking straight up or down
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    m = translation(position)
    m[:3, :3] = np.column_stack([right, np.cross(right, forward), -forward])
    a = np.radians(roll)
    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m @ turn


def _shifted(mesh: Mesh, offset: np.ndarray) -> Mesh:
    return dataclasses.replace(mesh, positions=mesh.positions + offset)


def _smooth_normals(positions: np.ndarray, tris: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-corner normals from smoothing groups: a corner averages the faces around its
    vertex that share a group bit with its own face; faces in no group stay flat."""
    p = positions[tris]
    face_n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    length = np.linalg.norm(face_n, axis=1, keepdims=True)
    face_n = np.divide(face_n, length, out=np.zeros_like(face_n), where=length > 0)
    # Each face counts by its angle at the corner, as in Blender.
    edge_a, edge_b = np.roll(p, -1, axis=1) - p, np.roll(p, 1, axis=1) - p
    cos = np.einsum("fki,fki->fk", edge_a, edge_b) / np.maximum(
        np.linalg.norm(edge_a, axis=2) * np.linalg.norm(edge_b, axis=2), 1e-300)
    weighted = face_n[:, None, :] * np.arccos(np.clip(cos, -1.0, 1.0))[:, :, None]  # (F, 3 corners, 3)
    out = np.repeat(face_n, 3, axis=0).reshape(-1, 3, 3)
    masks = np.unique(groups[groups != 0])

    def around(faces: np.ndarray) -> np.ndarray:
        """Per vertex: the angle-weighted sum of these faces' normals."""
        idx, w = tris[faces].ravel(), weighted[faces].reshape(-1, 3)
        return np.stack([np.bincount(idx, w[:, k], len(positions)) for k in range(3)], 1)

    if len(masks) <= 64:  # exact: each distinct mask gathers every face it shares a bit with
        for m in masks:
            own = groups == m
            out[own] = around((groups & m) != 0)[tris[own]]
    else:  # a hostile number of masks: sum per bit instead (faces sharing two bits count twice)
        acc = np.zeros_like(out)
        for bit in range(32):
            has = (groups >> bit) & 1 == 1
            if has.any():
                acc[has] += around(has)[tris[has]]
        out[groups != 0] = acc[groups != 0]
    out = out.reshape(-1, 3)
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, length, out=np.zeros_like(out), where=length > 0)


# ============================================================================ writing


def write(scene: Scene) -> bytes:
    """3DS bytes. Textures are referenced by file name; write_file also writes them."""
    return _Writer(scene).build()


def write_file(scene: Scene, path: str | os.PathLike) -> list[Path]:
    """Write path (.3ds) and the texture files it refers to next to it. Returns every file written."""
    path = Path(path)
    writer = _Writer(scene)
    data = writer.build()
    written = [path]
    for image, name in writer.image_files():
        (path.parent / name).write_bytes(image.data)
        written.append(path.parent / name)
    path.write_bytes(data)
    return written


def _chunk(cid: int, *parts: bytes) -> bytes:
    body = b"".join(parts)
    if len(body) + 6 > 0xFFFFFFFF:
        raise MeshportError("This scene is too big for a 3DS file (4 GB)")
    return _HEAD.pack(cid, len(body) + 6) + body


def _string(s: str) -> bytes:
    return s.encode("latin-1") + b"\0"


def _short_name(name: str, limit: int, taken: set[str]) -> str:
    """name cut to limit characters of plain ASCII, unique (ignoring case) among taken."""
    base = "".join(c if " " <= c <= "~" else "_" for c in name).strip()[:limit] or "object"
    out, n = base, 1
    while out.lower() in taken:
        suffix = f"_{n}"
        out = base[:limit - len(suffix)] + suffix
        n += 1
    taken.add(out.lower())
    return out


def _split(index: np.ndarray, limit: int = LIMIT) -> list[tuple[int, int]]:
    """Ranges of triangles that each use at most limit vertices and hold at most limit faces."""
    out, start, total = [], 0, len(index)
    while start < total:
        stop = min(total, start + limit)
        if len(np.unique(index[start:stop])) > limit:
            lo, hi = min(total, start + limit // 3), stop - 1  # limit // 3 triangles always fit
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(np.unique(index[start:mid])) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            stop = lo
        out.append((start, stop))
        start = stop
    return out


class _Writer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.mat_names: dict[int, str] = {}
        self.mat_taken: set[str] = set()
        self.materials: list[bytes] = []
        self.images: dict[int, tuple[Image, str]] = {}
        self.object_taken: set[str] = set()

    def image_files(self) -> list[tuple[Image, str]]:
        return [(img, name) for img, name in self.images.values() if img.data]

    def build(self) -> bytes:
        objects = []
        for node, world in self.scene.walk():
            if node.mesh is not None and len(node.mesh.corners):
                objects += self._objects(node, world)
        if not objects:
            raise MeshportError("No 3D geometry to write")
        edit = _chunk(EDIT, _chunk(MESH_VERSION, struct.pack("<I", 3)), *self.materials,
                      _chunk(MASTER_SCALE, struct.pack("<f", 1.0)), *objects)
        return _chunk(MAIN, _chunk(VERSION, struct.pack("<I", 3)), edit)

    def _objects(self, node: Node, world: np.ndarray) -> list[bytes]:
        mesh = node.mesh
        matrix = np.linalg.inv(_Z_UP) @ world  # the object's axes, in 3DS space
        tris = mesh.triangles()
        if not len(tris):
            return []
        tri_face = mesh.triangle_faces()
        if np.linalg.det(world[:3, :3]) < 0:  # mirrored: the points go out in world space, so flip
            tris = tris[:, ::-1]
        corner = tris.ravel()
        # One 3DS vertex per distinct (vertex, uv, normal): 3DS has one UV per vertex, and
        # splitting where normals change keeps hard edges hard under a single smoothing group.
        # Vertices the mesh keeps apart stay apart, even where they sit at the same spot.
        columns = [transform_points(matrix, mesh.positions)[mesh.corners[corner]]]
        if mesh.uvs:
            columns.append(mesh.uvs[0][corner])
        if mesh.normals is not None:
            columns.append(mesh.normals[corner])
        source = mesh.corners[corner].astype(np.int32).view(np.float32)[:, None]  # compared by its bits
        table = np.hstack([np.hstack(columns).astype(np.float32), source])
        first, inverse = unique_rows(table)
        vertices = table[first]
        index = inverse.reshape(-1, 3)

        slots = None
        if node.materials:
            slots = np.zeros(len(tris), np.int64) if mesh.face_materials is None else \
                np.clip(mesh.face_materials[tri_face], 0, len(node.materials) - 1)
            names = [self._material(m) for m in node.materials]
        matrix_bytes = _chunk(MESH_MATRIX, matrix[:3, :].T.astype("<f4").tobytes())
        out = []
        for start, stop in _split(index):
            used, local = np.unique(index[start:stop], return_inverse=True)
            local = local.reshape(-1, 3)
            v = vertices[used]
            parts = [_chunk(POINT_ARRAY, struct.pack("<H", len(v)), v[:, :3].astype("<f4").tobytes())]
            if mesh.uvs:
                parts.append(_chunk(TEX_VERTS, struct.pack("<H", len(v)), v[:, 3:5].astype("<f4").tobytes()))
            parts.append(matrix_bytes)
            faces = np.hstack([local, np.full((len(local), 1), 7)]).astype("<u2")  # 7: all edges visible
            subs = []
            if slots is not None:
                part = slots[start:stop]
                for slot in np.unique(part):
                    members = np.nonzero(part == slot)[0].astype("<u2")
                    subs.append(_chunk(MSH_MAT_GROUP, _string(names[int(slot)]), struct.pack("<H", len(members)),
                                       members.tobytes()))
            if mesh.normals is not None:
                subs.append(_chunk(SMOOTH_GROUP, np.ones(len(local), "<u4").tobytes()))
            parts.append(_chunk(FACE_ARRAY, struct.pack("<H", len(local)), faces.tobytes(), *subs))
            name = _short_name(node.name or mesh.name, _OBJECT_NAME, self.object_taken)
            out.append(_chunk(OBJECT, _string(name), _chunk(TRIMESH, *parts)))
        return out

    def _material(self, mat: Material) -> str:
        if id(mat) in self.mat_names:
            return self.mat_names[id(mat)]
        name = _short_name(mat.name or "material", _MATERIAL_NAME, self.mat_taken)
        self.mat_names[id(mat)] = name
        r, g, b, a = (float(np.clip(c, 0, 1)) for c in mat.base_color)
        specular = 0.5 * (1 - mat.roughness)
        parts = [_chunk(MAT_NAME, _string(name)),
                 _chunk(MAT_AMBIENT, _color((r, g, b))), _chunk(MAT_DIFFUSE, _color((r, g, b))),
                 _chunk(MAT_SPECULAR, _color((specular,) * 3)),
                 _chunk(MAT_SHININESS, _percent(1 - mat.roughness)), _chunk(MAT_SHIN2PCT, _percent(specular)),
                 _chunk(MAT_TRANSPARENCY, _percent(1 - a))]
        if mat.double_sided:
            parts.append(_chunk(MAT_TWO_SIDE))
        if any(mat.emissive):
            parts.append(_chunk(MAT_SELF_ILPCT, _percent(min(max(mat.emissive), 1.0))))
        for cid, image in ((MAT_TEXMAP, mat.base_color_texture), (MAT_BUMPMAP, mat.normal_texture),
                           (MAT_SELFIMAP, mat.emissive_texture)):
            file = self._image_name(image)
            if file:
                parts.append(_chunk(cid, _chunk(INT_PERCENTAGE, struct.pack("<h", 100)),
                                    _chunk(MAT_MAPNAME, _string(file))))
        self.materials.append(_chunk(MATERIAL, *parts))
        return name

    def _image_name(self, image: Image | None) -> str | None:
        """An 8.3 file name, the most every 3DS reader accepts."""
        if image is None or not (image.data or image.path):
            return None
        if id(image) not in self.images:
            path = Path((image.path or "").replace("\\", "/"))
            ext = _EXT.get(image.mime or "") or path.suffix.lstrip(".").lower() or "png"
            ext = "".join(c for c in ext if c.isascii() and c.isalnum())[:3] or "png"
            stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in (path.stem or image.name or "texture"))
            taken = {name.lower() for _, name in self.images.values()}
            name = _short_name(stem or "texture", 8, {n.rsplit(".", 1)[0] for n in taken})
            self.images[id(image)] = (image, f"{name}.{ext}")
        return self.images[id(image)][1]


def _color(rgb) -> bytes:
    """24-bit color (what every reader takes) plus the exact linear float color."""
    rgb = [float(np.clip(c, 0, 1)) for c in rgb]
    return (_chunk(COLOR_24, bytes(round(c * 255) for c in rgb))
            + _chunk(LIN_COLOR_F, struct.pack("<3f", *rgb)))


def _percent(v: float) -> bytes:
    return _chunk(INT_PERCENTAGE, struct.pack("<h", round(float(np.clip(v, 0, 1)) * 100)))
