"""FBX <-> Scene.

Reads binary and ASCII FBX, versions 6.x and 7.x (everything from 2006 on). Writes binary
FBX 7.4, which every FBX importer accepts.

What comes across: the node hierarchy with full FBX transforms (pivots, pre/post rotation,
rotation order, geometric offsets), polygons, normals, every UV set, vertex colors,
per-face materials and their textures (embedded or found next to the file). Skinning,
blend shapes, animation, cameras and lights are skipped.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .fbxtree import CREATION_TIME, FILE_ID, Record, as_array, decode_content, parse, write_binary
from .geometry import (EULER_ORDERS, compose, decompose, euler, scaling, transform_normals, transform_points,
                       translation, unique_rows)
from .scene import Image, Material, Mesh, MeshportError, Node, Scene

# ============================================================================ reading


@dataclass(eq=False)
class _Obj:
    id: object
    cls: str        # Model, Geometry, Material, Texture, Video, ...
    sub: str        # Mesh, Null, LimbNode, Clip, ...
    name: str
    rec: Record
    props: dict = field(default_factory=dict)
    template: dict = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return bool(self.props.get(name) or self.template.get(name))

    def prop(self, name: str, default=None):
        vals = self.props.get(name, self.template.get(name))
        if not vals:
            return default
        if isinstance(default, tuple) or (isinstance(default, np.ndarray)):
            nums = [float(v) for v in vals if isinstance(v, (int, float))]
            return tuple(nums[:len(default)]) if len(nums) >= len(default) else default
        v = vals[0]
        if isinstance(default, float):
            return float(v) if isinstance(v, (int, float)) else default
        if isinstance(default, int) and not isinstance(default, bool):
            return int(v) if isinstance(v, (int, float)) else default
        return v


def _split_name(s) -> tuple[str, str]:
    """('Cube', 'Model') from binary 'Cube\\x00\\x01Model' or ASCII 'Model::Cube'."""
    s = s if isinstance(s, str) else str(s)
    if "\x00\x01" in s:
        name, cls = s.split("\x00\x01", 1)
        return name, cls
    if "::" in s:
        cls, name = s.split("::", 1)
        return name, cls
    return s, ""


def _properties(rec: Record) -> dict:
    """Properties70 (P: name, type, label, flags, values...) or the 6.x Properties60
    (Property: name, type, flags, values...) as {name: [values]}."""
    out = {}
    p70 = rec.find("Properties70")
    if p70 is not None:
        for p in p70.find_all("P"):
            if p.props:
                out[str(p.props[0])] = p.props[4:]
        return out
    p60 = rec.find("Properties60")
    if p60 is not None:
        for p in p60.find_all("Property"):
            if p.props:
                out[str(p.props[0])] = p.props[3:]
    return out


def read(data: bytes, base_dir: str | os.PathLike | None = None) -> Scene:
    """Parse FBX bytes into a Scene. base_dir is where to look for external textures."""
    version, records = parse(data)
    top = {}
    for r in records:
        top.setdefault(r.name, r)
    if "Objects" not in top:
        raise MeshportError("This FBX file has no objects in it")
    return _Reader(version, top, Path(base_dir) if base_dir else None).scene()


def read_file(path: str | os.PathLike) -> Scene:
    path = Path(path)
    return read(path.read_bytes(), path.parent)


class _Reader:
    def __init__(self, version: int, top: dict, base_dir: Path | None):
        self.version, self.top, self.base_dir = version, top, base_dir
        self.legacy = version < 7000
        self.objects: dict[object, _Obj] = {}
        self.children: dict[object, list[tuple[str, object, str | None]]] = {}  # parent -> [(kind, child, prop)]
        self.parent_of: dict[object, list[object]] = {}
        self.images: dict[object, Image] = {}
        self._load_templates()
        self._load_objects()
        self._load_connections()

    # ---- object graph

    def _load_templates(self):
        self.templates = {}
        defs = self.top.get("Definitions")
        if defs is None:
            return
        for ot in defs.find_all("ObjectType"):
            tpl = ot.find("PropertyTemplate")
            if ot.props and tpl is not None:
                self.templates[str(ot.props[0])] = _properties(tpl)

    def _load_objects(self):
        for rec in self.top["Objects"].children:
            if self.legacy:
                if not rec.props:
                    continue
                oid = self._legacy_id(rec.props[0], rec.name)
                name = _split_name(rec.props[0])[0]
                sub = str(rec.props[1]) if len(rec.props) > 1 else ""
            else:
                if len(rec.props) < 2:
                    continue
                oid = rec.props[0]
                name = _split_name(rec.props[1])[0]
                sub = str(rec.props[2]) if len(rec.props) > 2 else ""
            obj = _Obj(oid, rec.name, sub, name, rec, _properties(rec), self.templates.get(rec.name, {}))
            self.objects[oid] = obj
            if self.legacy and rec.name == "Model" and rec.find("Vertices") is not None:
                # 6.x keeps the mesh inside its model; give it a geometry object of its own
                gid = ("geometry", oid)
                self.objects[gid] = _Obj(gid, "Geometry", "Mesh", name, rec)
                self._connect("OO", gid, oid, None)

    def _legacy_id(self, raw, cls: str):
        name, c = _split_name(raw)
        return f"{c or cls}::{name}"

    def _connect(self, kind, child, parent, prop):
        self.children.setdefault(parent, []).append((kind, child, prop))
        self.parent_of.setdefault(child, []).append(parent)

    def _load_connections(self):
        conns = self.top.get("Connections")
        if conns is None:
            return
        for c in conns.children:
            if len(c.props) < 3:
                continue
            kind = str(c.props[0])
            child, parent = c.props[1], c.props[2]
            if self.legacy:
                child, parent = self._legacy_id(child, ""), self._legacy_id(parent, "")
                if parent == "Model::Scene":
                    parent = 0
            prop = str(c.props[3]) if len(c.props) > 3 else None
            self._connect(kind, child, parent, prop)

    def _kids(self, oid, cls: str | None = None, kind: str = "OO"):
        for k, child, prop in self.children.get(oid, []):
            obj = self.objects.get(child)
            if obj is not None and k == kind and (cls is None or obj.cls == cls):
                yield obj, prop

    # ---- scene

    def scene(self) -> Scene:
        models = {oid: o for oid, o in self.objects.items() if o.cls == "Model"}
        nodes = {oid: self._node(o) for oid, o in models.items()}
        roots = []
        for oid, node in nodes.items():
            parents = [p for p in self.parent_of.get(oid, []) if p in nodes]
            if parents and parents[0] != oid:
                nodes[parents[0]].children.append(node)
            else:
                roots.append(node)
        # Viewport cameras ("Producer Perspective", ...) and lights carry nothing we keep.
        junk = {id(nodes[oid]) for oid, o in models.items()
                if (o.sub in ("Camera", "Light", "CameraSwitcher") or o.name == "Camera Switcher")
                and not nodes[oid].children}
        roots = [r for r in roots if id(r) not in junk]
        for node, _ in Scene(roots).walk():
            node.children = [c for c in node.children if id(c) not in junk]
        conversion = self._axis_conversion()
        for root in roots:
            root.matrix = conversion @ root.matrix
        scene = Scene(roots)
        if not scene.meshes:
            raise MeshportError("No 3D geometry found in this file")
        return scene

    def _axis_conversion(self) -> np.ndarray:
        gs = self.top.get("GlobalSettings") or self.top["Objects"].find("GlobalSettings")  # 6.x: inside Objects
        props = _properties(gs) if gs is not None else {}

        def get(name, default):
            v = props.get(name)
            return v[0] if v and isinstance(v[0], (int, float)) else default

        up, up_sign = int(get("UpAxis", 1)), get("UpAxisSign", 1)
        front, front_sign = int(get("FrontAxis", 2)), get("FrontAxisSign", 1)
        coord, coord_sign = int(get("CoordAxis", 0)), get("CoordAxisSign", 1)
        unit = float(get("UnitScaleFactor", 1.0)) or 1.0
        m = np.eye(4)
        if len({up, front, coord}) == 3 and max(up, front, coord) <= 2:
            m[:3, :3] = 0
            m[0, coord] = 1 if coord_sign >= 0 else -1
            m[1, up] = 1 if up_sign >= 0 else -1
            m[2, front] = 1 if front_sign >= 0 else -1
        return scaling([unit / 100.0] * 3) @ m

    def _node(self, model: _Obj) -> Node:
        node = Node(model.name, self._local_matrix(model))
        geoms = [g for g, _ in self._kids(model.id, "Geometry") if g.sub in ("Mesh", "")]
        materials = [self._material(m) for m, _ in self._kids(model.id, "Material")]
        if self.legacy:  # 6.x hangs the diffuse texture on the model, not the material
            images = [img for t, _ in self._kids(model.id, "Texture") if (img := self._texture_image(t))]
            if images:
                materials = materials or [Material(f"{model.name}_material")]
                for mat in materials:
                    if mat.base_color_texture is None:
                        mat.base_color_texture = images[0]
                        mat.base_color = (1.0, 1.0, 1.0, mat.base_color[3])
        geometric = self._geometric_matrix(model)
        for i, geom in enumerate(geoms):
            mesh = self._mesh(geom, geometric)
            if mesh is None:
                continue
            if node.mesh is None:
                node.mesh, node.materials = mesh, materials
            else:  # a second mesh on one model: hang it underneath
                node.children.append(Node(f"{model.name}.{i}", mesh=mesh, materials=materials))
        return node

    def _local_matrix(self, o: _Obj) -> np.ndarray:
        zero, one = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        t = o.prop("Lcl Translation", zero)
        r = o.prop("Lcl Rotation", zero)
        s = o.prop("Lcl Scaling", one)
        r_off, r_piv = o.prop("RotationOffset", zero), o.prop("RotationPivot", zero)
        s_off, s_piv = o.prop("ScalingOffset", zero), o.prop("ScalingPivot", zero)
        pre, post = o.prop("PreRotation", zero), o.prop("PostRotation", zero)
        order = EULER_ORDERS[min(max(o.prop("RotationOrder", 0), 0), 5)]
        return (translation(t) @ translation(r_off) @ translation(r_piv)
                @ euler(pre) @ euler(r, order) @ np.linalg.inv(euler(post))
                @ translation(-np.array(r_piv)) @ translation(s_off) @ translation(s_piv)
                @ scaling(s) @ translation(-np.array(s_piv)))

    def _geometric_matrix(self, o: _Obj) -> np.ndarray | None:
        zero, one = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        t = o.prop("GeometricTranslation", zero)
        r = o.prop("GeometricRotation", zero)
        s = o.prop("GeometricScaling", one)
        if t == zero and r == zero and s == one:
            return None
        return compose(t, r, s)

    # ---- meshes

    def _mesh(self, geom: _Obj, geometric: np.ndarray | None) -> Mesh | None:
        key = (geom.id, None if geometric is None else geometric.tobytes())
        cache = self.__dict__.setdefault("_mesh_cache", {})
        if key not in cache:
            cache[key] = self._build_mesh(geom, geometric)
        return cache[key]

    def _build_mesh(self, geom: _Obj, geometric: np.ndarray | None) -> Mesh | None:
        rec = geom.rec
        verts = rec.array("Vertices")
        pvi = rec.array("PolygonVertexIndex", np.int64)
        if verts is None or pvi is None or len(verts) < 9 or not len(pvi):
            return None
        positions = verts[: len(verts) // 3 * 3].reshape(-1, 3)
        ends = np.nonzero(pvi < 0)[0]
        if not len(ends):
            return None
        pvi = pvi[: ends[-1] + 1]
        corners = np.where(pvi < 0, ~pvi, pvi)
        sizes = np.diff(np.concatenate(([-1], ends)))
        n_faces, n_corners = len(sizes), len(corners)
        face_of_corner = np.repeat(np.arange(n_faces), sizes)
        if corners.max() >= len(positions):
            raise MeshportError(f"Mesh {geom.name!r} refers to vertices it doesn't have")

        ctx = (corners, face_of_corner, n_corners, n_faces, len(positions))
        normals = self._layer(rec, "LayerElementNormal", "Normals", "NormalsIndex", 3, ctx)
        colors = self._layer(rec, "LayerElementColor", "Colors", "ColorIndex", 4, ctx)
        uvs = []
        for elem in sorted(rec.find_all("LayerElementUV"), key=lambda e: e.props[0] if e.props else 0):
            uv = self._layer_values(elem, "UV", "UVIndex", 2, ctx)
            if uv is not None:
                uvs.append(uv)
        face_materials = None
        mat_elem = rec.find("LayerElementMaterial")
        if mat_elem is not None:
            mats = mat_elem.array("Materials", np.int64)
            mapping = str(mat_elem.value("MappingInformationType", "AllSame"))
            if mats is not None and len(mats):
                if mapping == "ByPolygon" and len(mats) >= n_faces:
                    face_materials = mats[:n_faces].copy()
                else:
                    face_materials = np.full(n_faces, mats[0], np.int64)
                face_materials[face_materials < 0] = 0

        keep_face = sizes >= 3
        if not keep_face.all():  # drop points and lines
            keep_corner = np.repeat(keep_face, sizes)
            sizes, corners = sizes[keep_face], corners[keep_corner]
            normals = None if normals is None else normals[keep_corner]
            colors = None if colors is None else colors[keep_corner]
            uvs = [uv[keep_corner] for uv in uvs]
            face_materials = None if face_materials is None else face_materials[keep_face]
            if not len(sizes):
                return None

        if geometric is not None:
            positions = transform_points(geometric, positions)
            if normals is not None:
                normals = transform_normals(geometric, normals)
            if np.linalg.det(geometric[:3, :3]) < 0:
                sizes, corners, normals, colors, uvs = _flip_winding(sizes, corners, normals, colors, uvs)
        mesh = Mesh(geom.name, positions.astype(np.float64), sizes.astype(np.int64), corners.astype(np.int64),
                    normals, uvs, colors, face_materials)
        mesh.validate()
        return mesh

    def _layer(self, rec, elem_name, data_name, index_name, width, ctx):
        elems = sorted(rec.find_all(elem_name), key=lambda e: e.props[0] if e.props else 0)
        return self._layer_values(elems[0], data_name, index_name, width, ctx) if elems else None

    @staticmethod
    def _layer_values(elem: Record, data_name, index_name, width, ctx) -> np.ndarray | None:
        corners, face_of_corner, n_corners, n_faces, n_verts = ctx
        data = elem.array(data_name)
        if data is None or len(data) < width:
            return None
        data = data[: len(data) // width * width].reshape(-1, width)
        mapping = str(elem.value("MappingInformationType", "ByPolygonVertex"))
        ref = str(elem.value("ReferenceInformationType", "Direct"))
        if mapping == "ByPolygonVertex":
            per = np.arange(n_corners)
        elif mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
            per = corners
        elif mapping == "ByPolygon":
            per = face_of_corner
        elif mapping == "AllSame":
            per = np.zeros(n_corners, np.int64)
        else:  # ByEdge and anything else carries nothing we can use per corner
            return None
        if ref in ("IndexToDirect", "Index"):
            index = elem.array(index_name, np.int64)
            if index is not None and len(index):
                if per.max(initial=0) >= len(index):
                    return None
                per = index[per]
        valid = (per >= 0) & (per < len(data))
        out = data[np.where(valid, per, 0)]
        if not valid.all():
            out[~valid] = 0
        return out.astype(np.float64)

    # ---- materials and textures

    def _material(self, m: _Obj) -> Material:
        cache = self.__dict__.setdefault("_mat_cache", {})
        if m.id in cache:
            return cache[m.id]
        diffuse = m.prop("DiffuseColor" if m.has("DiffuseColor") else "Diffuse", (0.8, 0.8, 0.8))
        factor = m.prop("DiffuseFactor", 1.0)
        if factor > 0:
            diffuse = tuple(c * factor for c in diffuse)
        if "Opacity" in m.props:
            alpha = m.prop("Opacity", 1.0)
        elif "TransparencyFactor" in m.props:
            tint = m.prop("TransparentColor", (1.0, 1.0, 1.0))
            alpha = 1.0 - m.prop("TransparencyFactor", 0.0) * (sum(tint) / 3)
        else:
            alpha = 1.0
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha < 0.01:  # a fully invisible material is an exporter mix-up, not intent
            alpha = 1.0
        emissive = m.prop("EmissiveColor" if m.has("EmissiveColor") else "Emissive", (0.0, 0.0, 0.0))
        emissive = tuple(min(c * m.prop("EmissiveFactor", 1.0), 1.0) for c in emissive)
        shininess = m.prop("ShininessExponent", m.prop("Shininess", 20.0))
        roughness = float(np.clip(np.sqrt(2.0 / (max(shininess, 0.0) + 2.0)), 0.0, 1.0))
        mat = Material(m.name or "material", (*[float(np.clip(c, 0, 1)) for c in diffuse], alpha),
                       roughness=roughness, emissive=emissive)
        if alpha < 1.0:
            mat.alpha_mode = "BLEND"
        for tex, prop in self._kids(m.id, kind="OP"):
            image = self._texture_image(tex)
            if image is None or prop is None:
                continue
            p = prop.lower()
            if p in ("diffusecolor", "diffuse", "maya|basecolor", "maya|texturebasecolor", "3dsmax|parameters|base_color_map"):
                mat.base_color_texture = mat.base_color_texture or image
                mat.base_color = (1.0, 1.0, 1.0, alpha)
            elif p in ("normalmap", "maya|normalcamera", "maya|texturenormal") or (
                    p == "bump" and "norm" in (image.name + (image.path or "")).lower()):
                mat.normal_texture = mat.normal_texture or image
            elif p in ("emissivecolor", "emissive", "maya|emissioncolor"):
                mat.emissive_texture = mat.emissive_texture or image
                if not any(mat.emissive):
                    mat.emissive = (1.0, 1.0, 1.0)
        cache[m.id] = mat
        return mat

    def _texture_image(self, tex: _Obj) -> Image | None:
        if tex.cls == "LayeredTexture":
            inner = next((t for t, _ in self._kids(tex.id, "Texture")), None)
            return self._texture_image(inner) if inner else None
        if tex.cls != "Texture":
            return None
        if tex.id in self.images:
            return self.images[tex.id]
        video = next((v for v, _ in self._kids(tex.id, "Video")), None)
        data = b""
        names = []
        for o in (tex, video):
            if o is None:
                continue
            for key in ("RelativeFilename", "FileName", "Filename"):
                v = o.rec.value(key)
                if isinstance(v, str) and v:
                    names.append(v)
            path = o.prop("Path", "")
            if isinstance(path, str) and path:
                names.append(path)
        if video is not None:
            content = video.rec.find("Content")
            if content is not None:
                data = decode_content(content.props)
        if not data:
            data = self._find_file(names) or b""
        label = tex.name or (Path(names[0].replace("\\", "/")).stem if names else "texture")
        image = Image(label, data or None, names[0] if names else None)
        if image.data is None and image.path is None:
            image = None
        self.images[tex.id] = image
        return image

    _TEXTURE_DIRS = {"texture", "textures", "tex", "materials", "images", "maps"}

    def _find_file(self, names: list[str]) -> bytes | None:
        """Texture paths in FBX files usually point at the artist's own drive. Try the path
        as written, then the file name next to the model, in texture folders up to two
        levels above it (Pack/Models/x.fbx -> Pack/Textures/), and finally anywhere under
        the folder above the model."""
        bases = []
        candidates = []
        for n in names:
            n = n.replace("\\", "/")
            p = Path(n)
            bases.append(p.name.lower())
            if self.base_dir is not None and not p.is_absolute() and ":" not in n:
                candidates.append(self.base_dir / n)
            candidates.append(p)
        for c in candidates:
            try:
                if c.is_file():
                    return c.read_bytes()
            except OSError:
                continue
        if self.base_dir is None:
            return None
        listing = self._listing()
        for base in bases:
            if base in listing:
                return listing[base].read_bytes()
        return None

    def _listing(self) -> dict[str, Path]:
        """Lower-case file name -> path, nearest folders first, capped so a model in a huge
        folder doesn't stall the read."""
        if hasattr(self, "_files"):
            return self._files
        files: dict[str, Path] = {}

        def add(folder: Path):
            try:
                for f in folder.iterdir():
                    if f.is_file():
                        files.setdefault(f.name.lower(), f)
            except OSError:
                pass

        add(self.base_dir)
        for fbm in self.base_dir.glob("*.fbm"):
            add(fbm)
        for up in (self.base_dir, self.base_dir.parent, self.base_dir.parent.parent):
            try:
                subdirs = [d for d in up.iterdir() if d.is_dir() and d.name.lower() in self._TEXTURE_DIRS]
            except OSError:
                subdirs = []
            for d in subdirs:
                add(d)
        budget = 20000
        for dirpath, dirnames, filenames in os.walk(self.base_dir.parent):
            dirnames.sort()
            for f in filenames:
                files.setdefault(f.lower(), Path(dirpath) / f)
            budget -= len(filenames) + 1
            if budget <= 0:
                break
        self._files = files
        return files


def _flip_winding(sizes, corners, normals, colors, uvs):
    """Reverse every polygon's corner order (for mirrored geometry)."""
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    face = np.repeat(np.arange(len(sizes)), sizes)
    local = np.arange(len(corners)) - starts[face]
    order = starts[face] + (sizes[face] - 1 - local)
    pick = (lambda a: None if a is None else a[order])
    return sizes, corners[order], pick(normals), pick(colors), [uv[order] for uv in uvs]


# ============================================================================ writing


def write(scene: Scene) -> bytes:
    """The scene as binary FBX 7.4 bytes, textures embedded."""
    return write_binary(_Writer(scene).records())


def write_file(scene: Scene, path: str | os.PathLike) -> None:
    Path(path).write_bytes(write(scene))


def _p(name: str, type_: str, label: str, flags: str, *values) -> Record:
    return Record("P", [name, type_, label, flags, *values])


def _p_vec(name, type_, label, flags, v) -> Record:
    return _p(name, type_, label, flags, *[float(x) for x in v])


def _fbx_name(name: str, cls: str) -> str:
    return f"{name.replace(chr(0), '').replace(chr(1), '')}\x00\x01{cls}"


_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/tiff": "tif",
        "image/bmp": "bmp", "image/vnd-ms.dds": "dds", "image/vnd.adobe.photoshop": "psd", "image/gif": "gif"}


class _Writer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.next_id = 1_000_000
        self.objects: list[Record] = []
        self.connections: list[Record] = []
        self.counts: dict[str, int] = {}
        self.geom_ids: dict[int, np.int64] = {}
        self.mat_ids: dict[int, np.int64] = {}
        self.tex_ids: dict[tuple[int, str], np.int64] = {}
        self.file_names: dict[int, str] = {}

    def _id(self) -> np.int64:
        self.next_id += 1
        return np.int64(self.next_id)

    def _add(self, rec: Record):
        self.objects.append(rec)
        self.counts[rec.name] = self.counts.get(rec.name, 0) + 1

    def _connect(self, kind, child, parent, prop=None):
        props = [kind, np.int64(child), np.int64(parent)]
        if prop:
            props.append(prop)
        self.connections.append(Record("C", props))

    def records(self) -> list[Record]:
        for root in self.scene.roots:
            self._model(root, np.int64(0))
        return [self._header(), Record("FileId", [FILE_ID]), Record("CreationTime", [CREATION_TIME]),
                Record("Creator", [self._creator()]), self._global_settings(), self._documents(),
                Record("References"), self._definitions(), Record("Objects", children=self.objects),
                Record("Connections", children=self.connections),
                Record("Takes", children=[Record("Current", [""])])]

    @staticmethod
    def _creator() -> str:
        return f"meshport {__version__}"

    def _header(self) -> Record:
        now = datetime.datetime.now()
        stamp = Record("CreationTimeStamp", children=[
            Record("Version", [1000]), Record("Year", [now.year]), Record("Month", [now.month]),
            Record("Day", [now.day]), Record("Hour", [now.hour]), Record("Minute", [now.minute]),
            Record("Second", [now.second]), Record("Millisecond", [now.microsecond // 1000])])
        info = Record("SceneInfo", [_fbx_name("GlobalInfo", "SceneInfo"), "UserData"], [
            Record("Type", ["UserData"]), Record("Version", [100]),
            Record("MetaData", children=[Record("Version", [100])] + [
                Record(k, [""]) for k in ("Title", "Subject", "Author", "Keywords", "Revision", "Comment")]),
            Record("Properties70", children=[
                _p("DocumentUrl", "KString", "Url", "", ""), _p("SrcDocumentUrl", "KString", "Url", "", ""),
                _p("Original", "Compound", "", ""),
                _p("Original|ApplicationVendor", "KString", "", "", "meshport"),
                _p("Original|ApplicationName", "KString", "", "", "meshport"),
                _p("Original|ApplicationVersion", "KString", "", "", __version__)])])
        return Record("FBXHeaderExtension", children=[
            Record("FBXHeaderVersion", [1003]), Record("FBXVersion", [7400]), Record("EncryptionType", [0]),
            stamp, Record("Creator", [self._creator()]), info])

    @staticmethod
    def _global_settings() -> Record:
        # Y up, Z front, X right, 1 unit = 100 cm: exactly the Scene convention, so nothing to convert.
        props = [_p("UpAxis", "int", "Integer", "", 1), _p("UpAxisSign", "int", "Integer", "", 1),
                 _p("FrontAxis", "int", "Integer", "", 2), _p("FrontAxisSign", "int", "Integer", "", 1),
                 _p("CoordAxis", "int", "Integer", "", 0), _p("CoordAxisSign", "int", "Integer", "", 1),
                 _p("OriginalUpAxis", "int", "Integer", "", 1), _p("OriginalUpAxisSign", "int", "Integer", "", 1),
                 _p("UnitScaleFactor", "double", "Number", "", 100.0),
                 _p("OriginalUnitScaleFactor", "double", "Number", "", 100.0),
                 _p_vec("AmbientColor", "ColorRGB", "Color", "", (0, 0, 0)),
                 _p("DefaultCamera", "KString", "", "", "Producer Perspective"),
                 _p("TimeMode", "enum", "", "", 6),
                 _p("TimeSpanStart", "KTime", "Time", "", np.int64(0)),
                 _p("TimeSpanStop", "KTime", "Time", "", np.int64(46186158000)),
                 _p("CustomFrameRate", "double", "Number", "", -1.0)]
        return Record("GlobalSettings", children=[Record("Version", [1000]), Record("Properties70", children=props)])

    @staticmethod
    def _documents() -> Record:
        doc = Record("Document", [np.int64(1_000_000), "Scene", "Scene"], [
            Record("Properties70", children=[_p("SourceObject", "object", "", ""),
                                             _p("ActiveAnimStackName", "KString", "", "", "")]),
            Record("RootNode", [np.int64(0)])])
        return Record("Documents", children=[Record("Count", [1]), doc])

    def _definitions(self) -> Record:
        kinds = [("GlobalSettings", 1)] + list(self.counts.items())
        children = [Record("Version", [100]), Record("Count", [sum(n for _, n in kinds)])]
        for kind, n in kinds:
            children.append(Record("ObjectType", [kind], [Record("Count", [n])]))
        return Record("Definitions", children=children)

    # ---- objects

    def _model(self, node: Node, parent_id: np.int64):
        mid = self._id()
        t, r, s = decompose(node.matrix)
        props = [_p_vec("Lcl Translation", "Lcl Translation", "", "A", t),
                 _p_vec("Lcl Rotation", "Lcl Rotation", "", "A", r),
                 _p_vec("Lcl Scaling", "Lcl Scaling", "", "A", s),
                 _p("InheritType", "enum", "", "", 1)]
        if node.mesh is not None:
            props.append(_p("DefaultAttributeIndex", "int", "Integer", "", 0))
        kind = "Mesh" if node.mesh is not None else "Null"
        self._add(Record("Model", [mid, _fbx_name(node.name or "node", "Model"), kind], [
            Record("Version", [232]), Record("Properties70", children=props),
            Record("Shading", [True]), Record("Culling", ["CullingOff"])]))
        self._connect("OO", mid, parent_id)
        if node.mesh is not None:
            self._connect("OO", self._geometry(node.mesh), mid)
            for mat in node.materials:
                self._connect("OO", self._material(mat), mid)
        for child in node.children:
            self._model(child, mid)

    def _geometry(self, mesh: Mesh) -> np.int64:
        if id(mesh) in self.geom_ids:
            return self.geom_ids[id(mesh)]
        gid = self._id()
        self.geom_ids[id(mesh)] = gid
        sizes = mesh.face_sizes.astype(np.int64)
        pvi = mesh.corners.astype(np.int64).copy()
        last = np.cumsum(sizes) - 1
        pvi[last] = ~pvi[last]
        children = [Record("GeometryVersion", [124]),
                    Record("Vertices", [np.ascontiguousarray(mesh.positions, np.float64).ravel()]),
                    Record("PolygonVertexIndex", [pvi.astype(np.int32)])]
        layer = [("LayerElementNormal", 0)] if mesh.normals is not None else []
        if mesh.normals is not None:
            children.append(Record("LayerElementNormal", [0], [
                Record("Version", [101]), Record("Name", [""]),
                Record("MappingInformationType", ["ByPolygonVertex"]),
                Record("ReferenceInformationType", ["Direct"]),
                Record("Normals", [np.ascontiguousarray(mesh.normals, np.float64).ravel()])]))
        if mesh.colors is not None:
            values, index = _indexed(mesh.colors)
            children.append(Record("LayerElementColor", [0], [
                Record("Version", [101]), Record("Name", ["Color"]),
                Record("MappingInformationType", ["ByPolygonVertex"]),
                Record("ReferenceInformationType", ["IndexToDirect"]),
                Record("Colors", [values.ravel()]), Record("ColorIndex", [index])]))
            layer.append(("LayerElementColor", 0))
        for k, uv in enumerate(mesh.uvs):
            values, index = _indexed(uv)
            children.append(Record("LayerElementUV", [k], [
                Record("Version", [101]), Record("Name", ["UVMap" if k == 0 else f"UVMap.{k}"]),
                Record("MappingInformationType", ["ByPolygonVertex"]),
                Record("ReferenceInformationType", ["IndexToDirect"]),
                Record("UV", [values.ravel()]), Record("UVIndex", [index])]))
        if mesh.uvs:
            layer.append(("LayerElementUV", 0))
        if mesh.face_materials is not None and len(np.unique(mesh.face_materials)) > 1:
            mapping, mats = "ByPolygon", mesh.face_materials.astype(np.int32)
        else:
            first = int(mesh.face_materials[0]) if mesh.face_materials is not None and len(mesh.face_materials) else 0
            mapping, mats = "AllSame", np.array([first], np.int32)
        children.append(Record("LayerElementMaterial", [0], [
            Record("Version", [101]), Record("Name", [""]),
            Record("MappingInformationType", [mapping]),
            Record("ReferenceInformationType", ["IndexToDirect"]), Record("Materials", [mats])]))
        layer.append(("LayerElementMaterial", 0))
        children.append(Record("Layer", [0], [Record("Version", [100])] + [
            Record("LayerElement", children=[Record("Type", [t]), Record("TypedIndex", [i])]) for t, i in layer]))
        for k in range(1, len(mesh.uvs)):
            children.append(Record("Layer", [k], [Record("Version", [100]), Record("LayerElement", children=[
                Record("Type", ["LayerElementUV"]), Record("TypedIndex", [k])])]))
        self._add(Record("Geometry", [gid, _fbx_name(mesh.name or "mesh", "Geometry"), "Mesh"], children))
        return gid

    def _material(self, mat: Material) -> np.int64:
        if id(mat) in self.mat_ids:
            return self.mat_ids[id(mat)]
        mid = self._id()
        self.mat_ids[id(mat)] = mid
        r, g, b, a = mat.base_color
        shininess = 2.0 / max(mat.roughness, 0.05) ** 2 - 2.0
        props = [_p("ShadingModel", "KString", "", "", "Phong"),
                 _p_vec("DiffuseColor", "Color", "", "A", (r, g, b)),
                 _p("DiffuseFactor", "Number", "", "A", 1.0),
                 _p_vec("EmissiveColor", "Color", "", "A", mat.emissive),
                 _p("EmissiveFactor", "Number", "", "A", 1.0),
                 _p_vec("AmbientColor", "Color", "", "A", (0, 0, 0)),
                 _p_vec("TransparentColor", "Color", "", "A", (1, 1, 1)),
                 _p("TransparencyFactor", "Number", "", "A", float(1.0 - a)),
                 _p("Opacity", "Number", "", "A", float(a)),
                 _p_vec("SpecularColor", "Color", "", "A", [0.5 * (1 - mat.roughness)] * 3),
                 _p("SpecularFactor", "Number", "", "A", 1.0),
                 _p("Shininess", "Number", "", "A", float(shininess)),
                 _p("ShininessExponent", "Number", "", "A", float(shininess)),
                 _p("ReflectionFactor", "Number", "", "A", float(mat.metallic))]
        self._add(Record("Material", [mid, _fbx_name(mat.name or "material", "Material"), ""], [
            Record("Version", [102]), Record("ShadingModel", ["phong"]), Record("MultiLayer", [0]),
            Record("Properties70", children=props)]))
        for image, prop in ((mat.base_color_texture, "DiffuseColor"), (mat.normal_texture, "NormalMap"),
                            (mat.emissive_texture, "EmissiveColor")):
            if image is not None:
                self._connect("OP", self._texture(image, prop), mid, prop)
        return mid

    def _file_name(self, image: Image) -> str:
        if id(image) not in self.file_names:
            ext = _EXT.get(image.mime or "")
            stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in (image.name or "texture"))
            if image.path:
                p = Path(image.path.replace("\\", "/"))
                stem, ext = p.stem, ext or p.suffix.lstrip(".") or "png"
            name = f"{stem}.{ext or 'png'}"
            taken = set(self.file_names.values())
            n = 1
            while name in taken:
                name = f"{stem}_{n}.{ext or 'png'}"
                n += 1
            self.file_names[id(image)] = name
        return self.file_names[id(image)]

    def _texture(self, image: Image, prop: str) -> np.int64:
        key = (id(image), prop)
        if key in self.tex_ids:
            return self.tex_ids[key]
        tid, vid = self._id(), self._id()
        self.tex_ids[key] = tid
        fname = self._file_name(image)
        name = image.name or Path(fname).stem
        video_children = [Record("Type", ["Clip"]),
                          Record("Properties70", children=[_p("Path", "KString", "XRefUrl", "", fname)]),
                          Record("UseMipMap", [0]), Record("Filename", [fname]), Record("RelativeFilename", [fname])]
        if image.data:
            video_children.append(Record("Content", [image.data]))
        self._add(Record("Video", [vid, _fbx_name(name, "Video"), "Clip"], video_children))
        self._add(Record("Texture", [tid, _fbx_name(name, "Texture"), ""], [
            Record("Type", ["TextureVideoClip"]), Record("Version", [202]),
            Record("TextureName", [_fbx_name(name, "Texture")]),
            Record("Properties70", children=[_p("UseMaterial", "bool", "", "", 1)]),
            Record("Media", [_fbx_name(name, "Video")]), Record("FileName", [fname]),
            Record("RelativeFilename", [fname]), Record("ModelUVTranslation", [0.0, 0.0]),
            Record("ModelUVScaling", [1.0, 1.0]), Record("Texture_Alpha_Source", ["None"]),
            Record("Cropping", [0, 0, 0, 0])]))
        self._connect("OO", vid, tid)
        return tid


def _indexed(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows plus an int32 index per corner (the IndexToDirect layout)."""
    values = np.ascontiguousarray(values, np.float64)
    first, index = unique_rows(values)
    return values[first], index.astype(np.int32)
