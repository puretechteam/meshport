"""FBX <-> Scene.

Reads binary and ASCII FBX, versions 6.x and 7.x (everything from 2006 on). Writes binary
FBX 7.4, which every FBX importer accepts, animation included.

What comes across: the node hierarchy with full FBX transforms (pivots, pre/post rotation,
rotation order, geometric offsets), polygons, normals, every UV set, vertex colors,
per-face materials and their textures (embedded or found next to the file), 7.x skins
(clusters) and blend shapes, and 7.x animation stacks (fbxanim.py has the curves), blend shape
weights included. Cameras and lights are skipped.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .fbxanim import (
    CONSTANT,
    FRAME_RATES,
    LINEAR,
    THIRDS,
    TICKS,
    Curve,
    continuous,
    euler_degrees,
    time_mode,
)
from .fbxtree import CREATION_TIME, FILE_ID, Record, decode_content, parse, write_binary
from .geometry import (
    EULER_ORDERS,
    compose,
    decompose,
    decompose_many,
    decompose_trs,
    euler,
    euler_many,
    rotations,
    scaling,
    transform_normals,
    transform_points,
    translation,
    unique_rows,
)
from .scene import (
    DAMAGED,
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
    frame_grid,
    frame_rate,
    inbetween_weights,
    linked_path,
    pack_influences,
    read_image_file,
    too_many_frames,
    with_crossings,
)

# ============================================================================ reading

_ANIMATED = ("Lcl Translation", "Lcl Rotation", "Lcl Scaling")


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
        if isinstance(default, (tuple, np.ndarray)):
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
    try:
        version, records = parse(data)
        top = {}
        for r in records:
            top.setdefault(r.name, r)
        if "Objects" not in top:
            raise MeshportError("This FBX file has no objects in it")
        return _Reader(version, top, Path(base_dir) if base_dir else None).scene()
    except DAMAGED as e:
        raise MeshportError(f"This FBX file is damaged ({type(e).__name__}: {e})") from None


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
        self.deformed: list[tuple[Node, _Obj]] = []  # every node given a mesh, and its geometry
        self.shape_values: dict[int, list[float]] = {}  # mesh -> its blend shape channels' own weights
        # mesh -> its blend shape channels: (channel id, the weights its shapes stand at, 0 to 1), their
        # shapes the mesh's targets in turn
        self.shape_channels: dict[int, list[tuple[object, np.ndarray]]] = {}
        self.finder = TextureFinder(base_dir)
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
        kept = {oid: n for oid, n in nodes.items() if id(n) not in junk}
        for node, geom in self.deformed:
            try:
                node.skin = self._skin(geom, node, kept)
            except (*DAMAGED, MeshportError):
                node.skin = None  # a damaged skin leaves the mesh where its node puts it
        scene.animations = self._animations(kept, roots, conversion)
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
                self.deformed.append((node, geom))
            else:  # a second mesh on one model: hang it underneath
                node.children.append(Node(f"{model.name}.{i}", mesh=mesh, materials=materials))
                self.deformed.append((node.children[-1], geom))
        return node

    def _local_matrix(self, o: _Obj) -> np.ndarray:
        zero, one = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        t = o.prop("Lcl Translation", zero)
        r = o.prop("Lcl Rotation", zero)
        s = o.prop("Lcl Scaling", one)
        before, after, last, order = self._transform_parts(o)
        return translation(t) @ before @ euler(r, order) @ after @ scaling(s) @ last

    @staticmethod
    def _transform_parts(o: _Obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """FBX's local transform is T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1:
        the fixed parts around the animated T, R and S, and R's euler order. Pre and post rotation only
        count when RotationActive is on (off by default), as in Autodesk's FBX SDK and Blender; the
        rotation order counts either way, as in the FBX SDK (Blender drops it too)."""
        zero = (0.0, 0.0, 0.0)
        r_off, r_piv = o.prop("RotationOffset", zero), o.prop("RotationPivot", zero)
        s_off, s_piv = o.prop("ScalingOffset", zero), o.prop("ScalingPivot", zero)
        active = bool(o.prop("RotationActive", 0))
        pre, post = (o.prop("PreRotation", zero), o.prop("PostRotation", zero)) if active else (zero, zero)
        order = EULER_ORDERS[min(max(o.prop("RotationOrder", 0), 0), 5)]
        before = translation(r_off) @ translation(r_piv) @ euler(pre)
        after = (np.linalg.inv(euler(post)) @ translation(-np.array(r_piv)) @ translation(s_off)
                 @ translation(s_piv))
        return before, after, translation(-np.array(s_piv)), order

    # ---- animation

    def _frame_rate(self) -> float:
        gs = self.top.get("GlobalSettings")
        props = _properties(gs) if gs is not None else {}
        mode = props.get("TimeMode", [0])[0]
        if mode == 14:
            custom = props.get("CustomFrameRate", [0])[0]
            if isinstance(custom, (int, float)) and custom > 0:
                return float(custom)
        return FRAME_RATES.get(int(mode) if isinstance(mode, (int, float)) else 0, 30.0)

    def _animations(self, nodes: dict, roots: list[Node], conversion: np.ndarray) -> list[Animation]:
        """Each animation stack, its layers' curves turned into node channels (where several
        layers animate one property, the first one's curves are used)."""
        stacks = [o for o in self.objects.values() if o.cls == "AnimationStack"]
        if not stacks:
            return []
        fps = self._frame_rate()
        roots_ids = {id(r) for r in roots}
        targets: dict[object, list[tuple[object, str]]] = {}  # curve node -> [(model, property)]
        shapes = {c for channels in self.shape_channels.values() for c, _ in channels}
        for parent, kids in self.children.items():
            for kind, child, prop in kids:
                if kind == "OP" and ((prop in _ANIMATED and parent in nodes) or (prop == "DeformPercent"
                                                                                 and parent in shapes)):
                    targets.setdefault(child, []).append((parent, prop))
        meshes = [n for root in roots for n, _ in root.walk() if n.mesh is not None and id(n.mesh) in self.shape_channels]
        out = []
        for stack in stacks:
            animated: dict[object, dict[str, list]] = {}  # model -> property -> per axis: Curve or a default
            percents: dict[object, Curve] = {}  # blend shape channel -> its DeformPercent curve
            for layer, _ in self._kids(stack.id, "AnimationLayer"):
                for cn, _ in self._kids(layer.id, "AnimationCurveNode"):
                    curves = {p: c for c, p in self._kids(cn.id, "AnimationCurve", "OP")}
                    for model_id, prop in targets.get(cn.id, []):
                        if prop == "DeformPercent":
                            if "d|DeformPercent" in curves and model_id not in percents:
                                try:
                                    percents[model_id] = Curve.from_record(curves["d|DeformPercent"].rec)
                                except DAMAGED:
                                    continue
                            continue
                        props = animated.setdefault(model_id, {})
                        if prop not in props:
                            props[prop] = [Curve.from_record(curves[f"d|{a}"].rec) if f"d|{a}" in curves
                                           else cn.prop(f"d|{a}", None) for a in "XYZ"]
            channels = []
            for model_id, props in animated.items():
                node = nodes[model_id]
                try:
                    channels += self._channels(self.objects[model_id], node, props, fps,
                                               conversion if id(node) in roots_ids else None)
                except DAMAGED:
                    continue  # a damaged curve loses its model's animation, not the file
            for node in meshes if percents else []:
                weights = self._weight_channel(node, percents, fps)
                if weights is not None:
                    channels.append(weights)
            if channels:
                out.append(Animation(stack.name, channels))
        return out

    @staticmethod
    def _key_times(curves: list[Curve], turning: bool, fps: float) -> tuple[np.ndarray, str]:
        """When to sample curves, and how to blend between: at their keys, and every frame between
        them too where they turn (euler angles, or cubic curves). Curves all constant are STEP."""
        keys = np.unique(np.concatenate([c.times for c in curves]))
        modes = set().union(*(c.interpolations for c in curves))
        interpolation = "STEP" if modes == {CONSTANT} else "LINEAR"
        if interpolation == "STEP" or (not turning and modes == {LINEAR}):
            return keys, interpolation
        grid = frame_grid(keys[0] * fps - 1e-6, keys[-1] * fps + 1e-6)
        if grid is None:  # too long to pose frame by frame: at the keys only
            return keys, interpolation
        times = np.unique(np.concatenate([keys, grid / fps]))
        return times[np.concatenate(([True], np.diff(times) > 1e-7))], interpolation

    def _weight_channel(self, node: Node, percents: dict, fps: float) -> Channel | None:
        """A mesh's blend shape weights over time (DeformPercent / 100), the channels nothing
        animates held at their own weight."""
        channels = self.shape_channels[id(node.mesh)]
        curves = [percents.get(c) for c, _ in channels]
        if not any(curves):
            return None
        try:
            times, interpolation = self._key_times([c for c in curves if c is not None], False, fps)
            rest = self.shape_values[id(node.mesh)]

            def levels() -> list[np.ndarray]:
                return [c.evaluate(times) / 100.0 if c is not None else np.full(len(times), rest[k])
                        for k, c in enumerate(curves)]
            for level, (_, knots) in zip(levels(), channels):
                if len(knots) > 1:  # in-between shapes: a key wherever the channel passes one
                    times = with_crossings(times, level, knots)
            values = np.hstack([inbetween_weights(level, knots) for level, (_, knots) in zip(levels(), channels)])
        except DAMAGED:
            return None
        return Channel(node, WEIGHTS, times, values, interpolation) if np.isfinite(values).all() else None

    def _channels(self, model: _Obj, node: Node, props: dict, fps: float, conversion: np.ndarray | None) -> list:
        """A model's curves as translation, rotation and scale channels: its local matrix posed
        at every key (and every frame between, where euler angles or cubic curves turn), then
        split up. Curves that are all constant become STEP channels."""
        curves = [c for axes in props.values() for c in axes if isinstance(c, Curve)]
        if not curves:
            return []
        times, interpolation = self._key_times(curves, "Lcl Rotation" in props, fps)

        def values(prop: str, rest: tuple) -> np.ndarray:
            axes, static = props.get(prop), model.prop(prop, rest)
            cols = []
            for i in range(3):
                c = axes[i] if axes else None
                cols.append(c.evaluate(times) if isinstance(c, Curve)
                            else np.full(len(times), float(c) if isinstance(c, (int, float)) else static[i]))
            return np.stack(cols, 1)

        before, after, last, order = self._transform_parts(model)
        moved = np.tile(np.eye(4), (len(times), 1, 1))
        moved[:, :3, 3] = values("Lcl Translation", (0.0, 0.0, 0.0))
        sizes = np.tile(np.eye(4), (len(times), 1, 1))
        sizes[:, [0, 1, 2], [0, 1, 2]] = values("Lcl Scaling", (1.0, 1.0, 1.0))
        m = moved @ before @ euler_many(values("Lcl Rotation", (0.0, 0.0, 0.0)), order) @ after @ sizes @ last
        if conversion is not None:
            m = conversion @ m
        out = []
        for path, vals, prop in zip(("translation", "rotation", "scale"), decompose_many(m), _ANIMATED):
            if prop in props or not np.allclose(vals, vals[0], atol=1e-9):
                out.append(Channel(node, path, times.copy(), vals, interpolation))
        return out

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
        shapes = self._blend_shapes(geom, geometric, len(positions))
        if shapes:
            mesh.targets = [m for _, morphs, _, _ in shapes for m in morphs]
            mesh.target_weights = np.hstack([inbetween_weights(w, knots)[0] for _, _, knots, w in shapes])
            self.shape_channels[id(mesh)] = [(c, knots) for c, _, knots, _ in shapes]
            self.shape_values[id(mesh)] = [w for _, _, _, w in shapes]
        mesh.validate()
        return mesh

    # ---- deformers

    def _blend_shapes(self, geom: _Obj, geometric: np.ndarray | None, count: int) -> list:
        """(channel id, its targets, the weights they stand at, its weight) for each of the geometry's
        blend shape channels: a target per shape, the full one and any in-between ones (FullWeights
        says where each stands, 0 to 100), which inbetween_weights() blends as FBX does. A shape holds
        offsets for the control points it lists. DeformPercent (0 to 100) is the channel's weight."""
        out = []
        for blend, _ in self._kids(geom.id, "Deformer"):
            if blend.sub != "BlendShape":
                continue
            for channel, _ in self._kids(blend.id, "Deformer"):
                shapes = [s for s, _ in self._kids(channel.id, "Geometry") if s.sub == "Shape"]
                if channel.sub != "BlendShapeChannel" or not shapes:
                    continue
                full = channel.rec.array("FullWeights")
                if full is None or len(full) < len(shapes) or not np.isfinite(full[:len(shapes)]).all():
                    full = np.arange(1, len(shapes) + 1) * 100.0 / len(shapes)  # missing: spread evenly
                order = np.argsort(full[:len(shapes)], kind="stable")
                knots = full[:len(shapes)][order] / 100.0
                if (knots <= 0).any() or (np.diff(knots) <= 0).any():
                    continue  # shapes that don't stand apart can't be blended between
                morphs = []
                for k in order:
                    shape = shapes[k]
                    index, delta = shape.rec.array("Indexes", np.int64), shape.rec.array("Vertices")
                    if index is None or delta is None or len(delta) < 3 * len(index):
                        break
                    delta = delta[:3 * len(index)].reshape(-1, 3)
                    keep = (index >= 0) & (index < count) & np.isfinite(delta).all(1)
                    offsets = np.zeros((count, 3))
                    offsets[index[keep]] = delta[keep]
                    if geometric is not None:
                        offsets = offsets @ geometric[:3, :3].T
                    name = channel.name or shape.name
                    morphs.append(Morph(name if k == order[-1] else f"{name} {full[k]:g}%", offsets))
                if len(morphs) != len(shapes):
                    continue
                percent = channel.rec.value("DeformPercent", 0.0)
                weight = channel.prop("DeformPercent", float(percent) if isinstance(percent, (int, float)) else 0.0)
                out.append((channel.id, morphs, knots, weight / 100.0 if np.isfinite(weight) else 0.0))
        return out

    def _skin(self, geom: _Obj, node: Node, nodes: dict) -> Skin | None:
        """The geometry's skin deformer: a joint per cluster, whose inverse bind matrix is the
        cluster's Transform (in the file it's already the mesh node's matrix in the bone's space at
        bind time, inv(TransformLink) @ the mesh's world matrix; Autodesk's FBX SDK hands out the
        mesh's world matrix instead). Weights are normalized, unless a cluster says Total1; whatever
        a vertex's weights leave over stays with the mesh's own node, as in the SDK."""
        mesh = node.mesh
        count = len(mesh.positions)
        for deformer, _ in self._kids(geom.id, "Deformer"):
            if deformer.sub != "Skin":
                continue
            joints, binds, rows, cols, vals, total_one = [], [], [], [], [], False
            for cluster, _ in self._kids(deformer.id, "Deformer"):
                links = [nodes[m.id] for m, _ in self._kids(cluster.id, "Model") if m.id in nodes]
                index, weight = cluster.rec.array("Indexes", np.int64), cluster.rec.array("Weights")
                if cluster.sub != "Cluster" or not links or index is None or weight is None:
                    continue
                n = min(len(index), len(weight))
                index, weight = index[:n], weight[:n]
                keep = (index >= 0) & (index < count) & np.isfinite(weight) & (weight > 0)
                rows.append(index[keep])
                cols.append(np.full(int(keep.sum()), len(joints)))
                vals.append(weight[keep])
                joints.append(links[0])
                binds.append(_matrix(cluster.rec.array("Transform")))
                total_one |= str(cluster.rec.value("Mode", "")) == "Total1"
            if not joints:
                continue
            rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
            total = np.bincount(rows, vals, count)
            if total_one:  # weights up to 1 stand as they are
                vals = vals / np.maximum(total, 1.0)[rows]
                total = np.minimum(total, 1.0)
            else:
                vals = vals / total[rows]
                total = (total > 0).astype(np.float64)
            left = 1.0 - total
            if (left > 1e-6).any():  # the mesh's node holds what's left over
                spare = np.flatnonzero(left > 1e-6)
                rows, vals = np.concatenate([rows, spare]), np.concatenate([vals, left[spare]])
                cols = np.concatenate([cols, np.full(len(spare), len(joints))])
                joints.append(node)
                binds.append(np.eye(4))
            indices, weights = pack_influences(rows, cols, vals, count)
            skin = Skin(joints, np.array(binds), indices, weights, deformer.name or "skin")
            skin.validate(mesh)
            return skin if np.isfinite(skin.inverse_binds).all() else None
        return None

    def _layer(self, rec, elem_name, data_name, index_name, width, ctx):
        elems = sorted(rec.find_all(elem_name), key=lambda e: e.props[0] if e.props else 0)
        return self._layer_values(elems[0], data_name, index_name, width, ctx) if elems else None

    @staticmethod
    def _layer_values(elem: Record, data_name, index_name, width, ctx) -> np.ndarray | None:
        corners, face_of_corner, n_corners, _, _ = ctx
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

    def _find_file(self, names: list[str]) -> bytes | None:
        return self.finder.find(names)


_TEXTURE_DIRS = frozenset({"texture", "textures", "tex", "materials", "images", "maps"})


class TextureFinder:
    """Finds the texture files a model points at. Shared by the FBX and 3DS readers."""

    def __init__(self, base_dir: Path | None):
        self.base_dir = base_dir.resolve() if base_dir is not None else None
        self.root = self.base_dir.parent if self.base_dir is not None else None
        self._files: dict[str, Path] | None = None

    def find(self, names: list[str]) -> bytes | None:
        """Texture paths in model files usually point at the artist's own drive. Files are only
        read from the model's folder and the folder above it (and below them): a path is tried
        as written when it's relative and stays in there, then its file name next to the model,
        in texture folders there or one level up (Pack/Models/x.fbx -> Pack/Textures/), and
        anywhere under the folder above the model. Absolute, network and URL paths are never
        opened, and only image files are taken, so a model can't pull other files into the
        output. Without a folder (bytes read on their own), nothing is looked up."""
        if self.base_dir is None:
            return None
        bases = []
        for n in names:
            bases.append(n.replace("\\", "/").rsplit("/", 1)[-1].lower())
            path = linked_path(n, self.base_dir, self.root)
            data = read_image_file(path) if path is not None else None
            if data is not None:
                return data
        listing = self._listing()
        for base in bases:
            if base in listing:
                path = linked_path(listing[base].name, listing[base].parent, self.root)  # a link can't lead out
                data = read_image_file(path) if path is not None else None
                if data is not None:
                    return data
        return None

    def _listing(self) -> dict[str, Path]:
        """Lower-case file name -> path, nearest folders first, capped so a model in a huge
        folder doesn't stall the read."""
        if self._files is not None:
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
        for up in (self.base_dir, self.base_dir.parent):
            try:
                subdirs = [d for d in up.iterdir() if d.is_dir() and d.name.lower() in _TEXTURE_DIRS]
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


def _matrix(values: np.ndarray | None) -> np.ndarray:
    """An FBX matrix (16 numbers, translation last) as a column-vector one, identity when missing."""
    if values is None or len(values) < 16:
        return np.eye(4)
    return values[:16].reshape(4, 4).T.astype(np.float64)


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
        self.model_ids: dict[int, np.int64] = {}
        self.rest = {id(n): w for n, w in scene.walk()}
        # skins whose joints are all in the scene, and the joints, written as skeleton nodes
        self.skins = {id(n): n.skin for n, _ in scene.walk() if n.skin is not None and n.mesh is not None
                      and all(id(j) in self.rest for j in n.skin.joints)}
        self.joints = {id(j) for skin in self.skins.values() for j in skin.joints}
        # a skinned mesh is written where it was bound (its node's own transform plays no part in
        # skinning), so importers that rebuild the bind from the bones as they stand agree
        self.parent = parent = {id(c): n for n, _ in scene.walk() for c in n.children}
        moving = {id(c.node) for a in scene.animations for c in a.channels if c.path in PATHS}
        self.placed: dict[int, np.ndarray] = {}  # node -> its local matrix as written
        for n, _ in scene.walk():
            if id(n) in self.skins and not n.children and id(n) not in moving:
                bind = self.skins[id(n)].bind_space(self.rest)
                if bind is not None:
                    above = self.rest[id(parent[id(n)])] if id(n) in parent else np.eye(4)
                    self.placed[id(n)] = np.linalg.inv(above) @ bind
                    self.rest[id(n)] = bind
        self.pending: list[tuple[np.int64, Node]] = []  # skinned geometries, once every model has an id
        self.skinned: set[tuple[int, int]] = set()
        self.shape_ids: dict[int, list[np.int64]] = {}  # mesh -> its blend shape channels
        self.bind_pose: dict[int, tuple[np.int64, np.ndarray]] = {}  # node -> (model, world matrix when bound)
        times = [c.times for a in scene.animations for c in a.channels if len(c.times)]
        self.fps = frame_rate(np.concatenate(times)) if times else 30.0
        self.span = (min(a.start for a in scene.animations), max(a.end for a in scene.animations)) \
            if times else (0.0, 1.0)

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
        for gid, node in self.pending:
            self._skin(gid, node)
        if self.bind_pose:
            self._add(Record("Pose", [self._id(), _fbx_name("BindPose", "Pose"), "BindPose"], [
                Record("Type", ["BindPose"]), Record("Version", [100]), Record("NbPoseNodes", [len(self.bind_pose)]),
                *[Record("PoseNode", children=[Record("Node", [mid]), Record("Matrix", [m.T.ravel()])])
                  for mid, m in self.bind_pose.values()]]))
        takes =[Record("Current", [self.scene.animations[0].name if self.scene.animations else ""])]
        for animation in self.scene.animations:
            if self._animation(animation):
                start, stop = (np.int64(round(t * TICKS)) for t in (animation.start, animation.end))
                takes.append(Record("Take", [animation.name], [
                    Record("FileName", [f"{animation.name}.tak"]), Record("LocalTime", [start, stop]),
                    Record("ReferenceTime", [start, stop])]))
        return [self._header(), Record("FileId", [FILE_ID]), Record("CreationTime", [CREATION_TIME]),
                Record("Creator", [self._creator()]), self._global_settings(), self._documents(),
                Record("References"), self._definitions(), Record("Objects", children=self.objects),
                Record("Connections", children=self.connections),
                Record("Takes", children=takes)]

    @staticmethod
    def _creator() -> str:
        return f"meshport {__version__}"

    def _header(self) -> Record:
        now = datetime.datetime.now().astimezone()
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

    def _global_settings(self) -> Record:
        # Y up, Z front, X right, 1 unit = 100 cm: exactly the Scene convention, so nothing to convert.
        mode, custom = time_mode(self.fps)
        start, stop = (np.int64(round(t * TICKS)) for t in self.span)
        props = [_p("UpAxis", "int", "Integer", "", 1), _p("UpAxisSign", "int", "Integer", "", 1),
                 _p("FrontAxis", "int", "Integer", "", 2), _p("FrontAxisSign", "int", "Integer", "", 1),
                 _p("CoordAxis", "int", "Integer", "", 0), _p("CoordAxisSign", "int", "Integer", "", 1),
                 _p("OriginalUpAxis", "int", "Integer", "", 1), _p("OriginalUpAxisSign", "int", "Integer", "", 1),
                 _p("UnitScaleFactor", "double", "Number", "", 100.0),
                 _p("OriginalUnitScaleFactor", "double", "Number", "", 100.0),
                 _p_vec("AmbientColor", "ColorRGB", "Color", "", (0, 0, 0)),
                 _p("DefaultCamera", "KString", "", "", "Producer Perspective"),
                 _p("TimeMode", "enum", "", "", mode),
                 _p("TimeSpanStart", "KTime", "Time", "", start),
                 _p("TimeSpanStop", "KTime", "Time", "", stop),
                 _p("CustomFrameRate", "double", "Number", "", custom)]
        return Record("GlobalSettings", children=[Record("Version", [1000]), Record("Properties70", children=props)])

    def _documents(self) -> Record:
        active = self.scene.animations[0].name if self.scene.animations else ""
        doc = Record("Document", [np.int64(1_000_000), "Scene", "Scene"], [
            Record("Properties70", children=[_p("SourceObject", "object", "", ""),
                                             _p("ActiveAnimStackName", "KString", "", "", active)]),
            Record("RootNode", [np.int64(0)])])
        return Record("Documents", children=[Record("Count", [1]), doc])

    # ---- animation

    def _animation(self, animation: Animation) -> bool:
        """One animation stack with one layer: a curve node per animated property of each node,
        a curve per axis. Rotations are written as XYZ euler angles that turn continuously."""
        by_node: dict[int, list[Channel]] = {}
        weights = []
        for c in animation.channels:
            if id(c.node) in self.model_ids and c.path in PATHS and len(c.times):
                by_node.setdefault(id(c.node), []).append(c)
            elif (c.path == WEIGHTS and c.node.mesh is not None and len(c.times)
                  and len(self.shape_ids.get(id(c.node.mesh), [])) == c.values.shape[-1]):
                weights.append(c)
        if not by_node and not weights:
            return False
        sid, lid = self._id(), self._id()
        start, stop = (np.int64(round(t * TICKS)) for t in (animation.start, animation.end))
        self._add(Record("AnimationStack", [sid, _fbx_name(animation.name or "Take 001", "AnimStack"), ""], [
            Record("Properties70", children=[_p("LocalStart", "KTime", "Time", "", start),
                                             _p("LocalStop", "KTime", "Time", "", stop),
                                             _p("ReferenceStart", "KTime", "Time", "", start),
                                             _p("ReferenceStop", "KTime", "Time", "", stop)])]))
        # Autodesk's FBX SDK only takes a layer that has a list of children, even an empty one
        self._add(Record("AnimationLayer", [lid, _fbx_name("BaseLayer", "AnimLayer"), ""], [Record("Properties70")]))
        self._connect("OO", lid, sid)
        for channels in by_node.values():
            self._node_curves(animation, channels, lid)
        for c in weights:
            self._weight_curves(c, lid)
        return True

    def _times(self, keys: np.ndarray, every_frame: bool) -> np.ndarray:
        if not every_frame:
            return keys
        grid = frame_grid(keys[0] * self.fps - 1e-6, keys[-1] * self.fps + 1e-6)
        if grid is None:
            raise too_many_frames((keys[-1] - keys[0]) * self.fps)
        times = np.unique(np.concatenate([keys, grid / self.fps]))
        return times[np.concatenate(([True], np.diff(times) > 1e-7))]

    def _curve(self, ticks: np.ndarray, values: np.ndarray, flags: np.ndarray, parent: np.int64, prop: str) -> None:
        cid = self._id()
        self._add(Record("AnimationCurve", [cid, _fbx_name("", "AnimCurve"), ""], [
            Record("Default", [float(values[0])]), Record("KeyVer", [4009]), Record("KeyTime", [ticks]),
            Record("KeyValueFloat", [values.astype(np.float32)]), Record("KeyAttrFlags", [flags]),
            Record("KeyAttrDataFloat", [np.array([0, 0, THIRDS, 0], np.float32)]),
            Record("KeyAttrRefCount", [np.array([len(ticks)], np.int32)])]))
        self._connect("OP", cid, parent, prop)

    def _weight_curves(self, c: Channel, layer_id: np.int64) -> None:
        """A DeformPercent curve (weight x 100) on each of the mesh's blend shape channels."""
        step = c.interpolation == "STEP"
        times = self._times(c.times, c.interpolation != "LINEAR")  # cubic splines and steps every frame
        percent = c.sample(times) * 100.0
        ticks = np.round(times * TICKS).astype(np.int64)
        flags = np.array([CONSTANT if step else LINEAR], np.int32)
        for k, channel_id in enumerate(self.shape_ids[id(c.node.mesh)]):
            cnid = self._id()
            self._add(Record("AnimationCurveNode", [cnid, _fbx_name("DeformPercent", "AnimCurveNode"), ""], [
                Record("Properties70", children=[_p("d|DeformPercent", "Number", "", "A", float(percent[0, k]))])]))
            self._connect("OO", cnid, layer_id)
            self._connect("OP", cnid, channel_id, "DeformPercent")
            self._curve(ticks, percent[:, k], flags, cnid, "d|DeformPercent")

    def _node_curves(self, animation: Animation, channels: list[Channel], layer_id: np.int64) -> None:
        node = channels[0].node
        keys = np.unique(np.concatenate([c.times for c in channels]))
        paths = {c.path for c in channels}
        step = all(c.interpolation == "STEP" for c in channels)
        # a key every frame: euler curves can't slerp, nor linear keys follow a cubic spline, and
        # steps are held frame by frame too, for importers (Blender's) that blend constant keys
        times = self._times(keys, step or "rotation" in paths or any(c.interpolation != "LINEAR" for c in channels))
        t0, q0, s0 = decompose_trs(node.matrix)
        sampled = {"translation": np.tile(t0, (len(times), 1)), "rotation": np.tile(q0, (len(times), 1)),
                   "scale": np.tile(s0, (len(times), 1))}
        for c in channels:  # as Animation.local_matrix: a later channel for the same property wins
            sampled[c.path] = c.sample(times)
        turns = continuous(euler_degrees(rotations(sampled["rotation"])))
        values = {"translation": sampled["translation"], "rotation": turns,
                  "scale": sampled["scale"]}
        ticks = np.round(times * TICKS).astype(np.int64)
        flags = np.array([CONSTANT if step else LINEAR], np.int32)
        for path, prop, letter in (("translation", "Lcl Translation", "T"), ("rotation", "Lcl Rotation", "R"),
                                   ("scale", "Lcl Scaling", "S")):
            if path not in paths:
                continue
            v = values[path]
            cnid = self._id()
            self._add(Record("AnimationCurveNode", [cnid, _fbx_name(letter, "AnimCurveNode"), ""], [
                Record("Properties70", children=[_p(f"d|{a}", "Number", "", "A", float(v[0, i]))
                                                 for i, a in enumerate("XYZ")])]))
            self._connect("OO", cnid, layer_id)
            self._connect("OP", cnid, self.model_ids[id(node)], prop)
            for i, axis in enumerate("XYZ"):
                self._curve(ticks, v[:, i], flags, cnid, f"d|{axis}")

    def _definitions(self) -> Record:
        kinds = [("GlobalSettings", 1)] + list(self.counts.items())
        children = [Record("Version", [100]), Record("Count", [sum(n for _, n in kinds)])]
        for kind, n in kinds:
            children.append(Record("ObjectType", [kind], [Record("Count", [n])]))
        return Record("Definitions", children=children)

    # ---- objects

    def _model(self, node: Node, parent_id: np.int64):
        mid = self._id()
        t, r, s = decompose(self.placed.get(id(node), node.matrix))
        props = [_p_vec("Lcl Translation", "Lcl Translation", "", "A", t),
                 _p_vec("Lcl Rotation", "Lcl Rotation", "", "A", r),
                 _p_vec("Lcl Scaling", "Lcl Scaling", "", "A", s),
                 _p("InheritType", "enum", "", "", 1)]
        if node.mesh is not None:
            props.append(_p("DefaultAttributeIndex", "int", "Integer", "", 0))
        kind = "Mesh" if node.mesh is not None else "LimbNode" if id(node) in self.joints else "Null"
        self._add(Record("Model", [mid, _fbx_name(node.name or "node", "Model"), kind], [
            Record("Version", [232]), Record("Properties70", children=props),
            Record("Shading", [True]), Record("Culling", ["CullingOff"])]))
        self._connect("OO", mid, parent_id)
        self.model_ids.setdefault(id(node), mid)
        if kind == "LimbNode":  # a skeleton bone, to importers that build armatures (Blender, Maya)
            aid = self._id()
            self._add(Record("NodeAttribute", [aid, _fbx_name(node.name or "node", "NodeAttribute"), "LimbNode"], [
                Record("Properties70", children=[_p("Size", "double", "Number", "", 100.0)]),
                Record("TypeFlags", ["Skeleton"])]))
            self._connect("OO", aid, mid)
        if node.mesh is not None:
            skin = self.skins.get(id(node))
            gid = self._geometry(node.mesh, skin)
            self._connect("OO", gid, mid)
            if skin is not None and (id(node.mesh), id(skin)) not in self.skinned:
                self.skinned.add((id(node.mesh), id(skin)))
                self.pending.append((gid, node))
            for mat in node.materials:
                self._connect("OO", self._material(mat), mid)
        for child in node.children:
            self._model(child, mid)

    def _geometry(self, mesh: Mesh, skin: Skin | None = None) -> np.int64:
        key = (id(mesh), id(skin))  # a skin deforms its geometry: one mesh, two skins, two geometries
        if key in self.geom_ids:
            return self.geom_ids[key]
        gid = self._id()
        self.geom_ids[key] = gid
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
        if mesh.targets and id(mesh) not in self.shape_ids:
            self._blend_shapes(gid, mesh)
        return gid

    def _blend_shapes(self, gid: np.int64, mesh: Mesh) -> None:
        """A BlendShape deformer with a channel per target, its shape holding the offsets of the
        vertices it moves. Shape normals are written unchanged (zero offsets)."""
        bid = self._id()
        self._add(Record("Deformer", [bid, _fbx_name(mesh.name or "mesh", "Deformer"), "BlendShape"],
                         [Record("Version", [100])]))
        self._connect("OO", bid, gid)
        weights = mesh.target_weights if mesh.target_weights is not None else np.zeros(len(mesh.targets))
        ids = []
        for m, w in zip(mesh.targets, weights):
            moved = np.flatnonzero(np.abs(m.positions).max(1) > 0)
            moved = moved if len(moved) else np.zeros(1, np.int64)  # a shape needs a vertex
            sid, cid = self._id(), self._id()
            self._add(Record("Geometry", [sid, _fbx_name(m.name or "shape", "Geometry"), "Shape"], [
                Record("Version", [100]), Record("Indexes", [moved.astype(np.int32)]),
                Record("Vertices", [np.ascontiguousarray(m.positions[moved], np.float64).ravel()]),
                Record("Normals", [np.zeros(3 * len(moved))])]))
            self._add(Record("Deformer", [cid, _fbx_name(m.name or "shape", "SubDeformer"), "BlendShapeChannel"], [
                Record("Version", [100]), Record("DeformPercent", [float(w) * 100.0]),
                Record("FullWeights", [np.array([100.0])]),
                Record("Properties70", children=[_p("DeformPercent", "Number", "", "A", float(w) * 100.0)])]))
            self._connect("OO", sid, cid)
            self._connect("OO", cid, bid)
            ids.append(cid)
        self.shape_ids[id(mesh)] = ids

    def _skin(self, gid: np.int64, node: Node) -> None:
        """A Skin deformer with a cluster per joint. Transform is the inverse bind matrix (the mesh
        in the bone's space), TransformLink the bone's world matrix when bound (where the mesh was
        bound, undone by the inverse bind matrix), TransformAssociateModel the armature's (the node
        the bones hang from), which Blender places the mesh by. The bind pose lists them all."""
        skin, world = node.skin, self.rest[id(node)]
        joints = {id(j) for j in skin.joints}
        arm = next((self.parent[id(j)] for j in skin.joints if id(j) in self.parent
                    and id(self.parent[id(j)]) not in joints), None)  # the armature: what the bones hang from
        arm_world = self.rest[id(arm)] if arm is not None else np.eye(4)
        did = self._id()
        self._add(Record("Deformer", [did, _fbx_name(skin.name or "skin", "Deformer"), "Skin"], [
            Record("Version", [101]), Record("Link_DeformAcuracy", [50.0])]))
        self._connect("OO", did, gid)
        count = len(node.mesh.positions)
        for k, joint in enumerate(skin.joints):
            weight = np.zeros(count)
            np.add.at(weight, np.nonzero(skin.indices == k)[0], skin.weights[skin.indices == k])
            moved = np.flatnonzero(weight > 0)
            bind = np.asarray(skin.inverse_binds[k], np.float64)
            try:
                link = world @ np.linalg.inv(bind)
            except np.linalg.LinAlgError:  # a joint that flattens the mesh: no bone matrix undoes it
                link = world
            cid = self._id()
            self._add(Record("Deformer", [cid, _fbx_name(joint.name or "joint", "SubDeformer"), "Cluster"], [
                Record("Version", [100]), Record("UserData", ["", ""]),
                Record("Indexes", [moved.astype(np.int32)]), Record("Weights", [weight[moved]]),
                Record("Transform", [bind.T.ravel()]), Record("TransformLink", [link.T.ravel()]),
                Record("TransformAssociateModel", [arm_world.T.ravel()])]))
            self._connect("OO", cid, did)
            self._connect("OO", self.model_ids[id(joint)], cid)
            self.bind_pose.setdefault(id(joint), (self.model_ids[id(joint)], link))
        self.bind_pose.setdefault(id(node), (self.model_ids[id(node)], world))
        for n in [node, *skin.joints]:  # and the nodes above them (the armature), as Blender writes it
            while id(n) in self.parent:
                n = self.parent[id(n)]
                self.bind_pose.setdefault(id(n), (self.model_ids[id(n)], self.rest[id(n)]))

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
