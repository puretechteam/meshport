"""Skins and blend shapes: the scene model's posing, glTF reading and writing, damaged skins and
targets, and Blender in both directions (data/make_skin_fixture.py builds a skinned, shape-keyed
tube and records where Blender puts every vertex)."""

import json
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import BLENDER, cube

import meshport
from meshport import Animation, Channel, MeshportError, Morph, Node, Scene, Skin
from meshport.fbx import _Writer
from meshport.fbxtree import Record, write_binary
from meshport.geometry import compose

DATA = Path(__file__).resolve().parent / "data"
Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])  # Blender's Z up to glTF's Y up
TIMES = np.array([0, 0.5, 1.0])


def rigged() -> Scene:
    """A cube hanging from a root and a bone: its top follows the bone, its bottom the root, its
    middle (none here) both; a blend shape lifts every vertex."""
    root = Node("Root", compose((0, 0, 1)))
    bone = Node("Bone", compose((0, 1, 0)))
    root.children = [bone]
    mesh = cube()
    mesh.targets = [Morph("Lift", np.tile([0, 0.5, 0.0], (8, 1))), Morph("Widen", mesh.positions * [0.5, 0, 0])]
    mesh.target_weights = np.array([0.2, 0.0])
    top = mesh.positions[:, 1] > 0
    weights = np.where(top[:, None], [0.75, 0.25], [1.0, 0.0])
    indices = np.where(top[:, None], [1, 0], [0, 1])
    binds = np.array([np.linalg.inv(root.matrix), np.linalg.inv(root.matrix @ bone.matrix)])
    body = Node("Body", compose((5, 0, 0)), mesh)  # skinned: its own transform plays no part
    body.skin = Skin([root, bone], binds, indices, weights, "Rig")
    scene = Scene([root, body])
    scene.animations = [Animation("Bend", [
        Channel(bone, "rotation", TIMES, np.array([[0, 0, 0, 1], [0, 0, np.sin(0.4), np.cos(0.4)], [0.3, 0, 0, 0.954]])),
        Channel(root, "translation", TIMES, np.array([[0, 0, 1], [1, 0, 1], [1, 2, 1.0]]), "STEP"),
        Channel(body, "weights", TIMES, np.array([[0, 0], [1, 0.5], [0.3, 1.0]]))])]
    return scene


def points(scene: Scene, t, animation=None) -> dict:
    return {n.name: p for n, p in scene.posed_points(t, animation)}


def same_points(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-5) -> bool:
    """The same set of points (glTF splits vertices where normals or UVs differ)."""
    def distinct(p):
        return np.unique(np.round(p / tolerance) * tolerance, axis=0)
    a, b = distinct(a), distinct(b)
    return a.shape == b.shape and np.allclose(a, b, atol=tolerance * 2)


# ---- the scene model

def test_bind_pose_puts_the_mesh_where_its_joints_hold_it():
    scene = rigged()
    scene.animations = []
    body = points(scene, 0)["Body"]
    mesh = scene.roots[1].mesh
    assert body == pytest.approx(mesh.morphed())  # at rest each joint undoes its inverse bind
    assert mesh.morphed() == pytest.approx(mesh.positions + [0, 0.1, 0])


def test_joints_and_weights_move_the_vertices():
    scene = rigged()
    anim = scene.animations[0]
    root, body = scene.roots
    bone = root.children[0]
    for t in (0.25, 0.5, 0.8, 2.0):
        worlds = {n.name: w for n, w in scene.walk_at(t)}
        p = body.mesh.morphed(anim.weights(body, t))
        joint = [worlds["Root"] @ body.skin.inverse_binds[0], worlds["Bone"] @ body.skin.inverse_binds[1]]
        expect = []
        for v, ix, w in zip(p, body.skin.indices, body.skin.weights):
            m = sum(wk * joint[k] for k, wk in zip(ix, w))
            expect.append(m[:3, :3] @ v + m[:3, 3])
        assert points(scene, t)["Body"] == pytest.approx(np.array(expect))
    assert anim.weights(body, 0.25) == pytest.approx([0.5, 0.25])
    assert anim.weights(bone, 0.25) is None
    assert anim.local_matrix(body, 0.5) is body.matrix  # weights don't move the node


def test_meshes_without_skins_follow_their_node():
    mesh = cube()
    mesh.targets = [Morph("Out", mesh.positions.copy())]
    node = Node("N", compose((1, 2, 3)), mesh)
    scene = Scene([node], [Animation("A", [Channel(node, "weights", TIMES, np.array([[0], [1], [0.5]]))])])
    assert points(scene, 0.5)["N"] == pytest.approx(mesh.positions * 2 + [1, 2, 3])
    assert points(scene, 0.75)["N"] == pytest.approx(mesh.positions * 1.75 + [1, 2, 3])


def test_bad_skins_and_targets_are_refused():
    mesh = cube()
    mesh.targets = [Morph("Short", np.zeros((3, 3)))]
    with pytest.raises(MeshportError, match="blend shape"):
        mesh.validate()
    mesh.targets, mesh.target_weights = [Morph("Fine", np.zeros((8, 3)))], np.ones(2)
    with pytest.raises(MeshportError, match="weights"):
        mesh.validate()
    with pytest.raises(MeshportError, match="past its joints"):
        Skin([Node("J")], np.eye(4)[None], np.full((8, 1), 1), np.ones((8, 1))).validate(cube())
    with pytest.raises(MeshportError, match="doesn't fit"):
        Skin([Node("J")], np.eye(4)[None], np.zeros((7, 1), int), np.ones((7, 1))).validate(cube())


def test_baking_keeps_weights():
    scene = rigged()
    baked = scene.animations[0].bake(30)
    body = scene.roots[1]
    for t in (0.1, 0.5, 0.9):
        assert baked.weights(body, t) == pytest.approx(scene.animations[0].weights(body, t), abs=1e-9)


# ---- glTF

@pytest.mark.parametrize("fmt", ["glb", "gltf"])
def test_gltf_round_trip_keeps_skins_targets_and_weights(fmt):
    scene = rigged()
    back = meshport.read(meshport.write(scene, fmt), fmt)
    body = next(n for n, _ in back.walk() if n.name == "Body")
    assert [j.name for j in body.skin.joints] == ["Root", "Bone"] and body.skin.name == "Rig"
    assert [m.name for m in body.mesh.targets] == ["Lift", "Widen"]
    assert body.mesh.target_weights == pytest.approx([0.2, 0])
    assert {c.path for c in back.animations[0].channels} == {"rotation", "translation", "weights"}
    for t in (0, 0.2, 0.5, 0.7, 1.0, 3.0):
        assert same_points(points(scene, t)["Body"], points(back, t)["Body"]), t


def test_gltf_keeps_cubic_and_step_weights():
    scene = rigged()
    body = scene.roots[1]
    v = np.zeros((3, 3, 2))
    v[:, 1] = [[0, 0], [1, 0.5], [0.2, 1]]
    v[:, 0], v[:, 2] = [[0, 0], [1, 1], [0, 0]], [[2, 0], [0, -1], [0, 0]]
    for interpolation, values in (("CUBICSPLINE", v), ("STEP", v[:, 1])):
        scene.animations = [Animation("W", [Channel(body, "weights", TIMES, values, interpolation)])]
        back = meshport.read(meshport.write(scene, "glb"))
        (c,) = back.animations[0].channels
        assert c.interpolation == interpolation and c.values == pytest.approx(values)
        for t in np.linspace(0, 1, 9):
            assert c.sample(t) == pytest.approx(scene.animations[0].channels[0].sample(t), abs=1e-6)


def test_gltf_vertices_in_the_same_place_keep_their_own_skin_and_targets():
    mesh = cube()
    mesh.positions = np.vstack([mesh.positions, mesh.positions[:1]])  # vertex 8 sits on vertex 0
    mesh.corners = mesh.corners.copy()
    mesh.corners[np.flatnonzero(mesh.corners == 0)[1]] = 8
    shift = np.zeros((9, 3))
    shift[8] = (0, 0, -3)
    mesh.targets, mesh.target_weights = [Morph("Pull", shift)], np.array([1.0])
    node = Node("N", mesh=mesh)
    back = meshport.read(meshport.write(Scene([node]), "glb"))
    (_, moved), = back.posed_points()
    assert same_points(moved, mesh.morphed())


def test_more_than_four_influences_take_two_sets():
    mesh = cube()
    joints = [Node(f"J{k}", compose((k, 0, 0))) for k in range(6)]
    weights = np.tile([0.3, 0.2, 0.15, 0.15, 0.1, 0.1], (8, 1))
    binds = np.array([np.linalg.inv(j.matrix) for j in joints])
    node = Node("Body", mesh=mesh)
    node.skin = Skin(joints, binds, np.tile(np.arange(6), (8, 1)), weights)
    scene = Scene([*joints, node])
    scene.animations = [Animation("A", [Channel(joints[4], "translation", TIMES, np.array([[4, 0, 0], [4, 5, 0], [4, 0, 0.0]]))])]
    data = meshport.write(scene, "gltf")
    attrs = json.loads(data)["meshes"][0]["primitives"][0]["attributes"]
    assert {"JOINTS_0", "JOINTS_1", "WEIGHTS_0", "WEIGHTS_1"} <= set(attrs)
    back = meshport.read(data, "gltf")
    assert same_points(points(scene, 0.5)["Body"], points(back, 0.5)["Body"])
    assert points(scene, 0.5)["Body"][:, 1] == pytest.approx(np.full(8, 0.5) + mesh.positions[:, 1])


def test_skins_whose_joints_are_left_out_are_dropped():
    scene = rigged()
    scene.roots = scene.roots[1:]  # the skeleton isn't in the scene
    back = meshport.read(meshport.write(scene, "glb"))
    assert back.roots[0].skin is None and "skins" not in json.loads(meshport.write(scene, "gltf"))


def _doc(scene: Scene) -> dict:
    return json.loads(meshport.write(scene, "gltf"))


def test_damaged_skins_and_targets_load_without_them():
    doc = _doc(rigged())
    doc["skins"][0]["joints"] = [0, 99]
    assert meshport.read(json.dumps(doc).encode(), "gltf").roots[1].skin is None
    doc = _doc(rigged())
    doc["accessors"][doc["skins"][0]["inverseBindMatrices"]]["count"] = 1
    assert meshport.read(json.dumps(doc).encode(), "gltf").roots[1].skin is None
    doc = _doc(rigged())
    prim = doc["meshes"][0]["primitives"][0]
    prim["targets"][0]["POSITION"] = prim["indices"]  # the wrong length and shape
    back = meshport.read(json.dumps(doc).encode(), "gltf")
    assert back.roots[1].mesh.targets == [] and not any(c.path == "weights" for c in back.animations[0].channels)
    doc = _doc(rigged())
    doc["nodes"][2]["skin"] = 7
    assert meshport.read(json.dumps(doc).encode(), "gltf").roots[1].skin is None


def test_zero_weights_go_to_the_first_joint():
    doc = _doc(rigged())
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"]]
    acc.pop("bufferView")  # no data: every weight zero
    body = meshport.read(json.dumps(doc).encode(), "gltf").roots[1]
    assert (body.skin.weights[:, 0] == 1).all() and (body.skin.indices[:, 0] == 0).all()


# ---- Blender

def _blender(script: Path, *args) -> None:
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", *map(str, args)], check=True, capture_output=True, timeout=600)


def _compare(scene: Scene, ref: dict, name: str, shift: float = 0.0, tolerance: float = 1e-5, animation=None):
    """Every vertex Blender placed at every frame against meshport's pose at (frame - shift) / fps,
    matched up by place at the first frame."""
    order = None
    for frame, theirs in ref.items():
        theirs = np.array(theirs) @ Z_UP.T
        ours = points(scene, (int(frame) - shift) / 24, scene.animations if animation is None else animation)[name]
        if order is None:
            gap = np.linalg.norm(ours[:, None] - theirs[None], axis=2)
            assert gap.min(1).max() < tolerance
            order = gap.argmin(1)
        assert ours == pytest.approx(theirs[order], abs=tolerance), frame


@pytest.fixture(scope="module")
def blender_skin(tmp_path_factory):
    if BLENDER is None:
        pytest.skip("Blender isn't installed")
    out = tmp_path_factory.mktemp("skin")
    _blender(DATA / "make_skin_fixture.py", out)
    return out


def test_we_pose_blenders_gltf_skin_and_shape_keys(blender_skin):
    ref = json.loads((blender_skin / "poses.json").read_text())["points"]
    scene = meshport.load(blender_skin / "skin.glb")
    body = next(n for n, _ in scene.walk() if n.skin is not None)
    assert [m.name for m in body.mesh.targets] == ["Bulge", "Shift"]
    assert len(scene.animations) == 2  # the armature's action and the shape keys'
    _compare(scene, ref, body.name)  # Blender's glTF export puts frame f at f / fps


_BLENDER_IMPORT = textwrap.dedent("""
    import bpy, sys, json
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = src.rsplit(".", 1)[1].lower()
    if ext == "glb":
        bpy.ops.import_scene.gltf(filepath=src)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=src)
    elif ext == "abc":
        bpy.ops.wm.alembic_import(filepath=src)
    else:
        bpy.ops.wm.usd_import(filepath=src)
    sc = bpy.context.scene
    points = {}
    for f in range(0, 25):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        points[f] = {}
        for o in sc.objects:
            if o.type == "MESH" and (o.data.shape_keys is not None
                                     or any(m.type in ("ARMATURE", "MESH_SEQUENCE_CACHE") for m in o.modifiers)):
                ev = o.evaluated_get(dg)
                m = ev.to_mesh()
                points[f][o.name] = [list(ev.matrix_world @ v.co) for v in m.vertices]
                ev.to_mesh_clear()
    json.dump(points, open(out, "w"))
""")


def _blender_poses(path: Path) -> dict:
    script = path.parent / "import.py"
    script.write_text(_BLENDER_IMPORT)
    _blender(script, path, path.with_suffix(".json"))
    return {f: next(iter(objs.values())) for f, objs in json.loads(path.with_suffix(".json").read_text()).items()}


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_poses_our_gltf_skin_and_blend_shapes(tmp_path):
    scene = rigged()
    meshport.save(scene, tmp_path / "ours.glb")
    ref = _blender_poses(tmp_path / "ours.glb")
    # at the keys: between them Blender blends quaternion components instead of slerping
    _compare(scene, {f: p for f, p in ref.items() if f in ("0", "12", "24")}, "Body")


def test_we_pose_blenders_fbx_skin_and_shape_keys(blender_skin):
    ref = json.loads((blender_skin / "poses.json").read_text())["points"]
    scene = meshport.load(blender_skin / "skin.fbx")
    body = next(n for n, _ in scene.walk() if n.skin is not None)
    assert [m.name for m in body.mesh.targets] == ["Bulge", "Shift"] and len(scene.animations) == 1
    _compare(scene, ref, body.name)  # Blender's FBX export puts frame f at f / fps too


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_poses_our_fbx_skin_and_blend_shapes(tmp_path):
    scene = rigged()
    meshport.save(scene, tmp_path / "ours.fbx")
    ref = _blender_poses(tmp_path / "ours.fbx")
    # Blender's FBX import puts time 0 at frame 1; it reads the keys meshport writes every frame
    # for the rotations as they are, so between keys it agrees as closely as the frame keys do
    _compare(scene, {f: p for f, p in ref.items() if f in ("1", "7", "13", "19", "25")}, "Body", shift=1)


# ---- FBX

def test_fbx_round_trip_keeps_skins_targets_and_weights():
    scene = rigged()
    back = meshport.read(meshport.write(scene, "fbx"), "fbx")
    body = next(n for n, _ in back.walk() if n.name == "Body")
    assert [j.name for j in body.skin.joints] == ["Root", "Bone"]
    assert [m.name for m in body.mesh.targets] == ["Lift", "Widen"] and body.mesh.target_weights == pytest.approx([0.2, 0])
    for t in np.linspace(0, 1.2, 13):
        assert points(back, t)["Body"] == pytest.approx(points(scene, t)["Body"], abs=1e-5), t


def _fbx_records(scene: Scene) -> list:
    return _Writer(scene).records()


def _objects(records: list) -> list:
    return next(r for r in records if r.name == "Objects").children


def test_fbx_writes_skeletons_clusters_and_a_bind_pose():
    objects = _objects(_fbx_records(rigged()))
    kinds = {(r.name, str(r.props[2])) for r in objects}
    assert {("Model", "LimbNode"), ("Deformer", "Skin"), ("Deformer", "Cluster"), ("Deformer", "BlendShape"),
            ("Deformer", "BlendShapeChannel"), ("Geometry", "Shape"), ("Pose", "BindPose")} <= kinds
    clusters = [r for r in objects if r.name == "Deformer" and r.props[2] == "Cluster"]
    assert [len(r.find("Indexes").props[0]) for r in clusters] == [8, 4]  # the root holds every vertex, the bone the top
    pose = next(r for r in objects if r.name == "Pose")
    assert pose.find("NbPoseNodes").props == [3]


def test_fbx_weights_short_of_one_leave_the_rest_with_the_mesh_node():
    """In Total1 mode what a vertex's clusters leave over follows the mesh's own node (written
    where the mesh was bound); by default (Normalize) the weights are scaled up to 1."""
    scene = rigged()
    body = scene.roots[1]
    full = {t: points(scene, t)["Body"] for t in (0, 0.5, 1)}
    body.skin.weights = body.skin.weights * 0.5
    records = _fbx_records(scene)
    for r in _objects(records):
        if r.name == "Deformer" and r.props[2] == "Cluster":
            r.children.append(Record("Mode", ["Total1"]))
    total_one = meshport.read(write_binary(records), "fbx")
    normalized = meshport.read(meshport.write(scene, "fbx"), "fbx")
    node = {n.name: w for n, w in total_one.walk()}["Body"]
    assert node == pytest.approx(np.eye(4))  # where rigged()'s joints bind it, not 5 along x where it was
    for t, whole in full.items():
        morphed = body.mesh.morphed(scene.animations[0].weights(body, t))
        assert points(total_one, t)["Body"] == pytest.approx(0.5 * whole + 0.5 * morphed, abs=1e-5)
        assert points(normalized, t)["Body"] == pytest.approx(whole, abs=1e-5)


def test_fbx_damaged_clusters_and_shapes_load_without_them():
    records = _fbx_records(rigged())
    for r in _objects(records):
        if r.name == "Deformer" and r.props[2] == "Cluster":
            r.children = [c for c in r.children if c.name != "Weights"]
        if r.name == "Geometry" and r.props[2] == "Shape":
            r.children = [c if c.name != "Vertices" else Record("Vertices", [np.zeros(2)]) for c in r.children]
    body = meshport.read(write_binary(records), "fbx").roots[1]
    assert body.mesh.targets == [] and body.skin is None  # the mesh stays where its node puts it


# ---- USD

def test_we_pose_blenders_usd_skin_and_shape_keys(blender_skin):
    ref = json.loads((blender_skin / "poses.json").read_text())["points"]
    scene = meshport.load(blender_skin / "skin.usdc")
    body = next(n for n, _ in scene.walk() if n.skin is not None)
    assert [j.name for j in body.skin.joints] == ["Lower", "Upper"]
    assert [m.name for m in body.mesh.targets] == ["Bulge", "Shift"]  # its animation names them in another order
    _compare(scene, ref, body.name)  # time code f is frame f, at 24 a second


def shapes_only() -> Scene:
    """A mesh with blend shapes and no skin, moving and changing weights."""
    mesh = cube()
    mesh.targets = [Morph("Out", mesh.positions * 0.5), Morph("Up", np.tile([0, 1.0, 0], (8, 1)))]
    node = Node("Shapes", compose((1, 2, 3), (0, 30, 0)), mesh)
    return Scene([node], [Animation("A", [
        Channel(node, "weights", TIMES, np.array([[0, 0], [1, 0.5], [0.2, 1.0]])),
        Channel(node, "translation", np.array([0, 1.0]), np.array([[1, 2, 3], [3, 2, 3.0]]))])])


# times a frame clear of rigged()'s steps, which USD samples every frame
CLEAR = (0, 0.2, 0.45, 0.7, 0.95, 1.5)


@pytest.mark.parametrize("fmt", ["usda", "usdc", "usdz"])
def test_usd_round_trip_keeps_skins_targets_and_weights(fmt):
    for scene, name in ((rigged(), "Body"), (shapes_only(), "Shapes")):
        back = meshport.read(meshport.write(scene, fmt), fmt)
        for t in CLEAR:
            assert points(back, t)[name] == pytest.approx(points(scene, t)[name], abs=1e-5), (name, t)
    body = next(n for n, _ in meshport.read(meshport.write(rigged(), fmt), fmt).walk() if n.skin is not None)
    assert [m.name for m in body.mesh.targets] == ["Lift", "Widen"]


def test_usd_writes_a_skel_root_skeleton_and_bindings():
    text = meshport.write(rigged(), "usda").decode()
    for line in ('def SkelRoot "Root"', 'def Skeleton "Rig"', 'uniform token[] joints = ["Root", "Root/Bone"]',
                 'def SkelAnimation "Animation"', "rel skel:skeleton = </Root/Rig>", "elementSize = 2",
                 'def BlendShape "Lift"', 'uniform token[] blendShapes = ["Lift", "Widen"]'):
        assert line in text, line
    assert "SkelRoot" not in meshport.write(Scene([Node("N", mesh=cube())]), "usda").decode()


def test_usd_damaged_bindings_and_shapes_load_without_them():
    text = meshport.write(rigged(), "usda").decode()
    short = text.replace("elementSize = 2", "elementSize = 3")  # no longer a set per point
    body = next(n for n, _ in meshport.read(short.encode(), "usda").walk() if n.mesh is not None)
    assert body.skin is None
    bad = text.replace("uniform int[] pointIndices = [", "uniform int[] pointIndices = [99, ", 1)
    body = next(n for n, _ in meshport.read(bad.encode(), "usda").walk() if n.mesh is not None)
    assert body.mesh.targets == [] and body.skin is not None


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
@pytest.mark.parametrize("make", [rigged, shapes_only])
def test_blender_poses_our_usd_skins_and_blend_shapes(tmp_path, make):
    scene = make()
    meshport.save(scene, tmp_path / "ours.usdc")
    ref = _blender_poses(tmp_path / "ours.usdc")
    name = scene.roots[-1].name
    _compare(scene, {f: p for f, p in ref.items() if int(f) / 24 in CLEAR or f in ("6", "18")}, name)


_PIXAR_BAKE = textwrap.dedent("""
    import json, sys
    import bpy  # sets up Blender's copy of Pixar's USD
    from pxr import Gf, Usd, UsdGeom, UsdSkel
    src, out = sys.argv[sys.argv.index("--") + 1:]
    stage = Usd.Stage.Open(src)
    assert UsdSkel.BakeSkinning(stage.Traverse(), Gf.Interval(-1, 40))
    points = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            for t in range(-1, 40):
                world = mesh.ComputeLocalToWorldTransform(Usd.TimeCode(t))
                points[t] = [list(world.Transform(Gf.Vec3d(p))) for p in mesh.GetPointsAttr().Get(Usd.TimeCode(t))]
    json.dump(points, open(out, "w"))
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
@pytest.mark.parametrize("make", [rigged, shapes_only])
@pytest.mark.parametrize("fmt", ["usda", "usdc"])
def test_pixars_usdskel_poses_ours_as_we_do(tmp_path, make, fmt):
    """Pixar's own skinning (UsdSkel.BakeSkinning, in the copy of USD inside Blender) puts every
    point where meshport does, at every time code."""
    scene = make()
    meshport.save(scene, tmp_path / f"ours.{fmt}")
    script = tmp_path / "bake.py"
    script.write_text(_PIXAR_BAKE)
    _blender(script, tmp_path / f"ours.{fmt}", tmp_path / "baked.json")
    name = scene.roots[-1].name
    for t, theirs in json.loads((tmp_path / "baked.json").read_text()).items():
        assert points(scene, int(t) / 24)[name] == pytest.approx(np.array(theirs), abs=1e-5), t


# ---- formats without skins

@pytest.mark.parametrize("fmt", ["abc", "3ds", "obj"])
def test_formats_without_skins_get_the_mesh_bent_where_it_stands(tmp_path, fmt):
    scene = rigged()  # its skinned node stands 5 along x, which the skin ignores
    if fmt == "abc":
        scene.animations = []  # a moving rig gets its points every frame in Alembic (tested below)
    still = points(Scene(scene.roots), 0)["Body"]
    meshport.save(scene, tmp_path / f"out.{fmt}")
    if fmt == "obj":
        text = (tmp_path / "out.obj").read_text()
        got = np.array([[float(x) for x in line.split()[1:4]] for line in text.splitlines() if line.startswith("v ")])
    else:
        back = meshport.load(tmp_path / f"out.{fmt}")
        got = np.concatenate([p for _, p in back.posed_points(0, [])])
    assert same_points(got, still, 1e-4)
    assert scene.roots[1].mesh.targets and scene.roots[1].skin is not None  # the scene itself is left as it was


def test_deformed_keeps_node_animation_on_the_copies():
    scene = rigged()
    flat = scene.deformed()
    assert flat is not scene and all(n.skin is None and (n.mesh is None or not n.mesh.targets) for n, _ in flat.walk())
    nodes = {id(n) for n, _ in flat.walk()}
    assert flat.animations and all(id(c.node) in nodes and c.path != "weights" for c in flat.animations[0].channels)
    plain = Scene([Node("N", mesh=cube())])
    assert plain.deformed() is plain


# ---- .blend

@pytest.mark.parametrize("name", ["skin", "skin_283"])  # Blender 5.1 and 2.83 (vertex groups kept two ways)
def test_blend_armatures_and_shape_keys_move_as_in_blender(name):
    import gzip
    with gzip.open(DATA / f"{name}_poses.json.gz", "rt") as f:
        ref = json.load(f)["points"]
    scene = meshport.load(DATA / f"{name}.blend")
    body = next(n for n, _ in scene.walk() if n.mesh is not None)
    assert [j.name for j in body.skin.joints] == ["Lower", "Upper"]
    assert [m.name for m in body.mesh.targets] == ["Bulge", "Shift"]
    assert {c.path for c in scene.animations[0].channels} >= {"rotation", "translation", "weights"}
    order = np.arange(len(body.mesh.positions))  # a .blend keeps Blender's own vertex order
    for frame, theirs in ref.items():
        ours = points(scene, int(frame) / 24)[body.name]
        assert ours == pytest.approx((np.array(theirs) @ Z_UP.T)[order], abs=1e-5), frame


@pytest.mark.parametrize("fmt", ["glb", "fbx", "usdc"])
def test_a_blend_files_skin_survives_conversion(fmt):
    scene = meshport.load(DATA / "skin.blend")
    back = meshport.read(meshport.write(scene, fmt), fmt)
    for t in np.linspace(0, 1, 9):
        ours = np.concatenate([p for _, p in scene.posed_points(t)])
        assert same_points(np.concatenate([p for _, p in back.posed_points(t)]), ours, 1e-4), t


# ---- in-between shapes

HALF, FULL = np.tile([0, 1.0, 0], (8, 1)), np.tile([0, 1.0, 0], (8, 1)) + cube().positions * [1, 0, 0]


def inbetween_fbx() -> bytes:
    """A box whose one blend shape channel has an in-between shape: at 50% it's raised by 1, at 100%
    raised and stretched along x. The channel goes 0 -> 100% -> 30% over 2 seconds."""
    mesh = cube()
    mesh.targets = [Morph("Half", HALF), Morph("Grow", FULL)]
    node = Node("Box", mesh=mesh)
    scene = Scene([node], [Animation("A", [Channel(node, "weights", TIMES * 2, np.array([[0, 0], [1, 0], [0.3, 0.0]]))])])
    records = _fbx_records(scene)
    objects = _objects(records)
    first, second = [r for r in objects if r.name == "Deformer" and r.props[2] == "BlendShapeChannel"]
    connections = next(r for r in records if r.name == "Connections").children
    for c in connections:  # the second channel's shape goes to the first, as its full shape
        if c.props[2] == second.props[0] and c.props[0] == "OO":
            c.props[2] = first.props[0]
    connections[:] = [c for c in connections if c.props[1] != second.props[0]]
    first.children = [c for c in first.children if c.name != "FullWeights"] + [Record("FullWeights", [np.array([50.0, 100.0])])]
    return write_binary(records)


def test_inbetween_weights_blend_neighbouring_shapes():
    from meshport.scene import inbetween_weights, with_crossings
    w = inbetween_weights([0, 0.25, 0.5, 0.75, 1.0, 1.2, -0.5], [0.5, 1.0])
    assert w == pytest.approx(np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [-0.4, 1.4], [-1, 0]]))
    assert inbetween_weights([0.3], [1.0]) == pytest.approx(np.array([[0.3]]))  # no in-betweens: the weight itself
    times = with_crossings(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.3]), [0.5, 1.0])
    assert times == pytest.approx([0, 0.5, 1, 12 / 7, 2])  # where it passes 50% on the way up and down


def test_fbx_inbetween_shapes_blend_as_fbx_has_them():
    scene = meshport.read(inbetween_fbx(), "fbx")
    box = scene.roots[0]
    assert [m.name for m in box.mesh.targets] == ["Half 50%", "Half"]
    for t in (0.25, 0.5, 0.9, 1.0, 1.5, 2.0):
        level = np.interp(t, [0, 1, 2], [0, 1, 0.3])
        half, full = (2 * level, 0) if level <= 0.5 else (2 - 2 * level, 2 * level - 1)
        assert points(scene, t)["Box"] == pytest.approx(cube().positions + half * HALF + full * FULL, abs=1e-6), t


def inbetween_usda() -> bytes:
    """inbetween_fbx()'s box as USD: one blend shape, its in-between at weight 0.5 (inbetweens:half)."""
    mesh = cube()
    mesh.targets = [Morph("Grow", FULL)]
    node = Node("Box", mesh=mesh)
    scene = Scene([node], [Animation("A", [Channel(node, "weights", TIMES * 2, np.array([[0], [1], [0.3]]))])])
    text = meshport.write(scene, "usda").decode()
    half = ", ".join(f"({x:g}, {y:g}, {z:g})" for x, y, z in HALF)
    anchor = "uniform int[] pointIndices"
    assert anchor in text
    return text.replace(anchor, f"uniform vector3f[] inbetweens:half = [{half}] (\n            weight = 0.5\n        )\n"
                        f"        {anchor}", 1).encode()


def test_usd_inbetween_shapes_blend_as_usdskel_has_them():
    scene = meshport.read(inbetween_usda(), "usda")
    box = next(n for n, _ in scene.walk() if n.mesh is not None)
    assert [m.name for m in box.mesh.targets] == ["Grow 50%", "Grow"]
    for t in (0.25, 0.5, 0.9, 1.0, 1.5, 2.0):
        level = np.interp(t, [0, 1, 2], [0, 1, 0.3])
        half, full = (2 * level, 0) if level <= 0.5 else (2 - 2 * level, 2 * level - 1)
        assert points(scene, t)["Box"] == pytest.approx(cube().positions + half * HALF + full * FULL, abs=1e-6), t


@pytest.mark.skipif(not os.environ.get("MESHPORT_PXR_PYTHON"), reason="set MESHPORT_PXR_PYTHON to a Python with usd-core")
def test_pixar_blends_usd_inbetweens_as_we_do(tmp_path):
    (tmp_path / "box.usda").write_bytes(inbetween_usda())
    script = tmp_path / "bake.py"
    script.write_text(textwrap.dedent("""
        import json, sys
        from pxr import Gf, Usd, UsdGeom, UsdSkel
        stage = Usd.Stage.Open(sys.argv[1])
        assert UsdSkel.BakeSkinning(stage.Traverse(), Gf.Interval(0, 48))
        mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        json.dump({t: [list(mesh.ComputeLocalToWorldTransform(t).Transform(Gf.Vec3d(p)))
                       for p in mesh.GetPointsAttr().Get(t)] for t in range(0, 49, 3)}, open(sys.argv[2], "w"))
    """))
    subprocess.run([os.environ["MESHPORT_PXR_PYTHON"], str(script), str(tmp_path / "box.usda"), str(tmp_path / "out.json")],
                   check=True, capture_output=True, timeout=300)
    scene = meshport.read(inbetween_usda(), "usda")
    for code, theirs in json.loads((tmp_path / "out.json").read_text()).items():
        assert points(scene, int(code) / 24)["Box"] == pytest.approx(np.array(theirs), abs=1e-5), code


# ---- Alembic: what bends is written as points every frame

@pytest.mark.parametrize("make", [rigged, shapes_only])
def test_alembic_keeps_what_skins_and_blend_shapes_bend(make):
    """Alembic has no skins or blend shapes: a mesh they bend gets its points every frame, which
    meshport reads back as a blend shape per sample (Alembic plays them linearly, as those blend)."""
    scene = make()
    name = scene.roots[-1].name
    data = meshport.write(scene, "abc")
    back = meshport.read(data, "abc")
    mesh = next(n for n, _ in back.walk() if n.mesh is not None).mesh
    assert len(mesh.targets) == 24  # 25 frames: the first is the mesh itself
    for t in np.arange(25) / 24:
        assert same_points(np.concatenate([p for _, p in back.posed_points(t)]), points(scene, t)[name]), t


def test_alembic_point_caches_past_the_budget_stand_still(monkeypatch):
    from meshport import alembic
    data = meshport.write(rigged(), "abc")
    monkeypatch.setattr(alembic, "CACHE_BUDGET", 10)
    back = meshport.read(data, "abc")
    assert all(n.mesh is None or not n.mesh.targets for n, _ in back.walk())


def test_still_meshes_stay_one_sample_in_alembic():
    scene = rigged()
    scene.animations[0].channels = [c for c in scene.animations[0].channels if c.path == "translation"]
    scene.animations[0].channels[0].node = Node("elsewhere")  # nothing that bends the mesh moves
    back = meshport.read(meshport.write(scene, "abc"), "abc")
    assert all(n.mesh is None or not n.mesh.targets for n, _ in back.walk())


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
@pytest.mark.parametrize("make", [rigged, shapes_only])
def test_blender_plays_our_alembic_point_caches(tmp_path, make):
    scene = make()
    meshport.save(scene, tmp_path / "ours.abc")
    ref = _blender_poses(tmp_path / "ours.abc")
    _compare(scene, {f: p for f, p in ref.items() if int(f) % 4 == 0}, scene.roots[-1].name)
