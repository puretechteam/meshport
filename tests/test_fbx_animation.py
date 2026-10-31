"""FBX animation: evaluating curves, reading stacks into channels (through pivots, pre/post
rotation and the axis conversion), writing them, damaged curves, and Blender both ways.
test_autodesk_fbx.py checks the same against Autodesk's FBX SDK."""

import json
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER, cube
from test_animation import TIMES, animated, poses

import meshport
from meshport import Animation, Channel, Node, Scene, fbx
from meshport.fbxanim import (
    CONSTANT,
    CUBIC,
    LINEAR,
    THIRDS,
    TICKS,
    Curve,
    continuous,
)
from meshport.fbxtree import Record, write_ascii, write_binary
from meshport.geometry import compose, decompose_many, decompose_trs, euler, euler_many

Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])


def curve(times, values, flags, data=None, counts=None) -> Curve:
    rec = Record("AnimationCurve", children=[
        Record("KeyTime", [np.round(np.asarray(times) * TICKS).astype(np.int64)]),
        Record("KeyValueFloat", [np.asarray(values, np.float32)]), Record("KeyAttrFlags", [np.asarray(flags, np.int32)]),
        Record("KeyAttrDataFloat", [np.asarray(data if data is not None else [[0, 0, THIRDS, 0]] * len(flags),
                                               np.float32).ravel()]),
        Record("KeyAttrRefCount", [np.asarray(counts if counts is not None else [len(times)] if len(flags) == 1
                                              else [1] * len(flags), np.int32)])])
    return Curve.from_record(rec)


# ---- curves

def test_constant_linear_and_constant_next_keys():
    c = curve([0, 1, 2], [0, 10, 20], [CONSTANT, CONSTANT | 0x100, LINEAR], counts=[1, 1, 1])
    assert c.evaluate(np.array([-1, 0, 0.5, 1, 1.5, 2, 3.0])) == pytest.approx([0, 0, 0, 10, 20, 20, 20])
    line = curve([0, 2], [1, 5], [LINEAR])
    assert line.evaluate(np.array([0.5, 1.0])) == pytest.approx([2, 3])
    assert curve([1], [7], [LINEAR]).evaluate(np.array([0, 5.0])) == pytest.approx([7, 7])


def test_cubic_keys_leave_each_key_at_its_slope():
    # handles a third of the way along: the same curve as a Hermite spline with those slopes
    c = curve([0, 2], [1, 3], [CUBIC], [[4, -1, THIRDS, 0], [0, 0, THIRDS, 0]], counts=[1, 1])
    t = np.linspace(0, 2, 9)
    u = t / 2
    hermite = (2 * u ** 3 - 3 * u ** 2 + 1) * 1 + (u ** 3 - 2 * u ** 2 + u) * 2 * 4 + (-2 * u ** 3 + 3 * u ** 2) * 3 \
        + (u ** 3 - u ** 2) * 2 * -1
    assert c.evaluate(t) == pytest.approx(hermite, abs=1e-6)


def test_weighted_handles():
    w = np.array([6000 | 1000 << 16], np.int32).view(np.float32)[0]  # weights in 1/9999ths
    w0, w1 = 6000 / 9999, 1000 / 9999
    c = curve([0, 1], [0, 1], [CUBIC | 0x03000000], [[2, 3, w, 0], [0, 0, THIRDS, 0]], counts=[1, 1])
    s = np.linspace(0, 1, 11)  # the Bezier with handles at (w0, 2 w0) and (1 - w1, 1 - 3 w1), by its own parameter
    x = 3 * (1 - s) ** 2 * s * w0 + 3 * (1 - s) * s ** 2 * (1 - w1) + s ** 3
    y = 3 * (1 - s) ** 2 * s * 2 * w0 + 3 * (1 - s) * s ** 2 * (1 - 3 * w1) + s ** 3
    assert c.evaluate(x) == pytest.approx(y, abs=1e-6)


def test_damaged_curves_are_dropped_or_cleaned():
    assert curve([], [], [LINEAR]) is None
    rec = Record("AnimationCurve", children=[Record("KeyTime", [np.array([0, TICKS], np.int64)]),
                                             Record("KeyValueFloat", [np.array([1], np.float32)])])
    assert Curve.from_record(rec) is None  # fewer values than keys
    c = curve([1, 2, 0], [5, np.nan, 3], [LINEAR])  # out of order, and a value that isn't a number
    assert list(c.times) == [0, 1] and list(c.values) == [3, 5]
    odd = curve([0, 1, 2], [0, 1, 2], [LINEAR, CONSTANT], [[0, 0, 0, 0]], counts=[5, 99])  # counts past the keys
    assert odd.evaluate(np.array([1.5])) == pytest.approx([1.5])


def test_batched_transforms_match_the_single_ones():
    rng = np.random.default_rng(4)
    angles = rng.uniform(-180, 180, (50, 3))
    for order in ("XYZ", "ZXY", "YZX"):
        assert euler_many(angles, order) == pytest.approx(np.array([euler(a, order) for a in angles]))
    m = np.array([compose(rng.normal(size=3), a, rng.uniform(0.3, 2, 3) * rng.choice([-1, 1], 3)) for a in angles])
    t, q, s = decompose_many(m)
    for k in range(len(m)):
        tt, qq, ss = decompose_trs(m[k])
        assert t[k] == pytest.approx(tt) and q[k] == pytest.approx(qq) and s[k] == pytest.approx(ss)
    spin = continuous(np.array([[0, 0, 170], [0, 0, -175], [0, 0, -160], [179, 0, 0], [-179, 0, 0]]))
    assert spin[:, 2].tolist()[:3] == [170, 185, 200] and abs(spin[4, 0] - spin[3, 0]) < 5


# ---- reading and writing

def test_fbx_round_trip_lands_on_every_frame():
    scene = animated()
    back = meshport.read(meshport.write(scene, "fbx"), "fbx")
    (anim,) = back.animations
    assert anim.name == "Move"
    kinds = {(c.node.name, c.path): c.interpolation for c in anim.channels}
    assert kinds[("B", "scale")] == "STEP" and kinds[("A", "rotation")] == "LINEAR"
    frames = np.arange(49) / 24  # written at 24 frames per second, the rate its keys sit on
    assert poses(back, frames) == pytest.approx(poses(scene, frames), abs=1e-5)
    # between frames the cubic spline (C) is a straight line from frame to frame
    between = np.linspace(0, 2, 301)
    assert poses(back, between) == pytest.approx(poses(scene, between), abs=0.02)


def test_linear_translation_keeps_its_own_keys_and_ascii_reads_the_same():
    box = Node("box", compose((0, 1, 0)), cube())
    scene = Scene([box], [Animation("slide", [Channel(box, "translation", TIMES, np.array([[0, 1, 0], [3, 1, 0], [3, 2, 0],
                                                                                          [0, 0, 0.0]]))])])
    records = fbx._Writer(scene).records()
    (c,) = fbx.read(write_binary(records)).animations[0].channels
    assert c.path == "translation" and list(c.times) == pytest.approx(TIMES)  # no extra keys needed
    ascii_back = fbx.read(write_ascii(records).encode())
    assert poses(ascii_back, TIMES) == pytest.approx(poses(scene, TIMES), abs=1e-6)


def test_pivots_pre_rotation_and_axis_conversion_are_posed():
    """Curves holding a model's own values pose it exactly as it stands, through every transform
    property and a Z-up file's conversion."""
    from test_fbx import _P, _patch, _settings
    box = Node("box", compose((1, 2, 3), (10, 20, 30), (1, 2, 1)), cube())
    q = decompose_trs(box.matrix)[1]
    scene = Scene([box], [Animation("still", [Channel(box, "rotation", np.array([0, 1.0]), np.array([q, q]))])])
    records = fbx._Writer(scene).records()
    _patch(records, "Model", "box", [_P("RotationPivot", 1, 0.5, 0), _P("ScalingPivot", 0, 1, 0),
                                     _P("RotationOffset", 0.2, 0, 0), _P("PreRotation", 10, 20, 30),
                                     _P("PostRotation", 5, -15, 40), Record("P", ["RotationOrder", "enum", "", "", 4]),
                                     Record("P", ["RotationActive", "bool", "", "", 1])])
    _settings(records, UpAxis=2, FrontAxis=1, FrontAxisSign=-1, UnitScaleFactor=2.54)
    read = fbx.read(write_binary(records))
    still = np.array([w for _, w in read.walk()])
    assert read.animations and np.array([w for _, w in read.walk_at(0.5)]) == pytest.approx(still, abs=1e-6)


def test_layers_and_stacks():
    scene = animated()
    scene.animations.append(Animation("Second", [Channel(scene.roots[1], "scale", TIMES, np.ones((4, 3)) * 2, "STEP")]))
    back = fbx.read(fbx.write(scene))
    assert [a.name for a in back.animations] == ["Move", "Second"]
    assert [c.path for c in back.animations[1].channels] == ["scale"]


def test_a_damaged_curve_loses_its_model_animation_only():
    records = fbx._Writer(animated()).records()
    objects = next(r for r in records if r.name == "Objects")
    first = next(r for r in objects.children if r.name == "AnimationCurve")
    first.find("KeyTime").props = [np.array([0, 5], np.int64)]  # 2 keys, many values: still usable
    first.find("KeyValueFloat").props = ["not numbers"]
    back = fbx.read(write_binary(records))
    assert len(back.meshes) == 3 and back.animations  # the others keep theirs


def test_other_writers_and_static_scenes_are_unchanged():
    static = Scene([Node("box", compose((1, 2, 3)), cube())])
    back = fbx.read(fbx.write(static))
    assert back.animations == []
    names = {r.name for r in next(r for r in fbx._Writer(static).records() if r.name == "Objects").children}
    assert "AnimationStack" not in names


# ---- Blender

_BLENDER_EXPORT = textwrap.dedent("""
    import bpy, sys, json
    out = sys.argv[sys.argv.index("--") + 1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add()
    a = bpy.context.object
    a.name = "Mover"
    for f, loc, rot in ((1, (0, 0, 0), (0, 0, 0)), (12, (2, 1, 0), (0.5, 0.2, 1.0)), (30, (3, -1, 2), (1.5, 0, 2.5))):
        a.location, a.rotation_euler = loc, rot
        a.keyframe_insert("location", frame=f)
        a.keyframe_insert("rotation_euler", frame=f)
    a.keyframe_insert("scale", frame=1)
    a.scale = (2, 0.5, 1)
    a.keyframe_insert("scale", frame=30)
    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 3, 0))
    b = bpy.context.object
    b.name = "Child"
    b.parent = a
    b.rotation_mode = "QUATERNION"
    b.keyframe_insert("rotation_quaternion", frame=1)
    b.rotation_quaternion = (0, 0, 1, 0)
    b.keyframe_insert("rotation_quaternion", frame=30)
    sc.frame_start, sc.frame_end = 1, 30
    # the whole scene as one stack, a key every frame (no simplifying)
    bpy.ops.export_scene.fbx(filepath=out + "/anim.fbx", bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                             bake_anim_simplify_factor=0.0)
    pts = {}
    for f in range(1, 31):
        sc.frame_set(f)
        pts[f] = {o.name: [list(o.matrix_world @ v.co) for v in o.data.vertices] for o in sc.objects}
    json.dump({"fps": sc.render.fps, "points": pts}, open(out + "/points.json", "w"))
""")


def _gap(ours: dict, theirs: dict) -> float:
    from scipy.spatial import cKDTree
    return max(cKDTree(ours[name]).query(np.array(p) @ Z_UP.T)[0].max() for name, p in theirs.items())


def _points(scene: Scene, t: float) -> dict:
    return {n.name: n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk_at(t, scene.animations)
            if n.mesh is not None}


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_we_read_blenders_fbx_animation(tmp_path):
    pytest.importorskip("scipy")
    script = tmp_path / "make.py"
    script.write_text(_BLENDER_EXPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path)], check=True, capture_output=True, timeout=300)
    ref = json.loads((tmp_path / "points.json").read_text())
    scene = meshport.load(tmp_path / "anim.fbx")
    assert [a.name for a in scene.animations] == ["Scene"]
    for frame, objects in ref["points"].items():  # Blender's FBX export puts frame f at f / fps seconds
        assert _gap(_points(scene, int(frame) / ref["fps"]), objects) < 1e-5, frame


_BLENDER_IMPORT = textwrap.dedent("""
    import bpy, sys, json
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=src)
    sc = bpy.context.scene
    pts = {}
    for f in range(1, 50):
        sc.frame_set(f)
        pts[f] = {o.name: [list(o.matrix_world @ v.co) for v in o.data.vertices] for o in sc.objects if o.type == "MESH"}
    json.dump({"fps": sc.render.fps, "points": pts}, open(out, "w"))
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_reads_our_fbx_animation(tmp_path):
    pytest.importorskip("scipy")
    scene = animated()
    meshport.save(scene, tmp_path / "ours.fbx")
    script = tmp_path / "import.py"
    script.write_text(_BLENDER_IMPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "ours.fbx"), str(tmp_path / "points.json")], check=True, capture_output=True,
                   timeout=300)
    ref = json.loads((tmp_path / "points.json").read_text())
    for frame, objects in ref["points"].items():  # Blender's FBX import starts the file's time 0 at frame 1
        assert _gap(_points(scene, (int(frame) - 1) / ref["fps"]), objects) < 1e-5, frame
