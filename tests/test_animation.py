"""Node animations: the scene model's sampling, glTF reading and writing, damaged channels, and
Blender in both directions."""

import json
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER, cube

import meshport
from meshport import Animation, Channel, Node, Scene
from meshport.geometry import compose, compose_trs, decompose_trs, euler, quaternion

# Blender's Z up to glTF's Y up, for comparing Blender's matrix_world with ours
Z_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
TIMES = np.array([0, 0.5, 1.25, 2.0])


def turn(angles, axis=(0, 0.6, 0.8)) -> np.ndarray:
    a = np.asarray(angles, np.float64)[:, None] / 2
    return np.hstack([np.sin(a) * axis, np.cos(a)])


def cubic_values() -> np.ndarray:
    v = np.zeros((4, 3, 3))
    v[:, 0] = [[0, 0, 0], [1, 0, 0], [0, -1, 2], [0, 0, 0]]  # in-tangents
    v[:, 1] = [[0, 0, 0], [1, 2, 0], [2, 2, 1], [3, 0, 0]]   # values
    v[:, 2] = [[2, 0, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0]]   # out-tangents
    return v


def animated() -> Scene:
    a = Node("A", compose((1, 0, 0)), cube())
    b = Node("B", compose((0, 2, 0), (0, 0, 30)), cube())
    a.children = [b]
    c = Node("C", compose((-3, 0, 0)), cube())
    scene = Scene([a, c])
    scene.animations = [Animation("Move", [
        Channel(a, "translation", TIMES, np.array([[1, 0, 0], [2, 1, 0], [2, 3, 1], [0, 0, 0.0]])),
        Channel(a, "rotation", TIMES, turn([0, 0.4, 1.3, 2.0])),
        Channel(b, "scale", TIMES, np.array([[1, 1, 1], [2, 1, 1], [1, 3, 1], [1, 1, 1.0]]), "STEP"),
        Channel(c, "translation", TIMES, cubic_values(), "CUBICSPLINE")])]
    return scene


def poses(scene: Scene, times, animation=None) -> np.ndarray:
    return np.array([[w for _, w in scene.walk_at(t, animation)] for t in times])


# ---- sampling

def test_linear_and_step_sampling_hold_outside_the_keys():
    node = Node("n")
    linear = Channel(node, "translation", np.array([1.0, 3.0]), np.array([[0, 0, 0], [4, 2, 0.0]]))
    assert linear.sample(2.0) == pytest.approx([2, 1, 0])
    assert linear.sample(np.array([0.0, 1.0, 3.0, 9.0])) == pytest.approx(np.array([[0, 0, 0], [0, 0, 0], [4, 2, 0],
                                                                                    [4, 2, 0]]))
    step = Channel(node, "scale", np.array([0, 1.0]), np.array([[1, 1, 1], [2, 2, 2.0]]), "STEP")
    assert step.sample(0.999) == pytest.approx([1, 1, 1]) and step.sample(1.0) == pytest.approx([2, 2, 2])
    single = Channel(node, "translation", np.array([0.5]), np.array([[1, 2, 3.0]]))
    assert single.sample(np.array([0.0, 7.0])) == pytest.approx(np.array([[1, 2, 3], [1, 2, 3]]))
    jump = Channel(node, "translation", np.array([0, 1, 1, 2.0]), np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0.0]]))
    assert jump.sample(1.0)[0] == pytest.approx(5) and jump.sample(0.5)[0] == pytest.approx(0.5)  # a cut at t=1


def test_rotations_slerp_the_short_way_round():
    node = Node("n")
    quarter = turn([0, np.pi / 2], (0, 1, 0))
    half = Channel(node, "rotation", np.array([0, 1.0]), quarter).sample(0.5)
    assert half == pytest.approx(turn([np.pi / 4], (0, 1, 0))[0])  # constant speed: halfway is 45 degrees
    flipped = Channel(node, "rotation", np.array([0, 1.0]), quarter * [[1], [-1]]).sample(0.5)  # -q is the same turn
    assert np.abs(np.abs(flipped @ half) - 1) < 1e-12


def test_cubic_splines_follow_gltfs_hermite_curve():
    c = Channel(Node("n"), "translation", TIMES, cubic_values(), "CUBICSPLINE")
    # halfway through the first half-second: 0.5 v0 + 0.0625 b0 + 0.5 v1 - 0.0625 a1 (tangents scaled by the span)
    assert c.sample(0.25) == pytest.approx([0.5625, 1, 0])
    assert c.sample(TIMES) == pytest.approx(cubic_values()[:, 1])
    spin = Channel(Node("n"), "rotation", np.array([0, 1.0]), np.stack([np.zeros((2, 4)), turn([0, 1]), np.zeros((2, 4))], 1),
                   "CUBICSPLINE")
    assert np.linalg.norm(spin.sample(0.3)) == pytest.approx(1)


def test_trs_round_trips_through_matrices():
    rng = np.random.default_rng(3)
    for _ in range(200):
        m = compose(rng.normal(size=3), rng.uniform(-180, 180, 3), rng.uniform(0.2, 3, 3) * rng.choice([-1, 1], 3))
        assert compose_trs(*decompose_trs(m)) == pytest.approx(m, abs=1e-9)
    for axis in np.eye(3) * 180:  # half turns, where the usual formula divides by zero
        m = euler(axis)
        assert quaternion(decompose_trs(m)[1]) == pytest.approx(m, abs=1e-12)


def test_animation_poses_bakes_and_plays_together():
    scene = animated()
    anim = scene.animations[0]
    assert (anim.start, anim.end) == (0, 2)
    static = [w for _, w in scene.walk()]
    assert np.allclose([w for _, w in Scene(scene.roots).walk_at(1.0)], static)  # no animation: as it stands
    baked = anim.bake(24)
    assert len(baked.channels[0].times) == 49 and baked.channels[0].times[-1] == 2
    frames = np.arange(49) / 24
    assert poses(scene, frames, baked) == pytest.approx(poses(scene, frames, anim), abs=1e-12)
    a_only = Animation("A", anim.channels[:2])
    c_only = Animation("C", anim.channels[3:])
    assert poses(scene, [0.3, 1.7], [a_only, c_only])[:, 2] == pytest.approx(poses(scene, [0.3, 1.7])[:, 2])


# ---- glTF

@pytest.mark.parametrize("fmt", ["glb", "gltf"])
def test_gltf_round_trip_keeps_every_interpolation(fmt):
    scene = animated()
    data = meshport.write(scene, fmt)
    back = meshport.read(data, fmt)
    (anim,) = back.animations
    assert anim.name == "Move"
    assert [(c.node.name, c.path, c.interpolation) for c in anim.channels] == \
        [("A", "translation", "LINEAR"), ("A", "rotation", "LINEAR"), ("B", "scale", "STEP"), ("C", "translation", "CUBICSPLINE")]
    frames = np.linspace(-0.5, 2.5, 61)
    assert poses(back, frames) == pytest.approx(poses(scene, frames), abs=1e-5)  # float32 in the file


def test_animated_nodes_are_written_as_trs():
    doc = json.loads(meshport.write(animated(), "gltf"))
    nodes = {n["name"]: n for n in doc["nodes"]}
    assert "matrix" not in nodes["A"] and nodes["A"]["translation"] == [1, 0, 0]  # glTF animates TRS only
    assert "matrix" not in nodes["B"] and nodes["B"]["rotation"] == pytest.approx([0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)])
    (anim,) = doc["animations"]
    assert len({s["input"] for s in anim["samplers"]}) == 1  # one accessor for the shared key times
    times = doc["accessors"][anim["samplers"][0]["input"]]
    assert times["min"] == [0] and times["max"] == [2]  # lists, as the spec requires
    still = Scene([Node("still", compose((1, 2, 3)), cube())])
    assert "matrix" in json.loads(meshport.write(still, "gltf"))["nodes"][0] and "animations" not in \
        json.loads(meshport.write(still, "gltf"))


def test_channels_of_nodes_left_out_are_dropped():
    scene = animated()
    scene.animations[0].channels.append(Channel(Node("elsewhere"), "translation", TIMES, np.zeros((4, 3))))
    scene.animations.append(Animation("Nothing", [Channel(Node("gone"), "scale", TIMES, np.ones((4, 3)))]))
    doc = json.loads(meshport.write(scene, "gltf"))
    assert len(doc["animations"]) == 1 and len(doc["animations"][0]["channels"]) == 4


def test_damaged_channels_are_skipped_and_the_model_still_loads():
    doc = json.loads(meshport.write(animated(), "gltf"))
    anim = doc["animations"][0]
    samplers, channels = anim["samplers"], anim["channels"]
    samplers[0]["output"] = samplers[2]["output"]                # 4 keys, but VEC3 values meant for scale: still fits
    samplers[1]["output"] = samplers[0]["input"]                 # SCALAR values for a rotation
    samplers[2]["interpolation"] = "SMOOTH"                      # not an interpolation glTF has
    channels.append({"sampler": 99, "target": {"node": 0, "path": "translation"}})       # no such sampler
    channels.append({"sampler": 0, "target": {"node": 50, "path": "translation"}})       # no such node
    channels.append({"sampler": 0, "target": {"node": 0, "path": "weights"}})            # morph weights: not yet
    doc["accessors"][samplers[3]["output"]]["count"] = 10 ** 6   # past the end of its buffer
    back = meshport.read(json.dumps(doc).encode(), "gltf")
    assert len(back.meshes) == 3
    assert [(c.node.name, c.path) for c in back.animations[0].channels] == [("A", "translation")]


def test_unsorted_and_nan_key_times_are_cleaned_up():
    node = Node("n", mesh=cube())
    times = np.array([1.0, np.nan, 0.0, 2.0])
    values = np.array([[1, 0, 0], [9, 9, 9], [0, 0, 0], [2, 0, 0.0]])
    scene = Scene([node], [Animation("a", [Channel(node, "translation", times, values)])])
    (c,) = meshport.read(meshport.write(scene, "gltf"), "gltf").animations[0].channels
    assert list(c.times) == [0, 1, 2] and c.values[:, 0] == pytest.approx([0, 1, 2])


def test_3ds_writes_no_animation():
    # FBX, USD and Alembic keep it (test_fbx_animation.py, test_usd_animation.py, test_alembic_animation.py);
    # meshport's 3DS writer has no keyframer, so the first frame is what's left
    back = meshport.read(meshport.write(animated(), "3ds"), "3ds")
    assert back.animations == [] and len(back.meshes) == 3


# ---- Blender

_BLENDER_EXPORT = textwrap.dedent("""
    import bpy, sys, json
    from mathutils import Euler
    out = sys.argv[sys.argv.index("--") + 1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add()
    a = bpy.context.object
    a.name = "Mover"
    # quaternion keys: Blender's export can only approximate euler curves with glTF's quaternion splines
    a.rotation_mode = "QUATERNION"
    for f, loc, rot in ((1, (0, 0, 0), (0, 0, 0)), (12, (2, 1, 0), (0.5, 0.2, 1.0)), (30, (3, -1, 2), (1.5, 0, 2.5))):
        a.location, a.rotation_quaternion = loc, Euler(rot).to_quaternion()
        a.keyframe_insert("location", frame=f)
        a.keyframe_insert("rotation_quaternion", frame=f)
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
    # once sampled per frame (Blender's default), once as Bezier curves written as cubic splines
    bpy.ops.export_scene.gltf(filepath=out + "/sampled.glb", export_format="GLB")
    bpy.ops.export_scene.gltf(filepath=out + "/curves.glb", export_format="GLB", export_force_sampling=False)
    mats, locations = {}, {}
    for f in range(1, 31):
        sc.frame_set(f)
        mats[f] = {o.name: [list(r) for r in o.matrix_world] for o in sc.objects}
        locations[f] = list(a.location)
    json.dump({"fps": sc.render.fps, "frames": mats, "locations": locations}, open(out + "/poses.json", "w"))
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_we_read_blenders_animation(tmp_path):
    script = tmp_path / "make.py"
    script.write_text(_BLENDER_EXPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path)], check=True, capture_output=True, timeout=300)
    ref = json.loads((tmp_path / "poses.json").read_text())
    # Blender's cubic splines for quaternions come out a little off its own curves (by up to 1e-3 in the
    # quaternion), so with curves only translation is exact: there meshport gives Blender's location
    for name, tolerance in (("sampled", 1e-5), ("curves", 5e-3)):
        scene = meshport.load(tmp_path / f"{name}.glb")
        assert len(scene.animations) == 2  # Blender exports each object's action as its own
        move = next(c for a in scene.animations for c in a.channels if c.node.name == "Mover" and c.path == "translation")
        assert move.interpolation == ("LINEAR" if name == "sampled" else "CUBICSPLINE")
        for frame, objects in ref["frames"].items():  # Blender's glTF export puts frame f at f / fps seconds
            t = int(frame) / ref["fps"]
            assert move.sample(t) == pytest.approx(Z_UP[:3, :3] @ ref["locations"][frame], abs=1e-5), (name, frame)
            ours = {n.name: w for n, w in scene.walk_at(t, scene.animations)}
            for obj, m in objects.items():
                assert ours[obj] == pytest.approx(Z_UP @ np.array(m) @ Z_UP.T, abs=tolerance), (name, frame, obj)


_BLENDER_IMPORT = textwrap.dedent("""
    import bpy, sys, json
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    sc = bpy.context.scene
    mats = {}
    for f in range(0, 49):
        sc.frame_set(f)
        mats[f] = {o.name: [list(r) for r in o.matrix_world] for o in sc.objects if o.type == "MESH"}
    json.dump({"fps": sc.render.fps, "frames": mats}, open(out, "w"))
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_reads_our_animation(tmp_path):
    scene = animated()
    meshport.save(scene, tmp_path / "ours.glb")
    script = tmp_path / "import.py"
    script.write_text(_BLENDER_IMPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "ours.glb"), str(tmp_path / "poses.json")], check=True, capture_output=True,
                   timeout=300)
    ref = json.loads((tmp_path / "poses.json").read_text())
    keys = {round(t * ref["fps"]) for t in TIMES}
    for frame, objects in ref["frames"].items():
        ours = {n.name: w for n, w in scene.walk_at(int(frame) / ref["fps"])}
        for obj, m in objects.items():
            theirs = Z_UP @ np.array(m) @ Z_UP.T
            if int(frame) in keys:
                assert ours[obj] == pytest.approx(theirs, abs=1e-5), (frame, obj)
            elif obj != "C":
                # between keys Blender blends quaternion components instead of slerping; its importer
                # also builds its own curve for cubic splines (C) rather than following their tangents
                assert ours[obj] == pytest.approx(theirs, abs=0.01), (frame, obj)


def test_keys_closer_than_32_bits_are_one_gltf_key():
    """A key 1e-7 s from a frame sample (as a trip through FBX or USD can leave) would make
    glTF's 32-bit times stand still, which glTF forbids: the later one is dropped."""
    node = Node("n", mesh=cube())
    times = np.array([0.0, 4.8, 4.8 + 1e-7, 5.0])
    scene = Scene([node], [Animation("A", [Channel(node, "translation", times, np.arange(12.0).reshape(4, 3))])])
    doc = json.loads(meshport.write(scene, "gltf"))
    acc = doc["accessors"][doc["animations"][0]["samplers"][0]["input"]]
    assert acc["count"] == 3
    (c,) = meshport.read(json.dumps(doc).encode(), "gltf").animations[0].channels
    assert (np.diff(c.times) > 0).all() and c.values[1] == pytest.approx([3, 4, 5])
