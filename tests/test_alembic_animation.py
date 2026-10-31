"""Alembic animation: Xform samples read at their time sampling's times (uniform, cyclic, acyclic)
and written on one uniform time sampling, damaged samplings, and Blender both ways."""

import json
import struct
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER, cube
from test_animation import animated

import meshport
from meshport import Node, Scene, alembic
from meshport.geometry import compose

Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")


def points(scene: Scene, t: float) -> np.ndarray:
    return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk_at(t, scene.animations)
                           if n.mesh is not None])


def gap(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def test_round_trip_lands_on_every_frame():
    scene = animated()
    back = meshport.read(meshport.write(scene, "abc"), "abc")
    assert {(c.node.name, c.path) for c in back.animations[0].channels} == \
        {("A", "translation"), ("A", "rotation"), ("B", "scale"), ("C", "translation")}
    for t in np.arange(49) / 24:  # a sample every frame at 24 frames per second, the rate its keys sit on
        assert np.abs(points(back, t) - points(scene, t)).max() < 1e-9


def test_one_uniform_time_sampling_is_written():
    data = meshport.write(animated(), "abc")
    reader = alembic._Reader(data)
    assert reader.samplings[1] == (pytest.approx(1 / 24), [0.0])
    still = meshport.write(Scene([Node("box", compose((1, 2, 3)), cube())]), "abc")
    assert len(alembic._Reader(still).samplings) == 1 and meshport.read(still, "abc").animations == []


def test_time_samplings():
    raw = (struct.pack("<IdI", 1, 1.0, 1) + struct.pack("<d", 0)          # 0: the default
           + struct.pack("<IdI", 5, 0.5, 1) + struct.pack("<d", 2)        # 1: uniform, from 2 s every 0.5 s
           + struct.pack("<IdI", 6, 1.0, 2) + struct.pack("<2d", 0, 0.25)  # 2: cyclic, two a second
           + struct.pack("<IdI", 3, np.finfo(np.float64).max / 32, 3) + struct.pack("<3d", 0, 0.1, 0.7))  # 3: acyclic
    samplings = alembic._time_samplings(raw)
    assert len(samplings) == 4
    assert alembic._sample_times(samplings, 1, 4) == pytest.approx([2, 2.5, 3, 3.5])
    assert alembic._sample_times(samplings, 2, 5) == pytest.approx([0, 0.25, 1, 1.25, 2])
    assert alembic._sample_times(samplings, 3, 3) == pytest.approx([0, 0.1, 0.7])
    assert alembic._sample_times(samplings, 3, 4) is None  # more samples than an acyclic sampling has times
    assert alembic._sample_times(samplings, 9, 4) is None
    assert alembic._time_samplings(raw[:-4])[-1:] == [samplings[2]]  # cut short: the whole ones are kept


def test_a_bad_time_sampling_leaves_the_object_still():
    w = alembic._Writer()
    shape = w._mesh("triShape", cube(), [])
    vals = w.scalar_samples(".vals", [compose((i, 0, 0)).T.ravel() for i in range(3)], "<f8", 7)  # no sampling 7
    xform = w.compound(".xform", [w.scalar(".ops", [0x30], "u1"), vals], alembic._XFORM_PROP)
    scene = meshport.read(w.finish([("t", alembic._XFORM, w.object([xform], [shape]))]), "abc")
    assert scene.animations == [] and len(scene.meshes) == 1


# ---- Blender

_BLENDER_EXPORT = textwrap.dedent("""
    import bpy, sys, json
    out = sys.argv[sys.argv.index("--") + 1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add()
    a = bpy.context.object
    for f, loc, rot in ((1, (0, 0, 0), (0, 0, 0)), (12, (2, 1, 0), (0.5, 0.2, 1.0)), (30, (3, -1, 2), (1.5, 0, 2.5))):
        a.location, a.rotation_euler = loc, rot
        a.keyframe_insert("location", frame=f)
        a.keyframe_insert("rotation_euler", frame=f)
    a.keyframe_insert("scale", frame=1)
    a.scale = (2, 0.5, 1)
    a.keyframe_insert("scale", frame=30)
    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 3, 0))
    b = bpy.context.object
    b.parent = a
    b.rotation_mode = "QUATERNION"
    b.keyframe_insert("rotation_quaternion", frame=1)
    b.rotation_quaternion = (0, 0, 1, 0)
    b.keyframe_insert("rotation_quaternion", frame=30)
    bpy.ops.wm.alembic_export(filepath=out + "/anim.abc", start=1, end=30)
    pts = {}
    for f in range(1, 31):
        sc.frame_set(f)
        pts[f] = [list(o.matrix_world @ v.co) for o in sc.objects for v in o.data.vertices]
    json.dump({"fps": sc.render.fps, "points": pts}, open(out + "/points.json", "w"))
""")


@needs_blender
def test_we_read_blenders_alembic_animation(tmp_path):
    pytest.importorskip("scipy")
    script = tmp_path / "make.py"
    script.write_text(_BLENDER_EXPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path)], check=True, capture_output=True, timeout=300)
    ref = json.loads((tmp_path / "points.json").read_text())
    scene = meshport.load(tmp_path / "anim.abc")
    for frame, pts in ref["points"].items():  # Blender writes frame f at f / fps seconds
        assert gap(points(scene, int(frame) / ref["fps"]), np.array(pts) @ Z_UP.T) < 1e-5, frame


_BLENDER_IMPORT = textwrap.dedent("""
    import bpy, sys, json
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.alembic_import(filepath=src)
    sc = bpy.context.scene
    pts = {}
    for f in range(0, 49):
        sc.frame_set(f)
        pts[f] = [list(o.matrix_world @ v.co) for o in sc.objects if o.type == "MESH" for v in o.data.vertices]
    json.dump({"fps": sc.render.fps, "points": pts}, open(out, "w"))
""")


@needs_blender
def test_blender_reads_our_alembic_animation(tmp_path):
    pytest.importorskip("scipy")
    scene = animated()
    meshport.save(scene, tmp_path / "ours.abc")
    script = tmp_path / "import.py"
    script.write_text(_BLENDER_IMPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "ours.abc"), str(tmp_path / "points.json")], check=True, capture_output=True,
                   timeout=300)
    ref = json.loads((tmp_path / "points.json").read_text())
    for frame, pts in ref["points"].items():
        assert gap(points(scene, int(frame) / ref["fps"]), np.array(pts) @ Z_UP.T) < 1e-5, frame
