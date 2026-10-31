"""USD animation: xform op time samples read (through layer offsets and timeCodesPerSecond) and
written, damaged samples, and Pixar's own library (the copy inside Blender) and Blender both ways."""

import json
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER
from test_animation import animated

import meshport

Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")


def points(scene: meshport.Scene, t: float) -> np.ndarray:
    return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk_at(t, scene.animations)
                           if n.mesh is not None])


def gap(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


@pytest.mark.parametrize("fmt", ["usda", "usdc", "usdz"])
def test_round_trip_lands_on_every_frame(fmt):
    scene = animated()
    back = meshport.read(meshport.write(scene, fmt), fmt)
    assert {(c.node.name, c.path) for c in back.animations[0].channels} == \
        {("A", "translation"), ("A", "rotation"), ("B", "scale"), ("C", "translation")}
    for t in np.arange(49) / 24:  # written at 24 time codes per second, the rate its keys sit on
        assert np.abs(points(back, t) - points(scene, t)).max() < 1e-5


def test_written_layer_has_time_samples_and_a_time_line():
    text = meshport.write(animated(), "usda").decode()
    assert "timeCodesPerSecond = 24" in text and "startTimeCode = 0" in text and "endTimeCode = 48" in text
    assert "double3 xformOp:translate.timeSamples = {" in text and "quatf xformOp:orient.timeSamples" in text
    assert '["xformOp:translate", "xformOp:orient", "xformOp:scale"]' in text
    still = meshport.Scene([meshport.Node("box", animated().roots[1].matrix, animated().roots[1].mesh)])
    plain = meshport.write(still, "usda").decode()
    assert "timeSamples" not in plain and "timeCodesPerSecond" not in plain


_LAYERS = {
    "root.usda": """#usda 1.0
(
    defaultPrim = "World"
    timeCodesPerSecond = 24
    subLayers = [@sub.usda@ (offset = 10; scale = 2)]
)
def Xform "World"
{
    def "Chair" (references = @asset.usda@</Asset> (offset = 5; scale = 0.5)) {}
    def "Twin" (references = </World/Template> (offset = -3)) {}
    def Xform "Template"
    {
        double3 xformOp:translate.timeSamples = {0: (0, 0, 0), 12: (0, 4, 0)}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Box" {}
    }
}
""",
    "sub.usda": """#usda 1.0
(
    timeCodesPerSecond = 24
)
over "World"
{
    def Xform "Spinner"
    {
        float xformOp:rotateY.timeSamples = {0: 0, 6: 90}
        uniform token[] xformOpOrder = ["xformOp:rotateY"]
        def Cube "Box"
        {
            double3 xformOp:translate = (2, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
""",
    "asset.usda": """#usda 1.0
(
    defaultPrim = "Asset"
    timeCodesPerSecond = 48
)
def Xform "Asset"
{
    double3 xformOp:translate.timeSamples = {0: (0, 0, 0), 48: (10, 0, 0)}
    float3 xformOp:scale.timeSamples = {0: (1, 1, 1), 24: (2, 2, 2)}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    def Cube "Box" {}
}
""",
}


@pytest.fixture
def layered(tmp_path):
    for name, text in _LAYERS.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def test_layer_offsets_retime_samples(layered):
    scene = meshport.load(layered / "root.usda")
    times = {c.node.name: (c.times[0] * 24, c.times[-1] * 24) for c in scene.animations[0].channels}
    # a sublayer at offset 10, scale 2: its 0..6 is the stage's 10..22
    assert times["Spinner"] == pytest.approx((10, 22))
    # a reference at offset 5, scale 0.5 into a layer at 48 time codes a second: its 0..48 (one second
    # there) is half a second here, 12 time codes, from 5
    assert times["Chair"] == pytest.approx((5, 17))
    assert times["Twin"] == pytest.approx((-3, 9)) and times["Template"] == pytest.approx((0, 12))
    nodes = {n.name: n for n, _ in scene.walk()}
    chair = scene.animations[0].local_matrix(nodes["Chair"], 17 / 24)
    assert chair[:3, 3] == pytest.approx([10, 0, 0]) and np.linalg.norm(chair[:3, 0]) == pytest.approx(2)
    spin = scene.animations[0].local_matrix(nodes["Spinner"], 16 / 24)  # halfway: 45 degrees about y
    assert spin[:3, :3] @ [1, 0, 0] == pytest.approx([np.cos(np.pi / 4), 0, -np.sin(np.pi / 4)])


def test_damaged_samples_leave_the_prim_still():
    text = b"""#usda 1.0
def Xform "Box"
{
    double3 xformOp:translate.timeSamples = {0: (0, 0, 0), 10: (1, 2)}
    float3 xformOp:scale.timeSamples = {0: (1, 1, 1), 10: (nan, 1, 1)}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    def Cube "Mesh" {}
}
"""
    scene = meshport.read(text, "usda")
    assert len(scene.meshes) == 1 and scene.animations == []


def test_one_sample_is_a_still_value():
    text = b"""#usda 1.0
def Xform "Box"
{
    double3 xformOp:translate.timeSamples = {5: (1, 2, 3)}
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Cube "Mesh" {}
}
"""
    scene = meshport.read(text, "usda")
    assert scene.animations == [] and scene.roots[0].matrix[:3, 3] == pytest.approx([0.01, 0.02, 0.03])  # cm


# ---- Pixar's library and Blender

_PIXAR = textwrap.dedent("""
    import sys, json
    from pxr import Usd, UsdGeom, UsdUtils
    folder, names, times = sys.argv[sys.argv.index("--") + 1], sys.argv[-2].split(","), json.loads(sys.argv[-1])
    out = {}
    for name in names:
        stage = Usd.Stage.Open(folder + "/" + name)
        frames = {}
        for t in times:
            cache = UsdGeom.XformCache(Usd.TimeCode(t))
            pts = []
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    m = cache.GetLocalToWorldTransform(prim)
                    pts += [list(m.Transform(v)) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode(t))]
                elif prim.IsA(UsdGeom.Cube):
                    pts.append(list(cache.GetLocalToWorldTransform(prim).ExtractTranslation()))
            frames[t] = pts
        checker = UsdUtils.ComplianceChecker(arkit=False, skipARKitRootLayerCheck=True)
        checker.CheckCompliance(folder + "/" + name)
        out[name] = {"frames": frames, "tcps": stage.GetTimeCodesPerSecond(),
                     "problems": checker.GetErrors() + checker.GetFailedChecks()}
    json.dump(out, open(folder + "/pixar.json", "w"))
""")


def _pixar(folder, names, times) -> dict:
    script = folder / "pixar.py"
    script.write_text(_PIXAR)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(folder), ",".join(names), json.dumps(list(times))], check=True, capture_output=True,
                   timeout=300)
    return json.loads((folder / "pixar.json").read_text())


@needs_blender
def test_pixars_library_reads_our_animation(tmp_path):
    pytest.importorskip("scipy")
    scene = animated()
    for fmt in ("usda", "usdc", "usdz"):
        meshport.save(scene, tmp_path / f"ours.{fmt}")
    got = _pixar(tmp_path, ["ours.usda", "ours.usdc", "ours.usdz"], range(49))
    for name, res in got.items():
        assert res["tcps"] == 24 and res["problems"] == [], name
        for t, pts in res["frames"].items():  # Pixar's library is Y up in meters here too
            assert gap(points(scene, int(t) / 24), np.array(pts)) < 1e-5, (name, t)


@needs_blender
def test_layer_offsets_like_pixars_library(layered):
    times = [x / 2 for x in range(-10, 60)]
    got = _pixar(layered, ["root.usda"], times)["root.usda"]["frames"]
    scene = meshport.load(layered / "root.usda")
    for t in times:  # each Cube's origin; the stage is in centimeters, as USD's default
        ours = np.array([w[:3, 3] for n, w in scene.walk_at(t / 24) if n.mesh is not None])
        theirs = np.array(got[str(t)]) * 0.01
        assert len(ours) == len(theirs) == 4 and gap(ours, theirs) < 1e-9, t


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
    sc.frame_start, sc.frame_end = 1, 30
    bpy.ops.wm.usd_export(filepath=out + "/anim.usda", export_animation=True)
    bpy.ops.wm.usd_export(filepath=out + "/anim.usdc", export_animation=True)
    pts = {}
    for f in range(1, 31):
        sc.frame_set(f)
        pts[f] = [list(o.matrix_world @ v.co) for o in sc.objects for v in o.data.vertices]
    json.dump({"fps": sc.render.fps, "points": pts}, open(out + "/points.json", "w"))
""")


@needs_blender
def test_we_read_blenders_usd_animation(tmp_path):
    pytest.importorskip("scipy")
    script = tmp_path / "make.py"
    script.write_text(_BLENDER_EXPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path)], check=True, capture_output=True, timeout=300)
    ref = json.loads((tmp_path / "points.json").read_text())
    for name in ("anim.usda", "anim.usdc"):
        scene = meshport.load(tmp_path / name)
        for frame, pts in ref["points"].items():  # time code = frame
            assert gap(points(scene, int(frame) / ref["fps"]), np.array(pts) @ Z_UP.T) < 1e-5, (name, frame)


_BLENDER_IMPORT = textwrap.dedent("""
    import bpy, sys, json
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=src)
    sc = bpy.context.scene
    pts = {}
    for f in range(0, 49):
        sc.frame_set(f)
        pts[f] = [list(o.matrix_world @ v.co) for o in sc.objects if o.type == "MESH" for v in o.data.vertices]
    json.dump({"fps": sc.render.fps, "points": pts}, open(out, "w"))
""")


@needs_blender
def test_blender_reads_our_usd_animation(tmp_path):
    pytest.importorskip("scipy")
    scene = animated()
    meshport.save(scene, tmp_path / "ours.usdz")
    script = tmp_path / "import.py"
    script.write_text(_BLENDER_IMPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "ours.usdz"), str(tmp_path / "points.json")], check=True, capture_output=True,
                   timeout=300)
    ref = json.loads((tmp_path / "points.json").read_text())
    for frame, pts in ref["points"].items():
        assert gap(points(scene, int(frame) / ref["fps"]), np.array(pts) @ Z_UP.T) < 1e-5, frame
