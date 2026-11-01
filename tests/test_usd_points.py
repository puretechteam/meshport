"""USD meshes with animated points, written by Pixar's library (the copy inside Blender) and posed by
meshport at the time codes Pixar evaluates them at."""

import json
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER

import meshport

_MAKE = textwrap.dedent("""
    import json, sys
    from pxr import Gf, Usd, UsdGeom, Vt
    path, out = sys.argv[sys.argv.index("--") + 1:]
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Wave")
    grid = [(x, 0.0, z) for z in range(3) for x in range(3)]
    mesh.CreateFaceVertexCountsAttr([4] * 4)
    mesh.CreateFaceVertexIndicesAttr([0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7])
    points = mesh.CreatePointsAttr()
    for code in (1, 4, 9, 13):
        points.Set(Vt.Vec3fArray([Gf.Vec3f(x, 0.3 * code * ((x + z) % 2), z) for x, _, z in grid]), code)
    still = UsdGeom.Mesh.Define(stage, "/Root/Still")
    still.CreateFaceVertexCountsAttr([3])
    still.CreateFaceVertexIndicesAttr([0, 1, 2])
    still.CreatePointsAttr([(5, 0, 0), (6, 0, 0), (5, 1, 0)])
    stage.Save()
    poses = {}
    for code in (0, 1, 2.5, 4, 6, 9, 11, 13, 20):
        poses[code] = [list(p) for p in points.Get(code)]
    open(out, "w").write(json.dumps(poses))
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender (and the copy of Pixar's library inside it) isn't installed")
def test_animated_points_play_as_pixar_plays_them(tmp_path):
    script = tmp_path / "make.py"
    script.write_text(_MAKE)
    path = tmp_path / "wave.usda"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(path), str(tmp_path / "poses.json")], check=True, capture_output=True, timeout=300)
    poses = json.loads((tmp_path / "poses.json").read_text())
    scene = meshport.load(path)
    (anim,) = scene.animations
    assert [c.path for c in anim.channels] == ["weights"]
    for code, theirs in poses.items():
        ours = {n.name: p for n, p in scene.posed_points(float(code) / 24, scene.animations)}
        np.testing.assert_allclose(ours["Wave"], np.array(theirs), atol=1e-6, err_msg=f"time code {code}")
    # the still mesh stays a plain mesh, and the animated one opens at its first sample
    nodes = {n.name: n for n, _ in scene.walk()}
    assert not nodes["Still"].mesh.targets
    assert len(nodes["Wave"].mesh.targets) == 3


def test_animated_points_round_trip_through_glb(tmp_path):
    """The blend shapes made of point samples keep playing after a trip through glTF."""
    stage = '''#usda 1.0
(
    defaultPrim = "Root"
    timeCodesPerSecond = 24
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "Root"
{
    def Mesh "Tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points.timeSamples = {
            0: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            24: [(0, 0, 2), (1, 0, 2), (0, 1, 2)],
        }
    }
}
'''
    (tmp_path / "tri.usda").write_text(stage)
    scene = meshport.load(tmp_path / "tri.usda")
    back = meshport.read(meshport.write(scene, "glb"), "glb")
    for t in (0.0, 0.5, 1.0):
        (ours,) = [p for _, p in back.posed_points(t, back.animations)]
        assert ours[:, 2] == pytest.approx([2 * t] * 3)
