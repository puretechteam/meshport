"""Real rigged and morphing models: Khronos' glTF sample models (CesiumMan, BrainStem, Fox,
RiggedFigure, RiggedSimple, SimpleSkin, AnimatedMorphCube, MorphStressTest, MorphPrimitivesTest)
in the folder named by MESHPORT_REAL_RIGS, from github.com/KhronosGroup/glTF-Sample-Assets.

- meshport poses each as Blender's own import does (at the keys: between them Blender blends
  quaternion components where glTF slerps);
- each survives glb, fbx, usdc and usdz point for point;
- Blender poses meshport's FBX and USD of each as meshport does;
- with MESHPORT_PXR_PYTHON (a Python with usd-core), Pixar's UsdSkel bakes meshport's USD of each
  to the same points; with MESHPORT_UNITY, Autodesk's FBX SDK (through Unity) bends meshport's FBX
  of each the same way; with MESHPORT_GLTF_VALIDATOR (Khronos' gltf_validator), every GLB meshport
  writes of them has no errors and no warnings the original doesn't have.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import BLENDER

import meshport

FOLDER = os.environ.get("MESHPORT_REAL_RIGS")
RIGS = sorted(p for p in Path(FOLDER).iterdir() if p.suffix in (".glb", ".gltf")) if FOLDER else []
ANIMATED = [p for p in RIGS if p.stem != "MorphPrimitivesTest"]
PXR = os.environ.get("MESHPORT_PXR_PYTHON")
VALIDATOR = os.environ.get("MESHPORT_GLTF_VALIDATOR")
pytestmark = pytest.mark.skipif(not RIGS, reason="set MESHPORT_REAL_RIGS to a folder of Khronos' sample models")
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
TOLERANCE = 1e-4  # of the model's size


def ids(paths):
    return [p.stem for p in paths]


def load(path: Path) -> meshport.Scene:
    """The model with its first animation only: what USD keeps, and what Blender's glTF import plays."""
    scene = meshport.load(path)
    scene.animations = scene.animations[:1]
    return scene


def points(scene: meshport.Scene, t: float) -> np.ndarray:
    return np.concatenate([p for _, p in scene.posed_points(t)])


def size(scene: meshport.Scene) -> float:
    start = scene.animations[0].start if scene.animations else 0.0
    return float(np.ptp(points(scene, start), axis=0).max())


def gap(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two sets of points are (both ways): vertex order and splitting differ by format."""
    from scipy.spatial import cKDTree
    return max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())


def keys(scene: meshport.Scene, most: int = 5) -> np.ndarray:
    if not scene.animations:
        return np.zeros(1)
    k = np.unique(np.concatenate([c.times for c in scene.animations[0].channels]))
    return k[np.linspace(0, len(k) - 1, min(most, len(k))).astype(int)]


_BLENDER_POINTS = textwrap.dedent("""
    import bpy, sys, json
    src, out, seconds, shift = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = src.rsplit(".", 1)[1].lower()
    if ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=src)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=src)
    else:
        bpy.ops.wm.usd_import(filepath=src)
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    if shift == "usd":  # Blender takes a USD time code as a frame
        from pxr import Usd
        fps, shift = Usd.Stage.Open(src).GetTimeCodesPerSecond(), 0
    shapes = {pb.custom_shape.name for o in sc.objects if o.type == "ARMATURE" for pb in o.pose.bones
              if pb.custom_shape}  # the bone-display shapes Blender's importer adds
    result = []
    for f in (float(x) * fps + float(shift) for x in seconds.split(",")):
        sc.frame_set(int(f), subframe=f - int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        pts = []
        for o in sc.objects:
            if o.type == "MESH" and o.name not in shapes:
                ev = o.evaluated_get(dg)
                m = ev.to_mesh()
                pts += [list(ev.matrix_world @ v.co) for v in m.vertices]
                ev.to_mesh_clear()
        result.append(pts)
    json.dump(result, open(out, "w"))
""")


def blender_points(path: Path, seconds, shift) -> list[np.ndarray]:
    """Where Blender's import of path puts every vertex at each time (frame = seconds x fps + shift)."""
    script = path.parent / "points.py"
    script.write_text(_BLENDER_POINTS)
    out = path.parent / "points.json"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(path), str(out), ",".join(repr(float(t)) for t in seconds), str(shift)],
                   check=True, capture_output=True, timeout=600)
    return [np.array(p) @ Z_UP.T for p in json.loads(out.read_text())]


@needs_blender
@pytest.mark.parametrize("path", ANIMATED, ids=ids(ANIMATED))
def test_we_pose_them_as_blender_does(path, tmp_path):
    scene = load(path)
    times = keys(scene)
    (tmp_path / path.name).write_bytes(path.read_bytes())
    for t, theirs in zip(times, blender_points(tmp_path / path.name, times, 0)):
        assert gap(points(scene, t), theirs) < TOLERANCE * size(scene), t


@pytest.mark.parametrize("fmt", ["glb", "fbx", "usdc", "usdz"])
@pytest.mark.parametrize("path", RIGS, ids=ids(RIGS))
def test_they_survive_every_format(path, fmt):
    scene = load(path)
    back = meshport.read(meshport.write(scene, fmt), fmt)
    span = (scene.animations[0].start, scene.animations[0].end) if scene.animations else (0.0, 0.0)
    for t in np.linspace(*span, 7):
        assert gap(points(scene, t), points(back, t)) < TOLERANCE * size(scene), t


@needs_blender
@pytest.mark.parametrize("fmt", ["fbx", "usdc"])
@pytest.mark.parametrize("path", ANIMATED, ids=ids(ANIMATED))
def test_blender_poses_ours_as_we_do(path, fmt, tmp_path):
    """Blender places a skinned mesh by its armature's bind matrix (FBX: TransformAssociateModel)
    and, for USD, by the geomBindTransform relative to the mesh's parent: meshport writes both so
    rigs whose skinned mesh hangs under a moved node (CesiumMan, BrainStem, ...) come out right."""
    scene = load(path)
    meshport.save(scene, tmp_path / f"ours.{fmt}")
    times = keys(scene)
    for t, theirs in zip(times, blender_points(tmp_path / f"ours.{fmt}", times, 1 if fmt == "fbx" else "usd")):
        assert gap(points(scene, t), theirs) < TOLERANCE * size(scene), t


_PIXAR_BAKE = textwrap.dedent("""
    import json, sys
    from pxr import Gf, Usd, UsdGeom, UsdSkel
    src, out, seconds = sys.argv[1:4]
    stage = Usd.Stage.Open(src)
    codes = [float(t) * stage.GetTimeCodesPerSecond() for t in seconds.split(",")]
    assert UsdSkel.BakeSkinning(stage.Traverse(), Gf.Interval(min(codes) - 1, max(codes) + 1))
    result = []
    for c in codes:
        pts = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                world = mesh.ComputeLocalToWorldTransform(Usd.TimeCode(c))
                pts += [list(world.Transform(Gf.Vec3d(p))) for p in mesh.GetPointsAttr().Get(Usd.TimeCode(c))]
        result.append(pts)
    json.dump(result, open(out, "w"))
""")


@pytest.mark.skipif(not PXR, reason="set MESHPORT_PXR_PYTHON to a Python with usd-core")
@pytest.mark.parametrize("path", ANIMATED, ids=ids(ANIMATED))
def test_pixar_bakes_our_usd_as_we_pose_it(path, tmp_path):
    scene = load(path)
    meshport.save(scene, tmp_path / "ours.usdc")
    (tmp_path / "bake.py").write_text(_PIXAR_BAKE)
    times = keys(scene)
    subprocess.run([PXR, str(tmp_path / "bake.py"), str(tmp_path / "ours.usdc"), str(tmp_path / "baked.json"),
                    ",".join(repr(float(t)) for t in times)], check=True, capture_output=True, timeout=600)
    for t, theirs in zip(times, json.loads((tmp_path / "baked.json").read_text())):
        assert gap(points(scene, t), np.array(theirs)) < TOLERANCE * size(scene), t


@pytest.mark.skipif(not os.environ.get("MESHPORT_UNITY"), reason="set MESHPORT_UNITY to Unity.exe")
def test_autodesk_bends_our_fbx_as_we_pose_it(tmp_path):
    """Every model at a spread of the frames Unity samples (Autodesk's FBX SDK evaluates them)."""
    from test_autodesk_fbx import _run_unity
    scenes = {}
    for path in ANIMATED:
        scenes[f"{path.stem}.fbx"] = load(path)
        meshport.save(scenes[f"{path.stem}.fbx"], tmp_path / f"{path.stem}.fbx")
    poses = _run_unity(tmp_path)
    for name, scene in scenes.items():
        frames = list(poses[name].items())
        for frame, (fps, meshes) in frames[::max(1, len(frames) // 8)]:
            theirs = np.concatenate(list(meshes.values()))
            assert gap(points(scene, scene.animations[0].start + frame / fps), theirs) < TOLERANCE * size(scene), \
                (name, frame)


def _validate(path: Path) -> tuple[int, set]:
    out = subprocess.run([VALIDATOR, "-o", str(path)], capture_output=True, text=True, timeout=120, check=False)
    issues = json.loads(out.stdout)["issues"]
    return issues["numErrors"], {m["code"] for m in issues["messages"] if m["severity"] == 1}


@pytest.mark.skipif(not VALIDATOR, reason="set MESHPORT_GLTF_VALIDATOR to Khronos' gltf_validator")
@pytest.mark.parametrize("path", RIGS, ids=ids(RIGS))
def test_khronos_validator_finds_nothing_new_in_our_glb(path, tmp_path):
    """meshport's GLB of each, written straight from it and after a trip through FBX and USD."""
    _, theirs = _validate(path)
    scene = meshport.load(path)
    for via in (None, "fbx", "usdc"):
        written = scene if via is None else meshport.read(meshport.write(scene, via), via)
        out = tmp_path / f"{via or 'direct'}.glb"
        meshport.save(written, out)
        errors, warnings = _validate(out)
        assert errors == 0, via
        assert warnings <= theirs, (via, warnings - theirs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
