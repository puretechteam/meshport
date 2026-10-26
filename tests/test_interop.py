"""Checks against other software. Each test skips when that software isn't installed."""

import json
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER, world_triangles

import meshport
from meshport.__main__ import main


def test_cli(scene, tmp_path, capsys):
    meshport.save(scene, tmp_path / "in.fbx")
    assert main([str(tmp_path / "in.fbx"), str(tmp_path / "out.glb")]) == 0
    assert world_triangles(meshport.load(tmp_path / "out.glb")) == world_triangles(scene)
    assert main([str(tmp_path / "in.fbx")]) == 0
    assert "box: 8 vertices, 6 faces" in capsys.readouterr().out
    (tmp_path / "junk.fbx").write_bytes(b"not a model")
    assert main([str(tmp_path / "junk.fbx"), str(tmp_path / "x.glb")]) == 1


def test_trimesh_reads_our_glb(scene):
    trimesh = pytest.importorskip("trimesh")
    import io
    loaded = trimesh.load(io.BytesIO(meshport.write(scene, "glb")), file_type="glb", force="scene")
    np.testing.assert_allclose(loaded.bounds, scene.bounds(), atol=1e-5)


_BLENDER_SCRIPT = textwrap.dedent("""
    import bpy, sys
    mode, src, dst = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if mode == "import":
        bpy.ops.import_scene.fbx(filepath=src)
        bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
    else:
        bpy.ops.mesh.primitive_monkey_add(rotation=(0.3, 0.2, 0.1), location=(1, 2, 3))
        bpy.ops.mesh.primitive_cylinder_add(vertices=7, location=(-2, 0, 1), scale=(1, 2, 0.5))
        bpy.ops.export_scene.fbx(filepath=src, axis_forward="Y", axis_up="Z", global_scale=2.0)
        bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
""")


def _blender(tmp_path, *args):
    script = tmp_path / "script.py"
    script.write_text(_BLENDER_SCRIPT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1",
                    "--python", str(script), "--", *map(str, args)], check=True, capture_output=True, timeout=300)


def _vertex_gap(a: meshport.Scene, b: meshport.Scene, b_scale: float = 1.0) -> float:
    from scipy.spatial import cKDTree

    def points(s):
        return np.unique(np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3]
                                         for n, w in s.walk() if n.mesh is not None]).round(6), axis=0)
    pa, pb = points(a), points(b) * b_scale
    return max(cKDTree(pb).query(pa)[0].max(), cKDTree(pa).query(pb)[0].max())


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_reads_our_fbx(scene, tmp_path):
    pytest.importorskip("scipy")
    meshport.save(scene, tmp_path / "ours.fbx")
    _blender(tmp_path, "import", tmp_path / "ours.fbx", tmp_path / "blender.glb")
    assert _vertex_gap(meshport.load(tmp_path / "blender.glb"), scene) < 1e-5


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_we_read_blenders_fbx(tmp_path):
    pytest.importorskip("scipy")
    _blender(tmp_path, "export", tmp_path / "blender.fbx", tmp_path / "blender.glb")
    # the FBX was exported Z-up with a 2x global scale; the GLB is the plain scene
    gap = _vertex_gap(meshport.load(tmp_path / "blender.fbx"), meshport.load(tmp_path / "blender.glb"), 2.0)
    assert gap < 1e-4


_BLENDER_3DS_SCRIPT = textwrap.dedent("""
    import addon_utils, bpy, json, sys
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)

    def have_importer():
        try:
            return bool(bpy.ops.import_scene.max3ds.get_rna_type())
        except KeyError:
            return False

    # Blender 4.2+ ships the 3DS importer as an extension, older versions as a bundled add-on.
    for module in ("io_scene_3ds", "bl_ext.blender_org.io_scene_3ds", "bl_ext.user_default.io_scene_3ds"):
        try:
            addon_utils.enable(module, default_set=False)
        except Exception:
            pass
        if have_importer():
            break
    if have_importer():
        names = bpy.ops.import_scene.max3ds.get_rna_type().properties.keys()
        options = {"constrain_size": 0.0} if "constrain_size" in names else {}
        bpy.ops.import_scene.max3ds(filepath=src, **options)
        meshes = [[len(o.data.vertices), len(o.data.polygons)] for o in bpy.data.objects if o.type == "MESH"]
        report = {"meshes": sorted(meshes)}
    else:
        report = {"skip": "Blender's 3DS importer (io_scene_3ds) isn't installed"}
    with open(out, "w") as f:
        json.dump(report, f)
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_reads_our_3ds(scene, tmp_path):
    meshport.save(scene, tmp_path / "ours.3ds")
    script = tmp_path / "script3ds.py"
    script.write_text(_BLENDER_3DS_SCRIPT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "ours.3ds"), str(tmp_path / "blender.json")],
                   check=True, capture_output=True, timeout=300)
    report = json.loads((tmp_path / "blender.json").read_text())
    if "skip" in report:
        pytest.skip(report["skip"])
    ours = meshport.load(tmp_path / "ours.3ds")
    want = sorted([len(n.mesh.positions), len(n.mesh.face_sizes)] for n, _ in ours.walk() if n.mesh is not None)
    assert report["meshes"] == want
