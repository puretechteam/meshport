"""USD against the reference implementations. Each test skips when that software isn't around.

- Pixar's own library (pip install usd-core), in the Python named by MESHPORT_PXR_PYTHON: it reads
  meshport's files to the same geometry and its validators find no problems; and meshport reads
  a corpus of tricky files it writes to the same geometry as it composes them.
- Blender: it reads meshport's USD to the same geometry, and meshport reads Blender's.
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

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
PXR = os.environ.get("MESHPORT_PXR_PYTHON")
needs_pxr = pytest.mark.skipif(not PXR, reason="set MESHPORT_PXR_PYTHON to a Python with usd-core")
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")


def world_points(scene: meshport.Scene) -> np.ndarray:
    return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk() if n.mesh is not None])


def close(a: np.ndarray, b: np.ndarray, tol: float = 1e-5) -> bool:
    from scipy.spatial import cKDTree
    return max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max()) <= tol


@needs_pxr
@pytest.mark.parametrize("fmt", ["usdz", "usda", "usdc"])
def test_pixar_reads_our_files_and_finds_no_problems(scene, tmp_path, fmt):
    path = tmp_path / f"model.{fmt}"
    meshport.save(scene, path)
    out = subprocess.run([PXR, str(BENCH / "usd_points.py"), str(path), "--check"], capture_output=True, text=True,
                         check=True, timeout=300)
    got = json.loads(out.stdout)
    assert got["problems"] == []
    theirs = np.concatenate([np.array(v).reshape(-1, 3) for v in got["meshes"].values()])
    assert len(got["meshes"]) == 3 and close(theirs, world_points(scene))


@needs_pxr
def test_we_read_pixars_corpus_like_pixar(tmp_path):
    subprocess.run([PXR, str(BENCH / "make_usd_corpus.py"), str(tmp_path)], check=True, capture_output=True,
                   timeout=600)
    sys.path.insert(0, str(BENCH))
    try:
        import compare_usd
    finally:
        sys.path.pop(0)
    assert compare_usd.main([str(tmp_path), "--pxr-python", PXR]) == 0


_BLENDER = textwrap.dedent("""
    import bpy, sys
    mode, src, dst = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if mode == "import":
        bpy.ops.wm.usd_import(filepath=src)
        bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
    else:
        bpy.ops.mesh.primitive_monkey_add(rotation=(0.3, 0.2, 0.1), location=(1, 2, 3))
        bpy.ops.mesh.primitive_cylinder_add(vertices=7, location=(-2, 0, 1), scale=(1, 2, 0.5))
        bpy.ops.wm.usd_export(filepath=src)
        bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
""")


def _blender(tmp_path, *args):
    script = tmp_path / "script.py"
    script.write_text(_BLENDER)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1",
                    "--python", str(script), "--", *map(str, args)], check=True, capture_output=True, timeout=300)


@needs_blender
@pytest.mark.parametrize("fmt", ["usdz", "usda", "usdc"])
def test_blender_reads_our_usd(scene, tmp_path, fmt):
    meshport.save(scene, tmp_path / f"ours.{fmt}")
    _blender(tmp_path, "import", tmp_path / f"ours.{fmt}", tmp_path / "theirs.glb")
    assert close(world_points(meshport.load(tmp_path / "theirs.glb")), world_points(scene))


@needs_blender
@pytest.mark.parametrize("fmt", ["usdz", "usdc"])
def test_we_read_blenders_usd(tmp_path, fmt):
    _blender(tmp_path, "export", tmp_path / f"theirs.{fmt}", tmp_path / "theirs.glb")
    ours = world_points(meshport.load(tmp_path / f"theirs.{fmt}"))
    assert close(ours, world_points(meshport.load(tmp_path / "theirs.glb")))
