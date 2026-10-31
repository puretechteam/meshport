"""Older .blend files: set MESHPORT_BLEND_CORPUS to a folder of .blend files, each next to Blender's own
GLB export of it (NAME_blender.glb, or NAME.glb), for example the test scenes written by Blender 2.83,
2.93, 3.3 and 3.5 with

    blender -b --factory-startup --python tests/data/make_blend_fixture.py -- OUT_DIR scene_283
    blender -b --factory-startup --python tests/data/make_modifier_fixture.py -- OUT_DIR modifiers_283
    blender -b --factory-startup --python tests/data/make_curve_fixture.py -- OUT_DIR curves_283

Each file is read with its modifiers on (those scripts export with them applied) and every object in
Blender's export must come out in the same place, point for point."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import meshport
from meshport import blend

CORPUS = os.environ.get("MESHPORT_BLEND_CORPUS")
FILES = sorted(Path(CORPUS).glob("*.blend")) if CORPUS else []
# objects of the fixture scripts that meshport only approximates (see test_blend.py): Blender's limit
# surface at smoothed corners (a little coarser in 2.83), Solidify's even thickness near a zero cosine,
# an open NURBS ribbon's frames, a stand-in for the built-in font, and metaball points (compared by surface)
TOLERANCE = {"SubsurfGridAll": 6e-3, "SubsurfGridCorners": 6e-3, "SolidifyEven": 0.05, "NurbsHelix": 0.04,
             "BuiltinText": 0.15}
SURFACE = {"Mball": 0.1, "Blob": 0.05}
SKIP = {"Unsupported"}  # Bevel, which meshport leaves off


def objects(scene: meshport.Scene) -> dict:
    out = {}
    for n, w in scene.walk():
        if n.mesh is not None:
            p = n.mesh.positions @ w[:3, :3].T + w[:3, 3]
            out[n.name] = (p[np.unique(n.mesh.corners)], p[n.mesh.corners[n.mesh.triangles()]])
    return out


@pytest.mark.skipif(not CORPUS, reason="set MESHPORT_BLEND_CORPUS to a folder of .blend files and their GLBs")
@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_old_blend_files_match_blender(path):
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from compare_blender import surface_gap

    ref = next((p for p in (path.with_name(path.stem + "_blender.glb"), path.with_suffix(".glb")) if p.exists()), None)
    if ref is None:
        pytest.skip(f"no GLB next to {path.name}")
    ours, theirs = objects(blend.read_file(path)), objects(meshport.load(ref))
    size = np.ptp(np.concatenate([p for p, _ in theirs.values()]), axis=0).max()
    off = []
    for name, (points, triangles) in theirs.items():
        if name in SKIP:
            continue
        if name not in ours:
            off.append(f"{name}: missing")
        elif name in SURFACE:
            g = surface_gap(ours[name][1], triangles, 4000)
            if g > SURFACE[name]:
                off.append(f"{name}: surface {g:.3g}")
        else:
            a = ours[name][0]
            g = max(cKDTree(a).query(points)[0].max(), cKDTree(points).query(a)[0].max())
            if g > TOLERANCE.get(name, max(1e-5, 1e-6 * size)):
                off.append(f"{name}: {g:.3g}")
    assert not off, ", ".join(off)
