"""Shading normals from .blend files, against Blender's own corner normals."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
from blendfile import STRUCTS, BlendWriter, mesh_object, scene_file

import meshport
from meshport import blend

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("name", ["Sphere", "Cylinder", "Cube", "Mixed", "Fin", "Flipped"])
def test_normals_match_blenders(name):
    """normals.blend (Blender 5.1) holds every way Blender shades a mesh: all smooth (weighted by corner
    angle), sharp edges on smooth faces, flat beside smooth, an edge three faces share, and neighbours
    wound opposite ways. Each corner's normal matches Blender's own (they used to be up to 150 degrees
    off where an edge was sharp or shared by three faces, and 6.6 degrees off on the smooth sphere)."""
    with gzip.open(DATA / "normals_corners.json.gz", "rt") as f:
        want = np.array(json.load(f)[name]).reshape(-1, 3)
    scene = blend.read_file(DATA / "normals.blend", False)
    node = next(n for n, _ in scene.walk() if n.name == name)
    assert node.mesh.normals.shape == want.shape
    assert np.abs(node.mesh.normals - want).max() < 1e-5  # Blender's values are rounded to 6 places


_OLD = {**STRUCTS,
        "Mesh": STRUCTS["Mesh"] + [("short", "flag"), ("float", "smoothresh"), ("int", "totedge"), ("MEdge", "*medge")],
        "MEdge": [("int", "v1"), ("int", "v2"), ("char", "crease"), ("char", "bweight"), ("short", "flag")]}
_CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 6), (6, 7), (7, 5), (5, 4), (0, 4), (1, 5), (3, 7), (2, 6)]


def _smooth_cube(flag=0, smoothresh=None, sharp=()):
    """A cube with every face smooth, in the Blender 2.8 to 3.4 layout, with Auto Smooth (flag 32)
    and its angle, and some edges marked sharp."""
    smoothresh = math.radians(30) if smoothresh is None else smoothresh
    w = BlendWriter(_OLD)
    p = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    verts = w.block(b"DATA", "MVert", [{"co": v} for v in p])
    polys = w.block(b"DATA", "MPoly", [{"loopstart": 4 * i, "totloop": 4, "flag": 1} for i in range(6)])
    loops = w.block(b"DATA", "MLoop", [{"v": v} for q in quads for v in q])
    edges = w.block(b"DATA", "MEdge", [{"v1": a, "v2": b, "flag": 512 if k in sharp else 0}
                                       for k, (a, b) in enumerate(_CUBE_EDGES)])
    me = w.block(b"ME\0\0", "Mesh", {"id": {"name": "MEcube"}, "totvert": 8, "totpoly": 6, "totloop": 24,
                                      "mvert": verts, "mpoly": polys, "mloop": loops, "flag": flag,
                                      "smoothresh": smoothresh, "totedge": 12, "medge": edges})
    return meshport.read(scene_file(w, [mesh_object(w, "cube", me)]), "blend").roots[0].mesh


def test_old_files_split_shading_only_with_auto_smooth():
    """Before 4.1, sharp edges and the auto smooth angle only counted with Auto Smooth on."""
    plain = _smooth_cube(sharp=range(12))  # Auto Smooth off: all smooth, sharp edges ignored
    corner = np.abs(plain.normals)
    assert np.allclose(corner, 1 / math.sqrt(3), atol=1e-6)
    split = _smooth_cube(flag=32)  # on, 30 degrees: every cube edge is sharper than that
    assert np.allclose(np.sort(np.abs(split.normals), 1), [0, 0, 1], atol=1e-6)
    loose = _smooth_cube(flag=32, smoothresh=math.radians(179))  # on, but no edge sharper than the angle
    assert np.allclose(np.abs(loose.normals), 1 / math.sqrt(3), atol=1e-6)
    one = _smooth_cube(flag=32, smoothresh=math.radians(179), sharp=[0, 1, 2, 3])  # one face ringed by sharp edges
    flat = np.isclose(np.sort(np.abs(one.normals), 1), [0, 0, 1], atol=1e-6).all(1)
    assert flat.sum() == 4  # that face's four corners are flat; the rest stay smooth
