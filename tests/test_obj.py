import numpy as np
import pytest
from conftest import world_triangles

import meshport
from meshport import MeshportError


def test_obj_writes_mesh_materials_and_textures(scene, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    files = meshport.save(scene, tmp_path / "out.obj")
    assert {p.name for p in files} == {"out.obj", "out.mtl", "checker.png"}
    text = (tmp_path / "out.obj").read_text()
    assert "mtllib out.mtl" in text and text.count("\no ") == 3 and "usemtl blue" in text
    assert "map_Kd checker.png" in (tmp_path / "out.mtl").read_text()
    loaded = trimesh.load(tmp_path / "out.obj", force="mesh", process=False)
    assert len(loaded.faces) == len(world_triangles(scene))
    np.testing.assert_allclose(loaded.bounds, scene.bounds(), atol=1e-5)


def test_obj_keeps_mirrored_faces_outward(scene, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    for node, _ in scene.walk():
        if node.name not in ("root", "box_mirror"):
            node.mesh = None
    meshport.save(scene, tmp_path / "m.obj")
    cube = trimesh.load(tmp_path / "m.obj", force="mesh", process=False)
    v = cube.vertices[cube.faces]
    signed = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6
    assert signed == pytest.approx(8.0, rel=1e-4)  # the mirrored 2x2x2 cube; inside-out would give -8


def test_obj_needs_save(scene):
    with pytest.raises(MeshportError, match="save"):
        meshport.write(scene, "obj")
