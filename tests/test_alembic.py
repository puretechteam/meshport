"""Alembic: round trips, the Ogawa container, damaged files, and Blender in both directions."""

import struct
import subprocess
import textwrap

import numpy as np
import pytest
from conftest import BLENDER, cube, nodes_by_name, world_triangles

import meshport
from meshport import Material, Mesh, Node, Scene, alembic
from meshport.geometry import compose


def test_round_trip_keeps_geometry_and_hierarchy(scene):
    back = meshport.read(meshport.write(scene, "abc"))
    assert world_triangles(back) == world_triangles(scene)
    assert {"root", "box", "box_mirror", "ell"} <= set(nodes_by_name(back))


def test_round_trip_keeps_attributes(scene, tmp_path):
    meshport.save(scene, tmp_path / "m.abc")
    back = meshport.load(tmp_path / "m.abc")
    node = nodes_by_name(back)["boxShape"][0]
    src = cube()
    np.testing.assert_array_equal(node.mesh.corners, src.corners)  # winding survives both reversals
    np.testing.assert_allclose(node.mesh.normals, src.normals, atol=1e-6)
    assert len(node.mesh.uvs) == 2
    for a, b in zip(node.mesh.uvs, src.uvs):
        np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(node.mesh.colors, src.colors, atol=1e-6)
    # Alembic has no materials: face sets named after them keep which face had which
    assert [node.materials[i].name for i in node.mesh.face_materials] == ["red"] * 3 + ["blue"] * 3


def test_file_layout():
    data = meshport.write(Scene([Node("a", mesh=cube())]), "abc")
    assert data[:8] == b"Ogawa\xff\x00\x01"
    root = struct.unpack_from("<Q", data, 8)[0]
    assert struct.unpack_from("<Q", data, root)[0] == 6  # version, library, objects, metadata, times, metadata table


def test_transform_ops_of_every_kind():
    w = alembic._Writer()
    m = Mesh("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([3]), np.arange(3))
    shape = w._mesh("triShape", m, [])
    vals = [2, 3, 4,  1, 2, 3,  0, 0, 1, 90,  45,  1, 1, 1]  # scale, translate, rotate about Z, rotateX, scale
    ops = [0x00, 0x10, 0x20, 0x40, 0x00]
    xform = w.compound(".xform", [w.scalar(".ops", ops, "u1"), w.scalar(".vals", vals, "<f8")], alembic._XFORM_PROP)
    data = w.finish([("t", alembic._XFORM, w.object([xform], [shape]))])
    _, world = nodes_by_name(meshport.read(data, "abc"))["triShape"]
    from meshport.geometry import euler, scaling, translation
    expect = scaling((2, 3, 4)) @ translation((1, 2, 3)) @ euler((0, 0, 90)) @ euler((45, 0, 0))
    np.testing.assert_allclose(world, expect, atol=1e-9)


def test_shared_and_nested_nodes():
    tri = Mesh("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([3]), np.arange(3))
    inner = Node("inner", compose((0, 1, 0)), tri)
    scene = Scene([Node("a", compose((1, 0, 0), (0, 30, 0)), tri, [Material("m")], [inner]),
                   Node("b", compose((5, 0, 0)), tri)])
    back = meshport.read(meshport.write(scene, "abc"))
    assert world_triangles(back) == world_triangles(scene)


@pytest.mark.parametrize("data", [
    b"Ogawa\xff\x00\x01" + struct.pack("<Q", 10 ** 9),
    b"Ogawa\xff\x00\x01" + struct.pack("<Q", 16) + struct.pack("<Q", 10 ** 15),
    b"Ogawa\xff\x00\x01" + struct.pack("<Q", 16) + struct.pack("<Q", 1) + struct.pack("<Q", 16),  # a loop
    b"\x89HDF\r\n\x1a\n" + bytes(64),
    b"Ogawa",
])
def test_damaged_files_raise_meshport_errors(data):
    with pytest.raises(meshport.MeshportError):
        meshport.read(data, "abc")


def test_damaged_mesh_raises(scene):
    data = bytearray(meshport.write(Scene([Node("a", mesh=cube())]), "abc"))
    key = np.array([4, 4, 4, 4, 4, 4], np.int32).tobytes()  # .faceCounts
    at = bytes(data).index(key)
    data[at:at + 4] = struct.pack("<i", 9)
    with pytest.raises(meshport.MeshportError):
        meshport.read(bytes(data), "abc")


_BLENDER = textwrap.dedent("""
    import bpy, sys
    mode, src, dst = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if mode == "import":
        bpy.ops.wm.alembic_import(filepath=src)
        bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
    else:
        bpy.ops.mesh.primitive_monkey_add(rotation=(0.3, 0.2, 0.1), location=(1, 2, 3))
        bpy.ops.mesh.primitive_cylinder_add(vertices=7, location=(-2, 0, 1), scale=(1, 2, 0.5))
        bpy.ops.wm.alembic_export(filepath=src)
        bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
""")


def _blender(tmp_path, *args):
    script = tmp_path / "script.py"
    script.write_text(_BLENDER)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1",
                    "--python", str(script), "--", *map(str, args)], check=True, capture_output=True, timeout=300)


def _facing(scene: Scene) -> float:
    """How many triangles face away from their mesh's middle, minus how many face into it."""
    total = 0.0
    for node, w in scene.walk():
        if node.mesh is None:
            continue
        m = node.mesh
        p = m.positions @ w[:3, :3].T + w[:3, 3]
        t = m.corners[m.triangles()]
        if np.linalg.det(w[:3, :3]) < 0:
            t = t[:, ::-1]
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        total += np.sign(np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3 - p.mean(0))).sum()
    return total


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_reads_our_alembic(scene, tmp_path):
    meshport.save(scene, tmp_path / "ours.abc")
    _blender(tmp_path, "import", tmp_path / "ours.abc", tmp_path / "theirs.glb")
    theirs = meshport.load(tmp_path / "theirs.glb")
    assert world_triangles(theirs) == world_triangles(scene)
    assert _facing(theirs) > 0 and _facing(scene) > 0  # (the flat L shape has no outside to count)


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_we_read_blenders_alembic(tmp_path):
    _blender(tmp_path, "export", tmp_path / "theirs.abc", tmp_path / "theirs.glb")
    ours = meshport.load(tmp_path / "theirs.abc")
    ref = meshport.load(tmp_path / "theirs.glb")
    # the same points (n-gon caps may be cut into triangles differently), facing the same way
    a, b = _points(ours), _points(ref)
    from scipy.spatial import cKDTree
    assert max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()) < 1e-5
    assert _facing(ours) > 0 and _facing(ref) > 0


def _points(scene: Scene) -> np.ndarray:
    return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk() if n.mesh is not None])
