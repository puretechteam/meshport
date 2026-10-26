import struct

import numpy as np
import pytest
from conftest import nodes_by_name, png, world_triangles

import meshport
from meshport import Image, Material, Mesh, MeshportError, Node, Scene, max3ds
from meshport.__main__ import main
from meshport.geometry import compose
from meshport.max3ds import _chunk, _chunks


def _same_geometry(a: Scene, b: Scene):
    np.testing.assert_allclose(np.array(world_triangles(b)), np.array(world_triangles(a)), atol=2e-5)


def _objects(data: bytes) -> dict[str, tuple[int, int]]:
    """Object name -> (vertex count, face count), straight from the chunks."""
    out = {}
    for _, s, e in _chunks(data, 6, len(data)):
        for cid, os_, oe in _chunks(data, s, e):
            if cid != max3ds.OBJECT:
                continue
            name, pos = max3ds._cstring(data, os_, oe)
            for _, ts, te in _chunks(data, pos, oe):
                counts = {cid: struct.unpack_from("<H", data, cs)[0] for cid, cs, _ in _chunks(data, ts, te)}
                out[name] = (counts[max3ds.POINT_ARRAY], counts[max3ds.FACE_ARRAY])
    return out


def test_round_trip_keeps_geometry_uvs_materials_and_names(scene):
    back = max3ds.read(max3ds.write(scene))
    _same_geometry(scene, back)
    want, got = nodes_by_name(scene), nodes_by_name(back)
    assert set(got) == {"box", "box_mirror", "ell"}  # 3DS has no hierarchy: just the meshes
    for name in got:
        node, world = want[name]
        other, other_world = got[name]
        np.testing.assert_allclose(other_world, world, atol=1e-6)
        a, b = node.mesh, other.mesh
        tris = a.triangles().ravel()
        np.testing.assert_allclose(b.positions[b.corners], a.positions[a.corners[tris]], atol=1e-6)
        if a.uvs:
            np.testing.assert_allclose(b.uvs[0], a.uvs[0][tris], atol=1e-6)
        if a.normals is not None:
            np.testing.assert_allclose(b.normals, a.normals[tris], atol=1e-6)
        else:
            assert b.normals is None
        slots = a.face_materials[a.triangle_faces()] if a.face_materials is not None else 0
        assert [m.name for m in other.materials] == [m.name for m in node.materials]
        np.testing.assert_array_equal(b.face_materials, np.broadcast_to(slots, len(b.face_sizes)))
    red, blue = got["box"][0].materials
    assert red is got["box_mirror"][0].materials[0]  # one material, used twice
    assert red.base_color_texture.path == "checker.png"
    assert red.roughness == pytest.approx(0.4, abs=0.01)
    assert blue.base_color == pytest.approx((0.1, 0.2, 0.9, 0.5), abs=1e-6)
    assert blue.alpha_mode == "BLEND"


def test_save_writes_textures_next_to_the_model(scene, tmp_path):
    files = meshport.save(scene, tmp_path / "out.3ds")
    assert {p.name for p in files} == {"out.3ds", "checker.png"}
    back = meshport.load(tmp_path / "out.3ds")
    red = nodes_by_name(back)["box"][0].materials[0]
    assert red.base_color_texture.data == png()
    _same_geometry(scene, back)


def test_smooth_normals_survive(tmp_path):
    # a UV sphere's normals point straight out from its center
    lat, lon = np.meshgrid(np.linspace(0.1, np.pi - 0.1, 12), np.linspace(0, 2 * np.pi, 24, endpoint=False))
    p = np.stack([np.sin(lat) * np.cos(lon), np.cos(lat), np.sin(lat) * np.sin(lon)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(24), np.arange(11), indexing="ij")
    quads = np.stack([i * 12 + j, (i + 1) % 24 * 12 + j, (i + 1) % 24 * 12 + j + 1, i * 12 + j + 1], -1).reshape(-1)
    mesh = Mesh("ball", p, np.full(len(quads) // 4, 4), quads, normals=p[quads])
    back = max3ds.read(max3ds.write(Scene([Node("ball", mesh=mesh)])))
    got = back.roots[0].mesh
    want = got.positions[got.corners]  # on the unit sphere already
    assert np.einsum("ij,ij->i", got.normals, want).min() > 0.99


def _model(points, faces, smooth=None, matrix=None, extra=b"") -> bytes:
    """A hand-made 3DS file holding one object."""
    points = np.asarray(points, "<f4")
    face_subs = [] if smooth is None else [_chunk(max3ds.SMOOTH_GROUP, np.asarray(smooth, "<u4").tobytes())]
    faces = np.hstack([np.asarray(faces), np.zeros((len(faces), 1))]).astype("<u2")
    parts = [_chunk(max3ds.POINT_ARRAY, struct.pack("<H", len(points)), points.tobytes()),
             _chunk(max3ds.FACE_ARRAY, struct.pack("<H", len(faces)), faces.tobytes(), *face_subs)]
    if matrix is not None:
        parts.append(_chunk(max3ds.MESH_MATRIX, np.asarray(matrix, "<f4").tobytes()))
    obj = _chunk(max3ds.OBJECT, b"thing\0", _chunk(max3ds.TRIMESH, *parts))
    return _chunk(max3ds.MAIN, _chunk(max3ds.VERSION, struct.pack("<I", 3)), _chunk(max3ds.EDIT, extra, obj))


_HINGE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  # two triangles folded along the 0-1 edge
_HINGE_FACES = [[0, 1, 2], [1, 0, 3]]


@pytest.mark.parametrize("groups, smooth", [([1, 1], True), ([1, 3], True), ([1, 2], False), ([0, 0], False)])
def test_smoothing_groups(groups, smooth):
    mesh = max3ds.read(_model(_HINGE, _HINGE_FACES, groups)).roots[0].mesh
    normals = mesh.normals.reshape(2, 3, 3)
    # corner 0 of face 0 and corner 1 of face 1 are both vertex 0
    shared = np.allclose(normals[0, 0], normals[1, 1])
    assert shared == smooth
    if not smooth:
        np.testing.assert_allclose(np.abs(normals[0, 0]), [0, 0, 1], atol=1e-6)


def test_no_smoothing_chunk_means_no_normals():
    assert max3ds.read(_model(_HINGE, _HINGE_FACES)).roots[0].mesh.normals is None


def test_z_up_and_the_mesh_matrix():
    # The points are stored in world space; the matrix says where the object's own axes are.
    matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1, 10, 20, 30]
    scene = max3ds.read(_model([[10, 20, 30], [11, 20, 30], [10, 21, 30]], [[0, 1, 2]], matrix=matrix))
    node = scene.roots[0]
    np.testing.assert_allclose(node.mesh.positions, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(node.matrix[:3, 3], [10, 30, -20])  # Z up -> Y up
    np.testing.assert_allclose(scene.bounds(), [[10, 30, -21], [11, 30, -20]])


def test_mirrored_matrix_keeps_faces_outward():
    matrix = [-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    node = max3ds.read(_model([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [1], matrix=matrix)).roots[0]
    world = node.mesh.normals @ np.linalg.inv(node.matrix[:3, :3])  # normals go by the inverse transpose
    np.testing.assert_allclose(world[0], [0, 1, 0], atol=1e-6)  # the file's face points up (+Z, so +Y here)


def test_big_meshes_are_split():
    rng = np.random.default_rng(1)
    n = 70_000  # separate triangles: 210,000 vertices, over the 65,535 limit
    loose = Mesh("loose", rng.random((3 * n, 3)), np.full(n, 3), np.arange(3 * n))
    side = 240  # a grid sharing its vertices: 57,600 vertices but 114,242 triangles
    g = np.arange(side * side).reshape(side, side)
    quads = np.stack([g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]], -1).reshape(-1)
    xy = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2)
    grid = Mesh("grid", np.hstack([xy, np.zeros((len(xy), 1))]).astype(float), np.full(len(quads) // 4, 4), quads)
    scene = Scene([Node("loose", mesh=loose), Node("grid", compose((0, 0, 5)), grid)])
    data = max3ds.write(scene)
    objects = _objects(data)
    assert all(v <= 65535 and f <= 65535 for v, f in objects.values())
    assert sum(f for name, (_, f) in objects.items() if name.startswith("loose")) == n
    assert sum(f for name, (_, f) in objects.items() if name.startswith("grid")) == 2 * (side - 1) ** 2
    assert sum(name.startswith("grid") for name in objects) == 2
    assert len(objects) == len(set(objects))
    back = max3ds.read(data)
    for name in ("loose", "grid"):
        src = nodes_by_name(scene)[name][0]
        parts = [(nd, w) for nd, w in back.walk() if nd.name.startswith(name)]
        got = np.concatenate([nd.mesh.positions[nd.mesh.corners] @ w[:3, :3].T + w[:3, 3] for nd, w in parts])
        want = src.mesh.positions[src.mesh.corners[src.mesh.triangles().ravel()]] + src.matrix[:3, 3]
        np.testing.assert_allclose(got, want, atol=1e-5)


def test_long_names_are_cut_and_kept_apart():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    mats = [Material("A very long material name", (1, 0, 0, 1)), Material("A very long material name, too", (0, 1, 0, 1))]
    mesh = Mesh("m", p, np.array([3, 3]), np.array([0, 1, 2, 1, 3, 2]), face_materials=np.array([0, 1]))
    nodes = [Node("Ünïcode object number one", mesh=mesh, materials=mats),
             Node("Ünïcode object number two", compose((0, 0, 2)), mesh, mats)]
    back = max3ds.read(max3ds.write(Scene(nodes)))
    names = [n.name for n in back.roots]
    assert len(set(names)) == 2 and all(len(n) <= 10 and n.isascii() for n in names)
    for node in back.roots:
        assert len({m.name for m in node.materials}) == 2 and all(len(m.name) <= 16 for m in node.materials)
        colors = [node.materials[s].base_color[:3] for s in node.mesh.face_materials]
        assert colors == [(1, 0, 0), (0, 1, 0)]


def test_unknown_chunks_are_skipped():
    junk = _chunk(0x1234, b"whatever") + _chunk(max3ds.OBJECT, b"light\0", _chunk(0x4600, bytes(12)))
    scene = max3ds.read(_model(_HINGE, _HINGE_FACES, extra=junk))
    assert [n.name for n in scene.roots] == ["thing"]


@pytest.mark.parametrize("data, message", [
    (b"", "isn't a 3DS"),
    (b"PK\x03\x04" + bytes(100), "isn't a 3DS"),
    (_chunk(max3ds.MAIN, _chunk(max3ds.VERSION, struct.pack("<I", 3))), "No 3D geometry"),
    (_model(_HINGE, [[0, 1, 9]]), "past its vertex list"),
])
def test_bad_files_raise_a_clear_error(data, message):
    with pytest.raises(MeshportError, match=message):
        max3ds.read(data)


def test_load_sniffs_3ds_and_cli_converts(scene, tmp_path, capsys):
    meshport.save(scene, tmp_path / "in.fbx")
    assert main([str(tmp_path / "in.fbx"), str(tmp_path / "out.3ds")]) == 0
    (tmp_path / "renamed.bin").write_bytes((tmp_path / "out.3ds").read_bytes())
    _same_geometry(scene, meshport.load(tmp_path / "renamed.bin"))
    assert main([str(tmp_path / "out.3ds")]) == 0
    assert "box: 24 vertices, 12 faces" in capsys.readouterr().out
    assert main([str(tmp_path / "out.3ds"), str(tmp_path / "back.glb")]) == 0
    _same_geometry(scene, meshport.load(tmp_path / "back.glb"))


def test_textures_without_data_are_still_named():
    mat = Material("m", base_color_texture=Image("x", None, "C:\\art\\Wood Planks Diffuse.tga"))
    p = np.eye(3)
    node = Node("n", mesh=Mesh("n", p, np.array([3]), np.arange(3)), materials=[mat])
    back = max3ds.read(max3ds.write(Scene([node])))
    assert back.roots[0].materials[0].base_color_texture.path == "Wood_Pla.tga"  # 8.3, what every reader takes
