from pathlib import Path

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


# ============================================================================ reading

DATA = Path(__file__).resolve().parent / "data"


def _corner_table(scene) -> dict:
    """Per object: each triangle corner's world position, unit normal, UV and color, side by side
    (flat normals where a mesh has none, as a viewer shades it), and its materials."""
    out = {}
    for node, w in scene.walk():
        m = node.mesh
        if m is None:
            continue
        t = m.triangles().ravel()
        p = m.positions[m.corners[t]] @ w[:3, :3].T + w[:3, 3]
        if m.normals is not None:
            n = m.normals[t] @ np.linalg.inv(w[:3, :3])
        else:
            n = np.repeat(np.cross(p[1::3] - p[0::3], p[2::3] - p[0::3]), 3, axis=0)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        uv = m.uvs[0][t] if m.uvs else np.zeros((len(t), 2))
        c = m.colors[t] if m.colors is not None else np.ones((len(t), 4))
        mats = [(x.name, *np.round(x.base_color, 4), round(x.roughness, 4), round(x.metallic, 4), x.alpha_mode,
                 x.base_color_texture is not None and x.base_color_texture.data is not None) for x in node.materials]
        out[node.name] = (np.hstack([p, n, uv, c]), len(t) // 3, mats)
    return out


@pytest.mark.parametrize("stem", ["objects", "objects_groups"])
def test_obj_reads_as_blender_imports_it(stem):
    # Blender 5.1 wrote the OBJ (with normals, or with smoothing groups instead), then imported it
    # and exported a GLB: data/make_obj_fixture.py
    from scipy.spatial import cKDTree
    ours = _corner_table(meshport.load(DATA / f"{stem}.obj"))
    theirs = _corner_table(meshport.load(DATA / f"{stem}_blender.glb"))
    assert sorted(ours) == sorted(theirs) == ["Cube", "Cylinder", "Grid", "Monkey", "Sphere", "Torus"]
    for name in ours:
        (a, ta, ma), (b, tb, mb) = ours[name], theirs[name]
        assert ta == tb, name
        gap = max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())
        assert gap < 2e-4, (name, gap)  # points, normals, UVs and colors (Blender writes normals to 4 places)
        assert ma == mb, name


def test_obj_reads_what_trimesh_writes():
    trimesh = pytest.importorskip("trimesh")
    sphere = trimesh.creation.icosphere(2)
    sphere.visual = trimesh.visual.ColorVisuals(sphere, vertex_colors=np.tile([255, 128, 0, 255], (len(sphere.vertices), 1)))
    text = trimesh.exchange.obj.export_obj(sphere, include_normals=True, include_color=True)
    mesh = meshport.read(text.encode(), "obj").meshes[0]
    np.testing.assert_allclose(mesh.positions[mesh.corners], sphere.vertices[sphere.faces.ravel()], atol=1e-7)
    np.testing.assert_allclose(mesh.normals, sphere.vertex_normals[sphere.faces.ravel()], atol=1e-7)
    np.testing.assert_allclose(mesh.colors[0], [1.0, 0.2158605, 0.0, 1.0], atol=1e-6)  # 128 in sRGB


def test_obj_round_trip(scene, tmp_path):
    box = scene.roots[0].children[0]
    box.mesh.colors = np.tile([0.2, 0.5, 1.0, 1.0], (len(box.mesh.corners), 1))
    meshport.save(scene, tmp_path / "out.obj")
    back = meshport.load(tmp_path / "out.obj")
    assert len(world_triangles(back)) == len(world_triangles(scene))
    np.testing.assert_allclose(back.bounds(), scene.bounds(), atol=1e-5)
    got = back.roots[0]
    assert [m.name for m in got.materials] == [m.name for m in box.materials]
    assert got.materials[0].base_color_texture.data == box.materials[0].base_color_texture.data
    np.testing.assert_allclose(got.mesh.colors, box.mesh.colors, atol=2e-3)
    assert got.mesh.normals is not None and len(got.mesh.uvs) == 1


def test_obj_materials_round_trip(tmp_path):
    from conftest import cube
    mat = meshport.Material("shiny", (0.1, 0.2, 0.3, 0.5), metallic=0.7, roughness=0.3, emissive=(0.5, 0, 0))
    meshport.save(meshport.Scene([meshport.Node("c", mesh=cube(), materials=[mat])]), tmp_path / "m.obj")
    assert "Ns 490" in (tmp_path / "m.mtl").read_text()  # Blender's (1 - roughness)^2 * 1000
    back = meshport.load(tmp_path / "m.obj").roots[0].materials[0]
    assert back.base_color == pytest.approx((0.1, 0.2, 0.3, 0.5)) and back.alpha_mode == "BLEND"
    assert (back.metallic, back.roughness) == pytest.approx((0.7, 0.3))
    assert back.emissive == pytest.approx((0.5, 0, 0))


def _read(text: str, tmp_path=None, mtl: str | None = None) -> meshport.Scene:
    if tmp_path is None:
        return meshport.read(text.encode(), "obj")
    (tmp_path / "m.obj").write_text(text)
    if mtl is not None:
        (tmp_path / "m.mtl").write_text(mtl)
    return meshport.load(tmp_path / "m.obj")


QUAD = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"


def test_obj_is_detected_from_its_content():
    assert meshport.read(("# a comment\n\n" + QUAD + "f 1 2 3 4\n").encode()).meshes[0].face_sizes.tolist() == [4]
    assert meshport._sniff(b"solid cube\n") is None


def test_obj_every_way_of_writing_corners():
    text = QUAD + "vt 0 0\nvt 1 0\nvt 1 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1//1 3//1 4//1\nf 1/1 3/3 4\nf -4 -3 -1\n"
    mesh = _read(text).meshes[0]
    assert mesh.face_sizes.tolist() == [3, 3, 3, 3]
    np.testing.assert_array_equal(mesh.positions[mesh.corners][:6, :2], [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]])
    assert mesh.uvs[0][:3].tolist() == [[0, 0], [1, 0], [1, 1]] and mesh.uvs[0][3].tolist() == [0, 0]
    np.testing.assert_allclose(mesh.normals, np.tile([0, 0, 1.0], (12, 1)))  # given, or worked out where not


def test_obj_negative_numbers_count_back_from_the_last_vertex():
    text = QUAD + "f -4 -3 -2\n" + "v 5 5 5\nv 6 5 5\nv 5 6 5\nf -3 -2 -1\n"
    mesh = _read(text).meshes[0]
    assert len(mesh.positions) == 6
    assert mesh.positions[mesh.corners[3:]].tolist() == [[5, 5, 5], [6, 5, 5], [5, 6, 5]]


def test_obj_objects_and_groups():
    by_object = _read("o A\n" + QUAD + "g x\nf 1 2 3\ng y\nf 1 3 4\no B\nf 1 2 4\n")
    assert [n.name for n in by_object.roots] == ["A", "B"]
    assert len(by_object.roots[0].mesh.face_sizes) == 2
    by_group = _read(QUAD + "f 1 2 3\ng x\nf 1 3 4\ng y\nf 1 2 4\ng x\nf 2 3 4\n")
    assert [n.name for n in by_group.roots] == ["mesh", "x", "y", "x"]


def test_obj_lines_joined_by_a_backslash_and_short_faces():
    mesh = _read(QUAD + "f 1 2 \\\n 3 4\nf 1 2\nl 1 2 3\n").meshes[0]
    assert mesh.face_sizes.tolist() == [4]


def test_obj_vertex_colors_are_srgb():
    mesh = _read("v 0 0 0 1 0.5 0\nv 1 0 0 0 0 0\nv 1 1 0\nf 1 2 3\n").meshes[0]
    np.testing.assert_allclose(mesh.colors, [[1, 0.21404, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1]], atol=1e-5)


def test_obj_materials_and_texture_options(tmp_path):
    from conftest import png
    (tmp_path / "my tex.png").write_bytes(png())
    mtl = "newmtl a\nKd 0.5\nmap_Kd -s 2 2 1 -bm 0.5 my tex.png\nnewmtl b\nKa 0.4 0.4 0.4\nillum 3\nNs 250\nTr 0.25\n"
    scene = _read("mtllib m.mtl\n" + QUAD + "usemtl a\nf 1 2 3\nusemtl b\nf 1 3 4\nusemtl missing\nf 1 2 4\n",
                  tmp_path, mtl)
    a, b, missing = scene.roots[0].materials
    assert a.base_color == (0.5, 0.5, 0.5, 1.0) and a.base_color_texture.data == png()
    assert (b.metallic, b.roughness, b.base_color[3]) == pytest.approx((0.4, 0.5, 0.75)) and b.alpha_mode == "BLEND"
    assert missing.name == "missing" and missing.base_color == meshport.Material("x").base_color
    assert scene.roots[0].mesh.face_materials.tolist() == [0, 1, 2]


def test_obj_texture_without_a_color_is_shown_as_it_is(tmp_path):
    from conftest import png
    (tmp_path / "t.png").write_bytes(png())
    mat = _read("mtllib m.mtl\n" + QUAD + "usemtl a\nf 1 2 3\n", tmp_path, "newmtl a\nmap_Kd t.png\n").roots[0].materials[0]
    assert mat.base_color == (1.0, 1.0, 1.0, 1.0)


def test_obj_smoothing_groups_shade_like_blender():
    # two quads folded along x = 1, meeting at a right angle at each shared vertex
    text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 1\nv 2 1 1\n"

    def normals(s1, s2):
        return _read(text + f"s {s1}\nf 1 2 3 4\ns {s2}\nf 2 5 6 3\n").meshes[0].normals

    assert normals("off", "off") is None  # flat: nothing to work out
    np.testing.assert_allclose(normals(1, 1)[1], [-0.3826834, 0, 0.9238795], atol=1e-6)  # halfway
    np.testing.assert_allclose(normals(1, 2)[1], [0, 0, 1], atol=1e-9)  # another group: kept apart
    np.testing.assert_allclose(normals(1, "off")[1], [0, 0, 1], atol=1e-9)


def test_obj_damaged_files():
    with pytest.raises(MeshportError, match="no faces"):
        _read(QUAD + "l 1 2\n")
    for bad in ("f 1 2 9\n", "f 1 2 -9\n", "f 1 2 0\n", "f 1/9 2/9 3/9\n", "f a b c\n", "v 1 x 2\nf 1 2 3\n"):
        with pytest.raises(MeshportError, match="damaged"):
            _read(QUAD + bad)


@pytest.mark.parametrize("how", ["absolute", "two levels up", "not an mtl"])
def test_obj_mtllib_outside_the_folder_rule_is_not_read(tmp_path, how):
    folder = tmp_path / "pack" / "models"
    folder.mkdir(parents=True)
    far = tmp_path / ("pack/far.txt" if how == "not an mtl" else "far.mtl")
    far.write_text("newmtl a\nKd 1 0 0\n")
    name = {"absolute": far.as_posix(), "two levels up": "../../far.mtl", "not an mtl": "../far.txt"}[how]
    (folder / "m.obj").write_text(f"mtllib {name}\n" + QUAD + "usemtl a\nf 1 2 3\n")
    assert meshport.load(folder / "m.obj").roots[0].materials[0].base_color[:3] == (0.8, 0.8, 0.8)
