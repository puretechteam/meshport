import numpy as np
import pytest
from conftest import nodes_by_name, world_triangles

import meshport
from meshport import MeshportError, fbx
from meshport.fbxtree import Record, parse, write_ascii, write_binary


def _records(scene):
    return fbx._Writer(scene).records()


def _patch(records, cls, name, extra):
    """Add P records to the Properties70 of the object called name."""
    objects = next(r for r in records if r.name == "Objects")
    for rec in objects.children:
        if rec.name == cls and str(rec.props[1]).startswith(name + "\x00"):
            p70 = rec.find("Properties70")
            p70.children = [p for p in p70.children if p.props[0] not in {e.props[0] for e in extra}] + extra
            return
    raise KeyError(name)


def _settings(records, **values):
    gs = next(r for r in records if r.name == "GlobalSettings").find("Properties70")
    for p in gs.children:
        if p.props[0] in values:
            p.props[4] = values[p.props[0]]


def test_round_trip_keeps_everything(scene):
    back = fbx.read(fbx.write(scene))
    want, got = nodes_by_name(scene), nodes_by_name(back)
    assert set(want) == set(got)
    for name, (node, world) in want.items():
        other, other_world = got[name]
        np.testing.assert_allclose(other_world, world, atol=1e-9)
        if node.mesh is None:
            continue
        a, b = node.mesh, other.mesh
        np.testing.assert_allclose(b.positions, a.positions)
        np.testing.assert_array_equal(b.face_sizes, a.face_sizes)
        np.testing.assert_array_equal(b.corners, a.corners)
        for x, y in [(a.normals, b.normals), (a.colors, b.colors), *zip(a.uvs, b.uvs)]:
            assert (x is None) == (y is None)
            if x is not None:
                np.testing.assert_allclose(y, x)
        assert len(b.uvs) == len(a.uvs)
        if len(node.materials) > 1:
            np.testing.assert_array_equal(b.face_materials, a.face_materials)
        assert [m.name for m in other.materials] == [m.name for m in node.materials]
    red = got["box"][0].materials[0]
    assert red.base_color_texture.data == want["box"][0].materials[0].base_color_texture.data
    blue = got["box"][0].materials[1]
    assert blue.base_color == pytest.approx((0.1, 0.2, 0.9, 0.5))
    assert blue.alpha_mode == "BLEND"
    assert red.roughness == pytest.approx(0.4, abs=1e-6)
    assert got["box"][0].mesh is got["box_mirror"][0].mesh  # instancing survives


def test_ascii_and_wide_binary_read_the_same(scene):
    records = _records(scene)
    expected = world_triangles(fbx.read(write_binary(records)))
    assert world_triangles(fbx.read(write_ascii(records).encode())) == expected
    assert world_triangles(fbx.read(write_binary(records, version=7500))) == expected


def test_record_types_round_trip():
    arrays = [np.arange(5, dtype=np.int32), np.arange(300, dtype=np.int64), np.linspace(0, 1, 7),
              np.linspace(0, 1, 300).astype(np.float32), np.array([True, False])]
    rec = Record("Thing", [True, np.int16(-3), 7, np.int64(1 << 40), np.float32(1.5), 2.25, "name\x00\x01Model",
                           b"\x00\xffraw", *arrays], [Record("Child", [1]), Record("Empty")])
    version, (back,) = parse(write_binary([rec]))
    assert version == 7400
    assert back.props[:8] == [True, -3, 7, 1 << 40, 1.5, 2.25, "name\x00\x01Model", b"\x00\xffraw"]
    for a, b in zip(arrays, back.props[8:]):
        np.testing.assert_array_equal(a, b)
    assert [c.name for c in back.children] == ["Child", "Empty"]


def test_z_up_inches_become_y_up_meters(scene):
    nodes_by_name(scene)["root"][0].matrix = np.eye(4)
    records = _records(scene)
    # 3ds Max style: Z up, -Y front, 1 unit = 2.54 cm
    _settings(records, UpAxis=2, FrontAxis=1, FrontAxisSign=-1, UnitScaleFactor=2.54)
    back = fbx.read(write_binary(records))
    # what was +Z in the file (up) is +Y now, file -Y (front) is +Z, scaled by 0.0254
    conv = nodes_by_name(back)["root"][1]
    np.testing.assert_allclose(conv[:3, :3] @ [0, 0, 1], [0, 0.0254, 0], atol=1e-12)
    np.testing.assert_allclose(conv[:3, :3] @ [0, -1, 0], [0, 0, 0.0254], atol=1e-12)


def _one_triangle_scene():
    from meshport import Mesh, Node, Scene
    mesh = Mesh("tri", np.array([[2.0, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([3]), np.array([0, 1, 2]))
    child = Node("child", mesh=Mesh("tri2", mesh.positions.copy(), np.array([3]), np.array([0, 1, 2])))
    return Scene([Node("m", mesh=mesh, children=[child])])


def _world_point(records, node="m", index=0):
    back = fbx.read(write_binary(records))
    n, world = nodes_by_name(back)[node]
    return world[:3, :3] @ n.mesh.positions[index] + world[:3, 3]


def _P(name, *v):
    return Record("P", [name, "Vector3D", "Vector", "", *map(float, v)])


def test_rotation_pivot():
    records = _records(_one_triangle_scene())
    _patch(records, "Model", "m", [_P("Lcl Rotation", 0, 0, 90), _P("RotationPivot", 1, 0, 0)])
    np.testing.assert_allclose(_world_point(records), [1, 1, 0], atol=1e-9)  # (2,0,0) turned about (1,0,0)


def test_pre_rotation_and_rotation_order():
    records = _records(_one_triangle_scene())
    _patch(records, "Model", "m", [_P("PreRotation", 90, 0, 0)])
    np.testing.assert_allclose(_world_point(records, index=1), [0, 0, 1], atol=1e-9)
    for order, expected in [(0, [0, 2, 0]), (5, [0, 0, 2])]:  # XYZ vs ZYX
        records = _records(_one_triangle_scene())
        _patch(records, "Model", "m", [_P("Lcl Rotation", 90, 0, 90),
                                       Record("P", ["RotationOrder", "enum", "", "", order])])
        np.testing.assert_allclose(_world_point(records), expected, atol=1e-9)


def test_geometric_offset_moves_only_the_mesh():
    records = _records(_one_triangle_scene())
    _patch(records, "Model", "m", [_P("GeometricTranslation", 0, 5, 0)])
    np.testing.assert_allclose(_world_point(records), [2, 5, 0], atol=1e-9)
    np.testing.assert_allclose(_world_point(records, "child"), [2, 0, 0], atol=1e-9)


def test_mirrored_geometric_scale_keeps_faces_outward():
    records = _records(_one_triangle_scene())
    _patch(records, "Model", "m", [_P("GeometricScaling", -1, 1, 1)])
    back = fbx.read(write_binary(records))
    mesh = nodes_by_name(back)["m"][0].mesh
    np.testing.assert_array_equal(mesh.corners, [2, 1, 0])


ASCII_7 = """; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Geometry: 100, "Geometry::Quad", "Mesh" {
		Vertices: *12 {
			a: 0,0,0,1,0,0,1,1,0,0,1,0
		}
		PolygonVertexIndex: *4 {
			a: 0,1,2,-4
		}
		LayerElementNormal: 0 {
			Version: 101
			MappingInformationType: "ByVertice"
			ReferenceInformationType: "Direct"
			Normals: *12 {
				a: 0,0,1,0,0,1,0,0,1,0,0,1
			}
		}
		LayerElementUV: 0 {
			Version: 101
			Name: "map1"
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *4 {
				a: 0,0,1,1
			}
			UVIndex: *4 {
				a: 0,1,1,-1
			}
		}
		LayerElementMaterial: 0 {
			MappingInformationType: "AllSame"
			ReferenceInformationType: "IndexToDirect"
			Materials: *1 {
				a: 0
			}
		}
	}
	Model: 200, "Model::Quad", "Mesh" {
		Version: 232
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",0,2,0
		}
		Shading: T
		Culling: "CullingOff"
	}
	Material: 300, "Material::Paint", "" {
		ShadingModel: "phong"
		Properties70:  {
			P: "DiffuseColor", "Color", "", "A",0.5,0.25,1
		}
	}
}
Connections:  {
	C: "OO",200,0
	C: "OO",100,200
	C: "OO",300,200
}
"""


def test_hand_written_ascii_7():
    scene = fbx.read(ASCII_7.encode())
    node, world = nodes_by_name(scene)["Quad"]
    np.testing.assert_allclose(world[:3, 3], [0, 2, 0])
    mesh = node.mesh
    assert list(mesh.face_sizes) == [4]
    np.testing.assert_allclose(mesh.normals, [[0, 0, 1]] * 4)
    np.testing.assert_allclose(mesh.uvs[0], [[0, 0], [1, 1], [1, 1], [0, 0]])  # -1 index -> 0
    assert node.materials[0].base_color[:3] == pytest.approx((0.5, 0.25, 1))


ASCII_6 = """; FBX 6.1.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 6100
}
Objects:  {
	Model: "Model::Tri", "Mesh" {
		Version: 232
		Properties60:  {
			Property: "Lcl Translation", "Lcl Translation", "A",1,0,0
			Property: "Lcl Scaling", "Lcl Scaling", "A",2,2,2
		}
		Vertices: 0,0,0,1,0,0,
		0,1,0
		PolygonVertexIndex: 0,1,-3
		LayerElementMaterial: 0 {
			MappingInformationType: "AllSame"
			Materials: 0
		}
	}
	Material: "Material::Old", "" {
		Properties60:  {
			Property: "DiffuseColor", "ColorRGB", "",0,1,0
		}
	}
	GlobalSettings:  {
		Version: 1000
		Properties60:  {
			Property: "UnitScaleFactor", "double", "",100
		}
	}
}
Connections:  {
	Connect: "OO", "Model::Tri", "Model::Scene"
	Connect: "OO", "Material::Old", "Model::Tri"
}
"""


def test_hand_written_ascii_6():
    scene = fbx.read(ASCII_6.encode())
    node, world = nodes_by_name(scene)["Tri"]
    pts = node.mesh.positions @ world[:3, :3].T + world[:3, 3]
    np.testing.assert_allclose(pts, [[1, 0, 0], [3, 0, 0], [1, 2, 0]])
    assert node.materials[0].base_color[:3] == pytest.approx((0, 1, 0))


def test_external_texture_found_in_a_textures_folder(tmp_path, scene):
    from conftest import png
    data = png((0, 255, 0))
    mat = nodes_by_name(scene)["box"][0].materials[0]
    mat.base_color_texture = meshport.Image("wood", None, "D:/artist/work/wood.png")  # path from someone's PC
    pack = tmp_path / "Pack"
    (pack / "Models").mkdir(parents=True)
    (pack / "Textures").mkdir()
    (pack / "Textures" / "wood.png").write_bytes(data)
    (pack / "Models" / "box.fbx").write_bytes(fbx.write(scene))
    back = meshport.load(pack / "Models" / "box.fbx")
    assert nodes_by_name(back)["box"][0].materials[0].base_color_texture.data == data


@pytest.mark.parametrize("data, message", [
    (b"Kaydara FBX Binary  \x00\x1a\x00\xe8\x1c\x00\x00" + b"\xff" * 40, "damaged"),
    (b"Kaydara FBX", "cut short"),
    (b"hello world", "isn't an FBX"),
])
def test_bad_files_raise_a_clear_error(data, message):
    with pytest.raises(MeshportError, match=message):
        fbx.read(data)


def test_animation_only_file_says_so(scene):
    records = _records(scene)
    objects = next(r for r in records if r.name == "Objects")
    objects.children = [r for r in objects.children if r.name == "Model" and "Null" in r.props]
    with pytest.raises(MeshportError, match="No 3D geometry"):
        fbx.read(write_binary(records))
