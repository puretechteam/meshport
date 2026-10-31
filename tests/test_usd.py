"""USD: .usda, .usdc and .usdz round trips, composition, and reading what other tools write."""

import io
import zipfile

import numpy as np
import pytest
from conftest import cube, nodes_by_name, png, world_triangles

import meshport
from meshport import Image, Material, Mesh, Node, Scene, usda, usdcrate
from meshport.geometry import compose

USD = ["usdz", "usda", "usdc"]


@pytest.mark.parametrize("fmt", USD)
def test_round_trip_keeps_geometry_and_hierarchy(scene, fmt):
    back = meshport.read(meshport.write(scene, fmt), fmt)
    assert world_triangles(back) == world_triangles(scene)
    names = nodes_by_name(back)
    assert {"box", "box_mirror", "ell"} <= set(names)


@pytest.mark.parametrize("fmt", USD)
def test_round_trip_keeps_attributes(scene, fmt, tmp_path):
    path = tmp_path / f"model.{fmt}"
    written = meshport.save(scene, path)
    back = meshport.load(path)
    box = nodes_by_name(back)["box"][0].mesh
    src = cube()
    assert len(box.positions) == 8 and list(box.face_sizes) == [4] * 6
    np.testing.assert_allclose(box.normals, src.normals, atol=1e-6)
    assert len(box.uvs) == 2
    np.testing.assert_allclose(box.uvs[0], src.uvs[0], atol=1e-6)
    np.testing.assert_allclose(box.uvs[1], src.uvs[1], atol=1e-6)
    np.testing.assert_allclose(box.colors, src.colors, atol=1e-6)
    node = nodes_by_name(back)["box"][0]
    mats = [node.materials[i].name for i in box.face_materials]
    assert mats == ["red"] * 3 + ["blue"] * 3
    red = next(m for m in node.materials if m.name == "red")
    blue = next(m for m in node.materials if m.name == "blue")
    assert red.base_color_texture is not None and red.base_color_texture.data == png()
    assert red.roughness == pytest.approx(0.4)
    assert blue.alpha_mode == "BLEND" and blue.base_color == pytest.approx((0.1, 0.2, 0.9, 0.5))
    assert blue.emissive == pytest.approx((0.1, 0, 0))
    if fmt == "usdz":
        assert written == [path]
    else:  # the texture goes in a folder next to the layer
        assert len(written) == 2 and written[1].parent.name == "textures"


def test_usdz_is_a_valid_package(scene):
    data = meshport.write(scene, "usdz")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        assert infos[0].filename.endswith(".usdc")
        for info in infos:
            assert info.compress_type == zipfile.ZIP_STORED
            assert z.read(info.filename)  # readable, CRC checked
    for info in infos:  # every file's data starts on a 64-byte boundary
        start = info.header_offset + 30 + len(info.filename.encode()) + \
            int.from_bytes(data[info.header_offset + 28:info.header_offset + 30], "little")
        assert start % 64 == 0


def test_usda_and_usdc_say_the_same():
    m = Mesh("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([3]), np.arange(3))
    scene = Scene([Node("a", compose((1, 2, 3), (10, 20, 30), (1, 2, 3)), m, [Material("m", (0.5, 0.25, 1, 1))])])
    text = usda.read(meshport.write(scene, "usda"))
    binary = usdcrate.read(meshport.write(scene, "usdc"))
    assert sorted(text.specs) == sorted(binary.specs)
    for path, spec in text.specs.items():
        other = binary.specs[path].fields
        for k, v in spec.fields.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_allclose(v, other[k], atol=1e-6)
            elif k != "properties":
                assert other[k] == v or other[k] == pytest.approx(v), (path, k)


# ---- reading: composition


LAYER = '''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "World"
{
    def "Chair" (
        prepend references = @./asset.usda@
    )
    {
        double3 xformOp:translate = (5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def "Inner" (
        references = </Proto>
    )
    {
    }

    def Xform "Switch" (
        variants = {
            string size = "big"
        }
        prepend variantSets = "size"
    )
    {
        variantSet "size" = {
            "big" {
                def Cube "Box"
                {
                    double size = 4
                }
            }
            "small" {
                def Cube "Box"
                {
                    double size = 1
                }
            }
        }
    }

    def Mesh "Hidden"
    {
        token visibility = "invisible"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}

class "Proto"
{
    def Mesh "Tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
'''

ASSET = '''#usda 1.0
(
    defaultPrim = "Asset"
)

def Xform "Asset"
{
    def Mesh "Seat" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </Asset/Looks/Wood>
    }

    def Scope "Looks"
    {
        def Material "Wood"
        {
            token outputs:surface.connect = </Asset/Looks/Wood/Surface.outputs:surface>

            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Asset/Looks/Wood/Tex.outputs:rgb>
                float inputs:roughness = 0.75
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @wood.png@
                float3 outputs:rgb
            }
        }
    }
}
'''


def test_composes_references_variants_and_units(tmp_path):
    (tmp_path / "scene.usda").write_text(LAYER)
    (tmp_path / "asset.usda").write_text(ASSET)
    (tmp_path / "wood.png").write_bytes(png((120, 80, 40)))
    scene = meshport.load(tmp_path / "scene.usda")
    nodes = nodes_by_name(scene)
    assert "Hidden" not in nodes and "Proto" not in nodes  # invisible, and a class
    seat, world = nodes["Seat"]
    # Z up -> Y up, then the chair moved 5 along X
    pts = seat.mesh.positions @ world[:3, :3].T + world[:3, 3]
    np.testing.assert_allclose(pts.min(0), (5, 0, -1), atol=1e-9)
    np.testing.assert_allclose(pts.max(0), (6, 0, 0), atol=1e-9)
    wood = seat.materials[0]
    assert wood.name == "Wood" and wood.roughness == 0.75 and wood.base_color_texture.data == png((120, 80, 40))
    assert len(seat.mesh.uvs) == 1
    tri, _ = nodes["Tri"]  # through an internal reference to a class
    assert len(tri.mesh.corners) == 3
    box, box_world = nodes["Box"]  # the selected variant: size 4
    p = box.mesh.positions @ box_world[:3, :3].T
    np.testing.assert_allclose(p.max(0) - p.min(0), (4, 4, 4))


TRI = '''def Mesh "{name}"
        {{
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        }}'''


def _centers(scene) -> dict[str, list]:
    out: dict[str, list] = {}
    for node, world in scene.walk():
        if node.mesh is not None:
            pts = node.mesh.positions @ world[:3, :3].T + world[:3, 3]
            out.setdefault(node.name, []).append(np.round(pts.mean(0), 6).tolist())
    return out


def test_inherits_and_specializes_in_strength_order():
    """(Binary files are checked against Pixar's own library by benchmarks/compare_usd.py.)"""
    text = f'''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
class "_tree"
{{
    double3 xformOp:translate = (0, 0, 7)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    {TRI.format(name="Leaves")}
}}
class "_base"
{{
    double3 xformOp:translate = (0, 50, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    {TRI.format(name="Detail")}
}}
def Xform "A" (inherits = </_tree>)
{{
}}
def Xform "B" (inherits = </_tree>)
{{
    double3 xformOp:translate = (20, 0, 0)
}}
def Xform "S" (specializes = </_base>)
{{
}}
def Xform "Both" (inherits = </_tree> specializes = </_base>)
{{
}}
def Xform "LoopA" (inherits = </LoopB>)
{{
    {TRI.format(name="G")}
}}
def Xform "LoopB" (inherits = </LoopA>)
{{
}}
'''
    c = _centers(meshport.read(text.encode(), "usda"))
    t = 1 / 3
    np.testing.assert_allclose(sorted(c["Leaves"]), [[t, t, 7], [t, t, 7], [20 + t, t, 0]], atol=1e-6)
    # "Both" takes the inherited class's translate over the specialized one's
    np.testing.assert_allclose(sorted(c["Detail"]), [[t, t, 7], [t, 50 + t, 0]], atol=1e-6)
    np.testing.assert_allclose(c["G"], [[t, t, 0]] * 2, atol=1e-6)  # each of the loop gets the other's child


def test_point_instancer():
    text = f'''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def PointInstancer "Forest"
{{
    point3f[] positions = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)]
    quath[] orientations = [(1, 0, 0, 0), (0.7071, 0, 0.7071, 0), (1, 0, 0, 0), (1, 0, 0, 0)]
    float3[] scales = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (1, 1, 1)]
    int[] protoIndices = [0, 0, 1, 0]
    int64[] ids = [5, 6, 7, 8]
    int64[] invisibleIds = [8]
    rel prototypes = [</Forest/Protos/Tri>, </Forest/Protos/Moved>]
    def Scope "Protos"
    {{
        {TRI.format(name="Tri")}
        def Xform "Moved"
        {{
            double3 xformOp:translate = (0, 5, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            {TRI.format(name="Inner")}
        }}
    }}
}}
'''
    scene = meshport.read(text.encode(), "usda")
    c = _centers(scene)
    third = 1 / 3
    assert "Tri" in c and len(c["Tri"]) == 2  # the prototype itself isn't drawn, nor the hidden id 8
    # the second one turned 90 degrees about Y (by a half-precision quaternion)
    np.testing.assert_allclose(sorted(c["Tri"]), [[third, third, 0], [10, third, -third]], atol=1e-3)
    np.testing.assert_allclose(c["Inner"], [[20 + 2 * third, 10 + 2 * third, 0]], atol=1e-6)
    assert len(scene.meshes) == 2  # instances share their prototype's mesh


def test_centimeters_and_y_up_by_default():
    text = b'''#usda 1.0
def Mesh "Tri"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (100, 0, 0), (0, 200, 0)]
}
'''
    scene = meshport.read(text, "usda")
    np.testing.assert_allclose(scene.bounds(), [[0, 0, 0], [1, 2, 0]])


def test_left_handed_faces_and_face_varying_normals_flip_together():
    text = b'''#usda 1.0
(
    metersPerUnit = 1
)
def Mesh "Quad"
{
    uniform token orientation = "leftHanded"
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    normal3f[] normals = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)] (
        interpolation = "faceVarying"
    )
}
'''
    mesh = meshport.read(text, "usda").meshes[0]
    assert list(mesh.corners) == [3, 2, 1, 0]
    assert list(mesh.normals[:, 2]) == [4, 3, 2, 1]


def test_geom_subsets_give_per_face_materials():
    text = b'''#usda 1.0
(
    metersPerUnit = 1
)
def Mesh "Two"
{
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    rel material:binding = </Red>

    def GeomSubset "second"
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [1]
        rel material:binding = </Blue>
    }
}
def Material "Red"
{
}
def Material "Blue"
{
}
'''
    node = meshport.read(text, "usda").roots[0]
    assert [node.materials[i].name for i in node.mesh.face_materials] == ["Red", "Blue"]


def test_display_color_without_a_material():
    text = b'''#usda 1.0
(
    metersPerUnit = 1
)
def Cube "Box"
{
    color3f[] primvars:displayColor = [(0.25, 0.5, 1)]
}
'''
    node = meshport.read(text, "usda").roots[0]
    assert node.materials[0].base_color == (0.25, 0.5, 1, 1)
    assert len(node.mesh.positions) == 8


def test_xform_ops_of_every_kind():
    text = b'''#usda 1.0
(
    metersPerUnit = 1
)
def Xform "A"
{
    double3 xformOp:translate = (1, 2, 3)
    float3 xformOp:rotateZYX = (0, 90, 0)
    quatf xformOp:orient = (0.70710677, 0, 0, 0.70710677)
    float3 xformOp:scale = (2, 2, 2)
    double3 xformOp:translate:pivot = (1, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:translate:pivot", "xformOp:rotateZYX", "xformOp:orient", "xformOp:scale", "!invert!xformOp:translate:pivot"]

    def Cube "C"
    {
        double size = 2
    }
}
'''
    _, world = nodes_by_name(meshport.read(text, "usda"))["C"]
    from meshport.geometry import euler, scaling, translation
    q = np.eye(4)
    q[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about Z
    expect = translation((1, 2, 3)) @ translation((1, 0, 0)) @ euler((0, 90, 0), "ZYX") @ q @ \
        scaling((2, 2, 2)) @ translation((-1, 0, 0))
    np.testing.assert_allclose(world, expect, atol=1e-6)


@pytest.mark.parametrize("data", [
    b"#usda 1.0\ndef Mesh \"x\" {\n int[] faceVertexCounts = [3\n",
    b"#usda 1.0\ndef Mesh \"x\" { int[] faceVertexCounts = [5] int[] faceVertexIndices = [0, 1] point3f[] points = [(0,0,0)] }",
    b"#usda 1.0\ndef Mesh \"x\" { int[] faceVertexCounts = [3] int[] faceVertexIndices = [0, 1, 9] point3f[] points = [(0,0,0), (1,0,0), (0,1,0)] }",
    b"PXR-USDC\x00\x08\x00" + bytes(100),
    usdcrate.MAGIC + bytes([0, 8, 0, 0, 0, 0, 0, 0]) + (10 ** 9).to_bytes(8, "little"),
    b"PK\x03\x04" + bytes(26) + b"x.usdc",
])
def test_damaged_files_raise_meshport_errors(data):
    with pytest.raises(meshport.MeshportError):
        meshport.read(data, "usd")


def test_textures_from_other_folders_must_be_images(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("password")
    (tmp_path / "m.usda").write_text(ASSET.replace("wood.png", str(secret).replace("\\", "/")))
    seat = nodes_by_name(meshport.load(tmp_path / "m.usda"))["Seat"][0]
    assert seat.materials[0].base_color_texture.data is None


def test_shared_mesh_written_and_read():
    tri = Mesh("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([3]), np.arange(3))
    scene = Scene([Node("a", mesh=tri), Node("b", compose((3, 0, 0)), tri)])
    back = meshport.read(meshport.write(scene, "usdz"))
    assert world_triangles(back) == world_triangles(scene)


def test_image_formats_usdz_cant_hold_become_png():
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), (1, 2, 3)).save(buf, "BMP")
    tri = Mesh("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([3]), np.arange(3),
               uvs=[np.zeros((3, 2))])
    scene = Scene([Node("a", mesh=tri, materials=[Material("m", base_color_texture=Image("bmp", buf.getvalue()))])])
    data = meshport.write(scene, "usdz")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert [n for n in z.namelist() if n.startswith("textures/")] == ["textures/bmp.png"]


def test_crate_integer_tables_wrap_like_usd():
    """Fieldsets end each spec's list with 0xFFFFFFFF. When jumping to and from that marker is
    the most common step (many specs with one field each, as in Khronos' BrainStem), the step is
    stored modulo 2**32, as Pixar's integer coding does, and still reads back."""
    from meshport.usdcrate import decode_ints, encode_ints, fast_decompress
    values = [7, 0xFFFFFFFF] * 40 + [3, 9, 0xFFFFFFFF]
    data = fast_decompress(encode_ints(values), 1 << 20)
    assert decode_ints(data, len(values), 4).astype(np.uint32).tolist() == values
