"""Reading .blend files: a fixture saved by Blender 5.1 (rebuilt by data/make_blend_fixture.py),
checked against Blender's own GLB export of it, plus damaged files and Blender itself."""

import gzip
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import BLENDER, nodes_by_name

import meshport

DATA = Path(__file__).resolve().parent / "data"


def world_points(scene: meshport.Scene) -> np.ndarray:
    return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk() if n.mesh is not None])


def gap(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


@pytest.fixture(scope="module")
def scene():
    return meshport.load(DATA / "scene.blend")


def test_same_geometry_as_blenders_own_export(scene):
    ref = meshport.load(DATA / "scene_blender.glb")
    assert len(scene.meshes) == len(ref.meshes) == 4
    assert gap(world_points(scene), world_points(ref)) < 1e-5


def test_hierarchy_rotation_modes_and_instances(scene):
    nodes = nodes_by_name(scene)
    parent, _ = nodes["Parent"]
    assert [c.name for c in parent.children] == ["Child"]  # parented, its inverse matrix applied
    child_world = nodes["Child"][1]
    assert np.linalg.det(child_world[:3, :3]) < 0  # the negative scale survives
    instance = nodes["Instance"][0]
    assert [c.name for c in instance.children] == ["Cone"]  # the collection it shows
    ref = meshport.load(DATA / "scene_blender.glb")
    for name in ("Parent", "Child", "AxisAngle"):  # quaternion, ZXY euler, axis-angle with a delta
        ours = nodes[name][0].mesh.positions @ nodes[name][1][:3, :3].T + nodes[name][1][:3, 3]
        theirs = nodes_by_name(ref)[name]
        assert gap(ours, theirs[0].mesh.positions @ theirs[1][:3, :3].T + theirs[1][:3, 3]) < 1e-5, name


def test_materials_textures_uvs_and_shading(scene):
    node = nodes_by_name(scene)["Parent"][0]
    painted, plain = node.materials
    assert painted.name == "Painted" and painted.roughness == pytest.approx(0.25)
    assert painted.metallic == pytest.approx(0.5)
    assert painted.base_color_texture is not None and painted.base_color_texture.data[:4] == b"\x89PNG"  # packed
    assert plain.base_color == pytest.approx((0.1, 0.2, 0.7, 1))
    mesh = node.mesh
    assert list(mesh.face_materials) == [0, 1] * 3
    assert len(mesh.uvs) == 2
    # the first three faces are smooth: their corners on a shared vertex share a normal
    flat = mesh.normals[mesh.face_starts[3]:mesh.face_starts[3] + 4]
    assert np.allclose(flat, flat[0])


def test_uncompressed_and_gzip_files_read_too(scene, tmp_path):
    from meshport.blend import _decompress
    raw = _decompress((DATA / "scene.blend").read_bytes())
    assert raw[:7] == b"BLENDER"
    for data in (raw, gzip.compress(raw)):
        back = meshport.read(data)
        assert gap(world_points(back), world_points(scene)) == 0


def test_writing_blend_is_refused(scene):
    with pytest.raises(meshport.MeshportError):
        meshport.write(scene, "blend")


@pytest.mark.parametrize("cut", [20, 2000, 30000, -40])
def test_damaged_files_raise_meshport_errors(cut):
    from meshport.blend import _decompress
    raw = _decompress((DATA / "scene.blend").read_bytes())
    with pytest.raises(meshport.MeshportError):
        meshport.read(raw[:cut], "blend")


def test_other_damage(tmp_path):
    for data in (b"BLENDER-V405" + bytes(64), b"BLENDER-v405" + b"DNA1" + bytes(40), b"\x28\xb5\x2f\xfd" + bytes(20)):
        with pytest.raises(meshport.MeshportError):
            meshport.read(data, "blend")


# ---- older files: the same scene saved by Blender 2.83 (gzip) and 3.5 (zstd), rebuilt with
# make_blend_fixture.py -- tests/data scene_283; both keep meshes in the old vertex, polygon and loop
# arrays and UVs in MLoopUV structs

@pytest.mark.parametrize("version", ["283", "35"])
def test_older_blender_files_match_blender(version):
    ours = meshport.load(DATA / f"scene_{version}.blend")
    ref = meshport.load(DATA / f"scene_{version}_blender.glb")
    assert len(ours.meshes) == len(ref.meshes) == 4
    assert gap(world_points(ours), world_points(ref)) < 1e-5

    def uv_pairs(scene):  # each corner's point with its UVs, in every UV map
        node, w = nodes_by_name(scene)["Parent"]
        p = node.mesh.positions[node.mesh.corners] @ w[:3, :3].T + w[:3, 3]
        return [np.unique(np.round(np.hstack([p, uv]), 4), axis=0) for uv in node.mesh.uvs]
    a, b = uv_pairs(ours), uv_pairs(ref)
    assert len(a) == len(b) == 2 and all(np.array_equal(x, y) for x, y in zip(a, b))
    node = nodes_by_name(ours)["Parent"][0]
    painted, plain = node.materials
    assert painted.roughness == pytest.approx(0.25) and painted.base_color_texture is not None
    assert plain.base_color == pytest.approx((0.1, 0.2, 0.7, 1))
    assert list(node.mesh.face_materials) == [0, 1] * 3


_BLENDER = textwrap.dedent("""
    import bpy, sys
    blend, glb = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_monkey_add(rotation=(0.3, 0.2, 0.1), location=(1, 2, 3))
    bpy.ops.mesh.primitive_torus_add(location=(-2, 0, 1), rotation=(1, 0, 0))
    bpy.context.object.rotation_mode = "YZX"
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB")
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_against_blender_now(tmp_path):
    script = tmp_path / "make.py"
    script.write_text(_BLENDER)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "s.blend"), str(tmp_path / "s.glb")], check=True, capture_output=True,
                   timeout=300)
    ours, theirs = meshport.load(tmp_path / "s.blend"), meshport.load(tmp_path / "s.glb")
    assert gap(world_points(ours), world_points(theirs)) < 1e-5


# ---- modifiers: data/modifiers.blend (rebuilt by data/make_modifier_fixture.py) against Blender's GLB
# export of it with the modifiers applied

def object_points(scene: meshport.Scene) -> dict:
    out = {}
    for node, w in scene.walk():
        if node.mesh is not None:
            m = node.mesh
            out[node.name] = (m.positions[np.unique(m.corners)] @ w[:3, :3].T + w[:3, 3], len(m.triangles()))
    return out


@pytest.fixture(scope="module")
def modified():
    return object_points(meshport.load(DATA / "modifiers.blend")), object_points(meshport.load(DATA / "modifiers_blender.glb"))


# Blender places smoothed boundary corners on an approximation of the limit surface, and Solidify's even
# thickness divides by a cosine near zero at one of the monkey's vertices (6 m out), where float precision tells
# A 0.7 crease (sharpness 4.9) next to a creased corner, three levels down, comes out 6% sharper than Blender's
TOLERANCE = {"SubsurfGridAll": 5e-3, "SolidifyEven": 0.05, "SubsurfCreaseVertex": 0.07}
MODIFIED = ["TriShortest", "TriBeauty", "WeldAll", "WeldConnected", "MirrorX", "MirrorBisect", "MirrorObject",
            "ArrayRelative", "ArrayMerge", "ArrayFit", "ArrayObject", "ArrayCaps", "ArrayCapsMerge", "SolidifyPlane",
            "SolidifyEven", "SolidifyRimOnly", "SubsurfCube", "SubsurfGridCorners", "SubsurfGridAll", "SubsurfNgon",
            "SubsurfMonkey", "SubsurfNoLimit", "SubsurfSimple", "SubsurfUV_NONE", "SubsurfUV_SMOOTH_ALL",
            "SubsurfUV_PRESERVE_CORNERS", "SubsurfCrease10", "SubsurfCrease5", "SubsurfCrease3", "SubsurfCreaseVertex",
            "Stack"]


@pytest.mark.parametrize("name", MODIFIED)
def test_modifiers_match_blender(modified, name):
    ours, theirs = modified
    assert ours[name][1] == theirs[name][1]  # the same number of triangles
    assert gap(ours[name][0], theirs[name][0]) < TOLERANCE.get(name, 1e-5)


def _corner_uvs(scene) -> dict:
    out = {}
    for node, w in scene.walk():
        m = node.mesh
        if m is not None and m.uvs:
            out[node.name] = np.hstack([m.positions[m.corners] @ w[:3, :3].T + w[:3, 3], m.uvs[0]])
    return out


@pytest.fixture(scope="module")
def modified_uvs():
    return _corner_uvs(meshport.load(DATA / "modifiers.blend")), _corner_uvs(meshport.load(DATA / "modifiers_blender.glb"))


# smoothed UV corners (UV Smooth "All") land where Blender's approximation of the limit puts them, as above
UV_TOLERANCE = {**TOLERANCE, "SubsurfUV_SMOOTH_ALL": 2e-3}


@pytest.mark.parametrize("name", [n for n in MODIFIED if not n.startswith("Weld")])  # the Weld grids have no UVs
def test_modifier_uvs_match_blender(modified_uvs, name):
    # each corner's point and UV together: subdivided UVs are smoothed as Blender smooths them
    ours, theirs = modified_uvs
    assert gap(ours[name], theirs[name]) < UV_TOLERANCE.get(name, 1e-5)


def test_unknown_and_hidden_modifiers_are_skipped(modified):
    ours, theirs = modified
    assert ours["Unsupported"][1] == 24  # Bevel skipped, the Array after it still applied
    # Stack ends with a Weld that's off in the viewport (with a 10 m distance, it would merge everything)
    assert ours["Stack"][1] == theirs["Stack"][1] == 384


def test_modifiers_can_be_left_off():
    from meshport import blend
    plain = object_points(blend.read_file(DATA / "modifiers.blend", modifiers=False))
    assert plain["SubsurfCube"][1] == plain["Stack"][1] == 12
    assert plain["MirrorX"][1] == 968


def test_modifier_meshes_are_per_object_and_valid():
    scene = meshport.load(DATA / "modifiers.blend")
    for node, _ in scene.walk():
        if node.mesh is not None:
            node.mesh.validate()
            assert np.isfinite(node.mesh.positions).all() and np.isfinite(node.mesh.normals).all()
    nodes = nodes_by_name(scene)
    plane = nodes["SolidifyPlane"][0]
    assert set(plane.mesh.face_materials.tolist()) == {0, 1}  # the copy and rim take the material offset
    back = meshport.read(meshport.write(scene, "glb"))
    assert len(back.meshes) == len(scene.meshes)


# ---- curves, text and metaballs: data/curves.blend (rebuilt by data/make_curve_fixture.py) against Blender's
# GLB export of it (its metaballs made meshes first, since the export leaves them out)

@pytest.fixture(scope="module")
def curved():
    return meshport.load(DATA / "curves.blend"), meshport.load(DATA / "curves_blender.glb")


def points_of(scene: meshport.Scene, name: str) -> np.ndarray:
    node, w = nodes_by_name(scene)[name]
    return node.mesh.positions[np.unique(node.mesh.corners)] @ w[:3, :3].T + w[:3, 3]


CURVES = ["BezierTube", "BezierRibbon", "BezierStadium", "BezierHalf", "BezierCaps", "BezierTapered", "BezierTilted",
          "BezierZUp", "NurbsCircleTube", "NurbsPathTube", "PolyHelix", "PolyLoop", "Disc", "Coin", "FrontOnly",
          "Ring2D", "Washer", "CurveArray", "Text", "FlatText"]


@pytest.mark.parametrize("name", CURVES)
def test_curves_and_text_match_blender(curved, name):
    ours, theirs = curved
    a, b = nodes_by_name(ours)[name][0].mesh, nodes_by_name(theirs)[name][0].mesh
    assert len(a.triangles()) == len(b.triangles())
    assert gap(points_of(ours, name), points_of(theirs, name)) < 1e-5


def test_filled_outlines_cover_the_same_area(curved):
    """Caps with holes (a washer, letters like o, e and d) fill exactly what Blender fills."""
    def cap_area(scene, name):
        node, w = nodes_by_name(scene)[name]
        p = node.mesh.positions @ w[:3, :3].T + w[:3, 3]
        t = p[node.mesh.corners[node.mesh.triangles()]]
        cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) / 2
        flat = np.abs(cross[:, 1]) > 0.999 * np.linalg.norm(cross, axis=1)
        return cross[flat & (cross[:, 1] > 0), 1].sum()
    ours, theirs = curved
    for name in ("Washer", "Text", "FlatText", "Coin"):
        assert cap_area(ours, name) == pytest.approx(cap_area(theirs, name), rel=1e-5), name


def test_open_nurbs_ribbon_turns_close_to_blender(curved):
    # an open NURBS curve's frames start a few degrees apart from Blender's, which shows on an extrusion
    ours, theirs = curved
    assert gap(points_of(ours, "NurbsHelix"), points_of(theirs, "NurbsHelix")) < 0.04


@pytest.mark.parametrize("name, resolution", [("Mball", 0.4), ("Blob", 0.2)])
def test_metaballs_match_blenders_surface(curved, name, resolution):
    import sys
    sys.path.insert(0, str(DATA.parent.parent / "benchmarks"))
    from compare_blender import surface_gap

    def triangles(scene):
        node, w = nodes_by_name(scene)[name]
        p = node.mesh.positions @ w[:3, :3].T + w[:3, 3]
        return p[node.mesh.corners[node.mesh.triangles()]]
    ours, theirs = curved
    # both sample the same surface at the same resolution, with different points
    assert surface_gap(triangles(ours), triangles(theirs), 4000) < 0.25 * resolution


def test_metaball_family_and_surface():
    from meshport import blendcurve
    assert blendcurve.family("Mball.001") == blendcurve.family("Mball") == "Mball"
    scene = meshport.load(DATA / "curves.blend")
    names = {n.name for n, _ in scene.walk() if n.mesh is not None}
    assert "Mball" in names and "Mball.001" not in names  # the family's surface is on its base object
    ball = [(0, 0, np.eye(4), 2.0, 2.0, np.ones(3))]
    mesh, smooth = blendcurve.metaball_mesh(ball, 0.1, 0.6)
    r = np.linalg.norm(mesh.positions, axis=1)
    assert np.allclose(r, 2 * np.sqrt(1 - 0.3 ** (1 / 3)), atol=1e-3)  # stiffness (1 - r²/R²)³ = threshold
    tri = mesh.positions[mesh.corners.reshape(-1, 3)]
    outward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri.mean(1))
    assert (outward > 0).all() and smooth.all()


def test_text_needs_fonttools(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "fontTools", None)
    monkeypatch.setitem(sys.modules, "fontTools.ttLib", None)
    names = {n.name for n, _ in meshport.load(DATA / "curves.blend").walk() if n.mesh is not None}
    assert "Text" not in names and "BezierTube" in names


def test_fill_handles_holes_and_flat_corners():
    from meshport import blendcurve
    square = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [0, 2], [0, 1]], float)  # points along straight sides
    hole = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]], float)
    tris = blendcurve._fill([square, hole])
    pts = np.concatenate([square, hole])
    t = pts[tris]
    area = 0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
    assert area.sum() == pytest.approx(3.0) and (area > -1e-12).all()


def test_damaged_curves_raise_or_skip():
    from meshport.blend import _decompress
    raw = bytearray(_decompress((DATA / "curves.blend").read_bytes()))
    rng = np.random.default_rng(3)
    for _ in range(20):
        damaged = bytearray(raw)
        for at in rng.integers(1000, len(raw), 40):
            damaged[at] = rng.integers(0, 256)
        try:
            meshport.read(bytes(damaged), "blend")
        except meshport.MeshportError:
            pass


def test_builtin_font_text_is_close(curved):
    from meshport import blendcurve
    if blendcurve._builtin_font() is None:
        pytest.skip("no stand-in for Blender's built-in font is installed")
    ours, theirs = curved
    a, b = points_of(ours, "BuiltinText"), points_of(theirs, "BuiltinText")
    size_a, size_b = np.ptp(a, axis=0), np.ptp(b, axis=0)
    assert np.allclose(size_a, size_b, rtol=0.1, atol=1e-6)  # capitals as tall as Blender's, widths close
