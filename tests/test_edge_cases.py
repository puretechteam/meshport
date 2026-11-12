"""Edge cases real files hit: objects scaled to 0, very deep scenes, long baked actions, big outlines and
curves, textures with the same name, files already next to the output, and numbers that aren't finite."""

import struct

import numpy as np
import pytest
from blendfile import (
    IDENTITY,
    BlendWriter,
    collection,
    cube_mesh,
    mesh_object,
    scene_file,
)
from conftest import cube
from test_3ds import _chunk, _keyed_file, _kf_node, _mesh_object

import meshport
from meshport import (
    Animation,
    Channel,
    Image,
    Light,
    Material,
    Mesh,
    MeshportError,
    Node,
    Scene,
    Skin,
    alembic,
    blendanim,
    blendcurve,
    blendmods,
    max3ds,
)
from meshport.fbxtree import Record, _read_ascii, write_ascii
from meshport.usd import _Source

PNG = b"\x89PNG\r\n\x1a\n"


def _tri(name="t") -> Mesh:
    return Mesh(name, np.eye(3), np.array([3]), np.arange(3))


def _world_points(scene: Scene) -> dict[str, np.ndarray]:
    return {n.name: n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk() if n.mesh is not None}


# ---- objects scaled to 0

def test_a_parent_scaled_to_zero_reads():
    """Scale 0 is a common way to hide things in Blender. The .blend was unreadable ("damaged
    (Singular matrix)"); now the child sits where Blender puts it, squashed into its parent's origin."""
    w = BlendWriter()
    me = cube_mesh(w)
    parent = mesh_object(w, "parent", me, size=[0, 0, 0], loc=[1, 2, 3])
    child = mesh_object(w, "child", me, parent=parent, loc=[2, 0, 0])
    points = _world_points(meshport.read(scene_file(w, [parent, child]), "blend"))
    for name in ("parent", "child"):
        assert np.allclose(points[name], [1, 3, -2])  # (1, 2, 3) in Blender's Z up is (1, 3, -2) in Y up


@pytest.mark.parametrize("fmt", ["obj", "3ds", "abc"])
def test_a_skinned_mesh_under_a_parent_scaled_to_zero_writes(tmp_path, fmt):
    bone = Node("bone")
    skinned = Node("skinned", mesh=cube())
    skinned.skin = Skin([bone], np.array([np.eye(4)]), np.zeros((8, 1), np.int64), np.ones((8, 1)))
    hidden = Node("hidden", np.diag([0.0, 0.0, 0.0, 1.0]), children=[skinned])
    meshport.save(Scene([bone, hidden]), tmp_path / f"out.{fmt}")  # was a bare LinAlgError
    assert (tmp_path / f"out.{fmt}").stat().st_size > 0


# ---- deep scenes

def _deep(depth: int) -> Scene:
    node = Node("leaf", mesh=_tri())
    for i in range(depth):
        node = Node(f"n{i}", children=[node])
    return Scene([node])


@pytest.mark.parametrize("fmt", ["fbx", "glb", "gltf"])
def test_very_deep_scenes_write_and_read_back(fmt):
    """3,000 levels: FBX and glTF writing recursed (RecursionError at about 900 for glb), and so did
    glTF reading."""
    data = meshport.write(_deep(3000), fmt)
    back = meshport.read(data, fmt)
    assert len(list(back.walk())) == 3001 and len(back.meshes) == 1


def test_very_deep_scenes_write_as_ascii_fbx():
    assert meshport.write(_deep(3000), "fbx", ascii=True)


def test_a_long_3ds_parent_chain_reads():
    """A chain of 3,000 dummies: the loop check climbed the whole chain for every node and the
    placing recursed, so any chain over about 1,000 failed."""
    nodes = [_kf_node("$$$DUMMY", 0)] + [_kf_node("$$$DUMMY", i, parent=i - 1) for i in range(1, 3000)]
    scene = max3ds.read(_keyed_file([_mesh_object("a", np.eye(4))], nodes))
    assert len(scene.meshes) == 1


def test_a_3ds_keyframer_without_a_frame_range():
    """KF_CURRENT with no KF_SEGMENT: the whole file failed ("damaged (TypeError)")."""
    data = _keyed_file([_mesh_object("a", np.eye(4))],
                       [_chunk(max3ds.KF_CURRENT, struct.pack("<I", 0)), _kf_node("a", 0, pos=(1, 2, 3))])
    assert len(max3ds.read(data).meshes) == 1


# ---- long animations

def test_a_baked_bezier_action_is_worked_out_all_at_once(monkeypatch):
    """A key on every frame: each Bezier segment was solved on its own (4,000 keys took 3 s per
    curve, now 0.016 s), with the same values."""
    n = 4000
    x = np.arange(n, dtype=float)
    y = np.sin(x / 10)
    keys = np.stack([np.stack([x - 0.3, y], 1), np.stack([x, y], 1), np.stack([x + 0.3, y], 1)], 1)
    curve = blendanim.FCurve("location", 0, keys, np.full(n, blendanim.BEZIER), np.zeros(n, int), np.zeros((n, 3)), 0)
    frames = np.arange(-2, n + 2) + 0.25
    calls = []
    real = blendanim.FCurve._segment
    monkeypatch.setattr(blendanim.FCurve, "_segment", lambda self, i, f: calls.append(i) or real(self, i, f))
    got = curve.evaluate(frames)
    assert len(calls) < 10
    for i in (5, 1234, 3998):  # each value as its own segment gives it
        inside = (frames >= x[i]) & (frames < x[i + 1])
        assert np.array_equal(got[inside], real(curve, i, frames[inside]))


def test_a_long_alembic_animation_reads_each_group_once(monkeypatch):
    """Every frame's sample re-read its property's whole list of samples: 8,000 frames took 1.40 s
    (and the count could trip the "tree loops" guard on a long valid file), now 0.13 s."""
    frames = 8000
    node = Node("box", mesh=cube())
    t = np.arange(frames) / 24.0
    data = alembic.write(Scene([node], [Animation("a", [Channel(node, "translation", t, np.c_[t, t * 0, t * 0])])]))
    reads = []
    real = alembic._File.group

    def group(self, ref):
        if ref not in getattr(self, "_groups", {}):
            reads.append(ref)
        return real(self, ref)
    monkeypatch.setattr(alembic._File, "group", group)
    back = alembic.read(data)
    assert len(back.animations[0].channels[0].times) == frames
    assert len(reads) < 500  # was 16,000


def test_rigs_are_posed_for_many_frames_at_once(monkeypatch):
    """The USD and Alembic writers posed every node on every frame, one sample at a time: a 30-bone
    chain over 200 frames now samples each channel a few times, and poses come out the same."""
    bones, parent, roots = [], None, []
    for i in range(30):
        b = Node(f"b{i}", np.eye(4))
        b.matrix[1, 3] = 0.1
        (parent.children if parent else roots).append(b)
        bones.append(b)
        parent = b
    worlds = {id(n): w for r in roots for n, w in r.walk()}
    skinned = Node("skinned", mesh=cube())
    skinned.skin = Skin(bones, np.array([np.linalg.inv(worlds[id(b)]) for b in bones]),
                        (np.arange(8) % 30)[:, None], np.ones((8, 1)))
    t = np.arange(200) / 24.0
    q = np.c_[np.zeros(200), np.zeros(200), np.sin(t * 0.3), np.cos(t * 0.3)]
    scene = Scene(roots + [skinned], [Animation("a", [Channel(b, "rotation", t, q) for b in bones])])
    for time in (0.0, 1.37, 8.29):
        many = {id(n): w for n, w in scene.walk_over([time, time])}
        for n, w in scene.walk_at(time):
            assert np.allclose(many[id(n)][1], w, atol=1e-12)
    calls = []
    real = Channel.sample
    monkeypatch.setattr(Channel, "sample", lambda self, at: calls.append(1) or real(self, at))
    for fmt in ("usdz", "abc"):
        calls.clear()
        meshport.write(scene, fmt)
        assert len(calls) < 1000, (fmt, len(calls))  # was 6,000 a format (30 bones x 200 frames)


def test_baking_an_endless_animation_is_refused():
    node = Node("n")
    long = Animation("a", [Channel(node, "translation", np.array([0.0, 1e9]), np.zeros((2, 3)))])
    with pytest.raises(MeshportError, match="too long"):
        long.bake(24)


# ---- curves and metaballs

def _star(n: int) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    r = 1 + 0.3 * (np.arange(n) % 2)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def test_a_huge_filled_outline_is_fanned_within_the_budget(monkeypatch):
    """Ear clipping is about n * n steps an outline (8,000 points took 15 s, 50,000 would take hours):
    past FILL_BUDGET an outline is fanned, as huge concave faces are."""
    monkeypatch.setattr(blendcurve, "FILL_BUDGET", 50 * 50)
    small = blendcurve._fill([_star(40)])
    assert len(small) == 38 and not (small == small[0, 0]).any(1).all()  # ear clipped
    big = blendcurve._fill([_star(60)])
    assert len(big) == 58 and (big == big[0, 0]).any(1).all()  # fanned from one point


def _dense_basis(k, order, u, count):  # Cox-de Boor over every control point, as meshport used to
    last = len(k) - 1
    b = np.zeros((len(u), last))
    for i in range(last):
        if k[i] < k[i + 1]:
            b[:, i] = (u >= k[i]) & (u < k[i + 1])
    for d in range(1, order):
        nb = np.zeros((len(u), last - d))
        for i in range(last - d):
            left, right = k[i + d] - k[i], k[i + d + 1] - k[i + 1]
            if left > 0:
                nb[:, i] += (u - k[i]) / left * b[:, i]
            if right > 0:
                nb[:, i] += (k[i + d + 1] - u) / right * b[:, i + 1]
        b = nb
    return b[:, :count]


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_nurbs_basis_works_out_only_the_functions_that_count(order):
    """The basis was a samples x control points matrix: 4,000 points took 146 s, now 0.014 s."""
    rng = np.random.default_rng(order)
    count = 15
    knots = np.sort(rng.integers(0, 9, count + order)).astype(float)
    u = np.linspace(knots[order - 1], np.nextafter(knots[count], -np.inf), 200)
    index, values = blendcurve._basis(knots, order, u)
    assert index.shape == (200, order)
    dense = np.zeros((200, count))
    for r in range(order):
        ok = (index[:, r] >= 0) & (index[:, r] < count)
        np.add.at(dense, (np.flatnonzero(ok), index[ok, r]), values[ok, r])
    assert np.allclose(dense, _dense_basis(knots, order, u, count), atol=1e-12)


def test_metaball_elements_are_worked_out_only_where_they_reach(monkeypatch):
    """Each element's field was worked out over the whole grid (50 elements: 27.9 s, now 0.56 s),
    and the number of elements had no limit."""
    rng = np.random.default_rng(0)
    elements = []
    for _ in range(20):
        m = np.eye(4)
        m[:3, 3] = rng.uniform(-10, 10, 3)
        elements.append((0, 0, m, 1.0, 2.0, np.ones(3)))
    sizes = []
    real = blendcurve._density
    monkeypatch.setattr(blendcurve, "_density", lambda kind, p, *a: sizes.append(len(p)) or real(kind, p, *a))
    got = blendcurve.metaball_mesh(elements, 0.1, 0.6)
    assert got is not None and len(got[0].corners)
    assert max(sizes) < 100_000  # the grid has about 8 million points
    monkeypatch.setattr(blendcurve, "MAX_METABALL_ELEMENTS", 10)
    with pytest.raises(MeshportError, match="too many elements"):
        blendcurve.metaball_mesh(elements, 0.1, 0.6)


# ---- modifiers

def _array(mesh, count):
    md = {"count": count, "offset_type": 2, "scale": np.array([1.0, 0, 0])}
    ctx = blendmods.Context(np.eye(4), lambda a: None, 1)
    return blendmods._array(mesh, np.ones(len(mesh.face_sizes), bool), md, ctx)


def test_array_keeps_every_copy_of_a_big_mesh():
    """Copies were cut to 1,000,000 points in all without a word: 100,000 points x 20 gave 11."""
    k = 317  # about 100,000 points
    x, y = np.meshgrid(np.arange(k, dtype=float), np.arange(k, dtype=float), indexing="ij")
    i, j = np.meshgrid(np.arange(k - 1), np.arange(k - 1), indexing="ij")
    quads = np.stack([i * k + j, (i + 1) * k + j, (i + 1) * k + j + 1, i * k + j + 1], -1).reshape(-1, 4)
    mesh = Mesh("grid", np.c_[x.ravel(), y.ravel(), np.zeros(k * k)], np.full(len(quads), 4), quads.ravel())
    out, _ = _array(mesh, 20)
    assert len(out.positions) == 20 * k * k


def test_array_still_refuses_a_hostile_count():
    with pytest.raises(MeshportError, match="too big"):
        _array(cube(), 2_000_000_000)


def test_an_instanced_object_gets_its_modifiers_applied_once(tmp_path, monkeypatch):
    """Shown by 50 collection instances, an object's modifiers ran 50 times, each spending the file's
    budget for generated geometry, so a real scene could be refused."""
    w = BlendWriter()
    array = w.address()
    w.block(b"DATA", "ArrayModifierData", {"modifier": {"mode": 1, "name": "Array"}, "count": 3,
                                           "offset_type": 2, "scale": [1, 0, 0]}, array)
    inner = collection(w, [mesh_object(w, "Cube", cube_mesh(w), modifiers={"first": array, "last": array})])
    empties = [w.block(b"OB\0\0", "Object", {"id": {"name": f"OBe{i}"}, "type": 0, "rotmode": 1, "transflag": 1 << 8,
                                              "instance_collection": inner, "size": [1, 1, 1], "loc": [0, 3 * i, 0],
                                              "parentinv": IDENTITY}) for i in range(50)]
    runs = []
    real = blendmods.apply
    monkeypatch.setattr(blendmods, "apply", lambda *a: runs.append(1) or real(*a))
    scene = meshport.read(scene_file(w, empties), "blend")
    assert len([n for n, _ in scene.walk() if n.mesh is not None]) == 50 and len(runs) == 1


# ---- files next to the output

def _textured(*names_and_bytes) -> Scene:
    nodes = []
    for k, (path, data) in enumerate(names_and_bytes):
        mat = Material(f"m{k}", base_color_texture=Image("wood", data, path))
        nodes.append(Node(f"n{k}", mesh=_tri(), materials=[mat]))
    return Scene(nodes)


def test_two_textures_with_one_name_both_survive_obj(tmp_path):
    """Two materials' wood.png were both written as wood.png, the second over the first."""
    scene = _textured(("x/wood.png", PNG + b"A" * 20), ("y/wood.png", PNG + b"B" * 20))
    written = meshport.save(scene, tmp_path / "o.obj")
    names = sorted(p.name for p in written if p.suffix == ".png")
    assert names == ["wood.png", "wood_1.png"]
    assert {(tmp_path / n).read_bytes()[-1:] for n in names} == {b"A", b"B"}


@pytest.mark.parametrize("fmt", ["obj", "3ds", "usda"])
def test_saving_never_overwrites_a_different_file_beside_it(tmp_path, fmt):
    """A picture of the user's own that happens to share a texture's name is left alone."""
    mine = tmp_path / ("textures" if fmt == "usda" else ".")
    mine.mkdir(exist_ok=True)
    (mine / "wood.png").write_bytes(b"the user's own picture")
    written = meshport.save(_textured(("wood.png", PNG + b"A" * 20)), tmp_path / f"o.{fmt}")
    assert (mine / "wood.png").read_bytes() == b"the user's own picture"
    assert any(p.suffix == ".png" and p.read_bytes().startswith(PNG) for p in written)
    again = meshport.save(_textured(("wood.png", PNG + b"A" * 20)), tmp_path / f"o.{fmt}")  # same bytes: same names
    assert sorted(p.name for p in again) == sorted(p.name for p in written)


def test_obj_keeps_world_space_points_far_from_the_origin(tmp_path):
    mesh = Mesh("far", np.array([[12345.678, 0.5, -98765.4321], [1, 0, 0], [0, 1, 0]]), np.array([3]), np.arange(3))
    meshport.save(Scene([Node("far", mesh=mesh)]), tmp_path / "o.obj")
    back = meshport.load(tmp_path / "o.obj")
    assert np.allclose(back.meshes[0].positions[0], [12345.678, 0.5, -98765.4321], rtol=0, atol=1e-4)


# ---- what files point at

def test_usd_reference_into_a_package_on_disk(tmp_path):
    """@asset.usdz[model.usdc]@ never resolved ("No 3D geometry found")."""
    (tmp_path / "asset.usdz").write_bytes(meshport.write(Scene([Node("tri", mesh=_tri())]), "usdz"))
    (tmp_path / "top.usda").write_text('#usda 1.0\n\ndef Xform "W" (\n    references = @./asset.usdz[model.usdc]@\n)\n{\n}\n')
    assert len(meshport.load(tmp_path / "top.usda").meshes) == 1


def test_usd_package_paths_keep_their_dots():
    """"../tex.png" from a layer in sub/ was read as sub/tex.png, and ".hidden/x" lost its dot."""
    src = _Source(None, object(), "sub")
    assert src._inner("../tex.png") == "tex.png"
    assert src._inner(".hidden/x.png") == "sub/.hidden/x.png"
    assert src._inner("./a/./b.png") == "sub/a/b.png"


def test_gltf_images_keep_to_the_folder_rule(tmp_path):
    """Images were read from anywhere relative to the model (../../../Pictures/x.jpg); now, as every
    other reader has it, from its folder and the one above."""
    import json
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (tmp_path / "private.png").write_bytes(PNG + b"secret" * 10)
    (deep.parent / "near.png").write_bytes(PNG + b"fine" * 10)
    doc = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "buffers": [{"uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"}],
           "bufferViews": [{"buffer": 0, "byteLength": 36}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
                         {"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}}],
           "textures": [{"source": 0}, {"source": 1}], "images": [{"uri": "../../../private.png"}, {"uri": "../near.png"}]}
    (deep / "m.gltf").write_text(json.dumps(doc))
    mat = meshport.load(deep / "m.gltf").meshes and meshport.load(deep / "m.gltf").roots[0].materials[0]
    assert mat.base_color_texture.data is None  # three folders up: not read
    doc["meshes"][0]["primitives"][0]["material"] = 1
    (deep / "m.gltf").write_text(json.dumps(doc))
    assert meshport.load(deep / "m.gltf").roots[0].materials[0].base_color_texture.data.startswith(PNG)


def test_gltf_node_weights_stay_on_their_own_node():
    """Two nodes showing one mesh: the last node's own weights were written onto the shared mesh."""
    import json
    import struct as st
    pts = st.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    import base64
    buf = base64.b64encode(pts + pts).decode()
    doc = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0, 1]}],
           "nodes": [{"mesh": 0}, {"mesh": 0, "weights": [0.75]}],
           "buffers": [{"uri": "data:application/octet-stream;base64," + buf, "byteLength": 72}],
           "bufferViews": [{"buffer": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 36}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                         {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "targets": [{"POSITION": 1}]}], "weights": [0.25]}]}
    scene = meshport.read(json.dumps(doc).encode(), "gltf")
    a, b = scene.roots
    assert np.allclose(a.mesh.target_weights, [0.25]) and np.allclose(b.mesh.target_weights, [0.75])


def test_ascii_fbx_numbers_that_arent_finite_read_as_numbers():
    """inf and nan were read as words, which shifted every value after them."""
    text = write_ascii([Record("X", [1.5, float("inf"), float("nan"), -float("inf"), "after", 2])])
    props = _read_ascii(text)[1][0].props
    assert props[0] == 1.5 and props[1] == np.inf and np.isnan(props[2]) and props[3] == -np.inf
    assert props[4:] == ["after", 2]


def test_deformed_scenes_keep_their_lights():
    lamp = Node("lamp", light=Light("lamp", "point"))
    shaped = cube()
    shaped.targets = [meshport.Morph("up", np.tile([0, 1.0, 0], (8, 1)))]  # something to deform
    flat = Scene([lamp, Node("mesh", mesh=shaped)]).deformed()
    assert next(n for n, _ in flat.walk() if n.name == "lamp").light is not None


def test_connected_weld_of_a_long_chain_in_scrambled_order_is_quick():
    import time

    from meshport import blendmods
    n = 60_000  # a strip of tiny triangles whose points are numbered at random: one chain of short edges
    order = np.random.default_rng(3).permutation(n)
    tris = np.stack([order[:-2], order[1:-1], order[2:]], 1)
    mesh = Mesh("strip", np.zeros((n, 3)) + np.arange(n)[order.argsort()][:, None] * 1e-6, np.full(len(tris), 3),
                tris.ravel())
    started = time.perf_counter()
    welded, _ = blendmods._weld(mesh, np.ones(len(tris), bool), {"mode": 1, "merge_dist": 0.001}, None)
    assert time.perf_counter() - started < 2.0  # about 5 s when each pass moved a label one step along the chain
    assert len(welded.face_sizes) == 0  # one point, so every triangle collapsed
