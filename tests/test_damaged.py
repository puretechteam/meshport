"""Damaged or hostile files must only ever raise MeshportError, never crash or hang."""

import random
import struct
import zlib

import numpy as np
import pytest
from conftest import nodes_by_name, png
from test_3ds import _keyed_file, _kf_node, _mesh_object

import meshport
from meshport import MeshportError, fbx, max3ds
from meshport.fbxtree import Record, write_ascii, write_binary
from meshport.scene import is_remote_path


def _keyed_3ds():
    """A 3DS file with a keyframer: a camera, a dummy group, an instance and a pivot."""
    objects = [_mesh_object("a", np.eye(4)), _mesh_object("b", np.eye(4))]
    nodes = [_kf_node("cam", 0, kind=max3ds.CAMERA_NODE, pos=(0, 0, 0)), _kf_node("cam", 1, kind=max3ds.TARGET_NODE,
             pos=(0, 5, 0)), _kf_node(max3ds.DUMMY, 2, 0, pos=(1, 2, 3), instance="g"),
             _kf_node("a", 3, 2, pos=(1, 0, 0), rot=(0.5, 0, 0, 1), scale=(1, 2, 1), pivot=(0.5, 0, 0)),
             _kf_node("a", 4, 3, pos=(0, 1, 0), instance="copy"), _kf_node("b", 5, 4, pos=(0, 0, 1))]
    return _keyed_file(objects, nodes)


def _samples(scene):
    return {"fbx": meshport.write(scene, "fbx"), "glb": meshport.write(scene, "glb"),
            "gltf": meshport.write(scene, "gltf"), "3ds": meshport.write(scene, "3ds"), "3ds-keyframer": _keyed_3ds(),
            "ascii": write_ascii(fbx._Writer(scene).records()).encode()}


@pytest.mark.parametrize("kind", ["fbx", "glb", "gltf", "ascii", "3ds", "3ds-keyframer"])
def test_corrupted_files_raise_meshport_error(scene, kind):
    data = _samples(scene)[kind]
    fmt = {"ascii": "fbx", "3ds-keyframer": "3ds"}.get(kind, kind)
    rng = random.Random(kind)
    for _ in range(300):
        b = bytearray(data)
        for _ in range(rng.randint(1, 8)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        if rng.random() < 0.2:
            b = b[:rng.randrange(len(b))]
        try:
            meshport.read(bytes(b), fmt)
        except MeshportError:
            pass


def test_zip_bomb_array_is_not_inflated():
    # An array that claims 3 doubles but whose zlib stream inflates to 100 MB.
    bomb = zlib.compress(bytes(100_000_000), 9)
    header = write_binary([])[:27]
    # a hand-made record holding one array property: type d, length 3, zlib-encoded
    body = b"d" + struct.pack("<III", 3, 1, len(bomb)) + bomb
    head = struct.Struct("<IIIB")
    name = b"Vertices"
    record = head.pack(27 + head.size + len(name) + len(body), 1, len(body), len(name)) + name + body
    data = header + record + bytes(head.size)
    _, (back,) = meshport.fbxtree.parse(data)
    np.testing.assert_array_equal(back.props[0], [0.0, 0.0, 0.0])  # only what was declared


def test_deep_nesting_is_an_error_not_a_crash():
    rec = Record("Deep")
    for _ in range(3000):
        rec = Record("Deep", children=[rec])
    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)  # enough to write it; the reader runs at the normal limit
    try:
        data = write_binary([Record("Objects", children=[rec])])
    finally:
        sys.setrecursionlimit(old)
    with pytest.raises(MeshportError):
        fbx.read(data)


def _3ds(*edit: bytes) -> bytes:
    return max3ds._chunk(max3ds.MAIN, max3ds._chunk(max3ds.EDIT, *edit))


def _3ds_mesh(*parts: bytes) -> bytes:
    return _3ds(max3ds._chunk(max3ds.OBJECT, b"x\0", max3ds._chunk(max3ds.TRIMESH, *parts)))


@pytest.mark.parametrize("data", [
    # a chunk shorter than its own header, which would loop forever if trusted
    b"MM" + struct.pack("<I", 18) + b"==" + struct.pack("<I", 0) + bytes(6),
    # a chunk that claims more bytes than its parent holds
    _3ds(struct.pack("<HI", max3ds.OBJECT, 1000) + b"x\0"),
    # 65,535 points claimed, 1 sent
    _3ds_mesh(max3ds._chunk(max3ds.POINT_ARRAY, struct.pack("<H3f", 65535, 0, 0, 0))),
    # 65,535 faces claimed, none sent
    _3ds_mesh(max3ds._chunk(max3ds.POINT_ARRAY, struct.pack("<H", 0)),
              max3ds._chunk(max3ds.FACE_ARRAY, struct.pack("<H", 65535))),
    # a name with no end
    _3ds(max3ds._chunk(max3ds.OBJECT, b"no terminator")),
])
def test_hostile_3ds_raises_meshport_error(data):
    with pytest.raises(MeshportError):
        meshport.read(data, "3ds")


def test_3ds_animation_key_cut_short():
    track = max3ds._chunk(max3ds.POS_TRACK, struct.pack("<H8xI", 0, 1))  # claims a key, then stops
    data = _keyed_file([_mesh_object("a", np.eye(4))], [max3ds._chunk(max3ds.OBJECT_NODE, track)])
    with pytest.raises(MeshportError, match="animation key is cut short"):
        meshport.read(data, "3ds")


def test_3ds_odd_but_readable_parts_are_tolerated():
    points = max3ds._chunk(max3ds.POINT_ARRAY, struct.pack("<H", 3), np.eye(3, dtype="<f4").tobytes())
    group = max3ds._chunk(max3ds.MSH_MAT_GROUP, b"nowhere\0", struct.pack("<HHH", 2, 0, 40000))  # face 40000 doesn't exist
    smooth = max3ds._chunk(max3ds.SMOOTH_GROUP, b"")  # fewer smoothing entries than faces
    faces = max3ds._chunk(max3ds.FACE_ARRAY, struct.pack("<H4H", 1, 0, 1, 2, 0), group, smooth)
    uvs = max3ds._chunk(max3ds.TEX_VERTS, struct.pack("<H2f", 1, 0, 0))  # fewer UVs than points
    singular = max3ds._chunk(max3ds.MESH_MATRIX, bytes(48))
    scene = meshport.read(_3ds_mesh(points, uvs, singular, faces), "3ds")
    mesh = scene.roots[0].mesh
    assert len(mesh.face_sizes) == 1 and mesh.uvs == [] and mesh.normals is not None
    assert scene.roots[0].materials[0].name == "nowhere"


def test_3ds_many_smoothing_masks_stay_fast():
    rng = np.random.default_rng(0)
    n = 20_000
    p = rng.random((n + 2, 3)).astype("<f4")
    tris = np.stack([np.arange(n), np.arange(n) + 1, np.arange(n) + 2, np.zeros(n)], 1).astype("<u2")
    masks = rng.integers(1, 2**32, n, dtype=np.uint64).astype("<u4")  # every face its own mix of groups
    data = _3ds_mesh(max3ds._chunk(max3ds.POINT_ARRAY, struct.pack("<H", n + 2), p.tobytes()),
                     max3ds._chunk(max3ds.FACE_ARRAY, struct.pack("<H", n), tris.tobytes(),
                                   max3ds._chunk(max3ds.SMOOTH_GROUP, masks.tobytes())))
    assert meshport.read(data, "3ds").roots[0].mesh.normals.shape == (3 * n, 3)


# ---- texture paths: a model names its textures by path, and a hostile one can name anything

SECRET = b"not an image: private notes"


def _with_texture(scene, path):
    mat = nodes_by_name(scene)["box"][0].materials[0]
    mat.base_color_texture = meshport.Image("tex", None, path)
    return scene


@pytest.mark.parametrize("how", ["absolute", "relative"])
def test_texture_path_cant_pull_in_other_files(tmp_path, scene, how):
    (tmp_path / "private").mkdir()
    (tmp_path / "models").mkdir()
    secret = tmp_path / "private" / "notes.txt"
    secret.write_bytes(SECRET)
    path = secret.as_posix() if how == "absolute" else "../private/notes.txt"
    src = tmp_path / "models" / "box.fbx"
    src.write_bytes(fbx.write(_with_texture(scene, path)))
    back = meshport.load(src)
    assert nodes_by_name(back)["box"][0].materials[0].base_color_texture.data is None
    (tmp_path / "out").mkdir()
    for ext in ("fbx", "3ds", "obj", "glb"):
        for f in meshport.save(back, tmp_path / "out" / f"box.{ext}"):
            assert SECRET not in f.read_bytes()


def test_texture_at_an_absolute_path_still_loads(tmp_path, scene):
    tex = tmp_path / "elsewhere" / "wood.png"
    tex.parent.mkdir()
    tex.write_bytes(png())
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "box.fbx").write_bytes(fbx.write(_with_texture(scene, str(tex))))
    back = meshport.load(tmp_path / "m" / "box.fbx")
    assert nodes_by_name(back)["box"][0].materials[0].base_color_texture.data == png()


def test_network_texture_paths_are_never_opened(tmp_path, scene, monkeypatch):
    opened = []
    real = fbx.read_image_file
    monkeypatch.setattr(fbx, "read_image_file", lambda p: opened.append(str(p)) or real(p))
    (tmp_path / "box.fbx").write_bytes(fbx.write(_with_texture(scene, r"\\attacker.example\share\wood.png")))
    meshport.load(tmp_path / "box.fbx")
    assert opened and not any(p.replace("\\", "/").startswith("//") for p in opened)
    assert is_remote_path(r"\\server\share\a.png") and is_remote_path("//server/a.png")
    assert is_remote_path("https://example.com/a.png") and not is_remote_path("C:/textures/a.png")


def _gltf_with(image_uri=None, buffer_uri="data.bin"):
    images = {"materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
              "textures": [{"source": 0}], "images": [{"uri": image_uri}]} if image_uri else {}
    prim = {"attributes": {"POSITION": 0}, **({"material": 0} if image_uri else {})}
    return {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [prim]}], "buffers": [{"byteLength": 36, "uri": buffer_uri}],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}], **images}


def _gltf_folder(tmp_path, doc):
    import json
    (tmp_path / "private").mkdir(exist_ok=True)
    (tmp_path / "private" / "notes.txt").write_bytes(SECRET)
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "data.bin").write_bytes(np.eye(3, dtype=np.float32).tobytes())
    (tmp_path / "model" / "m.gltf").write_text(json.dumps(doc))
    return tmp_path / "model" / "m.gltf"


@pytest.mark.parametrize("uri", ["../private/notes.txt", "ABSOLUTE", "file:///C:/Windows/win.ini",
                                 "//attacker.example/share/a.png"])
def test_gltf_image_uri_cant_pull_in_other_files(tmp_path, uri):
    if uri == "ABSOLUTE":
        uri = (tmp_path / "private" / "notes.txt").as_posix()
    scene = meshport.load(_gltf_folder(tmp_path, _gltf_with(uri)))
    assert scene.roots[0].materials[0].base_color_texture.data is None


def test_gltf_buffer_must_stay_in_its_folder(tmp_path):
    (tmp_path / "outside.bin").write_bytes(np.eye(3, dtype=np.float32).tobytes())
    with pytest.raises(MeshportError, match="outside its folder"):
        meshport.load(_gltf_folder(tmp_path, _gltf_with(buffer_uri="../outside.bin")))
    with pytest.raises(MeshportError, match="relative"):
        meshport.load(_gltf_folder(tmp_path, _gltf_with(buffer_uri=(tmp_path / "outside.bin").as_posix())))


def test_gltf_tga_texture_file_is_accepted(tmp_path):
    import io

    from PIL import Image as PILImage
    out = io.BytesIO()
    PILImage.new("RGB", (4, 4), (0, 0, 255)).save(out, "TGA")
    path = _gltf_folder(tmp_path, _gltf_with("wood.tga"))
    (path.parent / "wood.tga").write_bytes(out.getvalue())
    assert meshport.load(path).roots[0].materials[0].base_color_texture.data == out.getvalue()
