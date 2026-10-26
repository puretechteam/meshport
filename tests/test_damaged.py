"""Damaged or hostile files must only ever raise MeshportError, never crash or hang."""

import random
import struct
import zlib

import numpy as np
import pytest
from test_3ds import _keyed_file, _kf_node, _mesh_object

import meshport
from meshport import MeshportError, fbx, max3ds
from meshport.fbxtree import Record, write_ascii, write_binary


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
