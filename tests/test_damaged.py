"""Damaged or hostile files must only ever raise MeshportError, never crash or hang."""

import random
import struct
import zlib

import numpy as np
import pytest

import meshport
from meshport import MeshportError, fbx
from meshport.fbxtree import Record, write_ascii, write_binary


def _samples(scene):
    return {"fbx": meshport.write(scene, "fbx"), "glb": meshport.write(scene, "glb"),
            "gltf": meshport.write(scene, "gltf"),
            "ascii": write_ascii(fbx._Writer(scene).records()).encode()}


@pytest.mark.parametrize("kind", ["fbx", "glb", "gltf", "ascii"])
def test_corrupted_files_raise_meshport_error(scene, kind):
    data = _samples(scene)[kind]
    fmt = "fbx" if kind == "ascii" else kind
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
