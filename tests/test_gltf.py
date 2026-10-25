import base64
import json
import struct

import numpy as np
import pytest
from conftest import nodes_by_name, png, world_triangles

import meshport
from meshport import MeshportError, gltf


def test_glb_round_trip(scene):
    back = gltf.read(gltf.write(scene))
    assert world_triangles(back) == world_triangles(scene)
    got = nodes_by_name(back)
    box = got["box"][0]
    assert [m.name for m in box.materials] == ["red", "blue"]
    assert box.materials[0].base_color_texture.data == nodes_by_name(scene)["box"][0].materials[0].base_color_texture.data
    assert box.materials[1].alpha_mode == "BLEND"
    assert box.mesh is got["box_mirror"][0].mesh
    assert len(box.mesh.uvs) == 2 and box.mesh.colors is not None


def test_uvs_survive_the_v_flip(scene):
    back = gltf.read(gltf.write(scene))
    src, dst = nodes_by_name(scene)["box"][0].mesh, nodes_by_name(back)["box"][0].mesh
    want = {tuple(np.round(np.r_[src.positions[src.corners[i]], src.uvs[1][i]], 5)) for i in range(len(src.corners))}
    got = {tuple(np.round(np.r_[dst.positions[dst.corners[i]], dst.uvs[1][i]], 5)) for i in range(len(dst.corners))}
    assert got == want


def test_gltf_text_is_self_contained(scene, tmp_path):
    path = tmp_path / "a.gltf"
    gltf.write_file(scene, path)
    doc = json.loads(path.read_text())
    assert doc["buffers"][0]["uri"].startswith("data:")
    assert world_triangles(meshport.load(path)) == world_triangles(scene)


def _doc(**extra):
    return {"asset": {"version": "2.0"}, **extra}


def _glb(doc: dict, blob: bytes) -> bytes:
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_interleaved_strip_and_sparse():
    # 4 vertices interleaved as position + uv (20 bytes each), drawn as a triangle strip,
    # with a sparse accessor moving vertex 3 up.
    verts = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 1, 0, 1, 1]], np.float32)
    sparse_idx = np.array([3], np.uint16).tobytes() + b"\0\0"
    sparse_val = np.array([1, 1, 5], np.float32).tobytes()
    blob = verts.tobytes() + sparse_idx + sparse_val
    doc = _doc(
        scenes=[{"nodes": [0]}], nodes=[{"mesh": 0}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "mode": 5}]}],
        buffers=[{"byteLength": len(blob)}],
        bufferViews=[{"buffer": 0, "byteOffset": 0, "byteLength": 80, "byteStride": 20},
                     {"buffer": 0, "byteOffset": 80, "byteLength": 4}, {"buffer": 0, "byteOffset": 84, "byteLength": 12}],
        accessors=[{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                    "sparse": {"count": 1, "indices": {"bufferView": 1, "componentType": 5123},
                               "values": {"bufferView": 2}}},
                   {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 4, "type": "VEC2"}])
    mesh = gltf.read(_glb(doc, blob)).roots[0].mesh
    np.testing.assert_allclose(mesh.positions[3], [1, 1, 5])
    tris = mesh.corners.reshape(-1, 3)
    np.testing.assert_array_equal(tris, [[0, 1, 2], [2, 1, 3]])  # second strip triangle flipped back
    np.testing.assert_allclose(mesh.uvs[0][2], [0, 0])  # v=1 at the top in glTF is v=0 here


def test_external_buffer_and_missing_file(tmp_path):
    blob = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32).tobytes()
    (tmp_path / "data.bin").write_bytes(blob)
    doc = _doc(scenes=[{"nodes": [0]}], nodes=[{"mesh": 0}],
               meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
               buffers=[{"byteLength": 36, "uri": "data.bin"}],
               bufferViews=[{"buffer": 0, "byteLength": 36}],
               accessors=[{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}])
    (tmp_path / "m.gltf").write_text(json.dumps(doc))
    assert len(meshport.load(tmp_path / "m.gltf").meshes[0].positions) == 3
    (tmp_path / "data.bin").unlink()
    with pytest.raises(MeshportError, match="data.bin"):
        meshport.load(tmp_path / "m.gltf")


def test_draco_is_refused_clearly():
    doc = _doc(extensionsRequired=["KHR_draco_mesh_compression"])
    with pytest.raises(MeshportError, match="Draco"):
        gltf.read(json.dumps(doc).encode())


def test_non_png_texture_becomes_png(scene):
    from io import BytesIO

    from PIL import Image as PILImage
    out = BytesIO()
    PILImage.new("RGB", (4, 4), (9, 9, 9)).save(out, "TGA")
    mat = nodes_by_name(scene)["box"][0].materials[0]
    mat.base_color_texture = meshport.Image("tga", out.getvalue())
    back = gltf.read(gltf.write(scene))
    assert nodes_by_name(back)["box"][0].materials[0].base_color_texture.data.startswith(b"\x89PNG")


def test_data_uri_image():
    blob = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32).tobytes()
    uri = "data:image/png;base64," + base64.b64encode(png()).decode()
    doc = _doc(scenes=[{"nodes": [0]}], nodes=[{"mesh": 0}],
               meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
               materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
               textures=[{"source": 0}], images=[{"uri": uri}],
               buffers=[{"byteLength": 36}], bufferViews=[{"buffer": 0, "byteLength": 36}],
               accessors=[{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}])
    node = gltf.read(_glb(doc, blob)).roots[0]
    assert node.materials[0].base_color_texture.data == png()
