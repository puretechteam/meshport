"""Real-world .3ds files from the folder named by MESHPORT_3DS_CORPUS (skipped when it's unset).

The files aren't in the repository because of their licenses. Assimp's sample models
(test/models/3DS and test/models-nonbsd/3DS at github.com/assimp/assimp) are a good set:

    MESHPORT_3DS_CORPUS=path/to/folder pytest tests/test_corpus_3ds.py
"""

import os
import struct
from pathlib import Path

import numpy as np
import pytest

import meshport
from meshport import max3ds
from meshport.max3ds import _chunks, _counted, _cstring

_FOLDER = os.environ.get("MESHPORT_3DS_CORPUS")
FILES = sorted(p for p in Path(_FOLDER).rglob("*") if p.suffix.lower() == ".3ds") if _FOLDER else []


def _triangles(scene) -> int:
    return sum(len(n.mesh.triangles()) for n, _ in scene.walk() if n.mesh is not None)


def _gap(a, b) -> float:
    from scipy.spatial import cKDTree

    def points(s):
        return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in s.walk() if n.mesh is not None])
    pa, pb = points(a), points(b)
    return max(cKDTree(pb).query(pa)[0].max(), cKDTree(pa).query(pb)[0].max()) / max(np.ptp(pa, 0).max(), 1.0)


def _smoothing_by_the_rule(pts, faces, groups):
    """The smoothing-group rule read literally: a corner sums the unit normals of the faces
    around its vertex that share a group bit with its own face; a face in no group is flat."""
    unit, users = [], {}
    for i, face in enumerate(faces):
        p = pts[face]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        unit.append(n / (np.linalg.norm(n) or 1))
        for v in face:
            users.setdefault(v, []).append(i)
    out = []
    for i, face in enumerate(faces):
        for v in face:
            s = unit[i] if not groups[i] else sum((unit[j] for j in users[v] if groups[j] & groups[i]), np.zeros(3))
            out.append(s / (np.linalg.norm(s) or 1))
    return np.array(out)


def _smoothed_meshes(data):
    """(points, faces, groups) of each mesh that has smoothing groups, straight from the chunks."""
    found = []

    def walk(s, e):
        for cid, cs, ce in _chunks(data, s, e):
            if cid in (max3ds.MAIN, max3ds.EDIT):
                walk(cs, ce)
            elif cid == max3ds.OBJECT:
                walk(_cstring(data, cs, ce)[1], ce)
            elif cid == max3ds.TRIMESH:
                pts = faces = groups = None
                for c2, s2, e2 in _chunks(data, cs, ce):
                    if c2 == max3ds.POINT_ARRAY:
                        pts = _counted(data, s2, e2, "<f4", 3)[0].astype(float)
                    elif c2 == max3ds.FACE_ARRAY:
                        f, pos = _counted(data, s2, e2, "<u2", 4)
                        faces = f[:, :3].astype(int)
                        for c3, s3, _ in _chunks(data, pos, e2):
                            if c3 == max3ds.SMOOTH_GROUP:
                                groups = np.frombuffer(data, "<u4", len(faces), s3).astype(np.int64)
                if groups is not None:
                    found.append((pts, faces, groups))
    walk(0, min(struct.unpack_from("<I", data, 2)[0], len(data)))
    return found


@pytest.mark.skipif(not FILES, reason="set MESHPORT_3DS_CORPUS to a folder of .3ds files")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_real_3ds_file(path):
    pytest.importorskip("scipy")
    data = path.read_bytes()
    scene = meshport.load(path)
    assert scene.meshes

    # Re-saved in every format, nothing moves and no triangle is lost.
    for fmt in ("3ds", "glb", "fbx"):
        back = meshport.read(meshport.write(scene, fmt), fmt)
        assert _triangles(back) == _triangles(scene), fmt
        assert _gap(back, scene) < 1e-5, fmt

    # Saved on its first frame, the file's stored geometry is the pose meshport must give.
    reader = max3ds._Reader(data, None)
    reader.scene()
    if reader.saved_frame is not None and reader.saved_frame in (reader.segment_start, 160 * reader.segment_start):
        end = min(struct.unpack_from("<I", data, 2)[0], len(data))
        stored = max3ds.read(max3ds._chunk(max3ds.MAIN, *[data[s - 6:e] for cid, s, e in _chunks(data, 6, end)
                                                          if cid != max3ds.KFDATA]))
        assert _gap(scene, stored) < 1e-5

    # Normals follow the smoothing groups exactly.
    for pts, faces, groups in _smoothed_meshes(data):
        if len(faces) <= 20_000:
            want = _smoothing_by_the_rule(pts, faces, groups)
            np.testing.assert_allclose(max3ds._smooth_normals(pts, faces, groups), want, atol=1e-9)
