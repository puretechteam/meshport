"""Small numeric helpers: transforms and polygon triangulation."""

from __future__ import annotations

import numpy as np


def translation(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def scaling(s) -> np.ndarray:
    return np.diag([s[0], s[1], s[2], 1.0])


def _axis_rotation(axis: int, degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m = np.eye(4)
    m[i, i] = m[j, j] = c
    m[i, j], m[j, i] = -s, s
    return m


# FBX rotation orders, by enum value: the first letter is applied first.
EULER_ORDERS = ["XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX"]


def euler(degrees, order: str = "XYZ") -> np.ndarray:
    m = np.eye(4)
    for letter in order:
        axis = "XYZ".index(letter)
        m = _axis_rotation(axis, degrees[axis]) @ m
    return m


def compose(t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1)) -> np.ndarray:
    return translation(t) @ euler(r) @ scaling(s)


def decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 4x4 matrix into translation, XYZ euler degrees and scale (shear is dropped)."""
    t = m[:3, 3].copy()
    basis = m[:3, :3]
    s = np.linalg.norm(basis, axis=0)
    s[s == 0] = 1.0
    if np.linalg.det(basis) < 0:
        s[0] = -s[0]
    r = basis / s
    # r = Rz @ Ry @ Rx
    sy = -r[2, 0]
    if abs(sy) < 0.999999:
        y = np.arcsin(sy)
        x = np.arctan2(r[2, 1], r[2, 2])
        z = np.arctan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: fold z into x
        y = np.copysign(np.pi / 2, sy)
        x = np.arctan2(-r[1, 2], r[1, 1])
        z = 0.0
    return t, np.degrees([x, y, z]), s


def transform_points(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ m[:3, :3].T + m[:3, 3]


def transform_normals(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    inv_t = np.linalg.inv(m[:3, :3]).T
    out = n @ inv_t.T
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, length, out=np.zeros_like(out), where=length > 0)


def unique_rows(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index of each distinct row's first appearance, row -> distinct-row number), with the
    distinct rows numbered in order of first appearance. Rows compare by their bytes, which
    is much faster than np.unique(axis=0)."""
    table = np.ascontiguousarray(table)
    rows = table.view(np.dtype((np.void, table.dtype.itemsize * table.shape[1]))).ravel()
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return first[order], rank[inverse.ravel()]


# ---- triangulation

def triangulate(positions: np.ndarray, face_sizes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """(T, 3) corner indices, face by face, n-2 triangles per n-gon.

    Convex polygons are fanned (fast, vectorized); concave ones are ear-clipped.
    """
    sizes = np.asarray(face_sizes, dtype=np.int64)
    if not len(sizes):
        return np.zeros((0, 3), np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    per_face = sizes - 2
    tri_face = np.repeat(np.arange(len(sizes)), per_face)
    first_tri = np.concatenate(([0], np.cumsum(per_face)[:-1]))
    k = np.arange(len(tri_face)) - first_tri[tri_face] + 1
    base = starts[tri_face]
    tris = np.stack([base, base + k, base + k + 1], axis=1)

    for f in _concave_faces(positions, sizes, starts, corners):
        s, n = starts[f], sizes[f]
        ears = _ear_clip(positions[corners[s:s + n]])
        if ears is not None:
            tris[first_tri[f]:first_tri[f] + n - 2] = ears + s
    return tris


def _concave_faces(positions, sizes, starts, corners) -> np.ndarray:
    big = np.nonzero(sizes >= 4)[0]
    if not len(big):
        return big
    n = sizes[big]
    face = np.repeat(np.arange(len(big)), n)
    first = np.repeat(starts[big], n)
    local = np.arange(int(n.sum())) - np.repeat(np.concatenate(([0], np.cumsum(n)[:-1])), n)
    size = np.repeat(n, n)
    cur = first + local
    nxt = first + (local + 1) % size
    prv = first + (local - 1) % size
    p, pn, pp = positions[corners[cur]], positions[corners[nxt]], positions[corners[prv]]
    # Newell normal per face
    contrib = np.cross(p, pn)
    normal = np.zeros((len(big), 3))
    np.add.at(normal, face, contrib)
    turn = np.einsum("ij,ij->i", np.cross(p - pp, pn - p), normal[face])
    scale = np.linalg.norm(normal[face], axis=1) * 1e-9
    return big[np.unique(face[turn < -scale])]


def _ear_clip(points: np.ndarray) -> np.ndarray | None:
    """Triangulate one simple polygon given as (n, 3) points. None if it is degenerate."""
    normal = np.cross(points, np.roll(points, -1, axis=0)).sum(0)
    if not normal.any():
        return None
    drop = int(np.argmax(np.abs(normal)))
    keep = [a for a in range(3) if a != drop]
    p2 = points[:, keep]
    if normal[drop] < 0:  # make the 2D polygon counter-clockwise
        p2 = p2[:, ::-1]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    idx = list(range(len(points)))
    out = []
    guard = 0
    while len(idx) > 3 and guard < 4 * len(points) ** 2:
        guard += 1
        m = len(idx)
        for i in range(m):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % m]
            if cross(p2[a], p2[b], p2[c]) <= 0:
                continue
            inside = False
            for j in idx:
                if j in (a, b, c):
                    continue
                pt = p2[j]
                if cross(p2[a], p2[b], pt) >= 0 and cross(p2[b], p2[c], pt) >= 0 and cross(p2[c], p2[a], pt) >= 0:
                    inside = True
                    break
            if not inside:
                out.append((a, b, c))
                idx.pop(i)
                break
        else:
            return None
    if len(idx) != 3:
        return None
    out.append(tuple(idx))
    return np.array(out, dtype=np.int64)
