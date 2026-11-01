"""Blender's modifiers, applied to the meshes blend.py reads, the way Blender's viewport shows them.

Every modifier on an object is a struct of its own kind (MirrorModifierData, ArrayModifierData,
...) in the object's modifier list, led by a ModifierData header with its type and flags. Like
everything else in a .blend file, the settings are found by name through the file's struct
catalogue. The modifiers enabled in the viewport are applied in stack order; kinds meshport
doesn't know are skipped and the rest still applied.

- Triangulate: every quad-splitting method, n-gons by ear clipping (Blender's "Beauty" n-gon
  split may pick other diagonals; the points are the same).
- Weld: "All" and "Connected", merged points moved to their average, as Blender does.
- Mirror: any axes, merging along the seam (to the midpoint), bisecting, a mirror object, and
  flipped and offset UVs.
- Array: fixed count or fit length, relative, constant and object offsets, merging (also of the
  last copy with the first), UV offsets, and start and end caps (merged with the ends too).
  Fit-to-curve isn't applied.
- Solidify: simple mode with even thickness, rims, flipped normals, rim-only, and material
  offsets. Complex mode is approximated by simple mode with even thickness.
- Subdivision Surface: Catmull-Clark or simple at the viewport level, boundaries kept sharp (with
  "Keep Corners" or smoothed corners) and points pushed to the limit surface as Blender does.
  UVs are smoothed as a mesh of their own (split along seams), with its boundaries kept as
  the UV Smooth setting says; colors are interpolated linearly. Edge and vertex creases (Blender
  4.0 on) are OpenSubdiv's semi-sharp creases, resolved to the Quality level as Blender does.

Vertex groups (which can scale some of these) are ignored.
"""

from __future__ import annotations

import numpy as np

from .geometry import transform_points, translation, triangulate, unique_rows
from .scene import Budget, Mesh

MAX_GENERATED = 100_000_000  # corners one .blend file's modifiers and curves may make in all

# ModifierData.mode
_REALTIME = 1
# MirrorModifierData.flag
_MIR_NO_MERGE, _MIR_U, _MIR_V, _MIR_AXIS_X, _MIR_BISECT_X, _MIR_FLIP_X, _MIR_UDIM = 128, 2, 4, 8, 256, 2048, 16384
# ArrayModifierData
_ARR_CONST, _ARR_RELATIVE, _ARR_OBJECT = 1, 2, 4
_ARR_MERGE, _ARR_MERGE_FINAL = 1, 2
_ARR_FIT_LENGTH = 1
# SolidifyModifierData.flag
_SOL_RIM, _SOL_EVEN, _SOL_FLIP, _SOL_NO_SHELL = 1, 2, 32, 64
# SubsurfModifierData
_SUB_NO_LIMIT, _SUB_USE_CREASE = 64, 16
_SUB_SIMPLE = 1
_SUB_KEEP_CORNERS = 1
# SubsurfModifierData.uv_smooth: none (linear), keep corners, ... keep boundaries (the default), all
_UV_SMOOTH_NONE, _UV_KEEP_CORNERS, _UV_KEEP_BOUNDARIES, _UV_SMOOTH_ALL = 0, 1, 4, 5
_FLT_EPSILON = 1.1920929e-07


class Context:
    """What modifiers need from outside the mesh: the object's world matrix (Blender's axes),
    other objects' world matrices and meshes (modifiers applied, as (mesh, smooth) or None) by
    address, and the number of material slots."""

    def __init__(self, ob_world: np.ndarray, world_of, n_materials: int, budget: Budget | None = None,
                 mesh_of=None):
        self.ob_world, self.world_of, self.n_materials = ob_world, world_of, n_materials
        self.budget = budget or Budget(MAX_GENERATED, "This mesh is too big once its modifiers are applied")
        self.mesh_of = mesh_of or (lambda address: None)

    def relative(self, address: int) -> np.ndarray | None:
        """Another object's matrix in this object's space, or None if it isn't there."""
        world = self.world_of(address) if address else None
        return None if world is None else np.linalg.inv(self.ob_world) @ world


def stack(ob) -> list:
    """The object's modifiers that are on in the viewport, in order."""
    out = []
    for md in ob.listbase("modifiers"):
        head = md.get("modifier")
        if head is not None and int(head.get("mode", 0)) & _REALTIME:
            out.append(md)
    return out


def apply(mesh: Mesh, smooth: np.ndarray, modifiers: list, ctx: Context) -> tuple[Mesh, np.ndarray]:
    """The mesh (without normals) and per-face smooth flags after the modifiers."""
    for md in modifiers:
        fn = _APPLY.get(md.name)
        if fn is None or not len(mesh.face_sizes):
            continue
        mesh, smooth = fn(mesh, np.asarray(smooth, bool), md, ctx)
    return mesh, smooth


# ============================================================================ shared helpers

def _topology(sizes: np.ndarray):
    """For each corner: its face, the next corner and the previous corner in that face."""
    sizes = np.asarray(sizes, np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    face = np.repeat(np.arange(len(sizes)), sizes)
    c = np.arange(int(sizes.sum()))
    last = starts + sizes - 1
    nxt, prv = c + 1, c - 1
    nxt[last] = starts
    prv[starts] = last
    return face, nxt, prv


def _edges(corners: np.ndarray, nxt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(E, 2) undirected edges (lower index first) and each corner's edge (to the next corner)."""
    a, b = corners, corners[nxt]
    key = np.stack([np.minimum(a, b), np.maximum(a, b)], 1)
    if not len(key):
        return key.reshape(0, 2), np.zeros(0, np.int64)
    first, inverse = unique_rows(key)
    return key[first], inverse


def _materials(mesh: Mesh) -> np.ndarray:
    n = len(mesh.face_sizes)
    return np.zeros(n, np.int64) if mesh.face_materials is None else np.asarray(mesh.face_materials, np.int64)


def _face_normals(positions, sizes, corners, face, nxt) -> np.ndarray:
    p, pn = positions[corners], positions[corners[nxt]]
    cross = np.cross(p, pn)
    fn = np.zeros((len(sizes), 3))
    for k in range(3):
        fn[:, k] = np.bincount(face, cross[:, k], len(sizes))
    length = np.linalg.norm(fn, axis=1, keepdims=True)
    return np.divide(fn, length, out=np.zeros_like(fn), where=length > 0)


def _corner_angles(positions, corners, nxt, prv) -> np.ndarray:
    a = positions[corners[prv]] - positions[corners]
    b = positions[corners[nxt]] - positions[corners]
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(la * lb, 1e-30)
    return np.arccos(np.clip(cos, -1, 1))


def _vertex_normals(mesh: Mesh):
    """Angle-weighted vertex normals (Blender's), with face normals and corner angles."""
    face, nxt, prv = _topology(mesh.face_sizes)
    fn = _face_normals(mesh.positions, mesh.face_sizes, mesh.corners, face, nxt)
    angle = _corner_angles(mesh.positions, mesh.corners, nxt, prv)
    vn = np.zeros((len(mesh.positions), 3))
    for k in range(3):
        vn[:, k] = np.bincount(mesh.corners, fn[face, k] * angle, len(mesh.positions))
    length = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.divide(vn, length, out=np.zeros_like(vn), where=length > 0)
    return vn, fn, angle, face


def _rebuild(mesh: Mesh, smooth: np.ndarray, target: np.ndarray, positions: np.ndarray,
             drop_mapped: bool = False) -> tuple[Mesh, np.ndarray]:
    """Corners moved to target[their vertex] (positions indexed like target). A face loses
    corners that repeat the one before; faces left with under 3 corners go, and so do points
    no face uses. drop_mapped also drops faces whose points all merged into others (where two
    copies meet, as Mirror and Array do: one face of each coinciding pair is kept)."""
    target = np.asarray(target, np.int64)
    cv = target[mesh.corners]
    face, nxt, _ = _topology(mesh.face_sizes)
    keep_c = cv != cv[nxt]
    sizes = np.bincount(face[keep_c], minlength=len(mesh.face_sizes))
    keep_f = sizes >= 3
    if drop_mapped:
        moved = np.bincount(face, cv != mesh.corners, len(mesh.face_sizes))
        keep_f &= moved < mesh.face_sizes
    keep_c &= keep_f[face]
    used = np.unique(cv[keep_c])
    renumber = np.full(len(positions), -1, np.int64)
    renumber[used] = np.arange(len(used))
    out = Mesh(mesh.name, np.asarray(positions, np.float64)[used], sizes[keep_f].astype(np.int64),
               renumber[cv[keep_c]], None, [u[keep_c] for u in mesh.uvs],
               mesh.colors[keep_c] if mesh.colors is not None else None,
               mesh.face_materials[keep_f] if mesh.face_materials is not None else None)
    return out, np.asarray(smooth, bool)[keep_f]


def _close_pairs(p: np.ndarray, q: np.ndarray, dist: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (i, j, distance) with |p[i] - q[j]| <= dist, found through a grid of dist-sized cells."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    if not len(p) or not len(q):
        return empty
    size = max(float(dist), 1e-12)
    origin = np.minimum(p.min(0), q.min(0))
    cp = np.floor((p - origin) / size).astype(np.int64)
    cq = np.floor((q - origin) / size).astype(np.int64)
    order = np.lexsort(cq.T[::-1])
    sorted_q = cq[order]
    out_i, out_j = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                look = cp + (dx, dy, dz)
                both = np.concatenate([sorted_q, look])
                _, group = unique_rows(both)
                gq, gp = group[:len(sorted_q)], group[len(sorted_q):]
                # sorted_q's groups aren't sorted by number, so sort them to search
                qs = np.argsort(gq, kind="stable")
                gq_sorted = gq[qs]
                lo = np.searchsorted(gq_sorted, gp, "left")
                hi = np.searchsorted(gq_sorted, gp, "right")
                n = hi - lo
                if not n.any():
                    continue
                i = np.repeat(np.arange(len(p)), n)
                k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
                out_i.append(i)
                out_j.append(order[qs[k]])
    if not out_i:
        return empty
    i, j = np.concatenate(out_i), np.concatenate(out_j)
    d = np.linalg.norm(p[i] - q[j], axis=1)
    hit = d <= dist
    return i[hit], j[hit], d[hit]


def _reverse_faces(sizes: np.ndarray) -> np.ndarray:
    """Corner order that reverses each face but keeps its first corner first, as Blender does."""
    face, _, _ = _topology(sizes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    local = np.arange(int(np.sum(sizes))) - starts[face]
    return starts[face] + (sizes[face] - local) % sizes[face]


def _concat(meshes: list[Mesh], smooths: list[np.ndarray]) -> tuple[Mesh, np.ndarray]:
    """Meshes joined (they must have the same UV sets and all or none colors)."""
    offsets = np.cumsum([0] + [len(m.positions) for m in meshes[:-1]])
    first = meshes[0]
    mats = [m.face_materials for m in meshes]
    out = Mesh(first.name, np.concatenate([m.positions for m in meshes]),
               np.concatenate([m.face_sizes for m in meshes]).astype(np.int64),
               np.concatenate([m.corners + o for m, o in zip(meshes, offsets)]).astype(np.int64), None,
               [np.concatenate([m.uvs[k] for m in meshes]) for k in range(len(first.uvs))],
               np.concatenate([m.colors for m in meshes]) if first.colors is not None else None,
               None if all(x is None for x in mats) else np.concatenate([_materials(m) for m in meshes]))
    return out, np.concatenate(smooths)


def _copy(mesh: Mesh, positions=None, order=None, uvs=None) -> Mesh:
    """A copy with new positions and, optionally, its corners in another order."""
    o = slice(None) if order is None else order
    return Mesh(mesh.name, mesh.positions if positions is None else positions, mesh.face_sizes, mesh.corners[o],
                None, uvs if uvs is not None else [u[o] for u in mesh.uvs],
                mesh.colors[o] if mesh.colors is not None else None, mesh.face_materials)


# ============================================================================ Triangulate

def _triangulate(mesh: Mesh, smooth: np.ndarray, md, ctx: Context):
    sizes, corners, p = mesh.face_sizes, mesh.corners, mesh.positions
    quad_method = int(md.get("quad_method", 3))  # n-gons ("Beauty" or "Clip") are ear clipped either way
    min_vertices = max(int(md.get("min_vertices", 4)), 4)
    split = sizes >= min_vertices
    if not split.any():
        return mesh, smooth
    starts = mesh.face_starts
    face, _, _ = _topology(sizes)
    # every triangle as three corner indices, by the face it comes from
    tri_face, tri = [], []
    quads = np.nonzero(split & (sizes == 4))[0]
    if len(quads):
        q = starts[quads][:, None] + np.arange(4)
        v = p[corners[q]]
        d02 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
        d13 = np.linalg.norm(v[:, 1] - v[:, 3], axis=1)
        if quad_method == 1:  # fixed: 1st to 3rd corner
            use13 = np.zeros(len(quads), bool)
        elif quad_method == 2:  # fixed alternate: 2nd to 4th
            use13 = np.ones(len(quads), bool)
        elif quad_method == 4:  # longest diagonal
            use13 = d13 > d02
        elif quad_method == 0:  # beauty: the split whose worse triangle is the better shaped
            use13 = _quality(v[:, [1, 2, 3]], v[:, [1, 3, 0]]) > _quality(v[:, [0, 1, 2]], v[:, [0, 2, 3]])
        else:  # shortest diagonal
            use13 = d13 < d02
        a = np.where(use13[:, None], q[:, [1, 2, 3]], q[:, [0, 1, 2]])
        b = np.where(use13[:, None], q[:, [1, 3, 0]], q[:, [0, 2, 3]])
        tri_face += [quads, quads]
        tri += [a, b]
    ngons = np.nonzero(split & (sizes > 4))[0]
    if len(ngons):
        sub_corners = np.concatenate([np.arange(starts[f], starts[f] + sizes[f]) for f in ngons])
        t = triangulate(p, sizes[ngons], corners[sub_corners])
        tri_face.append(np.repeat(ngons, sizes[ngons] - 2))
        tri.append(sub_corners[t])
    tri_face = np.concatenate(tri_face)
    tri = np.concatenate(tri)
    # faces in their old order, each split face replaced by its triangles
    keep_c = ~split[face]
    src = np.concatenate([np.nonzero(keep_c)[0], tri.ravel()])
    key_face = np.concatenate([face[keep_c], np.repeat(tri_face, 3)])
    key_seq = np.concatenate([np.nonzero(keep_c)[0], len(corners) + np.arange(tri.size)])
    order = np.lexsort((key_seq, key_face))
    src = src[order]
    new_face = np.concatenate([np.nonzero(~split)[0], tri_face])
    new_sizes = np.concatenate([sizes[~split], np.full(len(tri_face), 3)])
    face_order = np.lexsort((np.concatenate([np.zeros((~split).sum()), np.arange(len(tri_face)) + 1]), new_face))
    new_face, new_sizes = new_face[face_order], new_sizes[face_order]
    out = Mesh(mesh.name, p, new_sizes.astype(np.int64), corners[src], None, [u[src] for u in mesh.uvs],
               mesh.colors[src] if mesh.colors is not None else None,
               mesh.face_materials[new_face] if mesh.face_materials is not None else None)
    return out, smooth[new_face]


def _quality(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The worse of two triangles' shapes: area over the sum of squared sides (equilateral is best)."""
    def one(t):
        area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        sides = sum(np.sum((t[:, (k + 1) % 3] - t[:, k]) ** 2, axis=1) for k in range(3))
        return area / np.maximum(sides, 1e-30)
    return np.minimum(one(a), one(b))


# ============================================================================ Weld

def _weld(mesh: Mesh, smooth: np.ndarray, md, ctx: Context):
    dist = float(md.get("merge_dist", 0.001))
    p = mesh.positions
    n = len(p)
    target = np.arange(n)
    if int(md.get("mode", 0)) == 1:  # connected: along edges shorter than the distance
        _, nxt, _ = _topology(mesh.face_sizes)
        edges, _ = _edges(mesh.corners, nxt)
        short = edges[np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1) <= dist]
        parent = np.arange(n)

        def root(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for a, b in short.tolist():
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        target = np.array([root(v) for v in range(n)], np.int64) if len(short) else target
    else:  # all: each point, in order, takes the points near it that nothing has taken yet
        i, j, _ = _close_pairs(p, p, dist)
        later = j > i
        i, j = i[later], j[later]
        if len(i):
            order = np.lexsort((j, i))
            i, j = i[order], j[order]
            taken = np.zeros(n, bool)
            starts = np.searchsorted(i, np.arange(n))
            ends = np.searchsorted(i, np.arange(n), "right")
            for v in np.unique(i).tolist():
                if taken[v]:
                    continue
                for w in j[starts[v]:ends[v]].tolist():
                    if not taken[w]:
                        taken[w] = True
                        target[w] = v
    if (target == np.arange(n)).all():
        return mesh, smooth
    count = np.bincount(target, minlength=n).astype(np.float64)
    merged = np.zeros_like(p)
    for k in range(3):
        merged[:, k] = np.bincount(target, p[:, k], n)
    merged = np.divide(merged, count[:, None], out=p.copy(), where=count[:, None] > 0)
    return _rebuild(mesh, smooth, target, merged)


# ============================================================================ Mirror

def _mirror(mesh: Mesh, smooth: np.ndarray, md, ctx: Context):
    flag = int(md.get("flag", _MIR_AXIS_X))
    tolerance = float(md.get("tolerance", 0.001))
    mirror_space = ctx.relative(int(md.get("mirror_ob", 0) or 0))
    uv_offset = np.asarray(md.get("uv_offset", np.zeros(2)), np.float64)
    uv_copy = np.asarray(md.get("uv_offset_copy", np.zeros(2)), np.float64)
    for axis in range(3):
        if not flag & (_MIR_AXIS_X << axis):
            continue
        to_plane = np.eye(4) if mirror_space is None else np.linalg.inv(mirror_space)  # object -> mirror space
        if flag & (_MIR_BISECT_X << axis):
            keep_negative = bool(flag & (_MIR_FLIP_X << axis))
            mesh, smooth = _bisect(mesh, smooth, to_plane, axis, keep_negative,
                                   float(md.get("bisect_threshold", 0.001)))
            if not len(mesh.face_sizes):
                return mesh, smooth
        flip = np.eye(4)
        flip[axis, axis] = -1
        reflect = np.linalg.inv(to_plane) @ flip @ to_plane
        p = mesh.positions
        q = transform_points(reflect, p)
        n = len(p)
        target = np.arange(2 * n)
        positions = np.concatenate([p, q])
        if not flag & _MIR_NO_MERGE:
            close = np.linalg.norm(p - q, axis=1) < tolerance
            target[n:][close] = np.nonzero(close)[0]
            positions[:n][close] = (p[close] + q[close]) / 2
        order = _reverse_faces(mesh.face_sizes)
        uvs = []
        for uv in mesh.uvs:
            uv = uv[order].copy()
            for k, bit in ((0, _MIR_U), (1, _MIR_V)):
                if flag & bit:
                    if flag & _MIR_UDIM:  # within each face's UDIM tile
                        face, _, _ = _topology(mesh.face_sizes)
                        tile = np.floor(np.bincount(face, uv[:, k], len(mesh.face_sizes)) / mesh.face_sizes)
                        uv[:, k] = 2 * tile[face] + 1 - uv[:, k] + uv_offset[k]
                    else:
                        uv[:, k] = 1 - uv[:, k] + uv_offset[k]
            uvs.append(uv + uv_copy)
        copy = _copy(mesh, q, order, uvs)
        joined, joined_smooth = _concat([_copy(mesh, p), copy], [smooth, smooth])
        joined.positions = positions
        mesh, smooth = _rebuild(joined, joined_smooth, target, positions, drop_mapped=True)
    return mesh, smooth


def _bisect(mesh: Mesh, smooth: np.ndarray, to_plane: np.ndarray, axis: int, keep_negative: bool,
            threshold: float) -> tuple[Mesh, np.ndarray]:
    """The mesh cut along a mirror plane, keeping one side. Points within the threshold of the
    plane are moved onto it; cut edges get a new point shared by the faces on both sides."""
    side = transform_points(to_plane, mesh.positions)[:, axis]
    if keep_negative:
        side = -side
    p = mesh.positions.copy()
    on = np.abs(side) <= threshold
    if on.any():  # snap onto the plane
        local = transform_points(to_plane, p[on])
        local[:, axis] = 0
        p[on] = transform_points(np.linalg.inv(to_plane), local)
        side = np.where(on, 0.0, side)
    face, nxt, _ = _topology(mesh.face_sizes)
    inside = side >= 0
    face_in = np.bincount(face, inside[mesh.corners], len(mesh.face_sizes)) == mesh.face_sizes
    face_out = np.bincount(face, side[mesh.corners] <= 0, len(mesh.face_sizes)) == mesh.face_sizes
    keep_whole = face_in
    cut = ~face_in & ~face_out
    positions = [p]
    new_points: dict[tuple[int, int], int] = {}
    sizes, corners, src_a, src_b, weights, face_of = [], [], [], [], [], []
    starts = mesh.face_starts
    for f in np.nonzero(cut)[0].tolist():
        s, n = int(starts[f]), int(mesh.face_sizes[f])
        ring = list(range(s, s + n))
        out_c = []
        for c in ring:
            cn = int(nxt[c])
            a, b = int(mesh.corners[c]), int(mesh.corners[cn])
            sa, sb = side[a], side[b]
            if sa >= 0:
                out_c.append((a, c, c, 0.0))
            if (sa > 0 > sb) or (sa < 0 < sb):
                t = sa / (sa - sb)
                key = (min(a, b), max(a, b))
                if key not in new_points:
                    new_points[key] = len(p) + len(new_points)
                    positions.append((p[a] + (p[b] - p[a]) * t)[None])
                out_c.append((new_points[key], c, cn, t))
        if len(out_c) >= 3:
            sizes.append(len(out_c))
            face_of.append(f)
            for v, ca, cb, t in out_c:
                corners.append(v)
                src_a.append(ca)
                src_b.append(cb)
                weights.append(t)
    whole_c = keep_whole[face]
    pieces = Mesh(mesh.name, mesh.positions, mesh.face_sizes[keep_whole], mesh.corners[whole_c], None,
                  [u[whole_c] for u in mesh.uvs], mesh.colors[whole_c] if mesh.colors is not None else None,
                  mesh.face_materials[keep_whole] if mesh.face_materials is not None else None)
    all_p = np.concatenate(positions)
    if sizes:
        ca, cb, w = np.array(src_a), np.array(src_b), np.array(weights)[:, None]
        face_of = np.array(face_of)

        def mix(attr):
            return attr[ca] * (1 - w) + attr[cb] * w

        extra = Mesh(mesh.name, all_p, np.array(sizes, np.int64), np.array(corners, np.int64), None,
                     [mix(u) for u in mesh.uvs], mix(mesh.colors) if mesh.colors is not None else None,
                     mesh.face_materials[face_of] if mesh.face_materials is not None else None)
        pieces.positions = all_p
        joined, joined_smooth = _concat_same([pieces, extra], [smooth[keep_whole], smooth[face_of]])
    else:
        pieces.positions = all_p
        joined, joined_smooth = pieces, smooth[keep_whole]
    return _rebuild(joined, joined_smooth, np.arange(len(all_p)), all_p)


def _concat_same(meshes: list[Mesh], smooths: list[np.ndarray]) -> tuple[Mesh, np.ndarray]:
    """Faces of meshes that share one point list, joined."""
    out, smooth = _concat(meshes, smooths)
    out.positions = meshes[0].positions
    out.corners = np.concatenate([m.corners for m in meshes]).astype(np.int64)
    return out, smooth


# ============================================================================ Array

def _array(mesh: Mesh, smooth: np.ndarray, md, ctx: Context):
    p = mesh.positions
    offset_type = int(md.get("offset_type", _ARR_RELATIVE))
    shift = np.zeros(3)
    if offset_type & _ARR_CONST:
        shift += np.asarray(md.get("offset", np.zeros(3)), np.float64)
    if offset_type & _ARR_RELATIVE:
        shift += np.asarray(md.get("scale", np.zeros(3)), np.float64) * (p.max(0) - p.min(0))
    step = translation(shift)
    if offset_type & _ARR_OBJECT:
        other = ctx.relative(int(md.get("offset_ob", 0) or 0))
        if other is not None:
            step = step @ other
    count = int(md.get("count", 2))
    if int(md.get("fit_type", 0)) == _ARR_FIT_LENGTH:
        dist = float(np.linalg.norm(step[:3, 3]))
        length = float(md.get("length", 0.0))
        count = int(length / dist * (1 + 1e-5)) + 1 if dist > _FLT_EPSILON else 1  # an exact fit counts
    count = max(1, min(count, 1_000_000 // max(len(p), 1) + 1))
    # caps: other objects' meshes, as they are in their own space, one step before the first copy and
    # one past the last (Blender doesn't look at where the cap objects themselves stand)
    caps = [ctx.mesh_of(int(md.get(field, 0) or 0)) for field in ("start_cap", "end_cap")]
    if count == 1 and caps == [None, None]:
        return mesh, smooth
    ctx.budget.need(count * len(mesh.corners) + sum(len(c[0].corners) for c in caps if c is not None))
    uv_offset = np.asarray(md.get("uv_offset", np.zeros(2)), np.float64)
    copies, matrix = [], np.eye(4)
    for k in range(count):
        copies.append(_copy(mesh, transform_points(matrix, p), uvs=[u + uv_offset * k for u in mesh.uvs]))
        matrix = step @ matrix
    n = len(p)
    parts, smooths = list(copies), [smooth] * count
    placed = []  # (first point, cap points, the copy it meets)
    for cap, place, meets in zip(caps, (np.linalg.inv(step), matrix), (0, count - 1)):
        if cap is None:
            continue
        cap_mesh = _like(cap[0], mesh, ctx.n_materials)
        cap_mesh.positions = transform_points(place, cap_mesh.positions)
        placed.append((sum(len(m.positions) for m in parts), cap_mesh.positions, meets))
        parts.append(cap_mesh)
        smooths.append(np.asarray(cap[1], bool))
    joined, joined_smooth = _concat(parts, smooths)
    target = np.arange(len(joined.positions))
    flags = int(md.get("flags", 0))
    dist = float(md.get("merge_dist", 0.01))
    if flags & _ARR_MERGE and count > 1:
        # each point of the second copy that lands on one of the first: the same for every later pair
        src, dst = _nearest(copies[1].positions, copies[0].positions, dist)
        for k in range(1, count):
            target[k * n + src] = (k - 1) * n + dst
    if flags & _ARR_MERGE_FINAL and count > 1:
        src, dst = _nearest(copies[-1].positions, copies[0].positions, dist)
        target[(count - 1) * n + src] = dst
    if flags & _ARR_MERGE:
        for start, points, meets in placed:  # a cap's points merge onto the copy it meets
            src, dst = _nearest(points, copies[meets].positions, dist)
            target[start + src] = meets * n + dst
    for _ in range(count + 3):  # follow chains of merges to their end
        nxt = target[target]
        if (nxt == target).all():
            break
        target = nxt
    if (target == np.arange(len(target))).all():
        return joined, joined_smooth
    return _rebuild(joined, joined_smooth, target, joined.positions, drop_mapped=True)


def _like(cap: Mesh, mesh: Mesh, n_materials: int) -> Mesh:
    """A copy of cap with mesh's UV sets and colors (zero UVs and white where cap has none), and its
    material numbers kept within the object's slots."""
    c = len(cap.corners)
    uvs = [cap.uvs[k] if k < len(cap.uvs) else np.zeros((c, 2)) for k in range(len(mesh.uvs))]
    colors = None
    if mesh.colors is not None:
        colors = cap.colors if cap.colors is not None else np.ones((c, 4))
    mats = _materials(cap)
    if mesh.face_materials is not None or mats.any():
        mats = np.minimum(mats, max(n_materials - 1, 0))
    else:
        mats = None
    return Mesh(cap.name, cap.positions.copy(), cap.face_sizes, cap.corners, None, uvs, colors, mats)


def _nearest(src: np.ndarray, dst: np.ndarray, dist: float) -> tuple[np.ndarray, np.ndarray]:
    """For points of src with a point of dst within dist: (src index, the nearest dst index)."""
    i, j, d = _close_pairs(src, dst, dist)
    if not len(i):
        return i, j
    order = np.lexsort((d, i))
    i, j = i[order], j[order]
    first = np.concatenate(([True], i[1:] != i[:-1]))
    return i[first], j[first]


# ============================================================================ Solidify

def _solidify(mesh: Mesh, smooth: np.ndarray, md, ctx: Context):
    flag = int(md.get("flag", _SOL_RIM))
    thickness = float(md.get("offset", 0.01))
    fac = float(md.get("offset_fac", -1.0))
    if int(md.get("mode", 0)) == 1:  # complex mode keeps the thickness even, like this
        flag |= _SOL_EVEN
    vn, fn, angle, face = _vertex_normals(mesh)
    n = len(mesh.positions)
    scale = np.ones(n)
    if flag & _SOL_EVEN:  # thicker at corners, so the faces keep their distance
        cos = np.abs(np.einsum("ij,ij->i", vn[mesh.corners], fn[face]))
        per_corner = np.where(cos > 1e-8, 1 / np.maximum(cos, 1e-8), 1.0)
        weight = np.pi - angle  # Blender weighs each face by the turn between its edges at the corner
        total = np.bincount(mesh.corners, weight, n)
        scale = np.divide(np.bincount(mesh.corners, per_corner * weight, n), total, out=np.ones(n), where=total > 0)
    ofs_orig = -((-fac + 1) * 0.5) * thickness
    ofs_new = thickness + ofs_orig
    outer, inner = (ofs_new, ofs_orig) if thickness >= 0 else (ofs_orig, ofs_new)
    p = mesh.positions
    a = p + vn * (scale * outer)[:, None]
    b = p + vn * (scale * inner)[:, None]
    flip = bool(flag & _SOL_FLIP)
    order = _reverse_faces(mesh.face_sizes)
    mat_max = max(ctx.n_materials - 1, 0)
    first = _copy(mesh, a, order if flip else None)
    second = _copy(mesh, b, None if flip else order)
    second.corners = second.corners + n
    mats = _materials(mesh)
    second.face_materials = np.minimum(mats + int(md.get("mat_ofs", 0)), mat_max)
    first.face_materials = mats
    parts, parts_smooth = [first], [smooth]
    if not flag & _SOL_NO_SHELL:  # "Only Rim" keeps the first shell, without its copy
        parts.append(second)
        parts_smooth.append(smooth)
    if flag & _SOL_RIM:
        _, nxt, _ = _topology(mesh.face_sizes)
        edges, corner_edge = _edges(mesh.corners, nxt)
        uses = np.bincount(corner_edge, minlength=len(edges))
        c = np.nonzero(uses[corner_edge] == 1)[0]  # corners starting a boundary edge
        if len(c):
            cn = nxt[c]
            va, vb = mesh.corners[c], mesh.corners[cn]
            quad = np.stack([vb, va, va + n, vb + n], 1)
            src = np.stack([cn, c, c, cn], 1)
            if flip:
                quad, src = quad[:, ::-1], src[:, ::-1]
            src = src.ravel()
            rim = Mesh(mesh.name, a, np.full(len(c), 4, np.int64), quad.ravel(), None, [u[src] for u in mesh.uvs],
                       mesh.colors[src] if mesh.colors is not None else None,
                       np.minimum(mats[face[c]] + int(md.get("mat_ofs_rim", 0)), mat_max))
            parts.append(rim)
            parts_smooth.append(smooth[face[c]])
    joined, joined_smooth = _concat_same(parts, parts_smooth)
    positions = np.concatenate([a, b])
    if mesh.face_materials is None and not (int(md.get("mat_ofs", 0)) or int(md.get("mat_ofs_rim", 0))):
        joined.face_materials = None
    return _rebuild(joined, joined_smooth, np.arange(2 * n), positions)


# ============================================================================ Subdivision Surface

def _subsurf(mesh: Mesh, smooth: np.ndarray, md, ctx: Context):
    levels = min(int(md.get("levels", 1)), 6)
    simple = int(md.get("subdivType", 0)) == _SUB_SIMPLE
    keep_corners = int(md.get("boundary_smooth", 0)) == _SUB_KEEP_CORNERS
    limit = levels > 0 and not simple and not int(md.get("flags", 0)) & _SUB_NO_LIMIT
    uv_smooth = int(md.get("uv_smooth", _UV_KEEP_BOUNDARIES))
    uv_pin = {_UV_KEEP_BOUNDARIES: "boundary", _UV_SMOOTH_ALL: keep_corners}.get(uv_smooth, True)
    uvs = mesh.uvs if not simple and uv_smooth != _UV_SMOOTH_NONE else []
    sizes, corners = mesh.face_sizes, mesh.corners
    creases = getattr(mesh, "creases", None) if int(md.get("flags", 0)) & _SUB_USE_CREASE else None
    for _ in range(levels):
        ctx.budget.need(4 * len(mesh.corners))  # each corner becomes a quad
        mesh, smooth, creases = _subdivide(mesh, smooth, simple, keep_corners, creases)
    if limit:
        extra = int(md.get("quality", 3)) - levels
        fine = mesh
        if creases is not None and extra > 0 and (creases.values.size or (creases.points is not None
                                                                         and creases.points.any())):
            # Blender's limit resolves creases down to its Quality level: refine that far (for the
            # creases' sake only; points keep their numbers) and take the limit of this level's points
            for _ in range(extra):
                ctx.budget.need(4 * len(fine.corners))
                fine, _, creases = _subdivide(fine, np.ones(len(fine.face_sizes), bool), False, keep_corners,
                                              creases)
        mesh.positions = _limit(fine, keep_corners, creases)[:len(mesh.positions)]
    if levels > 0 and uvs:
        mesh.uvs = [_smooth_uvs(sizes, corners, uv, levels, uv_pin, limit) for uv in uvs]
    return mesh, smooth


def _smooth_uvs(sizes, corners, uv, levels: int, pin, limit: bool) -> np.ndarray:
    """A UV layer subdivided as Catmull-Clark subdivides the mesh, over its own points (a mesh
    point with several UVs, along a seam, is several UV points), so seams are boundaries. pin is
    "boundary" (boundary points stay and boundary edges stay straight), True (corners stay) or
    False. The corners come out in the order the mesh's own subdivision gives them."""
    _, index = unique_rows(np.column_stack([corners.astype(np.float64), uv]))
    first = np.zeros(int(index.max()) + 1, np.int64)
    first[index] = np.arange(len(index))
    flat = Mesh("uv", np.column_stack([uv[first], np.zeros(len(first))]), sizes, index.astype(np.int64))
    smooth = np.ones(len(sizes), bool)
    for _ in range(levels):
        flat, smooth, _ = _subdivide(flat, smooth, False, pin)
    if limit:
        flat.positions = _limit(flat, pin)
    return flat.positions[flat.corners, :2]


_KEY = 1 << 31  # an edge's key: lower point * _KEY + higher point


def _edge_keys(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    return np.minimum(pairs[:, 0], pairs[:, 1]) * _KEY + np.maximum(pairs[:, 0], pairs[:, 1])


class Creases:
    """Semi-sharp creases as OpenSubdiv has them: a sharpness per edge (by its two points) and
    per point, each losing 1 a level. Blender's crease c is a sharpness of 10 c^2."""

    def __init__(self, keys: np.ndarray, values: np.ndarray, points: np.ndarray | None = None):
        order = np.argsort(keys)
        self.keys, self.values, self.points = keys[order], values[order], points

    @classmethod
    def from_blender(cls, pairs, edge_crease, vert_crease=None):
        values = 10.0 * np.clip(np.asarray(edge_crease, np.float64).ravel(), 0, 1) ** 2
        keep = values > 0
        points = None
        if vert_crease is not None:
            points = 10.0 * np.clip(np.asarray(vert_crease, np.float64).ravel(), 0, 1) ** 2
        return cls(_edge_keys(pairs)[keep], values[keep], points)

    def on(self, edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The sharpness of each of these edges, and of each of the n points."""
        found = np.zeros(len(edges))
        if len(self.keys) and len(edges):
            keys = _edge_keys(edges)
            at = np.clip(np.searchsorted(self.keys, keys), 0, len(self.keys) - 1)
            hit = self.keys[at] == keys
            found[hit] = self.values[at[hit]]
        points = np.zeros(n)
        if self.points is not None:
            m = min(n, len(self.points))
            points[:m] = self.points[:m]
        return found, points


def _less(sharpness: np.ndarray) -> np.ndarray:
    """A level down: 1 less, and nothing once it's at 1 or under (OpenSubdiv's uniform rule)."""
    return np.where(sharpness > 1, sharpness - 1, 0.0)


def _sharpness(mesh: Mesh, nxt: np.ndarray, keep_corners, creases: Creases | None = None):
    """Edges, each corner's edge, each edge's sharpness (infinite on a boundary or an edge shared
    by over two faces, else its crease), each point's edge count and sharpness, and which points
    are pinned (a kept corner; with keep_corners "boundary", every point on a boundary)."""
    edges, corner_edge = _edges(mesh.corners, nxt)
    n = len(mesh.positions)
    uses = np.bincount(corner_edge, minlength=len(edges))
    boundary = uses != 2
    if creases is not None:
        edge_sharp, point_sharp = creases.on(edges, n)
    else:
        edge_sharp, point_sharp = np.zeros(len(edges)), np.zeros(n)
    edge_sharp = np.where(boundary, np.inf, edge_sharp)
    ends = edges.ravel()
    valence = np.bincount(ends, minlength=n)
    boundary_count = np.bincount(ends, np.repeat(boundary, 2), n).astype(np.int64)
    pinned = np.zeros(n, bool)
    if keep_corners == "boundary":
        pinned = boundary_count > 0
    elif keep_corners:
        faces_at = np.bincount(mesh.corners, minlength=n)
        pinned = (faces_at == 1) & (boundary_count == 2)
    return edges, corner_edge, edge_sharp, valence, point_sharp, pinned


def _rule(edges, edge_sharp, point_sharp, pinned, p):
    """Each point's rule (0 smooth, 1 crease, 2 corner) and the sum of its sharp edges' far ends."""
    n = len(p)
    sharp = edge_sharp > 0
    count = np.bincount(edges.ravel(), np.repeat(sharp, 2), n)
    rule = np.where((count > 2) | (point_sharp > 0) | pinned, 2, np.where(count == 2, 1, 0))
    far = np.zeros((n, 3))
    se = edges[sharp]
    for k in range(3):
        far[:, k] = np.bincount(se[:, 0], p[se[:, 1], k], n) + np.bincount(se[:, 1], p[se[:, 0], k], n)
    return rule, far


def _subdivide(mesh: Mesh, smooth: np.ndarray, simple: bool, keep_corners, creases: Creases | None = None):
    """One level, and the creases a level down (on the new points), or None."""
    p, sizes, corners = mesh.positions, mesh.face_sizes, mesh.corners
    face, nxt, prv = _topology(sizes)
    nf, n = len(sizes), len(p)
    edges, corner_edge, edge_sharp, valence, point_sharp, pinned = _sharpness(mesh, nxt, keep_corners, creases)
    ne = len(edges)
    face_pt = np.zeros((nf, 3))
    for k in range(3):
        face_pt[:, k] = np.bincount(face, p[corners, k], nf) / sizes
    mid = (p[edges[:, 0]] + p[edges[:, 1]]) / 2
    child_edge, child_point = _less(edge_sharp), _less(point_sharp)
    if simple:
        edge_pt, vert_pt = mid, p
    else:
        fsum = np.zeros((ne, 3))
        for k in range(3):
            fsum[:, k] = np.bincount(corner_edge, face_pt[face, k], ne)
        # an edge under 1 sharp blends the smooth point with the midpoint by its sharpness
        w = np.clip(edge_sharp, 0, 1)[:, None]
        smooth_edge = (p[edges[:, 0]] + p[edges[:, 1]] + fsum) / 4
        edge_pt = np.where(edge_sharp[:, None] >= 1, mid, w * mid + (1 - w) * smooth_edge)
        # vertices: Catmull-Clark's (F + 2R + (n - 3)V) / n smooth, (a + 6V + b) / 8 on a crease, corners stay
        faces_at = np.bincount(corners, minlength=n).astype(np.float64)
        f_avg = np.zeros((n, 3))
        r_avg = np.zeros((n, 3))
        for k in range(3):
            f_avg[:, k] = np.bincount(corners, face_pt[face, k], n)
            r_avg[:, k] = np.bincount(edges.ravel(), np.repeat(mid[:, k], 2), n)
        f_avg /= np.maximum(faces_at, 1)[:, None]
        r_avg /= np.maximum(valence, 1)[:, None]
        val = np.maximum(valence, 1)[:, None].astype(np.float64)
        smooth_pt = (f_avg + 2 * r_avg + (val - 3) * p) / val

        def by_rule(rule, far):
            return np.where((rule == 2)[:, None], p, np.where((rule == 1)[:, None], (far + 6 * p) / 8, smooth_pt))

        rule, far = _rule(edges, edge_sharp, point_sharp, pinned, p)
        vert_pt = by_rule(rule, far)
        child_rule, child_far = _rule(edges, child_edge, child_point, pinned, p)
        change = rule != child_rule
        if change.any():  # a crease running out: blend the two rules by the sharpness that runs out
            fading = (edge_sharp > 0) & (child_edge <= 0)
            total = np.bincount(edges.ravel(), np.repeat(np.where(fading, edge_sharp, 0.0), 2), n)
            count = np.bincount(edges.ravel(), np.repeat(fading, 2), n).astype(np.float64)
            point_fading = (point_sharp > 0) & (child_point <= 0)
            total, count = total + np.where(point_fading, point_sharp, 0), count + point_fading
            weight = np.clip(np.divide(total, count, out=np.zeros(n), where=count > 0), 0, 1)[:, None]
            blended = weight * vert_pt + (1 - weight) * by_rule(child_rule, child_far)
            vert_pt = np.where(change[:, None], blended, vert_pt)
        vert_pt = np.where((valence == 0)[:, None], p, vert_pt)
    positions = np.concatenate([vert_pt, edge_pt, face_pt])
    # one quad per corner: the corner's point, its next edge's point, the face's point, its previous edge's point
    quad = np.stack([corners, n + corner_edge, n + ne + face, n + corner_edge[prv]], 1).ravel()

    def corner_attr(a):
        half_next, half_prev = (a + a[nxt]) / 2, (a + a[prv]) / 2
        centre = np.zeros((nf, a.shape[1]))
        for k in range(a.shape[1]):
            centre[:, k] = np.bincount(face, a[:, k], nf) / sizes
        return np.stack([a, half_next, centre[face], half_prev], 1).reshape(-1, a.shape[1])

    out = Mesh(mesh.name, positions, np.full(len(corners), 4, np.int64), quad.astype(np.int64), None,
               [corner_attr(u) for u in mesh.uvs], corner_attr(mesh.colors) if mesh.colors is not None else None,
               mesh.face_materials[face] if mesh.face_materials is not None else None)
    next_creases = None
    if creases is not None:  # each creased edge's two halves keep its sharpness, a level down
        e = np.nonzero(np.isfinite(child_edge) & (child_edge > 0))[0]
        halves = np.concatenate([np.stack([edges[e, 0], n + e], 1), np.stack([n + e, edges[e, 1]], 1)])
        points = np.concatenate([child_point, np.zeros(ne + nf)])
        next_creases = Creases(_edge_keys(halves), np.concatenate([child_edge[e], child_edge[e]]), points)
    return out, np.asarray(smooth, bool)[face], next_creases


def _limit(mesh: Mesh, keep_corners, creases: Creases | None = None) -> np.ndarray:
    """Points moved to Catmull-Clark's limit surface (all faces are quads after a level):
    (n^2 V + 4 * edge neighbours + diagonal neighbours) / (n(n + 5)) smooth, (a + 4V + b) / 6
    on a crease (with any sharpness left), and corners stay."""
    p, corners = mesh.positions, mesh.corners
    _, nxt, _ = _topology(mesh.face_sizes)
    n = len(p)
    edges, _, edge_sharp, valence, point_sharp, pinned = _sharpness(mesh, nxt, keep_corners, creases)
    e_sum = np.zeros((n, 3))
    d_sum = np.zeros((n, 3))
    diag = corners[nxt[nxt]]
    for k in range(3):
        e_sum[:, k] = np.bincount(edges[:, 0], p[edges[:, 1], k], n) + np.bincount(edges[:, 1], p[edges[:, 0], k], n)
        d_sum[:, k] = np.bincount(corners, p[diag, k], n)
    val = np.maximum(valence, 1)[:, None].astype(np.float64)
    smooth_pt = (val * val * p + 4 * e_sum + d_sum) / (val * (val + 5))
    rule, far = _rule(edges, edge_sharp, point_sharp, pinned, p)
    out = np.where((rule == 2)[:, None], p, np.where((rule == 1)[:, None], (far + 4 * p) / 6, smooth_pt))
    return np.where((valence == 0)[:, None], p, out)


_APPLY = {"TriangulateModifierData": _triangulate, "WeldModifierData": _weld, "MirrorModifierData": _mirror,
          "ArrayModifierData": _array, "SolidifyModifierData": _solidify, "SubsurfModifierData": _subsurf}
