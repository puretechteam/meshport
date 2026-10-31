"""Blender's curve, text and metaball objects, turned into meshes the way Blender's viewport shows them.

Curves: Bezier, poly and NURBS splines, each sampled at its resolution, then swept along a
profile: a round bevel (its depth and resolution), an extrusion, or both (two half circles
joined by straight sides). 3D curves are oriented by rotation-minimizing frames (the "Minimum"
twist, with the leftover twist of a closed curve spread along it) or by Z-up frames, and turned
by each point's tilt; the bevel is scaled by each point's radius. 2D curves lie flat, keep their
bevel the same width around corners, and are filled (front, back or both) when closed, with
curves inside other curves cut out as holes. Fill caps close open 3D curves' ends.
Left out: bevel and taper objects and custom bevel profiles (the bevel stays round), the
"Tangent" twist (treated as "Minimum"), bevel start and end factors, the path offset, text on
a curve, and NURBS surfaces.

Text: each character's outline from the font, laid out as Blender does (character, word and
line spacing, alignment, offsets and shear), then made into a 2D curve as above. Reading fonts
takes the optional fontTools package. A font packed in the .blend file, one in its folder or the folder above
it (the folder rule, see scene.linked_path), or a system font of the same file name is used
as is; Blender's built-in font is stood in for by a similar installed one, so its text comes
out close but not exact. Text boxes, small caps, underlines and kerning are left out.

Metaballs: ball, capsule, plane, ellipsoid and cube elements, negative ones included, each
adding stiffness · (1 - d²/r²)³ to a field whose threshold is the surface. Every metaball
object named like a "base" one (Mball, Mball.001, ...) joins the base object's surface, as in
Blender. The surface is found by marching tetrahedra at the viewport resolution, so it matches
Blender's surface, though not its exact points.
"""

from __future__ import annotations

import io
import math
import os
import re
from pathlib import Path

import numpy as np

from .scene import Mesh, linked_path

# Curve.flag
_CU_3D, _CU_FRONT, _CU_BACK, _CU_FILL_CAPS = 1, 2, 4, 16384
_TWIST_Z_UP = 0
# Nurb
_POLY, _BEZIER, _NURBS = 0, 1, 4
_CYCLIC, _ENDPOINT, _BEZIER_KNOTS = 1, 2, 4
_SMOOTH = 1
_HD_VECT = 2
_KEY_LINEAR, _KEY_CARDINAL, _KEY_BSPLINE, _KEY_EASE = 0, 1, 2, 3
# Curve.spacemode, Curve.align_y
_ALIGN_CENTER, _ALIGN_RIGHT = 1, 2
_TOP, _CENTER, _BOTTOM, _BOTTOM_BASELINE = 1, 2, 3, 4
# MetaElem
_MB_BALL, _MB_TUBE, _MB_PLANE, _MB_ELLIPSOID, _MB_CUBE = 0, 4, 5, 6, 7
_MB_TUBE_AXES = {1: 0, 2: 1, 3: 2}  # old single-axis tubes
_MB_NEGATIVE, _MB_HIDE = 2, 8
_MAX_POINTS = 4_000_000


class _Spline:
    __slots__ = ("cyclic", "ends", "material", "points", "radius", "smooth", "tilt")

    def __init__(self, points, tilt, radius, cyclic, material=0, smooth=False, ends=None):
        self.points, self.tilt, self.radius = np.asarray(points, np.float64), np.asarray(tilt, np.float64), \
            np.asarray(radius, np.float64)
        self.cyclic, self.material, self.smooth = bool(cyclic), int(material), bool(smooth)
        self.ends = ends  # an open Bezier spline's directions at its two ends (along its handles)


# ============================================================================ curves

def curve_mesh(cu, base_dir: Path | None = None) -> tuple[Mesh, np.ndarray] | None:
    """A curve or text object's data (a Curve struct) as a mesh without normals, with each face's
    smooth flag; None when it makes no faces (a curve with no bevel, extrusion or fill)."""
    if int(cu.get("type", 2)) == 4:
        splines = _text_splines(cu, base_dir)
    else:
        splines = [s for s in (_evaluate(nu, int(cu.get("resolu", 12) or 12)) for nu in cu.listbase("nurb", "Nurb"))
                   if s is not None]
    if not splines:
        return None
    return _sweep(cu, splines, name=cu.id_name() or "curve")


def _evaluate(nu, default_resolution: int) -> _Spline | None:
    kind = int(nu.get("type", 0)) & 7
    resolution = max(1, min(int(nu.get("resolu", 0) or default_resolution), 1024))
    cyclic = bool(int(nu.get("flagu", 0)) & _CYCLIC)
    material, smooth = int(nu.get("mat_nr", 0)), bool(int(nu.get("flag", 0)) & _SMOOTH)
    n = int(nu.get("pntsu", 0) or 0)
    if n <= 0 or n * resolution > _MAX_POINTS:
        return None
    f = nu.f
    if kind == _BEZIER:
        size = f.struct_size["BezTriple"]
        raw = nu.raw("bezt", "u1", size * n)
        if raw is None:
            return None
        fields = f.structs[f.struct_index["BezTriple"]][1]
        buf = raw.tobytes()

        def column(name, dtype, count=1):
            off = fields[name][0]
            shape = (n, count) if count > 1 else (n,)
            strides = (size, np.dtype(dtype).itemsize) if count > 1 else (size,)
            return np.ndarray(shape, dtype, buf, off, strides).astype(np.float64 if dtype[-2] == "f" else np.int64)

        vec = column("vec", "<f4", 9).reshape(n, 3, 3)
        tilt, radius = column("alfa", "<f4"), column("radius", "<f4")
        h1, h2 = column("h1", "u1"), column("h2", "u1")
        return _bezier(vec, tilt, radius, h1, h2, cyclic, resolution, int(nu.get("tilt_interp", 0)),
                       int(nu.get("radius_interp", 0)), material, smooth)
    size = f.struct_size["BPoint"]
    raw = nu.raw("bp", "u1", size * n)
    if raw is None:
        return None
    fields = f.structs[f.struct_index["BPoint"]][1]
    buf = raw.tobytes()
    vec = np.ndarray((n, 4), "<f4", buf, fields["vec"][0], (size, 4)).astype(np.float64)
    tilt = np.ndarray((n,), "<f4", buf, fields["alfa"][0], (size,)).astype(np.float64)
    radius = np.ndarray((n,), "<f4", buf, fields["radius"][0], (size,)).astype(np.float64)
    if kind == _NURBS:
        order = max(2, min(int(nu.get("orderu", 4) or 4), n))
        flag = int(nu.get("flagu", 0))
        total = n + order + (order - 1 if flag & _CYCLIC else 0)
        stored = nu.raw("knotsu", "<f4", total) if nu.get("knotsu", 0) else None
        knots = stored.astype(np.float64) if stored is not None else _knots(n, order, flag)
        return _nurbs(vec, tilt, radius, order, flag, resolution, material, smooth, knots)
    return _Spline(vec[:, :3], tilt, radius, cyclic, material, smooth)


def _interp_weights(t: np.ndarray, kind: int) -> np.ndarray:
    """(len(t), 4) weights on the values before, at, after and two after a segment's start."""
    t = np.asarray(t, np.float64)
    w = np.zeros((len(t), 4))
    if kind == _KEY_BSPLINE:
        t2, t3 = t * t, t * t * t
        w[:, 0] = (1 - t) ** 3 / 6
        w[:, 1] = (3 * t3 - 6 * t2 + 4) / 6
        w[:, 2] = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6
        w[:, 3] = t3 / 6
    elif kind == _KEY_CARDINAL:  # Catmull-Rom
        t2, t3 = t * t, t * t * t
        w[:, 0] = (-t3 + 2 * t2 - t) / 2
        w[:, 1] = (3 * t3 - 5 * t2 + 2) / 2
        w[:, 2] = (-3 * t3 + 4 * t2 + t) / 2
        w[:, 3] = (t3 - t2) / 2
    else:
        if kind == _KEY_EASE:
            t = t * t * (3 - 2 * t)
        w[:, 1], w[:, 2] = 1 - t, t
    return w


def _bezier(vec, tilt, radius, h1, h2, cyclic, resolution, tilt_interp, radius_interp, material, smooth) -> _Spline:
    n = len(vec)
    segments = n if cyclic else n - 1
    pts, tl, rd = [], [], []
    for i in range(segments):
        j = (i + 1) % n
        p0, p1, p2, p3 = vec[i, 1], vec[i, 2], vec[j, 0], vec[j, 1]
        straight = h2[i] == _HD_VECT and h1[j] == _HD_VECT
        t = np.zeros(1) if straight else np.arange(resolution) / resolution
        u = 1 - t
        pts.append(np.outer(u ** 3, p0) + np.outer(3 * u * u * t, p1) + np.outer(3 * u * t * t, p2) + np.outer(t ** 3, p3))
        idx = [(i - 1) % n if cyclic else max(i - 1, 0), i, j, (j + 1) % n if cyclic else min(j + 1, n - 1)]
        tl.append(_interp_weights(t, tilt_interp) @ tilt[idx])
        rd.append(_interp_weights(t, radius_interp) @ radius[idx])
    ends = None
    if not cyclic or n == 1:
        pts.append(vec[-1:, 1])
        tl.append(tilt[-1:])
        rd.append(radius[-1:])
        ends = (vec[0, 2] - vec[0, 1], vec[-1, 1] - vec[-1, 0])
    return _Spline(np.concatenate(pts), np.concatenate(tl), np.concatenate(rd), cyclic and n > 1, material, smooth,
                   ends)


def _knots(n: int, order: int, flag: int) -> np.ndarray:
    """Blender's knot vectors: uniform, clamped at the ends, or Bezier-like (each inner knot
    repeated order - 1 times, so the curve passes through every (order - 1)th point from the
    second, or from the first when clamped); a cyclic spline's points wrap around, so its knots
    run on."""
    cyclic = flag & _CYCLIC
    total = n + order + (order - 1 if cyclic else 0)
    if flag & _BEZIER_KNOTS and order > 2:
        head = order if flag & _ENDPOINT and not cyclic else 2
        k = np.concatenate([np.zeros(head), 1 + np.floor(np.arange(max(total - head, 0)) / (order - 1))])
        if flag & _ENDPOINT and not cyclic:
            k[total - order:] = k[total - order]
        return k
    if cyclic:
        return np.arange(total, dtype=np.float64)
    if flag & _ENDPOINT:
        inner = np.arange(1, n - order + 1, dtype=np.float64)
        return np.concatenate([np.zeros(order), inner, np.full(order, n - order + 1.0)])
    return np.arange(total, dtype=np.float64)


def _basis(knots: np.ndarray, order: int, u: np.ndarray, count: int) -> np.ndarray:
    """(len(u), count) B-spline basis values (Cox-de Boor)."""
    k = knots
    last = len(k) - 1
    b = np.zeros((len(u), last))
    for i in range(last):
        if k[i] < k[i + 1]:
            b[:, i] = (u >= k[i]) & (u < k[i + 1])
    # the end of the range belongs to the last non-empty span
    end = u >= k[last]
    if end.any():
        span = max(i for i in range(last) if k[i] < k[i + 1]) if (k[:-1] < k[1:]).any() else 0
        b[end] = 0
        b[end, span] = 1
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


def _nurbs(vec, tilt, radius, order, flag, resolution, material, smooth, knots) -> _Spline | None:
    n = len(vec)
    cyclic = bool(flag & _CYCLIC)
    if cyclic:  # the first order - 1 points again at the end
        wrap = np.arange(n + order - 1) % n
        vec, tilt, radius = vec[wrap], tilt[wrap], radius[wrap]
    count = len(vec)
    start, end = knots[order - 1], knots[count]
    if end <= start:
        return None
    # evenly over the range, resolution points per control point (as Blender does for bevels;
    # its bare wire curves sample each knot span instead, but those make no faces)
    samples = max(resolution * (n if cyclic else n - 1), 2)
    u = start + (end - start) * np.arange(samples) / (samples if cyclic else samples - 1)
    u = np.minimum(u, np.nextafter(end, -np.inf))  # the end of the range, from its own span
    basis = _basis(knots, order, u, count)
    w = vec[:, 3].copy()
    w[w <= 0] = 1.0
    wb = basis * w
    total = wb.sum(1, keepdims=True)
    total[total == 0] = 1.0
    wb /= total
    return _Spline(wb @ vec[:, :3], wb @ tilt, wb @ radius, cyclic, material, smooth)


# ---- sweeping

def _profile(cu, is_3d: bool) -> tuple[np.ndarray, np.ndarray, bool, float, float]:
    """The cross-section as (k, 2) bevel points (sideways, up), which each point's radius scales,
    plus each point's extrusion (up, which it doesn't), whether it closes, and the heights of the
    front and back caps."""
    pts, closed, top, bottom = _profile_points(cu, is_3d)
    ext = max(float(cu.get("ext1", 0.0)), 0.0)
    if not len(pts):
        return pts, np.zeros(0), closed, top, bottom
    shift = np.clip(pts[:, 1], -ext, ext)
    circle = pts - np.stack([np.zeros(len(pts)), shift], 1)
    return circle, shift, closed, top, bottom


def _profile_points(cu, is_3d: bool) -> tuple[np.ndarray, bool, float, float]:
    depth = max(float(cu.get("ext2", 0.0)), 0.0)
    ext = max(float(cu.get("ext1", 0.0)), 0.0)
    res = max(0, min(int(cu.get("bevresol", 4)), 64))
    flag = int(cu.get("flag", 0))
    front, back = bool(flag & _CU_FRONT), bool(flag & _CU_BACK)
    if is_3d:  # FULL, BACK, FRONT, HALF
        half, keep_top, keep_bottom = front and back, not back or front, not front or back
        if not front and not back:
            half = False
    else:  # NONE, BACK, FRONT, BOTH: a filled 2D curve only needs the outer half
        half = front or back
        keep_top, keep_bottom = front or not back, back or not front
    if depth == 0 and ext == 0:
        return np.zeros((0, 2)), False, 0.0, 0.0
    if depth == 0:
        top, bottom = (0.0, ext), (0.0, -ext)
        pts = [bottom if keep_bottom or not half else (0.0, 0.0), top if keep_top or not half else (0.0, 0.0)]
        return np.array(pts), False, ext if keep_top else 0.0, -ext if keep_bottom else 0.0
    if ext == 0 and not half:  # a full circle: 4 + 2 * resolution points, from the bottom
        k = 4 + 2 * res
        a = 2 * np.pi * np.arange(k) / k
        return np.stack([depth * np.sin(a), -depth * np.cos(a)], 1), True, depth, -depth
    # half circles of 2 * (resolution + 1) segments, the top one raised by ext and the bottom one lowered
    seg = 2 * (res + 1)
    a = np.pi * np.arange(seg + 1) / seg  # 0 at the bottom, pi at the top, through the + side
    s, h = depth * np.sin(a), -depth * np.cos(a)
    lower, upper = a <= np.pi / 2 + 1e-12, a >= np.pi / 2 - 1e-12
    right = np.concatenate([np.stack([s[lower], h[lower] - ext], 1), np.stack([s[upper], h[upper] + ext], 1)])
    if ext == 0:
        right = np.delete(right, int(lower.sum()), axis=0)  # the two middle points coincide
    if half:
        keep = np.ones(len(right), bool)
        if not keep_bottom:  # front only: from the side's foot, up and over the top
            keep &= (right[:, 1] >= -ext - 1e-12) & ~((right[:, 1] < -ext + 1e-12) & (right[:, 0] < depth - 1e-12))
        if not keep_top:
            keep &= (right[:, 1] <= ext + 1e-12) & ~((right[:, 1] > ext - 1e-12) & (right[:, 0] < depth - 1e-12))
        pts = right[keep]
        return pts, False, (ext + depth) if keep_top else ext, -(ext + depth) if keep_bottom else -ext
    left = right[::-1][1:-1] * (-1, 1)
    return np.concatenate([right, left]), True, ext + depth, -(ext + depth)


def _tangents(p: np.ndarray, cyclic: bool, ends=None) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents (the bisector of the segments either side, or along the handles at a
    Bezier spline's ends) and, for 2D bevels, how much to widen at each point so the bevel keeps
    its width around corners."""
    n = len(p)
    d = np.diff(p, axis=0)
    if cyclic:
        d = np.concatenate([d, (p[0] - p[-1])[None]])
    length = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, length, out=np.zeros_like(d), where=length > 0)
    if cyclic:
        before, after = np.roll(d, 1, axis=0), d
    else:
        before = np.concatenate([d[:1], d]) if len(d) else np.zeros((n, 3))
        after = np.concatenate([d, d[-1:]]) if len(d) else np.zeros((n, 3))
    if ends is not None and not cyclic:
        for row, d_end in ((0, ends[0]), (n - 1, ends[1])):
            length = np.linalg.norm(d_end)
            if length > 1e-12:
                before[row] = after[row] = d_end / length
    t = before + after
    tl = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(tl > 1e-9, t / np.maximum(tl, 1e-30), after)
    cos_half = np.einsum("ij,ij->i", t, after)
    widen = 1 / np.clip(cos_half, 0.05, 1)
    return t, widen


def _frames(p: np.ndarray, t: np.ndarray, tilt: np.ndarray, cyclic: bool, twist: int) -> np.ndarray:
    """(n, 3) normals ("up" of the profile) for a 3D curve."""
    def z_up(tt):
        up = np.array([0.0, 0, 1])
        n = up - (tt @ up)[..., None] * tt
        bad = np.linalg.norm(n, axis=-1) < 1e-6
        if np.any(bad):
            alt = np.array([0.0, 1, 0])
            n = np.where(bad[..., None], alt - (tt @ alt)[..., None] * tt, n)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    if twist == _TWIST_Z_UP:
        normals = z_up(t)
    else:  # minimum twist: each normal turned the way the tangent turns
        normals = np.zeros_like(t)
        normals[0] = z_up(t[0])
        for i in range(len(p) - 1):
            normals[i + 1] = _turn(normals[i], t[i], t[i + 1])
        if cyclic and len(p) > 2:  # spread the twist left over when coming back to the start
            closing = _turn(normals[-1], t[-1], t[0])
            side = np.cross(t[0], normals[0])
            angle = math.atan2(closing @ side, closing @ normals[0])
            normals = _rotate(normals, t, -angle * np.arange(len(p)) / len(p))
    return _rotate(normals, t, tilt)


def _turn(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """v turned by the smallest rotation taking unit vector a to unit vector b."""
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return v
    axis = axis / s
    angle = math.atan2(s, float(a @ b))
    c, sn = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * sn + axis * (axis @ v) * (1 - c)


def _rotate(v: np.ndarray, axis: np.ndarray, angle) -> np.ndarray:
    """v turned about each (unit) axis by angle (Rodrigues)."""
    angle = np.broadcast_to(np.asarray(angle, np.float64), (len(v),))
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (np.einsum("ij,ij->i", axis, v)[:, None]) * (1 - c)


def _sweep(cu, splines: list[_Spline], name: str) -> tuple[Mesh, np.ndarray] | None:
    flag = int(cu.get("flag", 0))
    is_3d = bool(flag & _CU_3D) and int(cu.get("type", 2)) != 4
    profile, lift, closed, cap_top, cap_bottom = _profile(cu, is_3d)
    twist = int(cu.get("twist_mode", 3))
    positions, sizes, corners, materials, smooth = [], [], [], [], []
    base = 0
    outlines = []

    def add(points, faces_sizes, faces, material, is_smooth):
        nonlocal base
        positions.append(points)
        sizes.append(faces_sizes)
        corners.append(faces + base)
        materials.append(np.full(len(faces_sizes), material, np.int64))
        smooth.append(np.full(len(faces_sizes), is_smooth))
        base += len(points)

    for sp in splines:
        p = sp.points.copy()
        if not is_3d:
            p[:, 2] = 0
        if sp.cyclic and len(p) > 1 and np.allclose(p[0], p[-1]):
            p, sp.tilt, sp.radius = p[:-1], sp.tilt[:-1], sp.radius[:-1]
        n = len(p)
        if n < 2:
            continue
        if not is_3d and sp.cyclic and n >= 3:
            outlines.append((p[:, :2], sp.material))
        if not len(profile):
            continue
        t, widen = _tangents(p, sp.cyclic, sp.ends)
        if is_3d:
            up = _frames(p, t, sp.tilt, sp.cyclic, twist)
            side = np.cross(up, t)
            widen = np.ones(n)
        else:
            up = np.tile([0.0, 0, 1], (n, 1))
            side = np.cross(up, t)
        sl = np.linalg.norm(side, axis=1, keepdims=True)
        side = np.divide(side, sl, out=np.zeros_like(side), where=sl > 0)
        radius = np.where(sp.radius > 0, sp.radius, 0.0)
        k = len(profile)
        # the bevel scaled by each point's radius (2D bevels widened at corners), the extrusion not
        s = profile[:, 0][None, :] * (radius * widen)[:, None]
        h = profile[:, 1][None, :] * radius[:, None] + lift[None, :]
        ring =p[:, None, :] + s[..., None] * side[:, None, :] + h[..., None] * up[:, None, :]
        rows = n if sp.cyclic else n - 1
        cols = k if closed else k - 1
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        i2, j2 = (i + 1) % n, (j + 1) % k
        quads = np.stack([i * k + j, i * k + j2, i2 * k + j2, i2 * k + j], -1).reshape(-1, 4)
        add(ring.reshape(-1, 3), np.full(len(quads), 4, np.int64), quads.ravel(), sp.material, sp.smooth)
        if is_3d and flag & _CU_FILL_CAPS and not sp.cyclic and closed and k >= 3:
            for row, reverse in ((0, False), (n - 1, True)):
                cap = np.arange(k) if not reverse else np.arange(k)[::-1]
                add(ring[row], np.array([k]), cap, sp.material, False)
    caps = []  # (height, facing up)
    if cap_top == cap_bottom:  # flat: one cap either way
        caps = [(cap_top, True)] if flag & (_CU_FRONT | _CU_BACK) else []
    else:
        caps = [c for c, on in (((cap_top, True), flag & _CU_FRONT), ((cap_bottom, False), flag & _CU_BACK)) if on]
    if outlines and caps:
        loops = [o for o, _ in outlines]
        tris = _fill(loops)
        if len(tris):
            flat = np.concatenate(loops)
            owner = np.concatenate([np.full(len(o), m) for o, m in outlines])
            for height, facing_up in caps:
                pts = np.concatenate([flat, np.full((len(flat), 1), height)], 1)
                faces = tris if facing_up else tris[:, ::-1]
                add(pts, np.full(len(faces), 3, np.int64), faces.ravel(), 0, False)
                materials[-1] = owner[faces[:, 0]].astype(np.int64)
    if not positions:
        return None
    mesh = Mesh(name, np.concatenate(positions), np.concatenate(sizes), np.concatenate(corners).astype(np.int64),
                face_materials=np.concatenate(materials))
    return mesh, np.concatenate(smooth)


# ---- filling flat outlines

def _area(p: np.ndarray) -> float:
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _inside(pt, poly: np.ndarray) -> bool:
    x, y = pt
    a, b = poly, np.roll(poly, -1, axis=0)
    cross = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(cross & (x < xs)) % 2)


def _fill(loops: list[np.ndarray]) -> np.ndarray:
    """(T, 3) triangles over the closed outlines (indices into them, joined in order); outlines
    inside an odd number of others are holes."""
    offsets = np.cumsum([0] + [len(lp) for lp in loops])
    depth = [sum(_inside(lp[0], other) for j, other in enumerate(loops) if j != i) for i, lp in enumerate(loops)]
    areas = [abs(_area(lp)) for lp in loops]
    out = []
    for i, lp in enumerate(loops):
        if depth[i] % 2:
            continue
        holes = []
        for j, other in enumerate(loops):
            if depth[j] == depth[i] + 1 and _inside(other[0], lp):
                # its nearest container must be this outline
                containers = [c for c, cl in enumerate(loops) if c != j and depth[c] == depth[i] and _inside(other[0], cl)]
                if min(containers, key=lambda c: areas[c], default=i) == i:
                    holes.append(j)
        idx = np.arange(offsets[i], offsets[i + 1])
        if _area(lp) < 0:
            idx = idx[::-1]
        ring = list(idx)
        flat = np.concatenate(loops)
        for j in sorted(holes, key=lambda h: -loops[h][:, 0].max()):
            hidx = np.arange(offsets[j], offsets[j + 1])
            if _area(loops[j]) > 0:
                hidx = hidx[::-1]
            ring = _bridge(flat, ring, list(hidx))
        out.append(_ear_clip(flat, ring))
    out = [t for t in out if len(t)]
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


def _bridge(pts: np.ndarray, ring: list, hole: list) -> list:
    """The hole joined into the (counter-clockwise) ring by a cut from its rightmost point to a
    ring point it can see (Eberly's method)."""
    h = np.array(hole)
    m = int(np.argmax(pts[h, 0]))
    mx, my = pts[h[m]]
    r = np.array(ring)
    a, b = pts[r], pts[np.roll(r, -1)]
    crosses = ((a[:, 1] <= my) & (b[:, 1] > my)) | ((b[:, 1] <= my) & (a[:, 1] > my))
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0] + (my - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    ok = crosses & (xs >= mx)
    if not ok.any():
        best = int(np.argmin(np.linalg.norm(pts[r] - (mx, my), axis=1)))
    else:
        e = int(np.argmin(np.where(ok, xs, np.inf)))
        ix = xs[e]
        best = e if pts[r[e], 0] > pts[r[(e + 1) % len(r)], 0] else (e + 1) % len(r)
        # a ring point inside the triangle (M, hit, candidate) would block the view: take the one at the smallest angle
        tri = np.array([[mx, my], [ix, my], pts[r[best]]])
        inside = _in_triangle(pts[r], tri[0], tri[1], tri[2])
        inside[best] = False
        if inside.any():
            cand = np.nonzero(inside)[0]
            d = pts[r[cand]] - (mx, my)
            ang = np.abs(np.arctan2(d[:, 1], d[:, 0]))
            best = int(cand[np.lexsort((np.linalg.norm(d, axis=1), ang))[0]])
    hole_seq = hole[m:] + hole[:m] + [hole[m]]
    return ring[:best + 1] + hole_seq + ring[best:]


def _in_triangle(p: np.ndarray, a, b, c) -> np.ndarray:
    """Which points are strictly inside the triangle (none, if it's flat)."""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[..., 1] - o[1]) - (u[1] - o[1]) * (v[..., 0] - o[0])
    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    return ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))


def _ear_clip(pts: np.ndarray, ring: list) -> np.ndarray:
    """Triangles of a counter-clockwise ring (which may touch itself along hole cuts). An ear is
    a convex corner with no other point strictly inside it; points that sit on a straight line
    are dropped, and if no ear is left the most convex corner is cut anyway."""
    idx = list(ring)
    out = []
    span = np.ptp(pts[idx], axis=0).max() if idx else 0.0
    eps = 1e-12 * span * span
    misses, i = 0, 0
    while len(idx) > 3:
        n = len(idx)
        i %= n
        a, b, c = idx[i - 1], idx[i], idx[(i + 1) % n]
        pa, pb, pc = pts[a], pts[b], pts[c]
        turn = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        ear = False
        if turn > eps:  # no reflex point inside it or on its sides (other than its own corners)
            q = pts[idx]
            prev, nxt = np.roll(q, 1, axis=0), np.roll(q, -1, axis=0)
            reflex = ((q[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
                      - (q[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0])) < -eps
            d1 = (pb[0] - pa[0]) * (q[:, 1] - pa[1]) - (pb[1] - pa[1]) * (q[:, 0] - pa[0])
            d2 = (pc[0] - pb[0]) * (q[:, 1] - pb[1]) - (pc[1] - pb[1]) * (q[:, 0] - pb[0])
            d3 = (pa[0] - pc[0]) * (q[:, 1] - pc[1]) - (pa[1] - pc[1]) * (q[:, 0] - pc[0])
            own = np.all(q == pa, axis=1) | np.all(q == pb, axis=1) | np.all(q == pc, axis=1)
            ear = not (reflex & ~own & (d1 >= -eps) & (d2 >= -eps) & (d3 >= -eps)).any()
        if ear:
            out.append((a, b, c))
            idx.pop(i)
            i, misses = i - 1, 0
            continue
        i += 1
        misses += 1
        if misses > n:  # no ear: drop a flat corner, or cut the most convex one
            turns = []
            for k in range(n):
                qa, qb, qc = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % n]]
                turns.append((qb[0] - qa[0]) * (qc[1] - qa[1]) - (qb[1] - qa[1]) * (qc[0] - qa[0]))
            turns = np.array(turns)
            flat = np.nonzero(np.abs(turns) <= eps)[0]
            k = int(flat[0]) if len(flat) else int(np.argmax(turns))
            if not len(flat):
                out.append((idx[k - 1], idx[k], idx[(k + 1) % n]))
            idx.pop(k)
            i, misses = max(k - 1, 0), 0
    if len(idx) == 3:
        out.append(tuple(idx))
    return np.array(out, np.int64).reshape(-1, 3)


# ============================================================================ text

_FONT_MAGIC = (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf", b"wOFF", b"wOF2")
# Stand-ins for Blender's built-in font, closest first; they're scaled so capitals come out as tall
# as Blender's (0.682 of the text size), which leaves widths a few percent off
_BUILTIN_STANDINS = ("Inter.ttf", "Inter-Regular.ttf", "InterVariable.ttf", "arial.ttf", "Arial.ttf",
                     "LiberationSans-Regular.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "segoeui.ttf")
_BUILTIN_CAP_HEIGHT = 0.682


def _font_dirs() -> list[Path]:
    dirs = []
    windir = os.environ.get("WINDIR")
    if windir:
        dirs.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library" / "Fonts",
             Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/TTF"), Path("/usr/share/fonts/truetype"),
             Path.home() / ".fonts"]
    return dirs


def _builtin_font() -> bytes | None:
    for d in _font_dirs():
        for name in _BUILTIN_STANDINS:
            data = _read_font(d / name)
            if data is not None:
                return data
    return None


def _read_font(path: Path) -> bytes | None:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            if head not in _FONT_MAGIC:
                return None
            return head + f.read()
    except (OSError, ValueError):
        return None


def _font_data(cu, base_dir: Path | None) -> tuple[bytes | None, bool]:
    """The text's font file, and whether it's a stand-in for Blender's built-in font."""
    vf = cu.follow("vfont", "VFont") if cu.get("vfont", 0) else None
    if vf is None:
        return _builtin_font(), True
    if vf.get("packedfile", 0):
        pf = vf.follow("packedfile", "PackedFile")
        raw = pf.raw("data", "u1", int(pf.get("size", 0))) if pf is not None else None
        if raw is not None and raw[:4].tobytes() in _FONT_MAGIC:
            return raw.tobytes(), False
    path = vf.get("name", "") or ""
    if path and path != "<builtin>":
        # the folder rule, as for textures: the path as written if it stays near the file (network
        # paths never), then its file name there and in the system's font folders (a font from the
        # machine the file was made on)
        rel = path.removeprefix("//")
        name = rel.replace("\\", "/").rsplit("/", 1)[-1]
        candidates = [linked_path(rel, base_dir), linked_path(name, base_dir),
                      linked_path(name, base_dir.parent, base_dir.parent)] if base_dir is not None else []
        candidates += [linked_path(name, d, d) for d in _font_dirs()]
        for c in candidates:
            data = _read_font(c) if c is not None else None
            if data is not None:
                return data, False
    return _builtin_font(), True


def _cap_height(font, glyphs, cmap) -> float:
    from fontTools.pens.boundsPen import BoundsPen
    name = cmap.get(ord("H"))
    if name in glyphs:
        pen = BoundsPen(glyphs)
        glyphs[name].draw(pen)
        if pen.bounds and pen.bounds[3] > 0:
            return float(pen.bounds[3])
    os2 = font.get("OS/2")
    cap = getattr(os2, "sCapHeight", 0) if os2 is not None else 0
    return float(cap) if cap > 0 else 0.7 * float(font["head"].unitsPerEm)


def _outline_pen(glyph_set):
    """A fontTools pen that collects a glyph's contours as line and cubic segments (quadratic
    ones are raised to cubic, exactly; components are drawn in place)."""
    from fontTools.pens.basePen import BasePen

    class Outline(BasePen):
        def __init__(self):
            super().__init__(glyph_set)
            self.contours: list[list] = []
            self.current: list | None = None

        def _moveTo(self, p):
            self._flush()
            self.current = [("move", np.array(p, np.float64))]

        def _lineTo(self, p):
            self.current.append(("line", np.array(p, np.float64)))

        def _curveToOne(self, p1, p2, p3):
            self.current.append(("cubic", *(np.array(p, np.float64) for p in (p1, p2, p3))))

        def _closePath(self):
            self._flush()

        _endPath = _closePath

        def _flush(self):
            if self.current and len(self.current) > 1:
                self.contours.append(self.current)
            self.current = None

    return Outline()


def _contour_points(contour, scale: float, offset, shear: float, resolution: int) -> np.ndarray:
    """A closed outline, each curved side sampled at the resolution and each straight one taken
    as its start point only (as Blender treats vector handles)."""
    pts = []
    start = contour[0][1]
    for seg in contour[1:]:
        if seg[0] == "line":
            pts.append(start[None])
            start = seg[1]
        else:
            _, c1, c2, end = seg
            t = np.arange(resolution)[:, None] / resolution
            u = 1 - t
            pts.append(u ** 3 * start + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t ** 3 * end)
            start = end
    if not np.allclose(start, contour[0][1]):
        pts.append(start[None])
    p = np.concatenate(pts) * scale
    p[:, 0] += shear * p[:, 1]
    return p + offset


def _text_splines(cu, base_dir: Path | None) -> list[_Spline]:
    text = _text(cu)
    if not text.strip():
        return []
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return []
    data, builtin = _font_data(cu, base_dir)
    if data is None:
        return []
    try:
        font = TTFont(io.BytesIO(data), fontNumber=0, lazy=True)
        glyphs, cmap = font.getGlyphSet(), font.getBestCmap() or {}
        head, hhea = font["head"], font["hhea"]
        unit = 1.0 / max(int(head.yMax) - int(head.yMin), 1)  # Blender scales glyphs by the font's height
        if builtin:
            unit = _BUILTIN_CAP_HEIGHT / _cap_height(font, glyphs, cmap)
    except Exception:  # noqa: BLE001 - a damaged or unsupported font: no text
        return []
    size = float(cu.get("fsize", 1.0))
    scale = unit * size
    spacing = float(cu.get("spacing", 1.0))
    wordspace = float(cu.get("wordspace", 1.0))
    linedist = float(cu.get("linedist", 1.0))
    shear = float(cu.get("shear", 0.0))
    resolution = max(1, min(int(cu.get("resolu", 12) or 12), 64))
    extra = (0.5 * spacing - 0.5) * size
    lines = text.split("\n")
    infos = _char_materials(cu, len(text))
    outlines, widths = [], []
    k = 0
    for row, line in enumerate(lines):
        x = 0.0
        placed = []
        for ch in line:
            name = cmap.get(ord(ch))
            glyph = glyphs.get(name) if name is not None else None
            advance = (glyph.width if glyph is not None else 0) * scale
            if ch == " ":
                advance *= wordspace
            if glyph is not None and ch.strip():
                pen = _outline_pen(glyphs)
                try:
                    glyph.draw(pen)
                except Exception:  # noqa: BLE001 - a broken glyph is left out
                    pen.contours = []
                placed.append((pen.contours, x, infos[k] if k < len(infos) else 0))
            x += advance + extra
            k += 1
        k += 1  # the newline
        outlines.append(placed)
        widths.append(x - extra if line else 0.0)
    ascent, descent = float(hhea.ascent) * scale, -float(hhea.descent) * scale
    n_lines = len(lines)
    align_y = int(cu.get("align_y", 0))
    shift_y = {_TOP: -ascent, _CENTER: ((n_lines - 1) * linedist * size + descent - ascent) / 2,
               _BOTTOM: (n_lines - 1) * linedist * size + descent,
               _BOTTOM_BASELINE: (n_lines - 1) * linedist * size}.get(align_y, 0.0)
    align_x = int(cu.get("spacemode", 0))
    xof, yof = float(cu.get("xof", 0.0)), float(cu.get("yof", 0.0))
    out = []
    for row, placed in enumerate(outlines):
        shift_x = {_ALIGN_CENTER: -widths[row] / 2, _ALIGN_RIGHT: -widths[row]}.get(align_x, 0.0)
        y = -row * linedist * size + shift_y + yof
        for contours, x, material in placed:
            for contour in contours:
                p = _contour_points(contour, scale, (x + shift_x + xof, y), shear, resolution)
                if len(p) >= 3:
                    pts = np.concatenate([p, np.zeros((len(p), 1))], 1)
                    out.append(_Spline(pts, np.zeros(len(p)), np.ones(len(p)), True, material, False))
    return out


def _text(cu) -> str:
    n = int(cu.get("len", 0) or 0)
    raw = cu.raw("str", "u1", n) if n > 0 else None
    return raw.tobytes().decode("utf-8", "replace") if raw is not None else ""


def _char_materials(cu, n: int) -> list[int]:
    if not cu.get("strinfo", 0) or n <= 0:
        return []
    f = cu.f
    size = f.struct_size["CharInfo"]
    raw = cu.raw("strinfo", "u1", size * n)
    if raw is None:
        return []
    off = f.structs[f.struct_index["CharInfo"]][1]["mat_nr"][0]
    return np.ndarray((n,), "<i2", raw.tobytes(), off, (size,)).astype(int).tolist()


# ============================================================================ metaballs

def family(name: str) -> str:
    """The base name a metaball object joins: "Mball.001" joins "Mball"."""
    return re.sub(r"\.\d+$", "", name)


def metaball_mesh(elements: list[tuple[int, int, np.ndarray, float, float, np.ndarray]], resolution: float,
                  threshold: float, name: str = "metaball") -> tuple[Mesh, np.ndarray] | None:
    """The surface of elements given as (type, flag, matrix into the base object's space, radius,
    stiffness, (x, y, z) sizes)."""
    elements = [e for e in elements if not e[1] & _MB_HIDE and e[3] > 0]
    if not elements or threshold <= 0:
        return None
    resolution = max(float(resolution), 1e-4)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for kind, flag, matrix, radius, stiff, dims in elements:
        if flag & _MB_NEGATIVE:
            continue
        if kind == _MB_ELLIPSOID:
            reach = radius * np.abs(np.asarray(dims, np.float64))
        else:
            reach = radius + _extent(kind, dims)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * reach
        world = corners @ matrix[:3, :3].T + matrix[:3, 3]
        lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
    if not np.isfinite(lo).all():
        return None
    lo, hi = lo - resolution, hi + resolution
    counts = np.ceil((hi - lo) / resolution).astype(np.int64) + 1
    if np.prod(counts.astype(np.float64)) > 8e6:  # keep the grid to a sane size
        resolution *= (np.prod(counts.astype(np.float64)) / 8e6) ** (1 / 3)
        counts = np.ceil((hi - lo) / resolution).astype(np.int64) + 1
    axes = [lo[k] + resolution * np.arange(counts[k]) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inverses = [np.linalg.inv(e[2][:3, :3]) for e in elements]

    def field(points):
        out = np.full(len(points), -threshold)
        for (kind, flag, matrix, radius, stiff, dims), inv in zip(elements, inverses):
            local = (points - matrix[:3, 3]) @ inv.T
            out += (-1 if flag & _MB_NEGATIVE else 1) * _density(kind, local, radius, stiff, dims)
        return out

    got = _march(field(grid).reshape(tuple(counts)), lo, resolution, field)
    if got is None:
        return None
    positions, tris = got
    mesh = Mesh(name, positions, np.full(len(tris), 3, np.int64), tris.ravel().astype(np.int64),
                face_materials=np.zeros(len(tris), np.int64))
    return mesh, np.ones(len(tris), bool)


def _extent(kind: int, dims: np.ndarray) -> np.ndarray:
    dims = np.abs(np.asarray(dims, np.float64))
    if kind == _MB_TUBE:
        return np.array([dims[0], 0, 0])
    if kind in _MB_TUBE_AXES:
        e = np.zeros(3)
        e[_MB_TUBE_AXES[kind]] = dims[_MB_TUBE_AXES[kind]]
        return e
    if kind == _MB_PLANE:
        return np.array([dims[0], dims[1], 0])
    if kind == _MB_CUBE:
        return dims
    return np.zeros(3)  # a ball, or an ellipsoid (scaled instead)


def _density(kind: int, p: np.ndarray, radius: float, stiff: float, dims) -> np.ndarray:
    dims = np.abs(np.asarray(dims, np.float64))
    q = p.copy()
    if kind == _MB_ELLIPSOID:
        q = q / np.where(dims > 1e-9, dims, 1e-9)
    else:
        e = _extent(kind, dims)
        q = np.sign(q) * np.maximum(np.abs(q) - e, 0)
    d2 = np.einsum("ij,ij->i", q, q) / (radius * radius)
    v = np.clip(1 - d2, 0, None)
    return stiff * v * v * v


# the six tetrahedra of a cube around its 0-7 diagonal (corner = x + 2y + 4z)
_TETS = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
_CORNER = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)])


def _march(f: np.ndarray, origin: np.ndarray, step: float, field=None):
    """Marching tetrahedra over a grid of field values (positive inside): points on grid edges
    shared between neighbouring cells (moved onto the surface by bisection when the field
    function is given), triangles facing out."""
    nx, ny, nz = f.shape
    if min(nx, ny, nz) < 2:
        return None
    inside = f > 0
    # cells with corners on both sides
    any_in = np.zeros((nx - 1, ny - 1, nz - 1), bool)
    all_in = np.ones_like(any_in)
    for dx, dy, dz in _CORNER:
        c = inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]
        any_in |= c
        all_in &= c
    active = np.argwhere(any_in & ~all_in)
    if not len(active):
        return None
    ijk = active[:, None, :] + _CORNER[None, :, :]
    corner_idx = (ijk[..., 0] * ny + ijk[..., 1]) * nz + ijk[..., 2]  # (cells, 8)
    tets = corner_idx[:, _TETS].reshape(-1, 4)  # (cells * 6, 4) grid point indices
    fv = f.ravel()
    val = fv[tets]
    ins = val > 0
    count = ins.sum(1)
    keep = (count > 0) & (count < 4)
    tets, val, ins, count = tets[keep], val[keep], ins[keep], count[keep]
    # put the inside corners first (stable), so each case is a fixed pattern
    order = np.argsort(~ins, axis=1, kind="stable")
    t = np.take_along_axis(tets, order, 1)
    edges = []  # (a, b) grid-point pairs per new triangle corner
    tri_inside = []  # an inside grid point per triangle, to face it outward

    one = count == 1
    if one.any():
        a = t[one]
        edges.append(np.stack([np.stack([a[:, 0], a[:, k]], 1) for k in (1, 2, 3)], 1))
        tri_inside.append(a[:, 0])
    three = count == 3
    if three.any():
        a = t[three]
        edges.append(np.stack([np.stack([a[:, k], a[:, 3]], 1) for k in (0, 1, 2)], 1))
        tri_inside.append(a[:, 0])
    two = count == 2
    if two.any():
        a = t[two]
        e = [np.stack([a[:, 0], a[:, 2]], 1), np.stack([a[:, 0], a[:, 3]], 1),
             np.stack([a[:, 1], a[:, 3]], 1), np.stack([a[:, 1], a[:, 2]], 1)]
        edges.append(np.stack([e[0], e[1], e[2]], 1))
        edges.append(np.stack([e[0], e[2], e[3]], 1))
        tri_inside += [a[:, 0], a[:, 0]]
    edges = np.concatenate(edges)  # (T, 3, 2)
    tri_inside = np.concatenate(tri_inside)
    pairs = np.sort(edges.reshape(-1, 2), axis=1)
    key = pairs[:, 0].astype(np.int64) * (nx * ny * nz) + pairs[:, 1]
    unique, inverse = np.unique(key, return_inverse=True)
    a, b = unique // (nx * ny * nz), unique % (nx * ny * nz)
    fa, fb = fv[a], fv[b]
    w = np.clip(fa / np.where(fa - fb != 0, fa - fb, 1), 0, 1)

    def point(i):
        ijk = np.stack([i // (ny * nz), (i // nz) % ny, i % nz], 1)
        return origin + ijk * step

    pa, pb = point(a), point(b)
    positions = pa + (pb - pa) * w[:, None]
    if field is not None:  # pa is inside (fa > 0) and pb outside, or the other way round
        inner = np.where((fa > 0)[:, None], pa, pb)
        outer = np.where((fa > 0)[:, None], pb, pa)
        for _ in range(10):
            mid = (inner + outer) / 2
            go_in = field(mid) > 0
            inner = np.where(go_in[:, None], mid, inner)
            outer = np.where(go_in[:, None], outer, mid)
        positions = (inner + outer) / 2
    tris = inverse.reshape(-1, 3)
    # face away from the inside
    p = positions[tris]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    outward = p.mean(1) - point(tri_inside)
    flip = np.einsum("ij,ij->i", normal, outward) < 0
    tris[flip] = tris[flip][:, ::-1]
    # drop triangles that collapsed onto a grid point
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[ok]
    area = np.linalg.norm(np.cross(positions[tris[:, 1]] - positions[tris[:, 0]],
                                   positions[tris[:, 2]] - positions[tris[:, 0]]), axis=1)
    tris = tris[area > 1e-14 * step * step]
    if not len(tris):
        return None
    used = np.unique(tris)
    renumber = np.full(len(positions), -1, np.int64)
    renumber[used] = np.arange(len(used))
    return positions[used], renumber[tris]
