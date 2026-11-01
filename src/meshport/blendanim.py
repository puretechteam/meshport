"""Blender actions -> the F-curves that animate an object, evaluated as Blender evaluates them.

An object's AnimData points at its action. Up to Blender 4.3 an action keeps its F-curves in a
list; from 4.4 actions are layered: layers hold strips, a keyframe strip's data holds a
channel bag per slot, and the object's AnimData says which slot is its own. Each F-curve has a
data path ("location", "rotation_euler", ...), an array index and its keys (BezTriple: the left
handle, the key and the right handle, as (frame, value)).

Between two keys the first key's interpolation applies: constant, linear, Bezier (the handles,
shortened when they reach past each other's key, as Blender does, with the curve solved for the
frame), or one of the easing modes (back, bounce, circular, cubic, elastic, exponential,
quadratic, quartic, quintic, sine), each easing in, out or both, by Robert Penner's published
equations with Blender's two changes: exponential easing is shifted and stretched by 2^-10 so it
starts and ends on its keys, and elastic easing's wobble is scaled by the key's amplitude (and
faded in where the amplitude is under the change). Before the first key and after the last, the
curve holds (constant extrapolation) or carries on in a line (linear).

F-curve modifiers apply as in Blender: Cycles (repeat, repeat with offset, mirrored; a number of
cycles or for ever, on either side), Stepped, Limits and Generator (expanded or factorised
polynomials, replacing the curve or added to it), each with its frame range, blend in and out,
and influence. Time modifiers go from the last to the first, value modifiers from the first to
the last, and Cycles only as the first. Noise, Envelope and Python modifiers, drivers and the
NLA are left out.
"""

from __future__ import annotations

import math

import numpy as np

CONSTANT, LINEAR, BEZIER = 0, 1, 2
BACK, BOUNCE, CIRC, CUBIC, ELASTIC, EXPO, QUAD, QUART, QUINT, SINE = range(3, 13)
EASE_AUTO, EASE_IN, EASE_OUT, EASE_IN_OUT = 0, 1, 2, 3
# FModifier types and flags
GENERATOR, CYCLES, LIMITS, STEPPED = 1, 4, 8, 9
_DISABLED, _MUTED, _RANGE, _INFLUENCE = 1, 8, 16, 32
_CYCLE_REPEAT, _CYCLE_OFFSET, _CYCLE_MIRROR = 1, 2, 3
_LIMIT_XMIN, _LIMIT_XMAX, _LIMIT_YMIN, _LIMIT_YMAX = 1, 2, 4, 8
_STEPPED_NO_BEFORE, _STEPPED_NO_AFTER = 1, 2


class Modifier:
    """One F-curve modifier: its type, flags, influence, frame range with blend in and out, and
    its own settings (a dict)."""

    def __init__(self, kind: int, flag: int, influence: float, start: float, end: float, blend_in: float,
                 blend_out: float, settings: dict):
        self.kind, self.flag, self.influence = kind, flag, influence
        self.start, self.end, self.blend_in, self.blend_out, self.settings = start, end, blend_in, blend_out, settings

    def weight(self, t: np.ndarray) -> np.ndarray:
        """How much the modifier counts at each frame (Blender's eval_fmodifier_influence)."""
        w = np.full(t.shape, self.influence if self.flag & _INFLUENCE else 1.0)
        if self.flag & _RANGE:
            a, b = self.start, self.end
            out = (t <= a) | (t >= b)
            if self.blend_in:
                ramp = (t >= a) & (t <= a + self.blend_in)
                w = np.where(ramp, w * (t - a) / self.blend_in, w)
            if self.blend_out:
                ramp = (t <= b) & (t >= b - self.blend_out) & ~((t >= a) & (t <= a + self.blend_in) & (self.blend_in != 0))
                w = np.where(ramp, w * (t - b) / -self.blend_out, w)
            w = np.where(out, 0.0, w)
        return w

    def active(self, t: np.ndarray) -> np.ndarray:
        if self.flag & _RANGE:
            return (t >= self.start) & (t <= self.end)
        return np.ones(t.shape, bool)


class FCurve:
    def __init__(self, path: str, index: int, keys: np.ndarray, ipo: np.ndarray, easing: np.ndarray,
                 params: np.ndarray, extend: int, modifiers: list[Modifier] | None = None):
        # keys: (N, 3, 2), each key's left handle, key and right handle as (frame, value);
        # params: (N, 3), each key's back, amplitude and period
        self.path, self.index, self.keys, self.ipo, self.easing, self.params, self.extend = \
            path, index, keys, ipo, easing, params, extend
        self.modifiers = [m for m in modifiers or [] if not m.flag & (_DISABLED | _MUTED)]
        if any(m.kind == CYCLES for m in self.modifiers[1:]):  # Blender runs Cycles only first in the stack
            self.modifiers = self.modifiers[:1] + [m for m in self.modifiers[1:] if m.kind != CYCLES]

    @property
    def carries_on(self) -> bool:
        """Whether the curve keeps changing past its keys (linear extrapolation, or modifiers)."""
        return bool(self.extend or self.modifiers)

    def evaluate(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, np.float64)
        if not self.modifiers:
            return self._keyed(frames)
        t, offset = frames, np.zeros(frames.shape)
        for m in reversed(self.modifiers):  # time first, last modifier to first
            if m.kind in (CYCLES, STEPPED, LIMITS):
                new, shift = self._time(m, t)
                w, on = m.weight(t), m.active(t)
                if m.kind == CYCLES:
                    offset = np.where(on, shift, offset)
                t = np.where(on, w * new + (1 - w) * t, t)
        value = self._keyed(t)
        for m in self.modifiers:  # then values, first to last
            w, on = m.weight(t), m.active(t)
            if m.kind == CYCLES:
                new = value + offset
            elif m.kind == LIMITS:
                new = value
                if m.settings["flag"] & _LIMIT_YMIN:
                    new = np.maximum(new, m.settings["ymin"])
                if m.settings["flag"] & _LIMIT_YMAX:
                    new = np.minimum(new, m.settings["ymax"])
            elif m.kind == GENERATOR:
                new = _generate(m.settings, t, value)
            else:
                continue
            value = np.where(on, w * new + (1 - w) * value, value)
        return value

    def _time(self, m: Modifier, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The frame a time modifier evaluates the keys at, and (for Cycles) the value offset."""
        st = m.settings
        if m.kind == STEPPED:
            step = st["step"] or 1.0
            snapped = np.trunc((t - st["offset"]) / step) * step + st["offset"]  # C's int(): towards zero
            keep = np.zeros(t.shape, bool)
            if st["flag"] & _STEPPED_NO_BEFORE:
                keep |= t < st["start"]
            if st["flag"] & _STEPPED_NO_AFTER:
                keep |= t > st["end"]
            return np.where(keep, t, snapped), np.zeros(t.shape)
        if m.kind == LIMITS:
            out = t
            if st["flag"] & _LIMIT_XMIN:
                out = np.where(out < st["xmin"], st["xmin"], out)
            if st["flag"] & _LIMIT_XMAX:
                out = np.where(out > st["xmax"], st["xmax"], out)
            return out, np.zeros(t.shape)
        return _cycle(self.keys[0, 1], self.keys[-1, 1], st, t)

    def _keyed(self, frames: np.ndarray) -> np.ndarray:
        keys = self.keys
        x, y = keys[:, 1, 0], keys[:, 1, 1]
        if len(keys) == 1:
            return np.full(frames.shape, y[0])
        out = np.empty(frames.shape)
        k = np.clip(np.searchsorted(x, frames, side="right") - 1, 0, len(keys) - 2)
        for seg in np.unique(k):
            sel = k == seg
            out[sel] = self._segment(int(seg), frames[sel])
        before, after = frames < x[0], frames > x[-1]
        out[before] = y[0] + self._slope(0) * (frames[before] - x[0]) if self.extend else y[0]
        out[after] = y[-1] + self._slope(-1) * (frames[after] - x[-1]) if self.extend else y[-1]
        return out

    def _slope(self, end: int) -> float:
        """The slope a linearly extrapolated curve carries on with past its first or last key:
        along the key's handle for a Bezier key, else towards its neighbour."""
        keys, n = self.keys, len(self.keys)
        i = 0 if end == 0 else n - 1
        if self.ipo[i if end == 0 else n - 2] == BEZIER or (end != 0 and self.ipo[i] == BEZIER):
            handle = keys[i, 0] if end == 0 else keys[i, 2]
            dx = keys[i, 1, 0] - handle[0]
            if dx:
                return (keys[i, 1, 1] - handle[1]) / dx
            return 0.0
        a, b = (keys[0, 1], keys[1, 1]) if end == 0 else (keys[n - 2, 1], keys[n - 1, 1])
        return (b[1] - a[1]) / (b[0] - a[0]) if b[0] != a[0] else 0.0

    def _segment(self, i: int, f: np.ndarray) -> np.ndarray:
        a, b = self.keys[i], self.keys[i + 1]
        x0, y0, x1, y1 = a[1, 0], a[1, 1], b[1, 0], b[1, 1]
        ipo = int(self.ipo[i])
        d = x1 - x0
        if ipo == CONSTANT or d <= 0:
            return np.where(f >= x1, y1, y0)
        t = np.clip(f - x0, 0.0, d)
        if ipo == LINEAR:
            return y0 + (y1 - y0) * t / d
        if ipo == BEZIER:
            return _bezier(a[1], a[2], b[0], b[1], np.clip(f, x0, x1))
        back, amplitude, period = self.params[i]
        return _ease(ipo, int(self.easing[i]), t, y0, y1 - y0, d, back, amplitude, period)


def _cycle(first, last, st: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blender's Cycles modifier (fcm_cycles_time): the frame within the keyed range, and the
    value offset of repeating with offset."""
    t = np.asarray(t, np.float64)
    out, offset = t.copy(), np.zeros(t.shape)
    dx, dy = last[0] - first[0], last[1] - first[1]
    if dx == 0:
        return out, offset
    for side, mode, count, at in ((-1, st["before_mode"], st["before_cycles"], first[0]),
                                  (1, st["after_mode"], st["after_cycles"], last[0])):
        sel = (t < first[0]) if side < 0 else (t > last[0])
        if not mode or not sel.any():
            continue
        e = t[sel]
        cycle = side * (e - at) / dx
        within = np.fmod(e - at, dx)
        go = np.ones(e.shape, bool) if count == 0 else cycle <= count
        new = first[0] + within
        mirror = mode == _CYCLE_MIRROR
        if mirror:
            odd = (np.trunc(cycle + 1).astype(np.int64) % 2) != 0
            new = np.where(odd, (first[0] if side < 0 else last[0]) - within, new)
        edge = within == 0
        edge_t = np.full(e.shape, last[0] if side == 1 else first[0])
        if mirror:
            edge_t = np.where(np.trunc(cycle).astype(np.int64) % 2 != 0, first[0] if side == 1 else last[0], edge_t)
        new = np.where(edge, edge_t, new)
        new = np.where(new < first[0], new + dx, new)
        shift = np.zeros(e.shape)
        if mode == _CYCLE_OFFSET:
            shift = (np.floor((e - at) / dx) if side < 0 else np.ceil((e - at) / dx)) * dy
        out[sel] = np.where(go, new, e)
        offset[sel] = np.where(go, shift, 0.0)
    return out, offset


def _generate(st: dict, t: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Blender's Generator modifier: an expanded polynomial (c0 + c1 t + c2 t^2 ...) or a
    factorised one ((a1 t + b1)(a2 t + b2)...), in place of the curve or added to it."""
    c, order = st["coefficients"], st["order"]
    if not order or not len(c):
        return value
    if st["mode"] == 1:
        made = np.ones(t.shape)
        for i in range(min(order, len(c) // 2)):
            made = made * (c[2 * i] * t + c[2 * i + 1])
    else:
        made = np.zeros(t.shape)
        power = np.ones(t.shape)
        for k in c:
            made = made + k * power
            power = power * t
    return value + made if st["flag"] & 1 else made


def _bezier(p0, p1, p2, p3, f: np.ndarray) -> np.ndarray:
    """The value of a keyframe's Bezier segment at frames f: its handles first shortened, together,
    so they don't reach past each other's key, then the curve solved for each frame."""
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    h1, h2 = p0 - p1, p3 - p2
    len1, len2, span = abs(h1[0]), abs(h2[0]), p3[0] - p0[0]
    if len1 + len2 > span > 0:
        fac = span / (len1 + len2)
        p1, p2 = p0 - fac * h1, p3 - fac * h2
    lo, hi = np.zeros_like(f), np.ones_like(f)
    for _ in range(48):  # x runs one way along the segment, so halving finds the parameter
        s = (lo + hi) / 2
        r = 1 - s
        x = r ** 3 * p0[0] + 3 * r * r * s * p1[0] + 3 * r * s * s * p2[0] + s ** 3 * p3[0]
        lo, hi = np.where(x < f, s, lo), np.where(x < f, hi, s)
    s = (lo + hi) / 2
    r = 1 - s
    return r ** 3 * p0[1] + 3 * r * r * s * p1[1] + 3 * r * s * s * p2[1] + s ** 3 * p3[1]


def _ease(ipo: int, easing: int, t, b: float, c: float, d: float, back: float, amplitude: float, period: float):
    """Robert Penner's easing equations: t time into the segment, b the start value, c the change,
    d the segment's length. Automatic easing eases in, except for the "dynamic" back, bounce and
    elastic, which ease out (as Blender has it)."""
    if easing == EASE_AUTO:
        easing = EASE_OUT if ipo in (BACK, BOUNCE, ELASTIC) else EASE_IN
    t = np.asarray(t, np.float64)

    def ease_in(t):
        return _IN[ipo](t, b, c, d, back, amplitude, period)

    def ease_out(t):
        return _OUT[ipo](t, b, c, d, back, amplitude, period)

    if easing == EASE_IN:
        return ease_in(t)
    if easing == EASE_OUT:
        return ease_out(t)
    return _IN_OUT[ipo](t, b, c, d, back, amplitude, period)


def _bounce_out(t, b, c, d, *_):
    t = t / d
    return np.where(t < 1 / 2.75, c * (7.5625 * t * t) + b,
                    np.where(t < 2 / 2.75, c * (7.5625 * (t - 1.5 / 2.75) ** 2 + 0.75) + b,
                             np.where(t < 2.5 / 2.75, c * (7.5625 * (t - 2.25 / 2.75) ** 2 + 0.9375) + b,
                                      c * (7.5625 * (t - 2.625 / 2.75) ** 2 + 0.984375) + b)))


def _bounce_in(t, b, c, d, *_):
    return c - _bounce_out(d - t, 0, c, d) + b


def _elastic(t, b, c, d, amplitude, period, side):
    """Blender's elastic easing (BLI_easing): Penner's, with the wobble scaled by amplitude / |c|
    when the amplitude is under the change (and faded in over the first quarter period).
    side: -1 in, 1 out, 0 in and out."""
    t = np.asarray(t, np.float64)
    if side == 0:
        u = t / (d / 2) - 1
        period = period if period else d * (0.3 * 1.5)
    else:
        u = t / d - 1 if side < 0 else -(t / d)
        period = period if period else d * 0.3
    f = np.ones_like(u)
    if not amplitude or amplitude < abs(c):
        s = period / 4
        if c:
            scale = amplitude / abs(c) if amplitude else 0.0
            near = np.abs(u * d) < abs(s)
            fade = np.abs(u * d) / abs(s) if s else np.zeros_like(u)
            f = np.where(near, scale * fade + (1 - fade), scale)
        amplitude = c
    else:
        s = period / (2 * math.pi) * math.asin(c / amplitude)
    with np.errstate(over="ignore", invalid="ignore"):
        if side == 0:  # the first half eases in, the second (time turned round) out
            in_half = u < 0
            w = np.where(in_half, u, -u)
            wave = amplitude * 2.0 ** (10 * w) * np.sin((w * d - s) * (2 * math.pi) / period)
            v = np.where(in_half, -0.5 * f * wave + b, 0.5 * f * wave + c + b)
        else:
            wave = amplitude * 2.0 ** (10 * u) * np.sin((u * d - s) * (2 * math.pi) / period)
            v = -f * wave + b if side < 0 else f * wave + c + b
    return np.where(t <= 0, b, np.where(t >= d, b + c, v))


def _elastic_in(t, b, c, d, _back, amplitude, period):
    return _elastic(t, b, c, d, amplitude, period, -1)


def _elastic_out(t, b, c, d, _back, amplitude, period):
    return _elastic(t, b, c, d, amplitude, period, 1)


def _elastic_in_out(t, b, c, d, _back, amplitude, period):
    return _elastic(t, b, c, d, amplitude, period, 0)


# Blender's exponential easing: 2^(10(x - 1)) shifted down by 2^-10 and stretched back to 0..1
_POW_MIN, _POW_SCALE = 2.0 ** -10, 1 / (1 - 2.0 ** -10)


def _expo_in(t, b, c, d, *_):
    return np.where(t <= 0, b, c * (2.0 ** (10 * (t / d - 1)) - _POW_MIN) * _POW_SCALE + b)


def _expo_out(t, b, c, d, *_):
    return np.where(t <= 0, b, c * (1 - (2.0 ** (-10 * t / d) - _POW_MIN) * _POW_SCALE) + b)


def _poly(n):
    def ease_in(t, b, c, d, *_):
        return c * (t / d) ** n + b

    def ease_out(t, b, c, d, *_):
        return c * (1 - (1 - t / d) ** n) + b

    def ease_in_out(t, b, c, d, *_):
        u = t / (d / 2)
        return np.where(u < 1, c / 2 * u ** n + b, c / 2 * (2 - (2 - u) ** n) + b)
    return ease_in, ease_out, ease_in_out


_QUAD, _CUBIC, _QUART, _QUINT = (_poly(n) for n in (2, 3, 4, 5))
_IN = {
    BACK: lambda t, b, c, d, s, *_: c * (t / d) ** 2 * ((s + 1) * (t / d) - s) + b,
    BOUNCE: _bounce_in,
    CIRC: lambda t, b, c, d, *_: -c * (np.sqrt(np.maximum(1 - (t / d) ** 2, 0)) - 1) + b,
    CUBIC: _CUBIC[0],
    ELASTIC: _elastic_in,
    EXPO: _expo_in,
    QUAD: _QUAD[0], QUART: _QUART[0], QUINT: _QUINT[0],
    SINE: lambda t, b, c, d, *_: -c * np.cos(t / d * (math.pi / 2)) + c + b,
}
_OUT = {
    BACK: lambda t, b, c, d, s, *_: c * ((t / d - 1) ** 2 * ((s + 1) * (t / d - 1) + s) + 1) + b,
    BOUNCE: _bounce_out,
    CIRC: lambda t, b, c, d, *_: c * np.sqrt(np.maximum(1 - (t / d - 1) ** 2, 0)) + b,
    CUBIC: _CUBIC[1],
    ELASTIC: _elastic_out,
    EXPO: _expo_out,
    QUAD: _QUAD[1], QUART: _QUART[1], QUINT: _QUINT[1],
    SINE: lambda t, b, c, d, *_: c * np.sin(t / d * (math.pi / 2)) + b,
}


def _back_in_out(t, b, c, d, s, *_):
    s *= 1.525
    u = t / (d / 2)
    return np.where(u < 1, c / 2 * (u * u * ((s + 1) * u - s)) + b,
                    c / 2 * ((u - 2) ** 2 * ((s + 1) * (u - 2) + s) + 2) + b)


def _bounce_in_out(t, b, c, d, *_):
    return np.where(t < d / 2, _bounce_in(t * 2, 0, c, d) * 0.5 + b, _bounce_out(t * 2 - d, 0, c, d) * 0.5 + c * 0.5 + b)


def _circ_in_out(t, b, c, d, *_):
    u = t / (d / 2)
    return np.where(u < 1, -c / 2 * (np.sqrt(np.maximum(1 - u * u, 0)) - 1) + b,
                    c / 2 * (np.sqrt(np.maximum(1 - (u - 2) ** 2, 0)) + 1) + b)


def _expo_in_out(t, b, c, d, *_):
    u = t / (d / 2)
    with np.errstate(over="ignore"):
        v = np.where(u < 1, c / 2 * (2.0 ** (10 * (u - 1)) - _POW_MIN) * _POW_SCALE + b,
                     c / 2 * (2 - (2.0 ** (-10 * (u - 1)) - _POW_MIN) * _POW_SCALE) + b)
    return np.where(t <= 0, b, v)


_IN_OUT = {
    BACK: _back_in_out, BOUNCE: _bounce_in_out, CIRC: _circ_in_out, CUBIC: _CUBIC[2], ELASTIC: _elastic_in_out,
    EXPO: _expo_in_out, QUAD: _QUAD[2], QUART: _QUART[2], QUINT: _QUINT[2],
    SINE: lambda t, b, c, d, *_: -c / 2 * (np.cos(math.pi * t / d) - 1) + b,
}


# ---- reading actions

def fcurves(f, adt) -> list[FCurve]:
    """The F-curves of an AnimData's action (the slot it uses, in a layered action)."""
    act = adt.follow("action", "bAction") if adt is not None and adt.get("action", 0) else None
    if act is None:
        return []
    found = list(act.listbase("curves", "FCurve"))
    if not found and act.has("layer_array"):
        handle = int(adt.get("slot_handle", 0) or 0)
        datas = _pointers(act, "strip_keyframe_data_array", int(act.get("strip_keyframe_data_array_num", 0) or 0))
        for layer in _structs(f, act, _pointers(act, "layer_array", int(act.get("layer_array_num", 0) or 0)),
                              "ActionLayer"):
            for strip in _structs(f, act, _pointers(layer, "strip_array", int(layer.get("strip_array_num", 0) or 0)),
                                  "ActionStrip"):
                i = int(strip.get("data_index", -1))
                if int(strip.get("strip_type", 0) or 0) != 0 or not 0 <= i < len(datas):
                    continue
                (data,) = _structs(f, act, [datas[i]], "ActionStripKeyframeData") or [None]
                if data is None:
                    continue
                bags = _structs(f, act, _pointers(data, "channelbag_array", int(data.get("channelbag_array_num", 0) or 0)),
                                "ActionChannelBag")
                for bag in bags:
                    if int(bag.get("slot_handle", -1)) == handle:
                        found += _structs(f, act, _pointers(bag, "fcurve_array", int(bag.get("fcurve_array_num", 0) or 0)),
                                          "FCurve")
    out = []
    for fc in found:
        curve = _fcurve(f, fc)
        if curve is not None:
            out.append(curve)
    return out


def _pointers(s, field: str, count: int) -> list[int]:
    if count <= 0 or count > 1_000_000:
        return []
    got = s.raw(field, s.f.ptr_dtype, count)
    return [] if got is None else [int(p) for p in got]


def _structs(f, owner, addresses: list[int], name: str) -> list:
    from .blend import _S
    out = []
    for address in addresses:
        got = f.at(address, owner.owner)
        if got is not None and name in f.struct_index:
            b, off = got
            out.append(_S(f, name, b.offset + off, b.scope))
    return out


def _fcurve(f, fc) -> FCurve | None:
    from .blend import _S
    path_at = f.at(fc.get("rna_path", 0), fc.owner)
    n = int(fc.get("totvert", 0) or 0)
    if path_at is None or n <= 0 or n > 1_000_000 or "BezTriple" not in f.struct_index:
        return None
    b, off = path_at
    path = f.data[b.offset + off:b.offset + b.length].split(b"\0", 1)[0].decode("utf-8", "replace")
    got = f.at(fc.get("bezt", 0), fc.owner)
    if got is None:
        return None
    blk, at = got
    size = f.struct_size["BezTriple"]
    if at + size * n > blk.length:
        return None
    keys, ipo, easing, params = [], [], [], []
    for i in range(n):
        bz = _S(f, "BezTriple", blk.offset + at + i * size, blk.scope)
        vec = np.asarray(bz.get("vec"), np.float64).reshape(3, 3)[:, :2]
        keys.append(vec)
        ipo.append(int(bz.get("ipo", BEZIER)))
        easing.append(int(bz.get("easing", 0) or 0))
        params.append((float(bz.get("back", 1.70158) or 0.0), float(bz.get("amplitude", 0.0) or 0.0),
                       float(bz.get("period", 0.0) or 0.0)))
    keys = np.array(keys)
    if not np.isfinite(keys).all():
        return None
    order = np.argsort(keys[:, 1, 0], kind="stable")
    return FCurve(path, int(fc.get("array_index", 0) or 0), keys[order], np.array(ipo)[order], np.array(easing)[order],
                  np.array(params, np.float64)[order], int(fc.get("extend", 0) or 0), _modifiers(f, fc))


_MODIFIER_DATA = {GENERATOR: "FMod_Generator", CYCLES: "FMod_Cycles", LIMITS: "FMod_Limits", STEPPED: "FMod_Stepped"}


def _modifiers(f, fc) -> list[Modifier]:
    """The F-curve's modifiers meshport applies, in stack order (others are left out)."""
    out = []
    for fm in fc.listbase("modifiers", "FModifier"):
        kind = int(fm.get("type", 0) or 0)
        name = _MODIFIER_DATA.get(kind)
        if name is None or name not in f.struct_index:
            continue
        data = fm.follow("data", name)
        if data is None:
            continue
        if kind == CYCLES:
            settings = {k: int(data.get(k, 0) or 0) for k in ("before_mode", "after_mode", "before_cycles",
                                                               "after_cycles")}
        elif kind == STEPPED:
            settings = {"step": float(data.get("step_size", 2.0) or 0.0), "offset": float(data.get("offset", 0.0)),
                        "start": float(data.get("start_frame", 0.0)), "end": float(data.get("end_frame", 0.0)),
                        "flag": int(data.get("flag", 0) or 0)}
        elif kind == LIMITS:
            rect = data.get("rect")
            values = [float(rect.get(k, 0.0)) for k in ("xmin", "xmax", "ymin", "ymax")] if rect is not None else [0] * 4
            settings = dict(zip(("xmin", "xmax", "ymin", "ymax"), values), flag=int(data.get("flag", 0) or 0))
        else:
            size = int(data.get("arraysize", 0) or 0)
            coefficients = data.raw("coefficients", "<f4", size) if 0 < size <= 1024 else None
            settings = {"coefficients": [] if coefficients is None else [float(x) for x in coefficients],
                        "order": int(data.get("poly_order", 0) or 0), "mode": int(data.get("mode", 0) or 0),
                        "flag": int(data.get("flag", 0) or 0)}
        out.append(Modifier(kind, int(fm.get("flag", 0) or 0), float(fm.get("influence", 1.0)),
                            float(fm.get("sfra", 0.0)), float(fm.get("efra", 0.0)), float(fm.get("blendin", 0.0)),
                            float(fm.get("blendout", 0.0)), settings))
    return out


# the object fields each data path animates: (field, first component) per array index
PATHS = {"location": "loc", "rotation_euler": "rot", "rotation_quaternion": "quat", "scale": "size",
         "delta_location": "dloc", "delta_rotation_euler": "drot", "delta_rotation_quaternion": "dquat",
         "delta_scale": "dscale"}
