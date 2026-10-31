"""FBX animation curves: evaluating them, and the euler-angle helpers the reader and writer
use to turn FBX's euler curves into Scene channels and back.

An FBX curve is a list of keys (times in ticks of 1/46186158000 s, float values) with one
attribute per run of keys (KeyAttrRefCount): flags saying how the segment after each key is
interpolated, and four floats (the right slope, the next key's left slope, their two weights
packed into one float's bits, and a velocity). Constant segments hold the key's value (or the
next key's, with the "constant next" flag), linear ones blend, and cubic ones are Bezier curves
in time and value whose handles leave each key at its slope, a third of the way along unless
weighted.
"""

from __future__ import annotations

import numpy as np

TICKS = 46186158000  # KTime units per second
CONSTANT, LINEAR, CUBIC = 0x2, 0x4, 0x8  # interpolation flags
_CONSTANT_NEXT = 0x100
_WEIGHTED_RIGHT, _WEIGHTED_NEXT_LEFT = 0x01000000, 0x02000000
# both weights 0.3333 (handles a third of the way along), packed into a float's bits as FBX does
THIRDS = np.array([3333 | 3333 << 16], np.int32).view(np.float32)[0]
# GlobalSettings TimeMode -> frames per second (14 is CustomFrameRate)
FRAME_RATES = {0: 30.0, 1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97,
               10: 25.0, 11: 24.0, 12: 1000.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94, 18: 119.88}


class Curve:
    def __init__(self, times: np.ndarray, values: np.ndarray, flags: np.ndarray, data: np.ndarray):
        self.times, self.values, self.flags, self.data = times, values, flags, data

    @classmethod
    def from_record(cls, rec) -> Curve | None:
        """An AnimationCurve record's keys, or None when it has none it can use. Keys with
        non-finite times or values are dropped and the rest put in time order."""
        times = rec.array("KeyTime", np.float64)
        values = rec.array("KeyValueFloat", np.float64)
        if times is None or values is None or not len(times) or len(values) < len(times):
            return None
        times, values = times / TICKS, values[:len(times)]
        n = len(times)
        flags = rec.array("KeyAttrFlags", np.float64)
        data = rec.array("KeyAttrDataFloat", np.float32)
        counts = rec.array("KeyAttrRefCount", np.float64)
        if flags is None or not len(flags):
            per_key = np.zeros(n, np.int64)
            flags, data = np.array([LINEAR], np.int64), np.zeros((1, 4), np.float32)
        else:
            flags = flags.astype(np.int64)
            data = data if data is not None else np.zeros(0, np.float32)
            data = np.resize(np.concatenate([data, np.zeros(4 * len(flags), np.float32)])[:4 * len(flags)],
                             (len(flags), 4))
            counts = counts.astype(np.int64) if counts is not None and len(counts) == len(flags) else np.ones(len(flags),
                                                                                                             np.int64)
            # flag k covers the next counts[k] keys, the last one any keys left over (found by where each
            # key falls in the running total, so huge counts cost nothing)
            ends = np.cumsum(np.clip(counts, 0, n))
            per_key = np.minimum(np.searchsorted(ends, np.arange(n), side="right"), len(flags) - 1)
        keep = np.isfinite(times) & np.isfinite(values)
        order = np.argsort(times[keep], kind="stable")
        if not len(order):
            return None
        pick = np.nonzero(keep)[0][order]
        return cls(times[pick], values[pick], flags[per_key[pick]], data[per_key[pick]])

    @property
    def interpolations(self) -> set[int]:
        return set((self.flags[:-1] & 0xE).tolist()) or {LINEAR}

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, np.float64)
        times, v = self.times, self.values
        if len(times) == 1:
            return np.full(t.shape, v[0])
        k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
        t0, dt = times[k], times[k + 1] - times[k]
        u = np.where(dt > 0, np.clip((t - t0) / np.where(dt > 0, dt, 1.0), 0.0, 1.0), 1.0)
        v0, v1, flags = v[k], v[k + 1], self.flags[k]
        out = v0 + u * (v1 - v0)
        mode = flags & 0xE
        constant = mode == CONSTANT
        # a constant key holds its value, or with "constant next" the next key's (from just after it)
        out = np.where(constant & (u < 1), np.where(((flags & _CONSTANT_NEXT) != 0) & (u > 0), v1, v0), out)
        cubic = mode == CUBIC
        if cubic.any():
            data = self.data[k]
            bits = data[:, 2].view(np.int32).astype(np.int64)
            w0 = np.where(flags & _WEIGHTED_RIGHT, (bits & 0xFFFF) / 9999.0, 1 / 3)
            w1 = np.where(flags & _WEIGHTED_NEXT_LEFT, (bits >> 16 & 0xFFFF) / 9999.0, 1 / 3)
            w0, w1 = np.clip(w0, 1e-4, 1.0), np.clip(w1, 1e-4, 1.0)
            s = _bezier_parameter(u, w0, 1 - w1)
            y1 = v0 + data[:, 0] * w0 * dt
            y2 = v1 - data[:, 1] * w1 * dt
            r = 1 - s
            bez = r ** 3 * v0 + 3 * r * r * s * y1 + 3 * r * s * s * y2 + s ** 3 * v1
            out = np.where(cubic & (dt > 0), bez, out)
        return np.where(t <= times[0], v[0], np.where(t >= times[-1], v[-1], out))


def _bezier_parameter(u: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """The Bezier parameter where x(s) = u, for x running 0, x1, x2, 1 (increasing in s while
    the handles stay inside their segment). Exactly u with handles at thirds."""
    thirds = (np.abs(x1 - 1 / 3) < 1e-6) & (np.abs(x2 - 2 / 3) < 1e-6)
    lo, hi = np.zeros_like(u), np.ones_like(u)
    for _ in range(40):
        s = (lo + hi) / 2
        r = 1 - s
        x = 3 * r * r * s * x1 + 3 * r * s * s * x2 + s ** 3
        lo, hi = np.where(x < u, s, lo), np.where(x < u, hi, s)
    return np.where(thirds, u, (lo + hi) / 2)


# ---- batched transforms (more in geometry.py)


def euler_degrees(r: np.ndarray) -> np.ndarray:
    """(N, 3) XYZ euler degrees (x applied first) of (N, 3, 3) rotations."""
    sy = -np.clip(r[:, 2, 0], -1, 1)
    locked = np.abs(sy) >= 0.999999
    x = np.where(locked, np.arctan2(-r[:, 1, 2], r[:, 1, 1]), np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    z = np.where(locked, 0.0, np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    return np.degrees(np.stack([x, np.arcsin(sy), z], 1))


def continuous(degrees: np.ndarray) -> np.ndarray:
    """XYZ euler angles that turn smoothly from key to key: each key takes, of its two equivalent
    forms (x, y, z) and (x + 180, 180 - y, z + 180), the one nearest the key before, each angle
    shifted by whole turns to land within half a turn of it."""
    rows = np.asarray(degrees, np.float64).tolist()  # plain floats: a key at a time is quicker that way
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        x, y, z = rows[i]
        best, best_cost = rows[i], None
        for option in ((x, y, z), (x + 180.0, 180.0 - y, z + 180.0)):
            near = [a - 360.0 * round((a - p) / 360.0) for a, p in zip(option, prev)]
            cost = sum(abs(a - p) for a, p in zip(near, prev))
            if best_cost is None or cost < best_cost:
                best, best_cost = near, cost
        rows[i] = best
    return np.array(rows, np.float64).reshape(-1, 3)




def time_mode(fps: float) -> tuple[int, float]:
    """GlobalSettings TimeMode and CustomFrameRate for a frame rate."""
    for mode, rate in FRAME_RATES.items():
        if mode not in (0, 7, 8) and abs(rate - fps) < 1e-9:
            return mode, -1.0
    return 14, fps
