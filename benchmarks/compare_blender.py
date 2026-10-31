"""Convert files to GLB with Blender and with meshport, and compare speed and geometry.

    python benchmarks/compare_blender.py FILE_OR_FOLDER... [--blender PATH] [--apply-modifiers] [--surface]

For each USD, Alembic or .blend file: Blender (in the background, the way ConvertAll ran it)
and meshport (in a fresh Python process, so start-up counts for both) each write a GLB. Both
GLBs are then read with meshport and compared: same world-space points (within 0.1 mm and
0.01% of the model's size) means the conversions agree. Prints a line per file and a summary.

--apply-modifiers has both apply .blend files' modifiers (meshport's default; without the flag
both leave them off). --surface compares surfaces instead of points, for geometry the two
build differently (curves, metaballs, text): the farthest any point of either surface is from
the other, within 0.5% of the model's size, means they agree.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import meshport

SUFFIXES = (".usd", ".usda", ".usdc", ".usdz", ".abc", ".blend")

BLENDER_SCRIPT = r'''
import bpy, sys
src, dst, apply = sys.argv[sys.argv.index("--") + 1:]
ext = src.rsplit(".", 1)[-1].lower()
if ext == "blend":
    bpy.ops.wm.open_mainfile(filepath=src)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext == "abc":
        bpy.ops.wm.alembic_import(filepath=src)
    else:
        bpy.ops.wm.usd_import(filepath=src)
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_apply=apply == "1")
'''


def find_blender() -> str | None:
    found = shutil.which("blender")
    if found:
        return found
    hits = sorted(Path("C:/Program Files/Blender Foundation").glob("*/blender.exe"), reverse=True)
    return str(hits[0]) if hits else None


def points(glb: Path) -> np.ndarray:
    scene = meshport.load(glb)
    return np.concatenate([n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk() if n.mesh is not None])


def gap(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max()))


def run_blender(blender: str, src: Path, dst: Path, apply: bool) -> float:
    script = dst.with_name("convert.py")
    script.write_text(BLENDER_SCRIPT)
    start = time.perf_counter()
    subprocess.run([blender, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python",
                    str(script), "--", str(src), str(dst), "1" if apply else "0"],
                   check=True, capture_output=True, timeout=1800)
    return time.perf_counter() - start


MESHPORT_BLEND = "import sys, meshport; from meshport import blend; "                  "meshport.save(blend.read_file(sys.argv[1], sys.argv[3] == '1'), sys.argv[2])"


def run_meshport(src: Path, dst: Path, apply: bool) -> float:
    start = time.perf_counter()
    if src.suffix.lower() == ".blend":
        command = [sys.executable, "-c", MESHPORT_BLEND, str(src), str(dst), "1" if apply else "0"]
    else:
        command = [sys.executable, "-m", "meshport", str(src), str(dst)]
    subprocess.run(command, check=True, capture_output=True, timeout=1800)
    return time.perf_counter() - start


def triangles(glb: Path) -> np.ndarray:
    """(T, 3, 3) world-space triangles."""
    scene = meshport.load(glb)
    out = []
    for n, w in scene.walk():
        if n.mesh is not None:
            p = n.mesh.positions @ w[:3, :3].T + w[:3, 3]
            out.append(p[n.mesh.corners[n.mesh.triangles()]])
    return np.concatenate(out)


def surface_gap(a: np.ndarray, b: np.ndarray, samples: int = 20000) -> float:
    """The farthest a sample of either surface (its corners and points spread by area) is from
    the other surface."""
    return max(_one_way(a, b, samples), _one_way(b, a, samples))


def _one_way(a: np.ndarray, b: np.ndarray, samples: int) -> float:
    rng = np.random.default_rng(0)
    area = np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1)
    pick = rng.choice(len(a), samples, p=area / area.sum()) if area.sum() > 0 else np.zeros(0, int)
    r = rng.random((len(pick), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    t = a[pick]
    corners = np.unique(a.reshape(-1, 3), axis=0)
    if len(corners) > samples:  # every corner of a big mesh would be slow: as many as the area samples
        corners = corners[rng.choice(len(corners), samples, replace=False)]
    pts = np.concatenate([corners, t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])])
    return float(_distance_to(pts, b).max())


def _distance_to(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Each point's exact distance to the nearest of the triangles. Small triangles are found
    through their centres (then every one that could still be nearer is checked); big ones are
    checked against every point."""
    from scipy.spatial import cKDTree
    centre = tris.mean(1)
    radius = np.linalg.norm(tris - centre[:, None], axis=2).max(1)
    limit = max(4 * float(np.median(radius)), float(np.quantile(radius, 0.995)))  # at most 0.5% checked one by one
    small, big = np.nonzero(radius <= limit)[0], np.nonzero(radius > limit)[0]
    best = np.full(len(pts), np.inf)
    for start in range(0, len(big), 64):
        chunk = tris[big[start:start + 64]]
        d = _point_triangle(np.repeat(pts, len(chunk), 0), np.tile(chunk, (len(pts), 1, 1)))
        best = np.minimum(best, d.reshape(len(pts), len(chunk)).min(1))
    if len(small):
        tree = cKDTree(centre[small])
        k = min(8, len(small))
        reach, near = tree.query(pts, k)
        reach, near = reach.reshape(len(pts), -1), near.reshape(len(pts), -1)
        for j in range(k):
            best = np.minimum(best, _point_triangle(pts, tris[small[near[:, j]]]))
        # a triangle further away could still be nearer while its centre is within best + its own size
        todo = np.nonzero(reach[:, -1] < best + limit)[0] if k < len(small) else np.zeros(0, int)
        if len(todo):
            lists = tree.query_ball_point(pts[todo], best[todo] + limit)
            counts = np.array([len(c) for c in lists])
            point = np.repeat(todo, counts)
            cand = small[np.concatenate([np.asarray(c, int) for c in lists])] if counts.sum() else np.zeros(0, int)
            keep = np.linalg.norm(pts[point] - centre[cand], axis=1) - radius[cand] < best[point]
            point, cand = point[keep], cand[keep]
            for start in range(0, len(point), 1 << 20):
                sl = slice(start, start + (1 << 20))
                np.minimum.at(best, point[sl], _point_triangle(pts[point[sl]], tris[cand[sl]]))
    return best


def _point_triangle(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Distance from each point to its triangle (Ericson's closest-point regions)."""
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp, cp = p - b, p - c
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v, w = vb / denom, vc / denom
        q = a + ab * v[:, None] + ac * w[:, None]  # inside
        e_ab = a + ab * np.clip(d1 / (d1 - d3), 0, 1)[:, None]
        e_ac = a + ac * np.clip(d2 / (d2 - d6), 0, 1)[:, None]
        e_bc = b + (c - b) * np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0, 1)[:, None]
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], e_ab, q)
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], e_ac, q)
    q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None], e_bc, q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)
    q = np.where(np.isfinite(q), q, a)
    return np.linalg.norm(p - q, axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--blender", default=find_blender())
    ap.add_argument("--apply-modifiers", action="store_true", help="apply .blend files' modifiers on both sides")
    ap.add_argument("--surface", action="store_true", help="compare surfaces rather than points")
    args = ap.parse_args(argv)
    if not args.blender:
        print("Blender not found: pass --blender", file=sys.stderr)
        return 2
    files = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    same = both = 0
    t_blender, t_meshport = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(files):
            b_out, m_out = tmp / f"b{i}.glb", tmp / f"m{i}.glb"
            notes = []
            try:
                tb = run_blender(args.blender, f, b_out, args.apply_modifiers)
            except subprocess.CalledProcessError as e:
                tb, notes = None, notes + [f"Blender failed: {e.stderr.decode(errors='replace').strip()[-120:]}"]
            try:
                tm = run_meshport(f, m_out, args.apply_modifiers)
            except subprocess.CalledProcessError as e:
                tm, notes = None, notes + [f"meshport failed: {e.stderr.decode(errors='replace').strip()[-160:]}"]
            verdict = ""
            if tb is not None and tm is not None:
                both += 1
                t_blender.append(tb)
                t_meshport.append(tm)
                try:
                    if args.surface:
                        tb_, tm_ = triangles(b_out), triangles(m_out)
                        size = max(np.ptp(tb_.reshape(-1, 3), axis=0).max(), 1e-9)
                        g = surface_gap(tb_, tm_)
                        ok = g <= max(1e-4, size * 5e-3)
                        detail = f"{len(tb_)} / {len(tm_)} triangles"
                    else:
                        pb, pm = points(b_out), points(m_out)
                        size = max(np.ptp(pb, axis=0).max(), 1e-9)
                        g = gap(pb, pm)
                        ok = g <= max(1e-4, size * 1e-4)
                        detail = f"{len(pb)} / {len(pm)} points"
                    same += ok
                    verdict = f"{'same' if ok else 'DIFFERENT'} {'surface' if args.surface else 'geometry'} "                               f"(gap {g:.3g} m, {detail})"
                except (meshport.MeshportError, ValueError) as e:
                    verdict = f"couldn't compare: {e}"
            times = f"Blender {tb:.2f} s" if tb is not None else "Blender -"
            times += f", meshport {tm:.2f} s" if tm is not None else ", meshport -"
            print(f"{f.name} ({f.stat().st_size / 1e6:.2f} MB): {times}; {verdict} {' '.join(notes)}")
    if t_blender:
        print(f"\n{both} files converted by both; {same} with the same geometry. "
              f"Median time: Blender {np.median(t_blender):.2f} s, meshport {np.median(t_meshport):.2f} s "
              f"({np.median(t_blender) / np.median(t_meshport):.1f}x).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
