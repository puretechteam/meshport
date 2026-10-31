"""Compare meshport's USD reader with Pixar's own library on a folder of USD files.

    python benchmarks/compare_usd.py FOLDER --pxr-python PATH

PATH is a Python that has Pixar's library (pip install usd-core); usd_points.py runs there and
reports each mesh's world-space points as Pixar composes the stage. meshport reads the same file
here. A file "agrees" when both give the same set of world-space points (within 0.1 mm and 0.01%
of the model's size), and the same number of meshes.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import meshport

HERE = Path(__file__).resolve().parent


def meshport_points(path: Path):
    start = time.perf_counter()
    scene = meshport.load(path)
    took = time.perf_counter() - start
    meshes = [node.mesh.positions @ world[:3, :3].T + world[:3, 3] for node, world in scene.walk()
              if node.mesh is not None]
    return meshes, took


def pxr_points(path: Path, python: str):
    start = time.perf_counter()
    out = subprocess.run([python, str(HERE / "usd_points.py"), str(path)], capture_output=True, text=True,
                         timeout=600, check=False)
    took = time.perf_counter() - start
    if out.returncode:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "failed")
    return [np.array(v, float).reshape(-1, 3) for v in json.loads(out.stdout)["meshes"].values()], took


def same_points(a: list, b: list) -> tuple[bool, float]:
    if not a or not b:
        return (not a and not b), 0.0
    pa, pb = np.concatenate(a), np.concatenate(b)
    size = max(np.ptp(pb, axis=0).max(), 1e-9)
    tol = max(1e-4, size * 1e-4)
    ua = np.unique(np.round(pa / tol).astype(np.int64), axis=0)
    ub = np.unique(np.round(pb / tol).astype(np.int64), axis=0)
    if len(ua) != len(ub):
        return False, float("inf")
    from scipy.spatial import cKDTree
    gap = max(cKDTree(pb).query(pa)[0].max(), cKDTree(pa).query(pb)[0].max())
    return gap <= 2 * tol, float(gap)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pxr-python", required=True)
    args = ap.parse_args(argv)
    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"))
    agree = 0
    rows = []
    for f in files:
        try:
            mine, t_mine = meshport_points(f)
        except meshport.MeshportError as e:
            mine, t_mine = ([], 0.0) if "No 3D geometry" in str(e) else (None, str(e))
        try:
            ref, t_ref = pxr_points(f, args.pxr_python)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            ref, t_ref = None, str(e)
        if mine is None or ref is None:
            ok, gap = False, None
        else:
            ok, gap = same_points(mine, ref)
            ok = ok and len(mine) == len(ref)
        agree += ok
        rows.append((f.name, ok, None if mine is None else len(mine), None if ref is None else len(ref), gap,
                     t_mine, t_ref))
        status = "agrees" if ok else "DIFFERS"
        print(f"{f.relative_to(args.folder)}: {status}  meshport {len(mine) if mine is not None else None} meshes "
              f"({t_mine if isinstance(t_mine, str) else f'{t_mine * 1000:.0f} ms'}), "
              f"Pixar {len(ref) if ref is not None else None} ({t_ref if isinstance(t_ref, str) else f'{t_ref * 1000:.0f} ms'})"
              + (f", largest gap {gap:.3g} m" if gap else ""))
    print(f"\n{agree} of {len(files)} files agree with Pixar's library")
    return 0 if agree == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
