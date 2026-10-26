"""Compare FBX readers on your own files: what loads, what crashes, how fast, and whether
they agree on where the geometry is.

    pip install meshport ufbx assimp-py
    python benchmarks/compare_readers.py path/to/folder/with/fbx/files

Each reader runs in its own process, so a reader that crashes takes down only itself (and is
counted). Readers that aren't installed are skipped. Textures are not searched for, since the
other readers don't do that either; this times geometry only.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

READERS = ("meshport", "ufbx", "assimp")


def _bounds(points):
    return [np.min([p.min(0) for p in points], 0).tolist(), np.max([p.max(0) for p in points], 0).tolist()] \
        if points else None


def read_meshport(path):
    import meshport
    import meshport.fbx
    meshport.fbx._Reader._find_file = lambda self, names: None  # geometry only, like the others
    try:
        scene = meshport.load(path)
    except meshport.MeshportError as e:
        return {"error": str(e)}
    return {"bounds": _bounds([n.mesh.positions @ w[:3, :3].T + w[:3, 3]
                               for n, w in scene.walk() if n.mesh is not None])}


def read_ufbx(path):
    import ufbx
    scene = ufbx.load_file(str(path), target_axes=ufbx.axes_right_handed_y_up, target_unit_meters=1.0)
    points = []
    for node in scene.nodes:
        if node.mesh is not None and len(node.mesh.vertices):
            m = node.geometry_to_world
            mat = np.array([[c.x, c.y, c.z] for c in (m.c0, m.c1, m.c2, m.c3)]).T
            v = np.array([[q.x, q.y, q.z] for q in node.mesh.vertices])
            points.append(v @ mat[:, :3].T + mat[:, 3])
    return {"bounds": _bounds(points)}


def read_assimp(path):
    import assimp_py
    scene = assimp_py.import_file(str(path), assimp_py.Process_Triangulate | assimp_py.Process_PreTransformVertices)
    return {"bounds": _bounds([np.frombuffer(m.vertices, np.float32).reshape(-1, 3).astype(float)
                               for m in scene.meshes if m.num_vertices])}


def worker(reader: str, listing: Path, out: Path) -> None:
    """Read every file in listing, appending a JSON line per file (resumes after a crash)."""
    read = globals()[f"read_{reader}"]
    done = {json.loads(line)["file"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for name in listing.read_text(encoding="utf-8").splitlines():
            if name in done:
                continue
            f.write(json.dumps({"file": name, "crashed": True}) + "\n")  # overwritten below unless it dies
            f.flush()
            start = time.perf_counter()
            try:
                result = read(name)
            except Exception as e:  # noqa: BLE001
                result = {"error": f"{type(e).__name__}: {e}"[:200]}
            result.update(file=name, seconds=time.perf_counter() - start)
            f.write(json.dumps(result) + "\n")
            f.flush()


def run(folder: Path) -> None:
    files = sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() == ".fbx")
    work = Path(tempfile.mkdtemp())
    listing = work / "files.txt"
    listing.write_text("\n".join(files), encoding="utf-8")
    results = {}
    for reader in READERS:
        try:
            __import__({"meshport": "meshport", "ufbx": "ufbx", "assimp": "assimp_py"}[reader])
        except ImportError:
            print(f"{reader}: not installed, skipped")
            continue
        out = work / f"{reader}.jsonl"
        for _ in range(len(files) + 1):  # one retry per crash at most
            if subprocess.run([sys.executable, __file__, "--worker", reader, str(listing), str(out)], check=False).returncode == 0:
                break
        rows = {}
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            rows[r["file"]] = r
        results[reader] = rows

    def ok(r) -> bool:
        return bool(r is not None and "seconds" in r and "error" not in r and r.get("bounds"))

    with_geometry = [f for f in files if any(ok(rows.get(f)) for rows in results.values())]
    everyone = [f for f in with_geometry if all(ok(rows.get(f)) for rows in results.values())]
    print(f"\n{len(files)} FBX files, {len(with_geometry)} with geometry, {len(everyone)} read by every reader\n")
    ref = results.get("ufbx") or results["meshport"]
    print(f"{'reader':10} {'loaded':>8} {'crashed':>8} {'agrees*':>8} {'median ms':>10}")
    for reader, rows in results.items():
        loaded = sum(ok(rows.get(f)) for f in with_geometry)
        crashed = sum("seconds" not in r for r in rows.values())
        agree = 0
        for f in with_geometry:
            if ok(rows.get(f)) and ok(ref.get(f)):
                a, b = np.array(rows[f]["bounds"]), np.array(ref[f]["bounds"])
                agree += np.abs(a - b).max() <= 1e-3 * (np.linalg.norm(b[1] - b[0]) or 1)
        median = statistics.median(rows[f]["seconds"] for f in everyone) * 1000 if everyone else float("nan")
        print(f"{reader:10} {loaded:8} {crashed:8} {agree:8} {median:10.2f}")
    print(f"\n* world-space bounds (in meters, Y up) within 0.1% of {'ufbx' if 'ufbx' in results else 'meshport'}'s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    elif len(sys.argv) == 2:
        run(Path(sys.argv[1]))
    else:
        sys.exit(__doc__)
