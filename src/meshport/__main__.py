"""meshport IN OUT: convert one 3D file to another (format from the extensions)."""

from __future__ import annotations

import argparse
import sys
import time

from . import FORMATS, MeshportError, __version__, load, save


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="meshport", description="Convert 3D scene files. "
                                 f"Formats: {', '.join(FORMATS)}.")
    ap.add_argument("src", help="file to read")
    ap.add_argument("dst", nargs="?", help="file to write (omit to print what's inside)")
    ap.add_argument("--version", action="version", version=f"meshport {__version__}")
    args = ap.parse_args(argv)
    try:
        start = time.perf_counter()
        scene = load(args.src)
        if args.dst is None:
            for node, _ in scene.walk():
                if node.mesh is not None:
                    m = node.mesh
                    print(f"{node.name}: {len(m.positions)} vertices, {len(m.face_sizes)} faces, "
                          f"{len(m.uvs)} UV sets, {len(node.materials)} materials")
            b = scene.bounds()
            if b is not None:
                print("size (m):", " x ".join(f"{v:.3g}" for v in b[1] - b[0]))
            return 0
        save(scene, args.dst)
        print(f"{args.src} -> {args.dst} in {time.perf_counter() - start:.2f}s")
        return 0
    except (MeshportError, OSError) as e:
        print(f"meshport: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
