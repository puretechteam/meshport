"""Fuzzed and cut-short USD, Alembic and .blend files: reading one may only raise MeshportError, and
each format's whole run has to finish in time (each runs in its own Python, so a hang fails the test
instead of stopping the run)."""

import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).parent

_FUZZ = r"""
import random, sys, time
import numpy as np
import meshport
from meshport import MeshportError
from conftest import cube
kind, rounds = sys.argv[1], int(sys.argv[2])

def scene():
    from meshport import Animation, Channel, Image, Material, Morph, Node, Scene, Skin
    from conftest import png
    mat = Material("red", (1, 0, 0, 1), Image("checker", png()))
    hip, knee = Node("hip"), Node("knee", np.eye(4))
    hip.children.append(knee)
    mesh = cube()
    mesh.targets = [Morph("up", np.tile([0, 0.5, 0], (8, 1)))]
    body = Node("body", mesh=mesh, materials=[mat])
    body.skin = Skin([hip, knee], np.tile(np.eye(4), (2, 1, 1)), np.tile([[0, 1]], (8, 1)), np.full((8, 2), 0.5))
    root = Node("root", children=[hip, body])
    t = np.array([0, 0.5, 1.0])
    anim = Animation("a", [Channel(knee, "rotation", t, np.array([[0, 0, 0, 1], [0, 0, 0.6, 0.8], [0, 0, 0, 1.0]])),
                           Channel(body, "weights", t, np.array([[0.0], [1.0], [0.0]]))])
    return Scene([root], [anim])

if kind in ("usda", "usdc", "usdz", "abc"):
    data, fmt = meshport.write(scene(), kind), kind
else:  # a .blend fixture, uncompressed, so the bytes reached are the file's own
    from meshport.blend import _decompress
    data, fmt = _decompress(open(f"data/{kind}.blend", "rb").read()), "blend"
rng = random.Random(kind)
bad, start = [], time.perf_counter()
for i in range(rounds):
    b = bytearray(data)
    for _ in range(rng.randint(1, 8)):
        b[rng.randrange(len(b))] = rng.randrange(256)
    r = rng.random()
    if r < 0.15:
        b = b[:rng.randrange(len(b))]
    elif r < 0.25:  # a run of bytes copied over another place: counts and offsets that point elsewhere
        a, n = rng.randrange(len(b)), rng.randrange(1, 64)
        at = rng.randrange(len(b))
        b[at:at + n] = b[a:a + n]
    try:
        meshport.read(bytes(b), fmt)
    except MeshportError:
        pass
    except BaseException as e:  # noqa: BLE001
        bad.append(f"{i}: {type(e).__name__}: {str(e)[:120]}")
print(f"{rounds} rounds in {time.perf_counter() - start:.1f} s")
for line in bad:
    print("BAD", line)
"""


def _run(code: str, kind: str, rounds: int) -> tuple[int, str]:
    from test_hostile import kill_tree
    proc = subprocess.Popen([sys.executable, "-c", code, kind, str(rounds)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=TESTS)
    try:
        out, _ = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        pytest.fail(f"{kind}: still running after 90 s (a damaged file hangs the reader)")
    return proc.returncode, out


@pytest.mark.parametrize("kind, rounds", [("usda", 300), ("usdc", 300), ("usdz", 200), ("abc", 300),
                                          ("scene", 150), ("skin", 150)])
def test_fuzzed_files_raise_only_meshport_error(kind, rounds):
    code, out = _run(_FUZZ, kind, rounds)
    assert code == 0, out
    assert "BAD" not in out, out


@pytest.mark.parametrize("kind", ["usda", "usdc", "usdz", "abc", "scene"])
def test_files_cut_short_raise_only_meshport_error(kind):
    code = _FUZZ.split("rng = random.Random(kind)")[0] + r"""
bad = []
step = max(1, len(data) // 400)
for n in range(0, len(data), step):  # cut short at 400 places
    try:
        meshport.read(data[:n], fmt)
    except MeshportError:
        pass
    except BaseException as e:  # noqa: BLE001
        bad.append(f"{n}: {type(e).__name__}: {str(e)[:120]}")
for line in bad:
    print("BAD", line)
"""
    code, out = _run(code, kind, 0)
    assert code == 0 and "BAD" not in out, out
