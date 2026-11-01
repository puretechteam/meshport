""".blend animation: actions (the list of F-curves up to Blender 4.3, layered from 4.4) checked
against the pose Blender gives each object (data/anim.blend from Blender 5.1 and anim_283.blend
from 2.83, rebuilt by data/make_anim_fixture.py), and F-curve evaluation on its own."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import meshport
from meshport import blend
from meshport.blendanim import BEZIER, CONSTANT, LINEAR, QUAD, FCurve

DATA = Path(__file__).resolve().parent / "data"
Z_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])  # how meshport places a .blend's roots


@pytest.mark.parametrize("name", ["anim", "anim_283"])  # layered actions, and the list before them
def test_every_object_moves_as_in_blender(name):
    with gzip.open(DATA / f"{name}_poses.json.gz", "rt") as f:
        ref = json.load(f)
    scene = blend.read_file(DATA / f"{name}.blend", modifiers=False)
    (anim,) = scene.animations
    assert len({c.node.name for c in anim.channels}) == len(next(iter(ref["poses"].values())))
    for frame, objects in ref["poses"].items():
        frame = int(frame)
        ours = {n.name: w for n, w in scene.walk_at(frame / ref["fps"])}  # frame f is f / fps seconds
        for obj, m in objects.items():
            if obj.startswith(("Extrapolated", "Mod")) and not ref["start"] <= frame <= ref["end"]:
                continue  # a curve that carries on is followed as far as the scene plays
            assert ours[obj] == pytest.approx(Z_UP @ np.array(m), abs=1e-4), (obj, frame)


def test_glb_keeps_a_blend_files_animation():
    scene = blend.read_file(DATA / "anim.blend", modifiers=False)
    back = meshport.read(meshport.write(scene, "glb"), "glb")
    for t in np.linspace(0, 2, 21):
        ours = {n.name: w for n, w in scene.walk_at(t)}
        for n, w in back.walk_at(t):
            if n.name in ours:
                assert w == pytest.approx(ours[n.name], abs=1e-4), (n.name, t)


def test_static_files_have_no_animation():
    assert meshport.load(DATA / "scene.blend").animations == []


def _curve(keys, ipo, extend=0):
    keys = np.array(keys, np.float64)
    n = len(keys)
    return FCurve("location", 0, keys, np.full(n, ipo), np.zeros(n, int), np.tile([1.70158, 0, 0], (n, 1)), extend)


def test_fcurve_interpolations_and_extrapolation():
    flat = [[[-1, 0], [0, 0], [1, 0]], [[9, 10], [10, 10], [11, 10]]]
    assert _curve(flat, LINEAR).evaluate(np.array([-5, 0, 2.5, 10, 20.0])) == pytest.approx([0, 0, 2.5, 10, 10])
    assert _curve(flat, CONSTANT).evaluate(np.array([0, 9.99, 10.0])) == pytest.approx([0, 0, 10])
    assert _curve(flat, LINEAR, extend=1).evaluate(np.array([-5, 15.0])) == pytest.approx([-5, 15])
    # flat handles a third of the way along: the smoothstep curve, 5 halfway
    bez = [[[-10 / 3, 0], [0, 0], [10 / 3, 0]], [[20 / 3, 10], [10, 10], [40 / 3, 10]]]
    assert _curve(bez, BEZIER).evaluate(np.array([5.0]))[0] == pytest.approx(5)
    # handles reaching past each other's key are shortened together, so the curve still runs forward
    long = [[[-30, 0], [0, 0], [30, 0]], [[-20, 10], [10, 10], [40, 10]]]
    v = _curve(long, BEZIER).evaluate(np.linspace(0, 10, 41))
    assert (np.diff(v) >= -1e-9).all() and v[[0, -1]] == pytest.approx([0, 10])
    # linear extrapolation of a Bezier key follows its handle
    tilted = [[[-1, -2], [0, 0], [1, 2]], [[9, 10], [10, 10], [11, 10]]]
    assert _curve(tilted, BEZIER, extend=1).evaluate(np.array([-2.0]))[0] == pytest.approx(-4)
    assert _curve(flat, QUAD).evaluate(np.array([5.0]))[0] == pytest.approx(2.5)  # automatic: eases in


def test_one_key_holds_its_value():
    assert _curve([[[-1, 3], [0, 3], [1, 3]]], BEZIER, extend=1).evaluate(np.array([-9, 0, 9.0])) == pytest.approx([3, 3, 3])


# ---- bones that leave out their parent's rotation or scale (data/make_bone_fixture.py)

# exact only while the parent scales evenly: under uneven scaling these poses shear, which a
# node's animation (translation, rotation, scale) can't hold
EVEN_ONLY = {"Hinge", "Scale_FIX_SHEAR", "Scale_ALIGNED"}


@pytest.mark.parametrize("name", ["bones", "bones_283"])  # Blender 5.1 and 2.83
def test_bones_follow_blenders_inherit_options(name):
    with gzip.open(DATA / f"{name}_poses.json.gz", "rt") as f:
        ref = json.load(f)
    scene = blend.read_file(DATA / f"{name}.blend")
    checked = 0
    for frame, bones in ref["poses"].items():
        ours = {n.name: w for n, w in scene.walk_at(int(frame) / ref["fps"])}
        for bone, m in bones.items():
            if bone in EVEN_ONLY and frame not in ("1", "15", "40"):
                continue
            assert ours[bone] == pytest.approx(Z_UP @ np.array(m), abs=1e-4), (bone, frame)
            checked += 1
    assert checked >= 50
