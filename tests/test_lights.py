"""Lights: read from .blend and glTF (KHR_lights_punctual), written to glTF, checked against
Blender's glTF export of data/lights.blend (rebuilt by data/make_light_fixture.py) and against
Blender reading meshport's files."""

import json
import math
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import BLENDER, cube

import meshport
from meshport import Light, Node, Scene

DATA = Path(__file__).resolve().parent / "data"


def _lights(scene: Scene) -> dict:
    """{name: (world position, direction it shines, light)}."""
    out = {}
    for node, w in scene.walk():
        if node.light is not None:
            out[node.name] = (w[:3, 3], -w[:3, 2] / np.linalg.norm(w[:3, 2]), node.light)
    return out


def test_blend_lights_match_blenders_gltf_export():
    ours, theirs = _lights(meshport.load(DATA / "lights.blend")), _lights(meshport.load(DATA / "lights_blender.glb"))
    assert sorted(ours) == sorted(theirs) == ["PointLamp", "RangeLamp", "SpotLamp", "SunLamp"]  # no area lights
    for name, (p, d, light) in ours.items():
        q, e, other = theirs[name]
        np.testing.assert_allclose(p, q, atol=1e-6)
        np.testing.assert_allclose(d, e, atol=1e-6)
        assert (light.kind, light.range) == (other.kind, other.range)
        assert light.color == pytest.approx(other.color, abs=1e-6)
        assert light.intensity == pytest.approx(other.intensity, rel=1e-6)  # watts at 683 lm/W
        assert (light.inner_cone, light.outer_cone) == pytest.approx((other.inner_cone, other.outer_cone), abs=1e-6)
    assert ours["SpotLamp"][2].outer_cone == pytest.approx(math.radians(30))
    assert ours["RangeLamp"][2].range == 12.0


def test_gltf_lights_round_trip():
    spot = Light("spot", "spot", (1.0, 0.2, 0.1), 250.0, 8.0, 0.2, 0.6)
    scene = Scene([Node("box", mesh=cube()), Node("lamp", np.diag([1, 1, 1, 1.0]), light=spot),
                   Node("sun", light=Light("sun", "directional", intensity=3.0, range=5.0))])
    data = meshport.write(scene, "gltf")
    doc = json.loads(data)
    assert doc["extensionsUsed"] == ["KHR_lights_punctual"]
    assert "range" not in doc["extensions"]["KHR_lights_punctual"]["lights"][1]  # a sun has no range
    back = _lights(meshport.read(data, "gltf"))
    got = back["lamp"][2]
    assert (got.kind, got.color, got.intensity, got.range, got.inner_cone, got.outer_cone) == \
        ("spot", (1.0, 0.2, 0.1), 250.0, 8.0, 0.2, 0.6)
    assert back["sun"][2].kind == "directional" and back["sun"][2].range is None


def test_damaged_light_references_are_left_out():
    doc = json.loads(meshport.write(Scene([Node("box", mesh=cube())]), "gltf"))
    doc["nodes"] += [{"name": "far", "extensions": {"KHR_lights_punctual": {"light": 9}}},
                     {"name": "odd", "extensions": {"KHR_lights_punctual": {"light": 0}}}]
    doc["scenes"][0]["nodes"] += [1, 2]
    doc["extensions"] = {"KHR_lights_punctual": {"lights": [{"type": "area", "intensity": "x"}]}}
    scene = meshport.read(json.dumps(doc).encode(), "gltf")
    assert _lights(scene) == {}


_IMPORT = textwrap.dedent("""
    import bpy, json, sys
    src, out = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    lights = {}
    for ob in bpy.data.objects:
        if ob.type == "LIGHT":
            d = ob.data
            lights[ob.name] = [d.type, d.energy, list(d.color), getattr(d, "spot_size", 0.0), getattr(d, "spot_blend", 0.0),
                               list(ob.matrix_world.translation), list(-ob.matrix_world.col[2].xyz.normalized())]
    open(out, "w").write(json.dumps(lights))
""")


@pytest.mark.skipif(BLENDER is None, reason="Blender isn't installed")
def test_blender_reads_our_lights_as_its_own(tmp_path):
    """meshport's GLB of lights.blend, imported by Blender: the lights it started with."""
    meshport.save(meshport.load(DATA / "lights.blend"), tmp_path / "ours.glb")
    script = tmp_path / "import.py"
    script.write_text(_IMPORT)
    subprocess.run([BLENDER, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "--python", str(script),
                    "--", str(tmp_path / "ours.glb"), str(tmp_path / "lights.json")], check=True, capture_output=True,
                   timeout=300)
    got = json.loads((tmp_path / "lights.json").read_text())
    expected = {"PointLamp": ("POINT", 1000.0, (1, 0.5, 0.25), (1, 2, 3)), "SunLamp": ("SUN", 3.5, (0.9, 0.9, 1), (0, 0, 5)),
                "SpotLamp": ("SPOT", 500.0, (0.2, 1, 0.3), (-2, 1, 4)), "RangeLamp": ("POINT", 50.0, (1, 1, 1), (3, 0, 1))}
    assert sorted(got) == sorted(expected)
    for name, (kind, watts, color, place) in expected.items():
        k, energy, c, _size, _blend, at, _ = got[name]
        assert k == kind and energy == pytest.approx(watts, rel=1e-5)
        assert c == pytest.approx(color, abs=1e-5) and at == pytest.approx(place, abs=1e-5)
    assert got["SpotLamp"][3:5] == pytest.approx([math.radians(60), 0.25], abs=1e-5)
