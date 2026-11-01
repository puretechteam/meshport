"""Run inside Blender to rebuild lights.blend (a point, a sun, a spot and a point with a custom
distance, plus an area light glTF has no kind for) and lights_blender.glb, Blender's own GLB
export of it (which the tests compare against):

    blender -b --factory-startup --python tests/data/make_light_fixture.py
"""

import math
from pathlib import Path

import bpy

out = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)


def lamp(name, kind, energy, color, location, rotation=(0, 0, 0), **settings):
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    data.color = color
    for key, value in settings.items():
        setattr(data, key, value)
    ob = bpy.data.objects.new(name, data)
    ob.location = location
    ob.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(ob)
    return ob


lamp("PointLamp", "POINT", 1000.0, (1.0, 0.5, 0.25), (1, 2, 3))
lamp("SunLamp", "SUN", 3.5, (0.9, 0.9, 1.0), (0, 0, 5), (0.4, 0.2, 0.0))
lamp("SpotLamp", "SPOT", 500.0, (0.2, 1.0, 0.3), (-2, 1, 4), (0.7, 0, 0.5), spot_size=math.radians(60), spot_blend=0.25)
lamp("RangeLamp", "POINT", 50.0, (1, 1, 1), (3, 0, 1), use_custom_distance=True, cutoff_distance=12.0)
lamp("AreaLamp", "AREA", 80.0, (1, 1, 1), (0, -3, 2))
bpy.ops.mesh.primitive_cube_add()
bpy.ops.wm.save_as_mainfile(filepath=str(out / "lights.blend"), compress=True)
bpy.ops.export_scene.gltf(filepath=str(out / "lights_blender.glb"), export_format="GLB", export_lights=True)
