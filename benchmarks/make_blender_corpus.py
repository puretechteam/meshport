"""Run inside Blender: build a few scenes and export each as USD (.usdc, .usda, .usdz), Alembic
and .blend, for compare_blender.py and the corpus tests.

    blender -b --factory-startup --python benchmarks/make_blender_corpus.py -- OUT_DIR
"""

import sys
from pathlib import Path

import bpy

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)


def checker_image(name: str, size: int = 64):
    img = bpy.data.images.new(name, size, size)
    px = []
    for y in range(size):
        for x in range(size):
            on = (x // 8 + y // 8) % 2
            px += [0.9, 0.2, 0.1, 1.0] if on else [0.95, 0.95, 0.95, 1.0]
    img.pixels = px
    img.filepath_raw = str(out / f"{name}.png")
    img.file_format = "PNG"
    img.save()
    return img


def material(name: str, color, image=None, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if image is not None:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def scene_small():
    bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 1), rotation=(0.3, 0.2, 0.1))
    bpy.context.object.data.materials.append(material("Skin", (0.8, 0.5, 0.3)))
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, location=(3, 0, 0))
    bpy.context.object.scale = (1, 2, 0.5)  # set afterwards: Blender 2.83's operators take no scale
    bpy.ops.mesh.primitive_cube_add(location=(-3, 1, 0))
    cube = bpy.context.object
    cube.scale = (-1, 1, 1)  # mirrored
    cube.data.materials.append(material("Checker", (1, 1, 1), checker_image("checker")))


def scene_medium():
    tex = checker_image("tiles")
    mats = [material(f"M{i}", (i / 5, 0.5, 1 - i / 5), tex if i % 2 else None, i / 5, 0.2 + i / 8) for i in range(5)]
    parent = None
    for i in range(40):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=(i % 8 * 2.5, i // 8 * 2.5, 0))
        obj = bpy.context.object
        obj.data.materials.append(mats[i % 5])
        obj.rotation_euler = (i * 0.1, i * 0.2, i * 0.3)
        if i % 10 == 0:
            parent = obj
        else:
            obj.parent = parent
    bpy.ops.mesh.primitive_torus_add(location=(0, 0, 4))
    bpy.context.object.data.uv_layers.new(name="Second")


def scene_large():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, radius=3)  # 327,680 triangles
    obj = bpy.context.object
    obj.data.materials.append(material("Clay", (0.6, 0.55, 0.5)))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=300, y_subdivisions=300, size=20, location=(0, 0, -3))
    bpy.context.object.data.materials.append(material("Ground", (0.2, 0.4, 0.2), checker_image("ground")))


for name, build in (("small", scene_small), ("medium", scene_medium), ("large", scene_large)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build()
    for ext in ("usdc", "usda", "usdz"):
        bpy.ops.wm.usd_export(filepath=str(out / f"blender_{name}.{ext}"))
    bpy.ops.wm.alembic_export(filepath=str(out / f"blender_{name}.abc"))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f"blender_{name}.blend"))
