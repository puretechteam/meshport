"""Run inside Blender to rebuild the OBJ test fixture: objects.obj (with objects.mtl and
checker.png) written by Blender's OBJ exporter with normals, objects_groups.obj written with
smoothing groups instead of normals, and objects_blender.glb / objects_groups_blender.glb,
Blender's GLB exports after importing each OBJ back (which the tests compare against):

    blender -b --factory-startup --python tests/data/make_obj_fixture.py
"""

from pathlib import Path

import bpy

out = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)


def checker():
    img = bpy.data.images.new("checker", 16, 16)
    px = []
    for y in range(16):
        for x in range(16):
            px += [0.9, 0.2, 0.1, 1.0] if (x // 4 + y // 4) % 2 else [0.95, 0.95, 0.95, 1.0]
    img.pixels = px
    img.filepath_raw = str(out / "checker.png")
    img.file_format = "PNG"
    img.save()
    return img


def material(name, color, image=None, alpha=1.0, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if image is not None:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def named(name, location):
    obj = bpy.context.object
    obj.name = name
    obj.location = location
    return obj


image = checker()
red = material("Red", (0.8, 0.1, 0.1), image)
blue = material("Blue", (0.1, 0.2, 0.8), metallic=0.7, roughness=0.3)
glass = material("Glass", (0.6, 0.9, 0.7), alpha=0.4)

bpy.ops.mesh.primitive_cube_add()
cube = named("Cube", (0, 0, 0))
cube.data.materials.append(red)
cube.rotation_euler = (0.2, 0.4, 0.6)

bpy.ops.mesh.primitive_monkey_add()
monkey = named("Monkey", (3, 0, 1))
monkey.data.materials.append(blue)
bpy.ops.object.shade_smooth()

bpy.ops.mesh.primitive_cylinder_add(vertices=10)  # n-gon caps, flat sides
cyl = named("Cylinder", (-3, 0, 0))
cyl.data.materials.append(red)
cyl.data.materials.append(glass)
for poly in cyl.data.polygons:
    poly.material_index = 1 if len(poly.vertices) > 4 else 0

bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8)
sphere = named("Sphere", (0, 3, 0))
sphere.scale = (1, 1, 2)
for i, poly in enumerate(sphere.data.polygons):
    poly.use_smooth = i % 3 != 0  # some faces smooth, some flat

bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4)
grid = named("Grid", (0, -3, 0))
colors = grid.data.color_attributes.new("Col", "BYTE_COLOR", "POINT")
for i, c in enumerate(colors.data):
    c.color = (i % 4 / 3, i // 4 / 4, 0.5, 1.0)

bpy.ops.mesh.primitive_torus_add(major_segments=16, minor_segments=8)
torus = named("Torus", (3, 3, 0))
torus.data.materials.append(glass)

common = {"export_selected_objects": False, "export_uv": True, "export_materials": True, "path_mode": "RELATIVE",
          "export_colors": True, "apply_modifiers": True}
bpy.ops.wm.obj_export(filepath=str(out / "objects.obj"), export_normals=True, **common)
bpy.ops.wm.obj_export(filepath=str(out / "objects_groups.obj"), export_normals=False,
                      export_smooth_groups=True, **common)

for stem in ("objects", "objects_groups"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=str(out / f"{stem}.obj"))
    bpy.ops.export_scene.gltf(filepath=str(out / f"{stem}_blender.glb"), export_format="GLB",
                              export_vertex_color="ACTIVE")
