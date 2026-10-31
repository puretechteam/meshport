"""Run inside Blender to rebuild scene.blend, the .blend test fixture (and scene_blender.glb, Blender's
own GLB export of it, which the tests compare against):

    blender -b --factory-startup --python tests/data/make_blend_fixture.py

or to write the same scene with another Blender version (2.8 onward) as NAME.blend and
NAME_blender.glb in another folder, for the old-file corpus (MESHPORT_BLEND_CORPUS):

    blender -b --factory-startup --python tests/data/make_blend_fixture.py -- OUT_DIR NAME
"""

import math
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
here = Path(args[0]) if args else Path(__file__).resolve().parent
name = args[1] if len(args) > 1 else "scene"
here.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)

# a packed 4x4 texture
img = bpy.data.images.new("tiles", 4, 4)
img.pixels = [v for i in range(16) for v in ((1, 0, 0, 1) if (i + i // 4) % 2 else (1, 1, 1, 1))]
img.pack()

mat = bpy.data.materials.new("Painted")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.25
bsdf.inputs["Metallic"].default_value = 0.5
tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = img
mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
plain = bpy.data.materials.new("Plain")
plain.use_nodes = True
plain.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.2, 0.7, 1)

# rotation modes, parenting with an inverse, negative scale, two UV maps, smooth and flat faces
bpy.ops.mesh.primitive_cube_add(location=(1, 2, 3))
parent = bpy.context.object
parent.name = "Parent"
parent.rotation_mode = "QUATERNION"
parent.rotation_quaternion = (math.cos(0.3), 0, math.sin(0.3), 0)
parent.data.materials.append(mat)
parent.data.materials.append(plain)
for i, poly in enumerate(parent.data.polygons):
    poly.material_index = i % 2
    poly.use_smooth = i < 3
parent.data.uv_layers.new(name="Second")

bpy.ops.mesh.primitive_cylinder_add(vertices=6, location=(-2, 0, 0))
child = bpy.context.object
child.name = "Child"
child.rotation_mode = "ZXY"
child.rotation_euler = (0.2, 0.4, 0.6)
child.scale = (1, -1, 2)
child.parent = parent
child.matrix_parent_inverse = parent.matrix_world.inverted()

bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, location=(0, -3, 0))
axis = bpy.context.object
axis.name = "AxisAngle"
axis.rotation_mode = "AXIS_ANGLE"
axis.rotation_axis_angle = (0.7, 1, 1, 0)
axis.delta_location = (0, 0, 1)

# a collection shown again through an empty
coll = bpy.data.collections.new("Props")
bpy.context.scene.collection.children.link(coll)
bpy.ops.mesh.primitive_cone_add(vertices=5, location=(0, 0, 5))
cone = bpy.context.object
cone.name = "Cone"
for c in cone.users_collection:
    c.objects.unlink(cone)
coll.objects.link(cone)
inst = bpy.data.objects.new("Instance", None)
inst.instance_type = "COLLECTION"
inst.instance_collection = coll
inst.location = (4, 0, 0)
bpy.context.scene.collection.objects.link(inst)

bpy.ops.wm.save_as_mainfile(filepath=str(here / f"{name}.blend"), compress=True)
# Blender's glTF export before 3.0 drops axis-angle rotations and delta transforms: export the same
# poses as plain quaternion transforms
for ob in bpy.data.objects:
    basis = ob.matrix_basis.copy()
    ob.rotation_mode = "QUATERNION"
    ob.delta_location, ob.delta_rotation_quaternion, ob.delta_scale = (0, 0, 0), (1, 0, 0, 0), (1, 1, 1)
    ob.matrix_basis = basis
bpy.ops.export_scene.gltf(filepath=str(here / f"{name}_blender.glb"), export_format="GLB")
