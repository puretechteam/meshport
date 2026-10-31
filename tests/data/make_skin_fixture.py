"""Run inside Blender to build a skinned tube with blend shapes, export it, and write where Blender
puts every vertex at each frame (poses.json, world space in Blender's Z-up axes):

    blender -b --factory-startup --python tests/data/make_skin_fixture.py -- OUT_DIR [FORMAT...]

FORMAT is glb, fbx or usdc (all three by default). The tube hangs from a two-bone armature that
the object and both bones animate, with each ring weighted between the bones; two shape keys,
animated too, bulge and shift it. tests/test_skinning.py runs this.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler

args = sys.argv[sys.argv.index("--") + 1:]
out, formats = Path(args[0]), args[1:] or ["glb", "fbx", "usdc"]
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = 24
sc.frame_start, sc.frame_end = 1, 24

# a tube of 9 rings of 8, from z = 0 to 4
rings, sides = 9, 8
verts = [(math.cos(2 * math.pi * s / sides), math.sin(2 * math.pi * s / sides), r * 0.5)
         for r in range(rings) for s in range(sides)]
faces = [(r * sides + s, r * sides + (s + 1) % sides, (r + 1) * sides + (s + 1) % sides, (r + 1) * sides + s)
         for r in range(rings - 1) for s in range(sides)]
data = bpy.data.meshes.new("Tube")
data.from_pydata(verts, [], faces)
body = bpy.data.objects.new("Body", data)
sc.collection.objects.link(body)

bpy.ops.object.armature_add(enter_editmode=True)
rig = bpy.context.object
rig.name = "Rig"
lower = rig.data.edit_bones[0]
lower.name, lower.head, lower.tail = "Lower", (0, 0, 0), (0, 0, 2)
upper = rig.data.edit_bones.new("Upper")
upper.head, upper.tail, upper.parent, upper.use_connect = (0, 0, 2), (0, 0, 4), lower, True
bpy.ops.object.mode_set(mode="OBJECT")
rig.rotation_euler = (0, 0, 0.4)

groups = {name: body.vertex_groups.new(name=name) for name in ("Lower", "Upper")}
for i, (_, _, z) in enumerate(verts):
    w = min(max((3 - z) / 2, 0), 1)  # all Lower up to z = 1, all Upper from z = 3
    if w > 0:
        groups["Lower"].add([i], w, "REPLACE")
    if w < 1:
        groups["Upper"].add([i], 1 - w, "REPLACE")
body.modifiers.new("Armature", "ARMATURE").object = rig
body.parent = rig

body.shape_key_add(name="Basis")
bulge = body.shape_key_add(name="Bulge")
shift = body.shape_key_add(name="Shift")
for i, (x, y, z) in enumerate(verts):
    bulge.data[i].co = (x * 1.4, y * 1.4, z)
    shift.data[i].co = (x + 0.5 * z / 4, y, z + 0.2)
for frame, a, b in ((1, 0.0, 0.5), (12, 1.0, 0.0), (24, 0.3, 1.0)):
    bulge.value, shift.value = a, b
    bulge.keyframe_insert("value", frame=frame)
    shift.keyframe_insert("value", frame=frame)

for frame, loc in ((1, (0, 0, 0)), (24, (1, 0, 0.5))):
    rig.location = loc
    rig.keyframe_insert("location", frame=frame)
poses = rig.pose.bones
for bone in poses:
    bone.rotation_mode = "QUATERNION"
for frame, a, b in ((1, (0, 0, 0), (0, 0, 0)), (12, (0.5, 0, 0), (0, 0.8, 0)), (24, (0.3, 0.2, 0), (-0.6, 0, 0.3))):
    poses["Lower"].rotation_quaternion = Euler(a).to_quaternion()
    poses["Upper"].rotation_quaternion = Euler(b).to_quaternion()
    for bone in poses:
        bone.keyframe_insert("rotation_quaternion", frame=frame)

bpy.ops.wm.save_as_mainfile(filepath=str(out / "skin.blend"), compress=True)
if "glb" in formats:
    bpy.ops.export_scene.gltf(filepath=str(out / "skin.glb"), export_format="GLB")
if "fbx" in formats:
    bpy.ops.export_scene.fbx(filepath=str(out / "skin.fbx"), bake_anim_use_all_actions=False,
                             bake_anim_use_nla_strips=False, add_leaf_bones=False)
if "usdc" in formats:
    bpy.ops.wm.usd_export(filepath=str(out / "skin.usdc"), export_animation=True)

points = {}
for frame in range(1, 25):
    sc.frame_set(frame)
    ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    m = ev.to_mesh()
    points[frame] = [[round(c, 6) for c in ev.matrix_world @ v.co] for v in m.vertices]
    ev.to_mesh_clear()
(out / "poses.json").write_text(json.dumps({"fps": sc.render.fps, "points": points}))
