"""Run inside Blender to write an armature whose bones use Blender's inheritance options, and the
pose Blender gives every bone at a spread of frames (NAME_poses.json.gz: each bone's world matrix,
armature object included, in Blender's Z-up axes):

    blender -b --factory-startup --python tests/data/make_bone_fixture.py -- OUT_DIR NAME

A parent bone moves, turns and scales (uniformly, then not) over 40 frames; under it hangs a child
for each option: inherit rotation off, each inherit scale mode, and both off together. Each child
also turns on its own.
"""

import gzip
import json
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out, name = Path(args[0]), args[1]
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = 24
sc.frame_start, sc.frame_end = 1, 40

bpy.ops.object.armature_add(enter_editmode=True)
rig = bpy.context.object
rig.name = "Rig"
eb = rig.data.edit_bones
parent = eb[0]
parent.name, parent.head, parent.tail = "Parent", (0, 0, 0), (0, 0, 1)
parent.roll = 0.3
PROPS = bpy.types.Bone.bl_rna.properties
MODES = set(PROPS["inherit_scale"].enum_items.keys()) if "inherit_scale" in PROPS else set()
SCALE_MODES = [m for m in ("FULL", "FIX_SHEAR", "ALIGNED", "AVERAGE", "NONE", "NONE_LEGACY") if m in MODES]
children = {"Hinge": (False, "FULL")}
children.update({f"Scale_{m}": (True, m) for m in SCALE_MODES})
children["HingeNoScale"] = (False, "NONE")
for i, label in enumerate(children):
    b = eb.new(label)
    b.head, b.tail = (0.3 * i - 1, 0.2, 1.2), (0.3 * i - 1, 0.4, 1.9)
    b.roll = 0.1 * i
    b.parent = parent
bpy.ops.object.mode_set(mode="OBJECT")
for label, (rotation, scale) in children.items():
    bone = rig.data.bones[label]
    bone.use_inherit_rotation = rotation
    if SCALE_MODES:
        bone.inherit_scale = scale
    elif scale == "NONE":
        bone.use_inherit_scale = False
rig.location = (0.5, 0, 0)
bpy.ops.mesh.primitive_cube_add(size=0.2)  # something to show: a file of bones alone has no geometry
bpy.context.object.name = "Marker"
bpy.context.object.parent = rig

pb = rig.pose.bones
for bone in pb:
    bone.rotation_mode = "XYZ"
for frame, loc, rot, size in ((1, (0, 0, 0), (0, 0, 0), (1, 1, 1)), (15, (0.2, 0.5, 0), (0.6, 0.2, 0.4), (1.5, 1.5, 1.5)),
                              (30, (0, 0.3, 0.4), (-0.3, 0.8, 1.2), (2.0, 0.7, 1.2)), (40, (0, 0, 0), (0, 0, 0), (1, 1, 1))):
    p = pb["Parent"]
    p.location, p.rotation_euler, p.scale = loc, rot, size
    for key in ("location", "rotation_euler", "scale"):
        p.keyframe_insert(key, frame=frame)
for i, label in enumerate(children):
    c = pb[label]
    for frame, angle in ((1, 0.0), (40, 0.5 + 0.1 * i)):
        c.rotation_euler = (angle, 0, -angle / 2)
        c.keyframe_insert("rotation_euler", frame=frame)

bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{name}.blend"), compress=True)
poses = {}
for frame in (1, 5, 10, 15, 22, 30, 35, 40):
    sc.frame_set(frame)
    poses[frame] = {b.name: [[round(v, 6) for v in r] for r in rig.matrix_world @ b.matrix] for b in pb}
with gzip.open(out / f"{name}_poses.json.gz", "wt") as f:
    json.dump({"fps": sc.render.fps, "modes": SCALE_MODES, "poses": poses}, f)
print("bones", {label: (rig.data.bones[label].use_inherit_rotation, scale) for label, (_, scale) in children.items()})

