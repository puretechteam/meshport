"""Run inside Blender to write an animated .blend file and the pose Blender gives each object at
a spread of frames (NAME_poses.json.gz, world matrices in Blender's Z-up axes), which the tests
compare against:

    blender -b --factory-startup --python tests/data/make_anim_fixture.py -- OUT_DIR NAME

It covers Bezier, linear and constant keys, every easing mode, linear extrapolation, every
rotation mode, delta transforms and a parent with an inverse matrix.
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
spot = iter(range(100))


def add(kind, label, rotation_mode="XYZ"):
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(location=(next(spot) * 4, 0, 0))
    obj = bpy.context.object
    obj.name = label
    obj.rotation_mode = rotation_mode
    return obj


def keys(obj, path, frames_values, interpolation=None, easing=None):
    for frame, value in frames_values:
        setattr(obj, path, value)
        obj.keyframe_insert(path, frame=frame)
    if interpolation or easing:
        action = obj.animation_data.action
        curves = list(action.fcurves) if hasattr(action, "fcurves") and len(action.fcurves) else [
            c for layer in action.layers for strip in layer.strips for bag in strip.channelbags for c in bag.fcurves]
        for curve in curves:
            if curve.data_path == path:
                for k in curve.keyframe_points:
                    if interpolation:
                        k.interpolation = interpolation
                    if easing:
                        k.easing = easing


bez = add("cube", "Bezier")
keys(bez, "location", [(1, (0, 0, 0)), (12, (2, 3, 1)), (30, (5, -1, 4)), (40, (6, 0, 0))])
keys(bez, "rotation_euler", [(1, (0, 0, 0)), (20, (0.5, 1.0, 2.0)), (40, (1.5, -0.5, 3.0))])
keys(bez, "scale", [(1, (1, 1, 1)), (40, (2, 0.5, 1.5))])

lin = add("cube", "Linear", "ZXY")
keys(lin, "location", [(1, (0, 0, 0)), (20, (4, 4, 0)), (40, (0, 8, 2))], "LINEAR")
keys(lin, "rotation_euler", [(1, (0, 0, 0)), (40, (1.0, 2.0, 3.0))], "LINEAR")

step = add("cube", "Constant")
keys(step, "location", [(1, (0, 0, 0)), (10, (0, 2, 0)), (25, (0, 2, 3))], "CONSTANT")

quat = add("ico_sphere", "Quaternion", "QUATERNION")
keys(quat, "rotation_quaternion", [(1, (1, 0, 0, 0)), (20, (0.7, 0.7, 0, 0)), (40, (0, 0, 1, 0))])

axis = add("cone", "AxisAngle", "AXIS_ANGLE")
keys(axis, "rotation_axis_angle", [(1, (0, 0, 0, 1)), (40, (3.0, 1, 1, 0))])

delta = add("cylinder", "Delta")
keys(delta, "delta_location", [(1, (0, 0, 0)), (40, (0, 0, 5))])
keys(delta, "delta_rotation_euler", [(1, (0, 0, 0)), (40, (0, 0, 1.5))])
keys(delta, "delta_scale", [(1, (1, 1, 1)), (40, (1, 2, 1))])

extend = add("cube", "Extrapolated")
keys(extend, "location", [(10, (0, 0, 0)), (20, (0, 0, 2))], "LINEAR")
for curve in (extend.animation_data.action.fcurves if hasattr(extend.animation_data.action, "fcurves")
              and len(extend.animation_data.action.fcurves) else
              [c for layer in extend.animation_data.action.layers for strip in layer.strips
               for bag in strip.channelbags for c in bag.fcurves]):
    curve.extrapolation = "LINEAR"

for i, mode in enumerate(("BACK", "BOUNCE", "CIRC", "CUBIC", "ELASTIC", "EXPO", "QUAD", "QUART", "QUINT", "SINE")):
    for easing in ("EASE_IN", "EASE_OUT", "EASE_IN_OUT", "AUTO"):
        ob = add("cube", f"{mode}_{easing}")
        keys(ob, "location", [(1, (ob.location.x, 0, 0)), (40, (ob.location.x, 0, 6))], mode, easing)

parent = add("cube", "Parent")
keys(parent, "location", [(1, (0, 0, 0)), (40, (0, 6, 0))])
keys(parent, "rotation_euler", [(1, (0, 0, 0)), (40, (0, 0, 2.0))])
child = add("ico_sphere", "Child")
child.parent = parent
child.matrix_parent_inverse = parent.matrix_world.inverted()
keys(child, "location", [(1, (child.location.x, 2, 0)), (40, (child.location.x, 2, 3))])


def curves_of(obj):
    action = obj.animation_data.action
    if hasattr(action, "fcurves") and len(action.fcurves):
        return list(action.fcurves)
    return [c for layer in action.layers for strip in layer.strips for bag in strip.channelbags for c in bag.fcurves]


def modified(label, kind, **settings):
    """A cube whose location curves carry an F-curve modifier (names start with "Mod")."""
    ob = add("cube", label)
    x = ob.location.x
    keys(ob, "location", [(5, (x, 0, 0)), (11, (x, 1, 2)), (16, (x, 0.5, 1))])
    for curve in curves_of(ob):
        m = curve.modifiers.new(kind)
        for key, value in settings.items():
            setattr(m, key, value)
    return ob


# F-curve modifiers, among them settings older Blenders also have
modified("ModCycles", "CYCLES")
modified("ModCyclesOffset", "CYCLES", mode_before="REPEAT_OFFSET", mode_after="REPEAT_OFFSET")
modified("ModCyclesMirror", "CYCLES", mode_before="MIRROR", mode_after="MIRROR", cycles_after=2, cycles_before=1)
modified("ModStepped", "STEPPED", frame_step=3.0, frame_offset=1.0)
modified("ModLimits", "LIMITS", use_max_y=True, max_y=1.5, use_min_x=True, min_x=7.0)
modified("ModGenerator", "GENERATOR", use_additive=True, poly_order=2, coefficients=(0.5, 0.05, 0.002))
modified("ModFactorised", "GENERATOR", mode="POLYNOMIAL_FACTORISED", poly_order=2, coefficients=(0.1, -1, 0.05, 0.5))
modified("ModRange", "CYCLES", use_restricted_range=True, frame_start=18, frame_end=34, blend_in=3, blend_out=2,
         use_influence=True, influence=0.7)
ob = modified("ModStack", "CYCLES", mode_after="REPEAT_OFFSET")
for curve in curves_of(ob):
    m = curve.modifiers.new("STEPPED")
    m.frame_step = 2.0

bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{name}.blend"), compress=True)
poses = {}
for frame in sorted({*range(-5, 51, 3), 1, 2, 10, 11, 12, 20, 21, 25, 30, 39, 40, 41, 50}):
    sc.frame_set(frame)
    poses[frame] = {o.name: [[round(v, 6) for v in r] for r in o.matrix_world] for o in sc.objects}
with gzip.open(out / f"{name}_poses.json.gz", "wt") as f:
    json.dump({"fps": sc.render.fps / sc.render.fps_base, "start": sc.frame_start, "end": sc.frame_end,
               "poses": poses}, f)
