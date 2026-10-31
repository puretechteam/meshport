"""Run inside Blender to rebuild curves.blend, the curve, text and metaball test fixture, and
curves_blender.glb, Blender's own GLB export of it (which the tests compare against):

    blender -b --factory-startup --python tests/data/make_curve_fixture.py

or to write the same objects with another Blender version as NAME.blend and NAME_blender.glb in
another folder:

    blender -b --factory-startup --python tests/data/make_curve_fixture.py -- OUT_DIR NAME

The text uses dejavu_subset.ttf (a few characters of DejaVu Sans, under its own license, kept in
the font's name table), packed into the file.
"""

import math
import sys
from pathlib import Path

import bpy

here = Path(__file__).resolve().parent
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = Path(args[0]) if args else here
stem = args[1] if len(args) > 1 else "curves"
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
spot = iter(range(100))


def place(obj, name):
    i = next(spot)
    obj.name = name
    obj.location = (i % 6 * 6, i // 6 * 6, 0)
    return obj


def curve(kind, name, **settings):
    getattr(bpy.ops.curve, f"primitive_{kind}_add")()
    obj = place(bpy.context.object, name)
    for key, value in settings.items():
        setattr(obj.data, key, value)
    return obj


def poly(name, points, cyclic=False, kind="POLY", **settings):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    sp = data.splines.new(kind)
    sp.points.add(len(points) - 1)
    for p, co in zip(sp.points, points):
        p.co = (*co, 1)
    sp.use_cyclic_u = cyclic
    for key, value in settings.items():
        setattr(data, key, value)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return place(obj, name)


# 3D curves
curve("bezier_curve", "BezierTube", bevel_depth=0.1)
curve("bezier_curve", "BezierRibbon", extrude=0.2)
curve("bezier_curve", "BezierStadium", bevel_depth=0.1, extrude=0.2, bevel_resolution=2)
curve("bezier_curve", "BezierHalf", bevel_depth=0.2, fill_mode="HALF")
curve("bezier_curve", "BezierCaps", bevel_depth=0.15, use_fill_caps=True)
obj = curve("bezier_curve", "BezierTapered", bevel_depth=0.2)
obj.data.splines[0].bezier_points[0].radius = 0.3
obj.data.splines[0].bezier_points[1].radius = 1.5
obj = curve("bezier_curve", "BezierTilted", extrude=0.3)
obj.data.splines[0].bezier_points[1].tilt = math.radians(90)
curve("bezier_curve", "BezierZUp", extrude=0.3, twist_mode="Z_UP")
curve("nurbs_circle", "NurbsCircleTube", bevel_depth=0.1)
curve("nurbs_path", "NurbsPathTube", bevel_depth=0.1, bevel_resolution=2)
helix = [(math.cos(a), math.sin(a), a / 4) for a in [i * 0.4 for i in range(20)]]
poly("PolyHelix", helix, bevel_depth=0.1)
poly("NurbsHelix", helix, kind="NURBS", bevel_depth=0.1, extrude=0.05)
poly("PolyLoop", [(0, 0, 0), (2, 0, 0.5), (2, 2, 0), (0, 2, 0.5)], cyclic=True, extrude=0.2)

# 2D curves: filled, with holes
curve("bezier_circle", "Disc", dimensions="2D", fill_mode="BOTH")
curve("bezier_circle", "Coin", dimensions="2D", fill_mode="BOTH", extrude=0.1, bevel_depth=0.05,
      bevel_resolution=2)
curve("bezier_circle", "FrontOnly", dimensions="2D", fill_mode="FRONT", extrude=0.2, bevel_depth=0.1)
curve("bezier_circle", "Ring2D", dimensions="2D", fill_mode="NONE", bevel_depth=0.1)
obj = curve("bezier_circle", "Washer", dimensions="2D", fill_mode="BOTH", extrude=0.1)
inner = obj.data.splines.new("BEZIER")
inner.bezier_points.add(3)
for p, (x, y) in zip(inner.bezier_points, ((-0.4, 0), (0, 0.4), (0.4, 0), (0, -0.4))):
    p.co = (x, y, 0)
    p.handle_left_type = p.handle_right_type = "AUTO"
inner.use_cyclic_u = True

# a curve with a modifier
obj = curve("bezier_curve", "CurveArray", bevel_depth=0.1)
m = obj.modifiers.new("Array", "ARRAY")
m.count = 3
m.relative_offset_displace = (0, 1.5, 0)

# text, in a packed font
font = bpy.data.fonts.load(str(here / "dejavu_subset.ttf"))
font.pack()
bpy.ops.object.text_add()
text = place(bpy.context.object, "Text")
text.data.font = font
text.data.body = "Hello\nWorld!"
text.data.extrude = 0.05
text.data.bevel_depth = 0.01
text.data.align_x = "CENTER"
bpy.ops.object.text_add()
flat = place(bpy.context.object, "FlatText")
flat.data.font = font
flat.data.body = "Big 2"
flat.data.size = 1.5
flat.data.space_character = 1.2
flat.data.shear = 0.2
bpy.ops.object.text_add()
place(bpy.context.object, "BuiltinText")  # Blender's own font, which meshport stands in for

# metaballs: a family of three objects, and one of its own with every element type
bpy.ops.object.metaball_add(type="BALL")
base = place(bpy.context.object, "Mball")
el = base.data.elements.new()
el.type = "CAPSULE"
el.co = (1.2, 0, 0)
el.size_x = 0.6
bpy.ops.object.metaball_add(type="ELLIPSOID")
member = bpy.context.object
member.name = "Mball.001"
member.location = (base.location.x, base.location.y + 1.3, 0.3)
member.data.elements[0].size_y = 0.5
bpy.ops.object.metaball_add(type="BALL")
blob = place(bpy.context.object, "Blob")
blob.data.resolution = 0.2
blob.data.elements[0].type = "CUBE"
blob.data.elements[0].size_x = 0.8
blob.data.elements[0].rotation = (math.cos(0.3), 0, 0, math.sin(0.3))
el = blob.data.elements.new()
el.type = "PLANE"
el.co = (0, 0, 1)
el.size_y = 0.7
el = blob.data.elements.new()
el.co = (0.9, 0, 0)
el.radius = 1.2
el.use_negative = True

bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{stem}.blend"), compress=True)
# Blender's glTF export leaves metaballs out: make them meshes first (the base objects take their families)
bpy.ops.object.select_all(action="DESELECT")
for obj in (base, blob):
    obj.select_set(True)
bpy.context.view_layer.objects.active = base
bpy.ops.object.convert(target="MESH")
for obj in list(bpy.data.objects):
    if obj.type == "META":
        bpy.data.objects.remove(obj)
for obj, name in zip(bpy.context.selected_objects, [o.name for o in bpy.context.selected_objects]):
    obj.name = name.split(".")[0]
if bpy.app.version < (2, 93, 0):  # and Blender 2.83's leaves out curves and text too
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        obj.select_set(obj.type in ("CURVE", "FONT"))
    bpy.context.view_layer.objects.active = next(o for o in bpy.data.objects if o.select_get())
    bpy.ops.object.convert(target="MESH")
bpy.ops.export_scene.gltf(filepath=str(out / f"{stem}_blender.glb"), export_format="GLB", export_apply=True)
