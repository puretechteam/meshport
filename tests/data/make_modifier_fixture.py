"""Run inside Blender to rebuild modifiers.blend, the modifier test fixture, and modifiers_blender.glb,
Blender's own GLB export of it with the modifiers applied (which the tests compare against):

    blender -b --factory-startup --python tests/data/make_modifier_fixture.py

or to write the same objects with another Blender version as NAME.blend and NAME_blender.glb in
another folder:

    blender -b --factory-startup --python tests/data/make_modifier_fixture.py -- OUT_DIR NAME
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy

here = Path(__file__).resolve().parent
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = Path(args[0]) if args else here
stem = args[1] if len(args) > 1 else "modifiers"
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
spot = iter(range(100))


def place(obj, name):
    i = next(spot)
    obj.name = name
    obj.location = (i % 6 * 6, i // 6 * 6, 0)
    return obj


def add(kind, name, **kw):
    scale = kw.pop("scale", None)  # set afterwards: Blender 2.83's operators take no scale
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(**kw)
    if scale is not None:
        bpy.context.object.scale = scale
    return place(bpy.context.object, name)


def modifier(obj, kind, **settings):
    m = obj.modifiers.new(kind.title(), kind)
    for key, value in settings.items():
        if hasattr(m, key):
            setattr(m, key, value)
        else:  # a setting older Blenders don't have yet: their own export shows what they make without it
            print(f"{obj.name}: no {key} in Blender {bpy.app.version_string}")
    return m


def loose_quads(name, jitter=0.0004):
    """A 3x3 grid whose faces don't share points (each copy a little apart), for Weld."""
    verts, faces = [], []
    for y in range(3):
        for x in range(3):
            base = len(verts)
            for k, (dx, dy) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                verts.append((x + dx + jitter * ((k + x) % 3 - 1), y + dy + jitter * ((k + y) % 2), 0))
            faces.append((base, base + 1, base + 2, base + 3))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return place(obj, name)


mat_a = bpy.data.materials.new("A")
mat_b = bpy.data.materials.new("B")

# Triangulate
modifier(add("cylinder", "TriShortest", vertices=7), "TRIANGULATE")
modifier(add("cube", "TriBeauty"), "TRIANGULATE", quad_method="BEAUTY", min_vertices=5)

# Weld
modifier(loose_quads("WeldAll"), "WELD")
modifier(loose_quads("WeldConnected", 0.0), "WELD", mode="CONNECTED")

# Mirror
obj = add("monkey", "MirrorX", location=(0.6, 0, 0))
obj.data.uv_layers.new(name="UVMap")
modifier(obj, "MIRROR", use_mirror_u=True, mirror_offset_u=0.1, offset_v=0.25)
obj = add("cube", "MirrorBisect")
bm = bmesh.new()
bm.from_mesh(obj.data)
bmesh.ops.translate(bm, verts=bm.verts, vec=(0.4, 0.2, 0))
bm.to_mesh(obj.data)
modifier(obj, "MIRROR", use_axis=(True, True, False), use_bisect_axis=(True, False, False),
         use_bisect_flip_axis=(False, True, False))
obj = add("cone", "MirrorObject", vertices=6, location=(1, 0, 0))
empty = bpy.data.objects.new("MirrorEmpty", None)
bpy.context.scene.collection.objects.link(empty)
empty.location = (obj.location.x - 1.5, obj.location.y + 0.3, 0)
empty.rotation_euler = (0, 0, math.radians(30))
modifier(obj, "MIRROR", mirror_object=empty, use_axis=(True, False, True))

# Array
modifier(add("cube", "ArrayRelative", scale=(0.5, 0.5, 0.5)), "ARRAY", count=4, relative_offset_displace=(1, 0.5, 0))
modifier(add("cylinder", "ArrayMerge", vertices=8, depth=1), "ARRAY", count=3, use_relative_offset=False,
         use_constant_offset=True, constant_offset_displace=(0, 0, 1.0), use_merge_vertices=True)
modifier(add("cube", "ArrayFit", scale=(0.3, 0.3, 0.3)), "ARRAY", fit_type="FIT_LENGTH", fit_length=4.2)
obj = add("cube", "ArrayObject", scale=(0.4, 0.4, 0.4))
step = bpy.data.objects.new("ArrayStep", None)
bpy.context.scene.collection.objects.link(step)
step.parent = obj
step.location = (1.5, 0, 0.3)
step.rotation_euler = (0, 0, math.radians(40))
modifier(obj, "ARRAY", count=5, use_relative_offset=False, use_object_offset=True, offset_object=step)
start_cap = add("cone", "CapCone", vertices=6, depth=0.6, radius1=0.3)  # its own place doesn't count, only its shape
start_cap.rotation_euler = (0.4, 0, 0.7)
end_cap = add("uv_sphere", "CapSphere", segments=8, ring_count=5, radius=0.35)
end_cap.data.materials.append(mat_b)
obj = add("cube", "ArrayCaps", scale=(0.3, 0.3, 0.3))
obj.data.materials.append(mat_a)
modifier(obj, "ARRAY", count=3, relative_offset_displace=(1.2, 0, 0.2), start_cap=start_cap, end_cap=end_cap)
plug = add("cylinder", "CapPlug", vertices=8, depth=0.5)  # meets the array's ends, so it merges with them
obj = add("cylinder", "ArrayCapsMerge", vertices=8, depth=1)
modifier(obj, "ARRAY", count=2, use_relative_offset=False, use_constant_offset=True,
         constant_offset_displace=(0, 0, 1.0), use_merge_vertices=True, start_cap=plug, end_cap=plug)

# Solidify
obj = add("grid", "SolidifyPlane", x_subdivisions=4, y_subdivisions=3, size=3)
obj.data.materials.append(mat_a)
obj.data.materials.append(mat_b)
modifier(obj, "SOLIDIFY", thickness=0.2, offset=0.3, material_offset=1)
modifier(add("monkey", "SolidifyEven"), "SOLIDIFY", thickness=0.05, use_even_offset=True, use_flip_normals=True)
modifier(add("circle", "SolidifyRimOnly", vertices=10, fill_type="NGON"), "SOLIDIFY", thickness=-0.3,
         use_rim_only=True)

# Subdivision Surface
modifier(add("cube", "SubsurfCube"), "SUBSURF", levels=2)
modifier(add("grid", "SubsurfGridCorners", x_subdivisions=3, y_subdivisions=3, size=3), "SUBSURF", levels=2,
         boundary_smooth="PRESERVE_CORNERS")
modifier(add("grid", "SubsurfGridAll", x_subdivisions=3, y_subdivisions=3, size=3), "SUBSURF", levels=1)
modifier(add("cylinder", "SubsurfNgon", vertices=7), "SUBSURF", levels=2)
modifier(add("monkey", "SubsurfMonkey"), "SUBSURF", levels=1)
modifier(add("cube", "SubsurfNoLimit"), "SUBSURF", levels=2, use_limit_surface=False)
modifier(add("cube", "SubsurfSimple"), "SUBSURF", levels=2, subdivision_type="SIMPLE")
for uv_smooth in ("NONE", "SMOOTH_ALL", "PRESERVE_CORNERS"):  # UVs (the default keeps boundaries)
    modifier(add("cylinder", f"SubsurfUV_{uv_smooth}", vertices=7), "SUBSURF", levels=2, uv_smooth=uv_smooth)


def crease(obj, value, vertex=None):
    """Crease the cube's top edges (and one vertex) by value."""
    me = obj.data
    top = [e.index for e in me.edges if all(me.vertices[v].co.z > 0 for v in e.vertices)]
    if hasattr(me.edges[0], "crease"):  # before Blender 4.0
        for i in top:
            me.edges[i].crease = value
        if vertex is not None and hasattr(me.vertices[0], "crease"):
            me.vertices[vertex].crease = 1.0
        return
    layer = me.attributes.new("crease_edge", "FLOAT", "EDGE")
    for i in top:
        layer.data[i].value = value
    if vertex is not None:
        me.attributes.new("crease_vert", "FLOAT", "POINT").data[vertex].value = 1.0


for value in (1.0, 0.5, 0.3):  # sharpness 10, 2.5 and 0.9: sharp, then fading over the levels
    obj = add("cube", f"SubsurfCrease{int(value * 10)}")
    crease(obj, value)
    modifier(obj, "SUBSURF", levels=2)
obj = add("cube", "SubsurfCreaseVertex")
crease(obj, 0.7, vertex=0)
modifier(obj, "SUBSURF", levels=3)

# a stack, one modifier off in the viewport, and one meshport doesn't know (skipped)
obj = add("cube", "Stack", location=(0.8, 0, 0))
modifier(obj, "MIRROR")
modifier(obj, "SUBSURF", levels=1)
modifier(obj, "ARRAY", count=2, relative_offset_displace=(0, 1.2, 0))
modifier(obj, "TRIANGULATE")
modifier(obj, "SOLIDIFY", thickness=0.1)
modifier(obj, "WELD", show_viewport=False, merge_threshold=10)
obj = add("cube", "Unsupported")
modifier(obj, "BEVEL", width=0.2)
modifier(obj, "ARRAY", count=2)

bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{stem}.blend"), compress=True)
bpy.ops.export_scene.gltf(filepath=str(out / f"{stem}_blender.glb"), export_format="GLB", export_apply=True)
