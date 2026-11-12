"""Run inside Blender (4.1 or later) to rebuild normals.blend and normals_corners.json.gz, Blender's
own corner normals for each object (in the object's space, corner by corner), which the tests
compare against:

    blender -b --factory-startup --python tests/data/make_normals_fixture.py

The objects cover each way Blender shades: all smooth (vertex normals, weighted by corner angle),
sharp edges marked on smooth faces (hard-surface models), flat and smooth faces side by side, an
edge three faces share, and two neighbours wound opposite ways.
"""

import gzip
import json
import math
from pathlib import Path

import bmesh
import bpy

out = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)


def add(name, build, location):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    build(bm)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = location
    bpy.context.scene.collection.objects.link(ob)
    return me


def sphere(bm):  # uneven faces, so angle and area weighting differ
    bmesh.ops.create_uvsphere(bm, u_segments=7, v_segments=5, radius=1.0)
    for v in bm.verts:
        v.co.x *= 1.0 + 0.6 * max(v.co.x, 0.0)
    for f in bm.faces:
        f.smooth = True


def cylinder(bm):  # caps marked sharp: a typical hard-surface part
    bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=1.0, radius2=1.0, depth=2.0)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-6 and all(abs(abs(v.co.z) - 1) < 1e-6 for v in e.verts):
            e.smooth = False  # sharp


def cube(bm):  # smooth faces, two edges marked sharp
    bmesh.ops.create_cube(bm, size=2.0)
    for f in bm.faces:
        f.smooth = True
    bm.edges.ensure_lookup_table()
    bm.edges[0].smooth = False
    bm.edges[5].smooth = False


def mixed(bm):  # a bent strip: flat and smooth faces side by side, no sharp edges
    rows = [[bm.verts.new((x, y, 0.3 * math.sin(x * 1.3) + 0.2 * y * y)) for x in range(5)] for y in range(3)]
    for y in range(2):
        for x in range(4):
            f = bm.faces.new((rows[y][x], rows[y][x + 1], rows[y + 1][x + 1], rows[y + 1][x]))
            f.smooth = (x + y) % 3 != 0


def fin(bm):  # three faces on one edge, all smooth, one sharp edge elsewhere
    a, b = bm.verts.new((0, 0, 0)), bm.verts.new((0, 0, 1))
    for k in range(3):
        ang = 2 * math.pi * k / 3
        c = bm.verts.new((math.cos(ang), math.sin(ang), 0.4 * k))
        d = bm.verts.new((math.cos(ang), math.sin(ang), 1.2))
        bm.faces.new((a, c, d, b)).smooth = True
    bm.edges.ensure_lookup_table()
    next(e for e in bm.edges if a not in e.verts and b not in e.verts).smooth = False


def flipped(bm):  # neighbours wound opposite ways, and a sharp edge so the corners are worked out
    v = [[bm.verts.new((x, y, 0.25 * x * x - 0.1 * y)) for x in range(4)] for y in range(2)]
    for x in range(3):
        quad = (v[0][x], v[0][x + 1], v[1][x + 1], v[1][x])
        bm.faces.new(quad[::-1] if x == 1 else quad).smooth = True
    bm.edges.ensure_lookup_table()
    next(e for e in bm.edges if set(e.verts) == {v[0][2], v[0][3]}).smooth = False


add("Sphere", sphere, (0, 0, 0))
add("Cylinder", cylinder, (3, 0, 0))
add("Cube", cube, (6, 0, 0))
add("Mixed", mixed, (0, 4, 0))
add("Fin", fin, (6, 4, 0))
add("Flipped", flipped, (0, 8, 0))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "normals.blend"), compress=True)
corners = {ob.name: [round(c, 6) for n in ob.data.corner_normals for c in n.vector]
           for ob in bpy.data.objects if ob.type == "MESH"}
with gzip.open(out / "normals_corners.json.gz", "wt") as f:
    json.dump(corners, f)
