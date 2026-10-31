# meshport

Read and write FBX, glTF, 3DS, USD and Alembic, and read Blender's .blend files, in pure Python. No FBX SDK, no 3D app, no compiled code: just NumPy.

What's new: see the [changelog](CHANGELOG.md).

```
pip install git+https://github.com/puretechteam/meshport
meshport character.fbx character.glb
```

```python
import meshport

scene = meshport.load("character.fbx")
meshport.save(scene, "character.glb")
```

## Why meshport?

To get FBX into or out of Python today, the usual options are:

- driving a 3D app (Blender);
- bindings to a native reader (ufbx, Assimp);
- Autodesk's FBX SDK, which is closed source and not on PyPI.

Here is how they compare on the same **2,079 real-world FBX files** from game asset packs: versions 6.0 to 7.7, binary and ASCII, exported by Autodesk's plugins and by Blender. All runs used one Windows PC with Python 3.13.

| | **meshport** | Blender 5.1 | ufbx 0.0.5 | assimp-py 1.2.0 |
|---|---|---|---|---|
| Install size | **610 KB** ⁴ | 963 MB | 911 KB | 5.1 MB |
| Reads FBX | ✓ 6.x and 7.x, binary and ASCII | binary 7.x only | ✓ | ✓ |
| Writes FBX | ✓ | ✓ | – | – |
| Converts to glTF | ✓ | ✓ | – | – |
| Files read, of the 2,036 with geometry | **2,036** | – ¹ | 2,034 | 2,033 |
| Times it crashed the Python process | **0** | – | 2 ² | 0 |
| Same world-space geometry as ufbx ³ | **2,034 of 2,034** | – | reference | 1,015 of 2,033 |
| Median read time per file | 1.15 ms | – | 0.30 ms | 0.58 ms |
| FBX → GLB, fresh process, at 90 KB / 1.5 MB / 8 MB | **0.24 / 0.29 / 1.06 s** | 1.86 / 1.87 / 3.53 s | – | – |

¹ Blender was measured on a 125-file sample; see below.<br>
² 3 in another run, on the same kind of files. A crash takes down whatever program loaded the file.<br>
³ World-space bounds in meters, Y up, within 0.1%.<br>
⁴ The whole package (every format), as installed; the FBX-only 0.2.0 was 116 KB.

**Against Blender.** meshport is over 1,500 times smaller and converts 3 to 8 times faster. Of 125 sampled files:

- 117 came out identical to Blender's import, vertex for vertex.
- 6 Blender couldn't open: ASCII and FBX 6 files.
- 2 Blender placed wrong. They use geometric rotation, and Unity, which imports through Autodesk's SDK, agrees with meshport.

**Against ufbx.** ufbx is an excellent C reader. meshport's geometry matches it on every one of the 2,034 files both can read, which is independent confirmation that meshport's transform and unit math is right. But ufbx only reads: it doesn't write FBX or convert to glTF. Its Python bindings also crashed the interpreter on 2 files, where meshport would raise a normal exception.

**Against Assimp.** With default settings, 1,018 of the files (half) came back in the file's own units, not meters: 100× too big for files in centimeters and 39× for files in inches. assimp-py only reads. The other Assimp bindings on PyPI (pyassimp, impasse) wouldn't even import on Windows without a separately installed Assimp library.

**Against the Autodesk FBX SDK.** It's the reference implementation, but it's a separate closed-source download. Files written by meshport import through it (via Unity) the same as the originals on all 18 models tested: same counts and bounds to 4 decimals, and embedded textures extract identically.

**Pure Python.** There's nothing compiled, so the same package installs anywhere NumPy does (Python 3.10+), and no native code can crash your program.

**Where the others are ahead.** Blender, ufbx and Assimp read more of what a rig can do (constraints, dual quaternion skinning). ufbx parses about 4 times faster, though at about a millisecond per file for either, that's small next to the 0.15 s it takes just to start Python and import meshport.

To check these numbers on your own files:

```
pip install git+https://github.com/puretechteam/meshport ufbx assimp-py
python benchmarks/compare_readers.py path/to/your/fbx/files
```

The asset-pack files themselves aren't included here because of their licenses.

## USD, Alembic and .blend

The other formats 3D tools trade in. Until now, reading them from Python meant installing Blender (1.01 GB) or, for USD only, Pixar's own library (49 MB). meshport handles all three in 610 KB, the whole package included (curves, text, metaballs and modifiers among it).

| | **meshport** | Blender 5.1 | Pixar's USD (usd-core 26.11) |
|---|---|---|---|
| Install size | **610 KB** (everything) | 1.01 GB | 49 MB |
| USD: `.usda`, `.usdc`, `.usdz` | ✓ read and write | ✓ | ✓ |
| Alembic `.abc` | ✓ read and write | ✓ | – |
| `.blend` | ✓ read | ✓ | – |
| Same geometry as Pixar on 17 USD files built to trip readers up | **17 of 17** | – | reference |
| Same geometry as Blender's own conversion of Blender's files | **19 of 19** | reference | – |
| Problems Pixar's validators find in its USD files | **0** | 0 | reference |
| To GLB in a fresh process (median over Blender's files) | **0.28–0.34 s** | 1.90–2.10 s | – |

**Against Pixar's library.** `benchmarks/make_usd_corpus.py` uses Pixar's library to write files that use the parts of USD readers most often get wrong: references to other files and within the file, payloads, sublayers, variant selections made on the referencing stage, instancing, every transform op (pivots and their inverses, `orient`, `resetXformStack`), left-handed meshes, holes, indexed primvars in every interpolation, time samples, Z up in centimeters, packages, and binary files in crate versions 0.7, 0.8, 0.9 and 0.10. `benchmarks/compare_usd.py` composes each with Pixar's library and with meshport: all 17 come out with the same world-space points, to within float precision (under a micrometer). In the other direction, Pixar's library reads meshport's `.usda`, `.usdc` and `.usdz` to the same geometry, and all of its validators (including the USDZ package rules) find nothing to report.

**Against Blender.** `benchmarks/compare_blender.py` converts files to GLB with Blender, the way ConvertAll used to, and with meshport, then compares the two. On Blender's own exports of three scenes (a few objects, 41 objects with 5 materials, and 660,000 triangles) as USD, USDZ, Alembic and .blend, plus the four .blend files with meshes that ship with Blender (saved by Blender 4.3 to 5.1), all 19 agree. meshport is 6 to 7 times faster, because Blender takes about 2 seconds just to start. Blender also reads meshport's USD and Alembic files to the same geometry. Rerun with modifiers applied on both sides (`--apply-modifiers`, meshport's default for .blend files), 15 of 16 files from that set agree. The exception is Blender's Sculpting template: its level-6 Subdivision Surface is past Blender's default Quality, where Blender approximates the surface (raising Quality to 6 makes the two agree). The modifier and curve test files are compared object by object in `tests/test_blend.py`. Older Blenders agree too: the same three benchmark scenes saved as .blend by Blender 2.83, 2.93, 3.3 and 3.5 (12 files) convert to the same geometry as each version's own conversion, and so do the test scene and the modifier and curve test files written by each of them (`tests/test_blend_corpus.py`).

**File size.** Written from the same scenes, meshport's files are as small as Blender's or smaller: USDZ 672 KB vs 759 KB, Alembic 909 KB vs 1.03 MB. Like USD and Alembic themselves, meshport stores repeated arrays once.

**What `.blend` reading covers.** Mesh, curve, text and metaball objects in the open scene, with their hierarchy, every rotation mode and collection instances; meshes stored any of the three ways Blender has used since 2.8 (checked with files saved by 2.83, 2.93, 3.3 and 3.5); UV maps, color attributes, per-face materials and flat or smooth shading; Principled BSDF materials with an image texture (packed in the file or next to it). Compressed .blend files (the default since Blender 3.0) need Python 3.14 or the `backports.zstd` package, and text objects need `fonttools`; both come with `pip install "meshport[blend] @ git+https://github.com/puretechteam/meshport"`.

**Modifiers** that are on in Blender's viewport are applied in order, as Blender shows them (`blend.read(..., modifiers=False)` leaves them off, as Blender's glTF export does by default). Six are covered, and others are skipped while the rest of the stack still applies:

- **Triangulate**, **Weld** (All and Connected), **Mirror** (any axes, merging, bisecting, a mirror object, UV flips and offsets) and **Array** (fixed count or fit length; relative, constant and object offsets; merging, also of the last copy with the first; UV offsets) give the same points as Blender, to float precision.
- **Solidify** (simple mode, even thickness, rims, flipped normals, rim only, material offsets) gives the same points too. Complex mode is approximated by simple mode with even thickness.
- **Subdivision Surface** (Catmull-Clark or simple, at the viewport level) puts points on the exact limit surface, which is where Blender puts them up to its Quality setting (3 by default). Past that, Blender approximates near points where three or five faces meet, so at level 6 the two differ by up to 0.3% of the model's size. Boundaries follow the Boundary Smooth setting; UVs are interpolated linearly and edge creases are ignored.
- Not applied: Array caps and fit-to-curve, vertex groups, and every other modifier (Bevel, Boolean, Geometry Nodes, ...).

**Curves, text and metaballs** become meshes the way Blender's viewport shows them:

- **Curves:** Bezier, poly and NURBS splines (uniform, endpoint and Bezier knots) at their resolution, with a round bevel, an extrusion or both, fill modes, fill caps, per-point radius and tilt, and minimum-twist or Z-up frames. 2D curves are filled, with curves inside others cut out as holes. Every curve in the test file gives Blender's points to float precision, except an open NURBS curve with an extrusion, whose frames start about 3° apart from Blender's. Not applied: bevel and taper objects, custom bevel profiles, the Tangent twist (treated as Minimum), bevel start and end, path offset and text on a curve. NURBS surfaces are skipped.
- **Text:** laid out as Blender does (character, word and line spacing, alignment, offsets, shear), in the font packed in the file, next to it or installed on the system, which gives Blender's points to float precision. Blender's built-in font isn't in the file, so meshport uses a similar installed font (Inter, Arial, Liberation Sans, Helvetica, DejaVu Sans or Segoe UI), scaled so capitals come out as tall as Blender's; widths differ by a few percent. Text boxes, small caps, underlines and kerning are left out.
- **Metaballs:** balls, capsules, planes, ellipsoids and cubes, negative ones included, joined by family (`Mball`, `Mball.001`, ...) as in Blender, at the viewport resolution. meshport finds the same surface, with its own points: measured against a fine reference, its mesh is as close to the true surface as Blender's (8.6 cm for both at Blender's default 0.4 m resolution).

**What USD and Alembic reading leave out.** Curves and points, and animated points (those open at their first sample; animated transforms, USD skeletons and blend shapes come along, see [Animation](#what-it-does)). USD composition covers sublayers, inherits, variants, references, payloads and specializes in USD's strength order, and point instancers become one node per instance sharing the prototype's meshes. Alembic files from before 2013 (HDF5) are refused with a clear error.

## Where it came from

meshport was written for [ConvertAll](https://github.com/puretechteam/ConvertAll), a drag-and-drop file converter. ConvertAll used to need Blender for FBX:

- Users had to install 963 MB of 3D software just to convert one format.
- Every conversion waited about 2 seconds for Blender to start.
- ASCII and FBX 6 files didn't open at all.

With meshport, FBX is built into ConvertAll:

- **Size:** the app grew by 74 KB.
- **Speed:** all 12 conversions measured (three real models to GLB, STL, OBJ and 3MF, full app runs) got faster, by 1.6× to 4.9×.

  | Model | Before (Blender) | After (meshport) |
  |---|---|---|
  | Car, 90 KB → GLB | 2.28 s | 0.46 s |
  | Material ball, 1.5 MB → GLB | 2.35 s | 0.49 s |
  | Nature scene, 7.8 MB → 3MF | 6.68 s | 2.73 s |

- **Coverage:** files Blender couldn't open now convert.

With USD, Alembic and .blend, ConvertAll no longer needs Blender at all. Every conversion between those
formats and the others got faster, by 1.8 to 5.5 times, in full app runs on this PC:

  | Conversion | Before (Blender) | After (meshport) |
  |---|---|---|
  | .blend, 3 objects → GLB | 2.13 s | 0.43 s |
  | .blend, 660,000 triangles → GLB | 2.41 s | 0.93 s |
  | USDZ → 3MF | 3.33 s | 1.31 s |
  | Alembic → STL | 2.01 s | 1.13 s |
  | GLB → USDZ | 2.31 s | 0.44 s |
  | GLB → Alembic | 2.23 s | 0.40 s |

## What it does

| Format | Read | Write |
|---|---|---|
| FBX binary, 6.x and 7.x (2006 onward, either byte order) | ✓ | ✓ (7.4, the version every importer accepts) |
| FBX ASCII, 6.x and 7.x | ✓ | – |
| glTF 2.0 (`.gltf`, `.glb`) | ✓ | ✓ (everything embedded in one file) |
| Autodesk 3DS (`.3ds`) | ✓ | ✓ (with texture files; big meshes split to fit the format) |
| USD (`.usda`, `.usdc`, `.usd`, `.usdz`) | ✓ (with composition: references, payloads, sublayers, variants) | ✓ (textures inside `.usdz`, or in a `textures` folder next to the others) |
| Alembic (`.abc`, the Ogawa kind written since 2013) | ✓ | ✓ |
| Blender (`.blend`, 2.8 onward) | ✓ | – |
| Wavefront OBJ | – | ✓ (with .mtl and texture files) |

What comes across in both directions:

- **Hierarchy and transforms.** FBX transforms are applied in full: pivots, offsets, pre/post rotation (when the model's RotationActive is on, as Autodesk's FBX SDK has it), all six rotation orders, and geometric transforms. Axis systems (Y-up, Z-up, mirrored) and units (cm, m, inches) are converted.
- **Geometry.** Polygons are kept as polygons until a format needs triangles. Concave faces are triangulated correctly, and mirrored geometry keeps its faces pointing outward.
- **Normals, every UV set, vertex colors.**
- **Instancing.** One mesh used by several nodes stays shared.
- **Materials.** Per-face materials, base color, opacity, emissive, roughness (from FBX shininess), and base color, normal and emissive textures. FBX textures can be embedded or external. External ones are found even when the file still points at the original artist's drive: meshport looks next to the model, in `Textures/` folders there and one level up, and anywhere under the folder above the model.

**Animation** of nodes (translation, rotation and scale) is read from every format that has it and written to glTF, FBX, USD and Alembic. glTF and FBX hold several animations a file; USD and Alembic have one time line, so the first is written.

- glTF keeps all three of its interpolations: step, linear (rotations slerped) and cubic splines.
- FBX curves come in every kind FBX has: constant (both kinds), linear and cubic (with weighted handles too). FBX rotations are euler angles inside the model's pivots and pre/post rotation, so each model is posed at every key and every frame between (where its curves turn), then split into translation, rotation and scale. Only 7.x animation stacks are read, and where several layers animate one property, the first layer's curves are used.
- Writing FBX, rotations become euler curves that turn continuously, keyed at every frame of the animation's frame rate (the rate its keys sit on, or 30 frames per second); steps are held frame by frame too, since Blender blends FBX's constant keys. Translation and scale that are already linear keep their own keys.
- USD: every xform op's time samples, interpolated as USD does (linearly, quaternions slerped), and retimed through layer offsets on sublayers, references and payloads and each layer's timeCodesPerSecond. Writing gives each animated node translate, orient and scale ops with time samples; USD's blending is glTF's linear, so linear keys stay as they are and steps and cubic splines get a sample every frame.
- Alembic: an Xform's samples at their time sampling's times (uniform, cyclic or acyclic). Writing puts a transform sample every frame on one uniform time sampling.
- 3DS (read): the keyframer's position, rotation and scale tracks, at 30 frames a second (3ds Max's default). Each rotation key turns on from the one before, and keys blend in a straight line: that's how 3ds Max's own poses, stored in files saved part way through an animation, come out (checked to float precision on two such files). The spline terms on keys are read past.
- .blend (read): each object's action, the F-curves of Blender up to 4.3 and the layered actions of 4.4 on, at the scene's frame rate: Bezier keys (handles shortened as Blender shortens them), linear, constant and every easing mode, with constant or linear extrapolation (followed to the end of the scene's frame range), driving location, every rotation mode, scale and the delta transforms. Easing uses Robert Penner's published equations; Blender's exponential easing differs from them by up to 0.1% of the change, and its elastic easing scales the wobble by the key's amplitude. F-curve modifiers, drivers and the NLA are left out.

**Skins and blend shapes** (glTF's skins and morph targets) are read from glTF, FBX, USD and .blend and written to glTF, FBX and USD, with blend shape weights animated like any other channel. A skinned vertex follows up to any number of joints by weight; the mesh's own node then plays no part, as in glTF.

- glTF: skins (every JOINTS_n and WEIGHTS_n set, inverse bind matrices), morph targets (positions and normals, the targets' names Blender and three.js write, default weights on the mesh or the node) and weights channels in all three interpolations. A skin is written only if its joints are in the scene.
- FBX: Skin deformers with a cluster per bone (a cluster's Transform is the inverse bind matrix; Total1 clusters leave the rest of each vertex with the mesh's node, as Autodesk's FBX SDK does, and other weights are normalized), BlendShape deformers (in-between shapes included: each becomes a target of its own, weighted so the blend is FBX's, linear between neighbouring shapes; Autodesk's SDK blends them the same) and their DeformPercent curves. Writing adds skeleton (LimbNode) models, a bind pose (the armature included) with TransformAssociateModel, and shapes holding only the vertices they move; a skinned mesh is written where its joints bound it, so importers that rebuild the bind from the bones (Blender's) agree.
- USD (UsdSkel): Skeletons with their SkelAnimation, meshes bound to them (jointIndices and jointWeights, skel:joints, geomBindTransform) and BlendShapes (in-between shapes included, as for FBX) weighted by the animation's blendShapeWeights, matched by name. Writing makes the root prim a SkelRoot with a Skeleton per skin (sampled every frame while its joints move) and, for a mesh with blend shapes but no skin, a Skeleton without joints to carry its weights; bindTransforms are the joints at rest and geomBindTransform where the mesh was bound, and a skinned mesh under moved prims is written under the SkelRoot, where Blender's reading of geomBindTransform agrees with UsdSkel's. Pixar's own skinning (UsdSkel.BakeSkinning) poses meshport's files exactly as meshport does, and its compliance checker finds nothing.
- .blend (read): armatures (bones become nodes, posed by the action's pose bone curves in every rotation mode), the armature modifier's vertex groups (normalized as Blender normalizes them; a vertex in no bone's group stays with the mesh), and relative shape keys (offsets from the key each is relative to, scaled by its vertex group, animated by the shape keys' own action). Bones follow Blender's inheritance options (inherit rotation off; inherit scale Full, Average, None and None (Legacy)) exactly, as Blender poses them; inherit rotation off with Full scale, Fix Shear and Aligned are exact while the parent scales evenly, and under uneven scaling (a pose with shear, which animation channels can't hold) come out close. Envelopes, preserve volume and constraints aren't followed.
- Alembic: a mesh its skin or blend shapes bend over time gets its points every frame (a point cache), which Blender plays as meshport poses it; reading, a point cache (points changing, faces not) becomes a blend shape per stored sample, weighted so they play back as Alembic does. That takes the samples' offsets plus a row of weights per sample, up to 30 million values in all (`alembic.CACHE_BUDGET`); a bigger cache stands as its first sample.
- Formats without skins (3DS, OBJ, and Alembic for what doesn't move) get each mesh bent where it stands, its shape keys at their weights.

Not handled: cameras and lights. glTF files compressed with Draco or meshopt are refused with a clear error. meshport's GLBs pass Khronos' glTF Validator: a skinned mesh is written as a root without a transform (as glTF recommends), inverse bind matrices end in an exact (0, 0, 0, 1), and keys closer than 32-bit times can tell apart are one key.

3DS is an old format with hard limits, so a few things change on the way out:

- Meshes are triangles, with at most 65,535 vertices and 65,535 faces per object. Bigger meshes are split into several objects.
- There is one UV set, and normals are stored as smoothing groups. Vertices are split wherever a UV or normal changes, so hard edges stay hard.
- There is no hierarchy: every mesh is written as its own object, in place.
- Object names are cut to 10 characters and material names to 16, kept unique. Texture files get 8.3 names, which every 3DS reader accepts.

When reading 3DS, the keyframer section supplies the hierarchy, pivots, dummy groups and instances. Animated files open at their first frame, as in Blender and Assimp, with the objects' animation as channels (above). Lights, cameras and cameras' animation are skipped; a camera that other objects hang from is kept as an empty group.

Textures that glTF can't hold (TGA, PSD, TIFF, ...) are re-encoded to PNG when [Pillow](https://pypi.org/project/Pillow/) is installed (`pip install "meshport[images] @ git+https://github.com/puretechteam/meshport"`), and left out otherwise.

Linked files are followed carefully, since a model from the internet can name any file. Texture paths (FBX, 3DS, USD, .blend), font paths (.blend) and USD sublayers, references and payloads are only read from the model's own folder and the folder above it, and below them: a relative path that stays in there is tried as written, and for textures and fonts the file name is looked for in those folders (for fonts also in the system's font folders, so a font from the machine the file was made on is still found). Absolute paths, network paths (`\\server\share\...`), URLs (`file:` included) and paths reaching further out are never opened, and a chain of USD references can't walk outward from one file to the next. Only files that really are images are taken as textures. glTF files may only point to paths relative to themselves, and their buffers must sit in the model's own folder.

Hostile files can't stall meshport or fill the memory either: what one file can make it build is limited far above what real models need (a million nodes, `scene.MAX_NODES`, instances and glTF nodes listed under several parents included; 100,000 frames on an animation's frame grid, `scene.MAX_FRAMES`; and limits on decoding, decompressing and composing for each format). Reaching one raises `MeshportError` with a plain message. `tests/test_hostile.py` builds a file for every case, and `tests/test_fuzz.py` reads thousands of damaged and cut-short USD, Alembic and .blend files.

## The scene model

Every reader produces the same `Scene` and every writer takes one, so adding a format means writing one reader and one writer.

```python
scene.roots                  # [Node]
node.name, node.matrix       # local 4x4 transform, column vectors
node.mesh, node.materials    # Mesh or None; material slots for the mesh's faces
node.children
mesh.positions               # (V, 3)
mesh.face_sizes, mesh.corners  # polygons: corners per face, vertex index per corner
mesh.normals, mesh.uvs, mesh.colors  # per corner
mesh.face_materials          # slot per face
scene.animations             # [Animation]
animation.name, animation.channels   # Channel(node, path, times, values, interpolation)
channel.sample(t)            # a "translation", "rotation" (x, y, z, w) or "scale" value at t seconds
animation.local_matrix(node, t), animation.bake(rate)
scene.walk_at(t)             # (node, world matrix) posed at t, like scene.walk()
node.skin                    # Skin(joints, inverse_binds, indices, weights) or None: bends the mesh with joints
mesh.targets                 # [Morph(name, positions, normals)]: blend shapes, as offsets
mesh.target_weights          # their weights when nothing animates them; "weights" channels animate them
scene.posed_points(t)        # (node, world-space points) with blend shapes and skins applied at t
scene.deformed()             # the scene with skins and blend shapes applied where they stand
```

Conventions are glTF's: right-handed, +Y up, 1 unit = 1 meter. UVs have v = 0 at the bottom of the image.

## How it's tested

- **Real-world files.** 2,079 FBX files from game asset packs, exported by Autodesk's FBX plugins and by Blender: versions 6.0 to 7.7, binary in both byte orders, and ASCII. All 2,036 that contain geometry are read; the other 43 are animation-only clips and get a clear error saying so. Each file is also written back as FBX and GLB and re-read, with identical bounds.
- **Against the Autodesk FBX SDK.** In Unity, which imports through the FBX SDK, files rewritten by meshport import the same as the originals: same vertex, triangle, material and node counts, same UV sets, vertex colors and embedded textures, and bounds equal to 4 decimals.
- **FBX animation against the Autodesk FBX SDK.** `tests/test_autodesk_fbx.py` (set `MESHPORT_UNITY`) has Unity sample, at every frame, the SDK's evaluation of probe files: cubic keys with their own slopes and with weighted handles, both kinds of constant key, and models with rotation and scaling offsets and pivots, pre and post rotation and ZXY order, with RotationActive on and off. meshport poses all of them as the SDK does (within 0.1 mm), and the SDK poses meshport's own animated files as meshport does. Unity's own pivot conversion (which runs after the SDK) moves a model with a post rotation and a rotation pivot, scaling offset or scaling pivot; Blender reads those as meshport does, so they're checked apart.
- **Against Blender's importer.** 117 of 125 sampled files matched Blender vertex for vertex. Of the other 8:
  - **6 Blender can't open:** ASCII, FBX 6.0 and FBX 6.1 files.
  - **2 Blender gets wrong:** both are copies of one model that uses geometric rotation, and Unity agrees with meshport.

  Where a file has back-to-back duplicate faces (two-sided walls), Blender deletes them and meshport keeps them.
- **3DS against Autodesk's own reader.** Unity imports `.3ds` through Autodesk's FBX SDK, whose 3DS reader is built on Autodesk's 3DS File Toolkit. `tests/test_autodesk_3ds.py` runs it (set `MESHPORT_UNITY` to Unity.exe):
  - Files written by meshport (rotated, mirrored, textured, and a mesh split in two) import with every point, normal and UV where meshport put it.
  - Keyframer cases (rotation keys, pivots, a non-uniformly scaled parent) are placed the same way by both.
  - Assimp's 15 sample `.3ds` files, mostly from 3ds Max, and meshport's re-saved copies of them: same points, normals and UVs, with one exception. Autodesk's reader ignores a camera's aim when placing an object hung from it. 3ds Max doesn't: the one such file stores the object exactly where meshport puts it.
- **3DS against Blender.** Blender's 3DS add-on reads meshport's files with the same counts and positions, and meshport reads Blender's. On the sample files, positions, UVs and materials agree. Normals differ on some, because Blender weights faces by angle and adds sharp edges of its own; meshport follows Autodesk's rule.
- **Real-world 3DS files, on their own.** Every file saved on its first frame comes out exactly as its stored geometry, hierarchy included; one saved part way through comes out, posed on that frame, as the geometry 3ds Max stored; and normals match a literal implementation of the smoothing-group rule. To rerun on your own files: `MESHPORT_3DS_CORPUS=path/to/folder pytest tests/test_corpus_3ds.py` (add `MESHPORT_UNITY` to compare with Autodesk's reader too).
- **Against ufbx and Assimp.** See [Why meshport?](#why-meshport) and `benchmarks/compare_readers.py`.
- **USD against Pixar's library.** `tests/test_usd_interop.py` (set `MESHPORT_PXR_PYTHON` to a Python with `usd-core`) has Pixar's library read meshport's files and run its validators, and builds the 17-file corpus above for meshport to read. With the copy of Pixar's library inside Blender, set `PXR_WORK_THREAD_LIMIT=1` too: its validators trip over their own threads otherwise. See [USD, Alembic and .blend](#usd-alembic-and-blend).
- **USD, Alembic and .blend against Blender.** Run whenever Blender is installed: Blender reads meshport's USD and Alembic, meshport reads Blender's USD, Alembic and .blend, and both sides agree point for point. `tests/data/scene.blend` (rebuilt by `tests/data/make_blend_fixture.py`) covers every rotation mode, parenting with an inverse matrix, negative scale, a collection instance, a packed texture and two UV maps. `tests/data/modifiers.blend` (23 objects: every supported modifier and setting, a stack, a modifier that's off and one meshport skips) and `tests/data/curves.blend` (25 objects: every kind of curve, fill, text in a packed font and in Blender's own, and metaball families) are compared with Blender's GLB exports of them object by object. `tests/data/scene_283.blend` and `scene_35.blend` are the same scene saved by Blender 2.83 and 3.5, which store meshes the old way. `tests/test_blend_corpus.py` (with `MESHPORT_BLEND_CORPUS`) checks a folder of .blend files against Blender's exports of them, object by object; the three fixture scripts take `-- OUT_DIR NAME` to write their files with any Blender from 2.83 on.
- **Real rigs.** `tests/test_real_rigs.py` (set `MESHPORT_REAL_RIGS` to a folder of Khronos' sample models: CesiumMan, BrainStem, Fox, RiggedFigure, RiggedSimple, SimpleSkin, AnimatedMorphCube, MorphStressTest, MorphPrimitivesTest) poses each as Blender's own import does, round-trips each through GLB, FBX, USDC and USDZ, has Blender pose meshport's FBX and USD of each, and, with `MESHPORT_PXR_PYTHON`, `MESHPORT_UNITY` and `MESHPORT_GLTF_VALIDATOR` (Khronos' gltf_validator) set, has Pixar's UsdSkel, Autodesk's FBX SDK and the validator check them too: every one agrees to 1e-4 of the model's size (most to 1e-6), and the validator finds no errors and no warnings the originals don't have.
- **The test suite** (`pytest`) builds its own files. It checks round trips, hand-written ASCII 6 and 7 files, every transform feature, glTF corner cases (interleaved buffers, sparse accessors, strips, external buffers), animation in every format (sampling, every kind of FBX key, pivots under animation, USD layer offsets, Alembic time samplings, 3DS keyframer tracks, Blender's actions in both layouts, round trips, damaged channels, curves and samples; skins and blend shapes in glTF, FBX, USD and .blend, a skinned, shape-keyed tube from `tests/data/make_skin_fixture.py` posed as Blender poses it at every frame from Blender's glTF, FBX, USD and .blend (5.1 and 2.83) and Blender posing meshport's, Pixar's UsdSkel baking meshport's USD skins, damaged skins, clusters, bindings and targets; Blender in both directions for glTF, FBX, USD and Alembic, meshport posing Blender's exports as Blender does at every frame and Blender posing meshport's; and Pixar's own library, the copy inside Blender, posing meshport's USD and its layer offsets as meshport does, with its compliance checker finding nothing), 3DS smoothing groups and mesh splitting, damaged and hostile files in every format and, when Blender is installed, both directions of Blender interop.

## Development

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest
```

Code layout:

- `src/meshport/fbxtree.py`: FBX as a raw record tree (binary and ASCII parsing, binary writing).
- `src/meshport/fbx.py`: the record tree ↔ `Scene`.
- `src/meshport/fbxanim.py`: FBX animation curves.
- `src/meshport/blendanim.py`: Blender actions and their F-curves.
- `src/meshport/blendskin.py`: Blender armatures, vertex groups and shape keys.
- `src/meshport/gltf.py`: glTF ↔ `Scene`.
- `src/meshport/max3ds.py`: 3DS ↔ `Scene`.
- `src/meshport/obj.py`: `Scene` → OBJ.
- `src/meshport/usdlayer.py`: one USD layer in memory; `usda.py` and `usdcrate.py`: text and binary layers ↔ that; `usd.py`: composing layers into a `Scene`, a `Scene` into a layer, and `.usdz` packages.
- `src/meshport/alembic.py`: Alembic ↔ `Scene`.
- `src/meshport/blend.py`: `.blend` → `Scene`.
- `src/meshport/blendmods.py`: Blender's modifiers, applied to those meshes.
- `src/meshport/blendcurve.py`: curve, text and metaball objects, made into meshes.
- `src/meshport/geometry.py`: transforms and triangulation.

## License

MIT, see [LICENSE](LICENSE).
