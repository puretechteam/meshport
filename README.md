# meshport

Read and write FBX, glTF and 3DS in pure Python. No FBX SDK, no 3D app, no compiled code: just NumPy.

What's new: see the [changelog](CHANGELOG.md).

```
pip install meshport
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
| Install size | **116 KB** | 963 MB | 911 KB | 5.1 MB |
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
³ World-space bounds in meters, Y up, within 0.1%.

**Against Blender.** meshport is over 8,000 times smaller and converts 3 to 8 times faster. Of 125 sampled files:

- 117 came out identical to Blender's import, vertex for vertex.
- 6 Blender couldn't open: ASCII and FBX 6 files.
- 2 Blender placed wrong. They use geometric rotation, and Unity, which imports through Autodesk's SDK, agrees with meshport.

**Against ufbx.** ufbx is an excellent C reader. meshport's geometry matches it on every one of the 2,034 files both can read, which is independent confirmation that meshport's transform and unit math is right. But ufbx only reads: it doesn't write FBX or convert to glTF. Its Python bindings also crashed the interpreter on 2 files, where meshport would raise a normal exception.

**Against Assimp.** With default settings, 1,018 of the files (half) came back in the file's own units, not meters: 100× too big for files in centimeters and 39× for files in inches. assimp-py only reads. The other Assimp bindings on PyPI (pyassimp, impasse) wouldn't even import on Windows without a separately installed Assimp library.

**Against the Autodesk FBX SDK.** It's the reference implementation, but it's a separate closed-source download. Files written by meshport import through it (via Unity) the same as the originals on all 18 models tested: same counts and bounds to 4 decimals, and embedded textures extract identically.

**Pure Python.** There's nothing compiled, so the same package installs anywhere NumPy does (Python 3.10+), and no native code can crash your program.

**Where the others are ahead.** Blender, ufbx and Assimp read animation and skinning, and meshport doesn't yet. ufbx parses about 4 times faster, though at about a millisecond per file for either, that's small next to the 0.15 s it takes just to start Python and import meshport.

To check these numbers on your own files:

```
pip install meshport ufbx assimp-py
python benchmarks/compare_readers.py path/to/your/fbx/files
```

The asset-pack files themselves aren't included here because of their licenses.

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

ConvertAll still uses Blender for `.blend`, USD and Alembic.

## What it does

| Format | Read | Write |
|---|---|---|
| FBX binary, 6.x and 7.x (2006 onward, either byte order) | ✓ | ✓ (7.4, the version every importer accepts) |
| FBX ASCII, 6.x and 7.x | ✓ | – |
| glTF 2.0 (`.gltf`, `.glb`) | ✓ | ✓ (everything embedded in one file) |
| Autodesk 3DS (`.3ds`) | ✓ | ✓ (with texture files; big meshes split to fit the format) |
| Wavefront OBJ | – | ✓ (with .mtl and texture files) |

What comes across in both directions:

- **Hierarchy and transforms.** FBX transforms are applied in full: pivots, offsets, pre/post rotation, all six rotation orders, and geometric transforms. Axis systems (Y-up, Z-up, mirrored) and units (cm, m, inches) are converted.
- **Geometry.** Polygons are kept as polygons until a format needs triangles. Concave faces are triangulated correctly, and mirrored geometry keeps its faces pointing outward.
- **Normals, every UV set, vertex colors.**
- **Instancing.** One mesh used by several nodes stays shared.
- **Materials.** Per-face materials, base color, opacity, emissive, roughness (from FBX shininess), and base color, normal and emissive textures. FBX textures can be embedded or external. External ones are found even when the file still points at the original artist's drive: meshport looks next to the model and in `Textures/` folders up to two levels above it.

Not handled: animation, skinning, blend shapes, cameras and lights. Rigged characters come out in their bind pose. glTF files compressed with Draco or meshopt are refused with a clear error.

3DS is an old format with hard limits, so a few things change on the way out:

- Meshes are triangles, with at most 65,535 vertices and 65,535 faces per object. Bigger meshes are split into several objects.
- There is one UV set, and normals are stored as smoothing groups. Vertices are split wherever a UV or normal changes, so hard edges stay hard.
- There is no hierarchy: every mesh is written as its own object, in place.
- Object names are cut to 10 characters and material names to 16, kept unique. Texture files get 8.3 names, which every 3DS reader accepts.

When reading 3DS, the keyframer section supplies the hierarchy, pivots, dummy groups and instances. Animated files open at their first frame, as in Blender and Assimp. The animation itself, lights and cameras are skipped; a camera that other objects hang from is kept as an empty group.

Textures that glTF can't hold (TGA, PSD, TIFF, ...) are re-encoded to PNG when [Pillow](https://pypi.org/project/Pillow/) is installed (`pip install meshport[images]`), and left out otherwise.

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
```

Conventions are glTF's: right-handed, +Y up, 1 unit = 1 meter. UVs have v = 0 at the bottom of the image.

## How it's tested

- **Real-world files.** 2,079 FBX files from game asset packs, exported by Autodesk's FBX plugins and by Blender: versions 6.0 to 7.7, binary in both byte orders, and ASCII. All 2,036 that contain geometry are read; the other 43 are animation-only clips and get a clear error saying so. Each file is also written back as FBX and GLB and re-read, with identical bounds.
- **Against the Autodesk FBX SDK.** In Unity, which imports through the FBX SDK, files rewritten by meshport import the same as the originals: same vertex, triangle, material and node counts, same UV sets, vertex colors and embedded textures, and bounds equal to 4 decimals.
- **Against Blender's importer.** 117 of 125 sampled files matched Blender vertex for vertex. Of the other 8:
  - **6 Blender can't open:** ASCII, FBX 6.0 and FBX 6.1 files.
  - **2 Blender gets wrong:** both are copies of one model that uses geometric rotation, and Unity agrees with meshport.

  Where a file has back-to-back duplicate faces (two-sided walls), Blender deletes them and meshport keeps them.
- **Real-world 3DS files.** Assimp's 15 sample `.3ds` files, mostly exported from 3ds Max, all read. 14 match Blender's import vertex for vertex. In the 15th, a box hangs from an animated camera. That file was saved on frame 0, so its stored geometry shows where 3ds Max had the box, and meshport puts it there; Blender's camera handling puts it 41 units away. Every file saved on frame 0 comes out exactly as its stored geometry. Each file also re-saves as 3DS, glTF and FBX with every triangle in place.
- **Against ufbx and Assimp.** See [Why meshport?](#why-meshport) and `benchmarks/compare_readers.py`.
- **The test suite** (`pytest`) builds its own files. It checks round trips, hand-written ASCII 6 and 7 files, every transform feature, glTF corner cases (interleaved buffers, sparse accessors, strips, external buffers), 3DS smoothing groups and mesh splitting, damaged and hostile files in every format and, when Blender is installed, both directions of Blender interop.

## Development

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest
```

Code layout:

- `src/meshport/fbxtree.py`: FBX as a raw record tree (binary and ASCII parsing, binary writing).
- `src/meshport/fbx.py`: the record tree ↔ `Scene`.
- `src/meshport/gltf.py`: glTF ↔ `Scene`.
- `src/meshport/max3ds.py`: 3DS ↔ `Scene`.
- `src/meshport/obj.py`: `Scene` → OBJ.
- `src/meshport/geometry.py`: transforms and triangulation.

## License

MIT, see [LICENSE](LICENSE).
