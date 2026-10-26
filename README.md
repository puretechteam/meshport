# meshport

Read and write FBX and glTF in pure Python. No FBX SDK, no 3D app, no compiled code: just NumPy.

```
pip install meshport
meshport character.fbx character.glb
```

```python
import meshport

scene = meshport.load("character.fbx")
meshport.save(scene, "character.glb")
```

## What it does

| Format | Read | Write |
|---|---|---|
| FBX binary, 6.x and 7.x (2006 onward, either byte order) | ✓ | ✓ (7.4, the version every importer accepts) |
| FBX ASCII, 6.x and 7.x | ✓ | – |
| glTF 2.0 (`.gltf`, `.glb`) | ✓ | ✓ (everything embedded in one file) |
| Wavefront OBJ | – | ✓ (with .mtl and texture files) |

What comes across in both directions:

- **Hierarchy and transforms.** FBX transforms are applied in full: pivots, offsets, pre/post rotation, all six rotation orders, and geometric transforms. Axis systems (Y-up, Z-up, mirrored) and units (cm, m, inches) are converted.
- **Geometry.** Polygons are kept as polygons until a format needs triangles. Concave faces are triangulated correctly, and mirrored geometry keeps its faces pointing outward.
- **Normals, every UV set, vertex colors.**
- **Instancing.** One mesh used by several nodes stays shared.
- **Materials.** Per-face materials, base color, opacity, emissive, roughness (from FBX shininess), and base color, normal and emissive textures. FBX textures can be embedded or external. External ones are found even when the file still points at the original artist's drive: meshport looks next to the model and in `Textures/` folders up to two levels above it.

Not handled: animation, skinning, blend shapes, cameras and lights. Rigged characters come out in their bind pose. glTF files compressed with Draco or meshopt are refused with a clear error.

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
- **The test suite** (`pytest`) builds its own files. It checks round trips, hand-written ASCII 6 and 7 files, every transform feature, glTF corner cases (interleaved buffers, sparse accessors, strips, external buffers) and, when Blender is installed, both directions of Blender interop.

Speed (fresh process per conversion, FBX → GLB): 0.24 s for a 90 KB model, 0.29 s for 1.5 MB and 1.06 s for 8 MB. That's 3 to 8 times faster than doing the same conversion with Blender.

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
- `src/meshport/obj.py`: `Scene` → OBJ.
- `src/meshport/geometry.py`: transforms and triangulation.

## License

MIT, see [LICENSE](LICENSE).
