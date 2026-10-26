# Changelog

Newest first. Each entry is dated by when it landed.

## 2026-10-25: meshport 0.2.0, Autodesk 3DS

- **Added:** reading and writing `.3ds` files (`meshport.load`/`save`, `meshport.read`/`write` with `"3ds"`, and the command line).
  - Reads mesh objects with their UVs, smoothing groups (turned into normals), per-face materials and object matrices, and materials with their diffuse color, opacity, shininess, two-sided flag, self-illumination, and texture, bump and self-illumination maps. Textures are found next to the model the same way as for FBX.
  - Writes every mesh as an object in world space. Meshes over the format's 65,535-vertex or 65,535-face limit are split into several objects. Hard edges are kept by splitting vertices under one smoothing group.
  - Names are cut to the format's limits (10 characters for objects, 16 for materials, 8.3 for texture files) and kept unique. `save()` writes the texture files next to the `.3ds`.
  - Reads the keyframer section for the hierarchy, pivots, dummy groups, instances and objects parented to cameras. Animated files open at their first frame, as in Blender and Assimp.
  - Skipped when reading: the animation itself, lights and cameras.
- **Added:** 3DS tests: round trips, smoothing groups, splitting, name limits, damaged and hostile files, and Blender interop in both directions. Blender's 3DS add-on reads meshport's files with the same vertex and face counts and every vertex in place, and meshport reads Blender's 3DS files (keyframer section included) to the same geometry and smoothing as Blender's own glTF export. These skip when Blender or its `autodesk_3ds_format` extension isn't installed.
- **Tested** on Assimp's 15 real-world `.3ds` sample files, mostly from 3ds Max. Positions, UVs, material colors, opacity and textures match Blender's import. Files saved on their first frame come out exactly as their stored geometry. Normals match a literal implementation of the smoothing-group rule; where Blender differs, its importer adds rules of its own. Blender reads meshport's re-saved copies with the same counts and positions. `tests/test_corpus_3ds.py` reruns these checks on any folder named by `MESHPORT_3DS_CORPUS`.
- **Fixed while testing on those files:**
  - Smoothing-group normals weight each face by its angle at the corner, as Blender does, instead of by its area.
  - A material whose texture file is missing keeps its own color instead of turning white.
  - The writer no longer merges vertices that the mesh keeps apart at the same spot. Merging them turned zero-area faces into faces other programs delete.
  - A file saved on its first frame takes object placement from its stored geometry, which is exact, and still takes the hierarchy from the keyframer.
- **Changed:** the FBX reader's texture search is now shared with the 3DS reader.
- **Changed:** the code passes `ruff check` with ruff 0.16's wider default rules. No behavior changes.

## 2026-10-25: Sturdier against damaged files

- **Fixed:** damaged or deliberately malformed files now always raise `MeshportError`. Before, a corrupted GLB or glTF could raise a raw `UnicodeDecodeError`, `KeyError` or `IndexError`, and a deeply nested FBX could hit a `RecursionError`. Found by feeding meshport 1,600 corrupted files.
- **Fixed:** compressed FBX arrays are never inflated beyond the size the file declares, so a tiny file can't claim gigabytes of memory (a "zip bomb").
- **Added:** fuzz, zip-bomb and deep-nesting tests (`tests/test_damaged.py`).
- **Changed:** removed an unused import found by linting.

## 2026-10-25: Comparisons you can check

- **Added:** a "Why meshport?" section in the README. It compares meshport with Blender, ufbx, Assimp and the Autodesk FBX SDK on 2,079 real FBX files: size, what each can read and write, crashes, correctness and speed. It also lists where the others are ahead.
- **Added:** `benchmarks/compare_readers.py`, which reruns that comparison on any folder of FBX files.
- **Added:** a "Where it came from" section: meshport replaced Blender for FBX in [ConvertAll](https://github.com/puretechteam/ConvertAll). The app grew by 74 KB instead of needing a 963 MB install, and every measured conversion got 1.6–4.9× faster.

## 2026-10-25: OBJ export

- **Added:** writing Wavefront OBJ with its `.mtl` file and texture files.
  - Polygons stay polygons.
  - Each node becomes its own object in world space.
  - Mirrored nodes keep their faces pointing outward.
  - `meshport.save()` returns every file it wrote.
- **Why:** converting a 7.8 MB, 821-part FBX scene to OBJ took 5.9 s through trimesh, and 1.5 s with meshport writing it directly.

## 2026-10-25: meshport 0.1.0, first release

- **Reads FBX:** binary and ASCII, versions 6.x and 7.x (2006 onward), in either byte order.
- **Writes FBX:** binary 7.4, the version every importer accepts, with textures embedded.
- **Reads and writes glTF 2.0:** `.glb` and self-contained `.gltf`. Reading handles interleaved buffers, sparse accessors, triangle strips and fans, and external or embedded buffers. Draco and meshopt files get a clear error.
- **Carries across:**
  - the node hierarchy, with full FBX transforms (pivots, offsets, pre/post rotation, all six rotation orders, geometric transforms);
  - axis and unit conversion (Y-up, Z-up, cm, m, inches);
  - polygons, with concave faces triangulated correctly;
  - normals, every UV set, vertex colors;
  - instancing;
  - per-face materials with base color, opacity, emissive, roughness, and base color, normal and emissive textures.
- **Finds textures** that FBX files point at on the original artist's drive, by looking next to the model and in `Textures` folders up to two levels above it.
- **Command line:** `meshport in.fbx out.glb`, or `meshport file.fbx` to list what's inside.
- **Tested** against 2,079 real-world FBX files, against Autodesk's FBX SDK (through Unity) and against Blender's importer.
- **Not yet:** animation, skinning, blend shapes, cameras and lights.
