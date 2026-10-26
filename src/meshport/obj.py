"""Scene -> Wavefront OBJ (+ .mtl and texture files next to it). Write only.

OBJ has no hierarchy, so every mesh is written in world space as its own object. Polygons
stay polygons; mirrored nodes get their winding flipped so faces keep pointing outward.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .fbx import _EXT
from .geometry import transform_normals, transform_points, unique_rows
from .scene import Image, Material, MeshportError, Scene


def write_file(scene: Scene, path: str | os.PathLike) -> list[Path]:
    """Write path (.obj), plus a .mtl and texture files when there are materials.
    Returns every file written."""
    path = Path(path)
    lines = ["# meshport"]
    mtl_name = f"{path.stem}.mtl"
    materials: dict[int, str] = {}
    images: dict[int, str] = {}
    written = [path]
    offsets = [1, 1, 1]  # OBJ indices are 1-based and global across objects
    used_names: set[str] = set()
    any_mesh = False
    for node, world in scene.walk():
        mesh = node.mesh
        if mesh is None or not len(mesh.corners):
            continue
        any_mesh = True
        lines.append(f"o {_name(node.name or mesh.name, used_names)}")
        lines.append(_rows("v", transform_points(world, mesh.positions)))
        corner_v = mesh.corners + offsets[0]
        offsets[0] += len(mesh.positions)
        corner_vt = corner_vn = None
        if mesh.uvs:
            first, index = unique_rows(np.ascontiguousarray(mesh.uvs[0]))
            lines.append(_rows("vt", mesh.uvs[0][first]))
            corner_vt = index + offsets[1]
            offsets[1] += len(first)
        if mesh.normals is not None:
            normals = transform_normals(world, mesh.normals)
            first, index = unique_rows(np.ascontiguousarray(normals))
            lines.append(_rows("vn", normals[first]))
            corner_vn = index + offsets[2]
            offsets[2] += len(first)
        tokens = _corner_tokens(corner_v, corner_vt, corner_vn)
        sizes = mesh.face_sizes
        starts = mesh.face_starts
        if np.linalg.det(world[:3, :3]) < 0:  # mirrored: reverse each face
            tokens = tokens[_reversed_corners(sizes, starts)]
        slots = mesh.face_materials if mesh.face_materials is not None else np.zeros(len(sizes), np.int64)
        for slot in np.unique(slots):
            if node.materials:
                mat = node.materials[min(int(slot), len(node.materials) - 1)]
                if id(mat) not in materials:
                    materials[id(mat)] = _name(mat.name or "material", set(materials.values()))
                lines.append(f"usemtl {materials[id(mat)]}")
            faces = np.nonzero(slots == slot)[0]
            lines.append("\n".join("f " + " ".join(tokens[starts[f]:starts[f] + sizes[f]]) for f in faces))
    if not any_mesh:
        raise MeshportError("No 3D geometry to write")
    if materials:
        lines.insert(1, f"mtllib {mtl_name}")
        mtl = []
        lookup = {id(m): m for node, _ in scene.walk() for m in node.materials}
        for mid, name in materials.items():
            mtl += _material(lookup[mid], name, path.parent, images, written)
        (path.parent / mtl_name).write_text("\n".join(mtl) + "\n", encoding="utf-8")
        written.append(path.parent / mtl_name)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def _name(name: str, taken: set[str]) -> str:
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "object"
    out, n = base, 1
    while out in taken:
        out = f"{base}_{n}"
        n += 1
    taken.add(out)
    return out


def _rows(tag: str, values: np.ndarray) -> str:
    fmt = f"{tag}" + " %.6g" * values.shape[1]
    return "\n".join(fmt % tuple(r) for r in values.tolist())


def _corner_tokens(v, vt, vn) -> np.ndarray:
    v = v.astype(str)
    if vt is None and vn is None:
        return v
    vt_s = vt.astype(str) if vt is not None else np.full(len(v), "")
    out = np.char.add(np.char.add(v, "/"), vt_s)
    if vn is not None:
        out = np.char.add(np.char.add(out, "/"), vn.astype(str))
    return out


def _reversed_corners(sizes, starts) -> np.ndarray:
    face = np.repeat(np.arange(len(sizes)), sizes)
    local = np.arange(int(sizes.sum())) - starts[face]
    return starts[face] + (sizes[face] - 1 - local)


def _material(mat: Material, name: str, folder: Path, images: dict, written: list) -> list[str]:
    r, g, b, a = mat.base_color
    out = [f"newmtl {name}", f"Kd {r:.6g} {g:.6g} {b:.6g}", "Ka 0 0 0", "Ks 0.2 0.2 0.2",
           f"Ns {2.0 / max(mat.roughness, 0.05) ** 2 - 2.0:.6g}", f"d {a:.6g}", "illum 2"]
    if any(mat.emissive):
        out.append("Ke " + " ".join(f"{c:.6g}" for c in mat.emissive))
    for tag, image in (("map_Kd", mat.base_color_texture), ("map_Ke", mat.emissive_texture),
                       ("norm", mat.normal_texture)):
        file = _image_file(image, folder, images, written)
        if file:
            out.append(f"{tag} {file}")
    return out + [""]


def _image_file(image: Image | None, folder: Path, images: dict, written: list) -> str | None:
    if image is None or not image.data:
        return None
    if id(image) not in images:
        ext = _EXT.get(image.mime or "") or (Path(image.path).suffix.lstrip(".") if image.path else "") or "png"
        stem = Path((image.path or "").replace("\\", "/")).stem if image.path else image.name
        name = _name(stem or "texture", set(images.values())) + f".{ext}"
        (folder / name).write_bytes(image.data)
        written.append(folder / name)
        images[id(image)] = name
    return images[id(image)]
