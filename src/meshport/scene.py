"""The in-memory scene every format reads into and writes from.

Conventions (the same as glTF): right-handed, +Y up, +Z forward, 1 unit = 1 meter.
Readers convert into this; writers convert out of it.

Meshes keep polygons as they are (no forced triangulation). Every per-corner attribute
(normals, UVs, colors) is stored per *corner*, so a cube is 8 positions, 6 faces and 24
corners. UV coordinates use the FBX/OpenGL convention: v = 0 at the bottom of the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


class MeshportError(Exception):
    """A file meshport can't read or write. The message is meant for end users."""


# What a damaged or hostile file can make the parsers raise. Readers turn these into
# MeshportError, so callers only ever have to catch one thing.
DAMAGED = (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError, RecursionError,
           MemoryError)


@dataclass(eq=False)
class Image:
    name: str
    data: bytes | None = None   # the encoded file (png, jpg, ...), when found or embedded
    path: str | None = None     # where it was referenced from, for files that only point to it

    @property
    def mime(self) -> str | None:
        return sniff_mime(self.data) if self.data else None


@dataclass(eq=False)
class Material:
    name: str
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    base_color_texture: Image | None = None
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_texture: Image | None = None
    normal_texture: Image | None = None
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"  # OPAQUE, MASK or BLEND, as in glTF


@dataclass(eq=False)
class Mesh:
    name: str
    positions: np.ndarray                    # (V, 3) float64
    face_sizes: np.ndarray                   # (F,) int64, corners per polygon (>= 3)
    corners: np.ndarray                      # (C,) int64, position index per corner
    normals: np.ndarray | None = None        # (C, 3)
    uvs: list[np.ndarray] = field(default_factory=list)  # each (C, 2)
    colors: np.ndarray | None = None         # (C, 4), linear 0..1
    face_materials: np.ndarray | None = None  # (F,) int64 into the node's material slots

    @property
    def face_starts(self) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(self.face_sizes)[:-1])).astype(np.int64)

    def validate(self) -> None:
        c = len(self.corners)
        if int(self.face_sizes.sum()) != c:
            raise MeshportError(f"Mesh {self.name!r}: face sizes don't add up to the corner count")
        if c and (self.corners.min() < 0 or self.corners.max() >= len(self.positions)):
            raise MeshportError(f"Mesh {self.name!r}: a corner points past the vertex list")
        for a in [self.normals, self.colors, *self.uvs]:
            if a is not None and len(a) != c:
                raise MeshportError(f"Mesh {self.name!r}: an attribute doesn't have one value per corner")

    def triangles(self) -> np.ndarray:
        """(T, 3) corner indices, triangulating every polygon (ear clipping for concave ones)."""
        from .geometry import triangulate
        return triangulate(self.positions, self.face_sizes, self.corners)

    def triangle_faces(self) -> np.ndarray:
        """For each row of triangles(): the polygon it came from."""
        return np.repeat(np.arange(len(self.face_sizes)), self.face_sizes - 2)


@dataclass(eq=False)
class Node:
    name: str
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))  # local, column vectors
    mesh: Mesh | None = None
    materials: list[Material] = field(default_factory=list)  # slots for mesh.face_materials
    children: list[Node] = field(default_factory=list)

    def walk(self, parent: np.ndarray | None = None):
        """Yield (node, world_matrix) for this node and everything under it."""
        world = self.matrix if parent is None else parent @ self.matrix
        yield self, world
        for child in self.children:
            yield from child.walk(world)


@dataclass(eq=False)
class Scene:
    roots: list[Node] = field(default_factory=list)

    def walk(self):
        for root in self.roots:
            yield from root.walk()

    @property
    def meshes(self) -> list[Mesh]:
        seen: dict[int, Mesh] = {}
        for node, _ in self.walk():
            if node.mesh is not None:
                seen.setdefault(id(node.mesh), node.mesh)
        return list(seen.values())

    def bounds(self) -> np.ndarray | None:
        """World-space (2, 3) min/max over every mesh, or None for an empty scene."""
        lo, hi = [], []
        for node, world in self.walk():
            if node.mesh is not None and len(node.mesh.positions):
                p = node.mesh.positions @ world[:3, :3].T + world[:3, 3]
                lo.append(p.min(0))
                hi.append(p.max(0))
        return np.array([np.min(lo, 0), np.max(hi, 0)]) if lo else None


_MORE_IMAGE_MAGIC = (b"\x76\x2f\x31\x01", b"#?RADIANCE", b"#?RGBE", b"\xabKTX ", b"\x01\xda")  # EXR, HDR, KTX, SGI
_BMP_INFO_SIZES = (12, 16, 40, 52, 56, 64, 108, 124)


def looks_like_image(head: bytes, name: str = "") -> bool:
    """Whether a file found on disk (its first 64 bytes or more) is an image. Model files name
    their textures by path, so without this a hostile model could point at any file (a document,
    a key) and have it copied into the converted output."""
    if head.startswith(b"BM"):
        return len(head) >= 18 and int.from_bytes(head[14:18], "little") in _BMP_INFO_SIZES
    if sniff_mime(head) or head.startswith(_MORE_IMAGE_MAGIC):
        return True
    if head[4:8] == b"ftyp" and head[8:12] in (b"avif", b"avis", b"heic", b"heix", b"mif1"):
        return True
    # TGA has no signature: take the extension's word for it when the header is a valid one
    return (name.lower().endswith(".tga") and len(head) >= 18 and head[1] in (0, 1)
            and head[2] in (1, 2, 3, 9, 10, 11) and head[16] in (8, 15, 16, 24, 32))


def is_remote_path(name: str) -> bool:
    """A UNC path (\\\\server\\share\\...) or a URL. Merely checking whether one exists makes
    Windows contact that server and offer it the user's login, so readers never touch them."""
    name = name.replace("\\", "/")
    return name.startswith("//") or "://" in name


def read_image_file(path: str | os.PathLike) -> bytes | None:
    """The file's bytes if it's an image, else None. Only the start of anything else is read."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
            if not looks_like_image(head, os.fspath(path)):
                return None
            return head + f.read()
    except (OSError, ValueError):
        return None


def sniff_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"II*\0", b"MM\0*")):
        return "image/tiff"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"DDS "):
        return "image/vnd-ms.dds"
    if data.startswith(b"8BPS"):
        return "image/vnd.adobe.photoshop"
    if data.startswith(b"GIF8"):
        return "image/gif"
    return None
