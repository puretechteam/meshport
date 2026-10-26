"""meshport: read and write 3D scene files in pure Python.

    import meshport
    scene = meshport.load("character.fbx")
    meshport.save(scene, "character.glb")

or from a terminal: ``meshport character.fbx character.glb``.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

from .scene import Image, Material, Mesh, MeshportError, Node, Scene  # noqa: E402

FORMATS = {"fbx": "FBX", "glb": "glTF binary", "gltf": "glTF", "obj": "Wavefront OBJ (write only)"}

__all__ = ["FORMATS", "Image", "Material", "Mesh", "MeshportError", "Node", "Scene", "load", "save",
           "read", "write", "__version__"]


def _module(fmt: str):
    fmt = fmt.lower().lstrip(".")
    if fmt == "fbx":
        from . import fbx
        return fbx
    if fmt in ("glb", "gltf"):
        from . import gltf
        return gltf
    raise MeshportError(f"meshport doesn't handle .{fmt} files (it handles {', '.join(FORMATS)})")


def _sniff(data: bytes) -> str | None:
    if data[:4] == b"glTF":
        return "glb"
    if data[:20] == b"Kaydara FBX Binary  ":
        return "fbx"
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b";") or b"FBXHeaderExtension" in head:
        return "fbx"
    if head.startswith(b"{") and b'"asset"' in data[:1 << 16]:
        return "gltf"
    return None


def read(data: bytes, fmt: str | None = None, base_dir: str | os.PathLike | None = None) -> Scene:
    """Parse file bytes. The format is detected from the content when not given."""
    fmt = fmt or _sniff(data)
    if fmt is None:
        raise MeshportError("Couldn't tell what kind of 3D file this is")
    return _module(fmt).read(data, base_dir)


def write(scene: Scene, fmt: str) -> bytes:
    fmt = fmt.lower().lstrip(".")
    if fmt == "obj":
        raise MeshportError("OBJ is a set of files (.obj, .mtl, textures): write it with save()")
    mod = _module(fmt)
    return mod.write(scene, binary=fmt == "glb") if fmt in ("glb", "gltf") else mod.write(scene)


def load(path: str | os.PathLike, fmt: str | None = None) -> Scene:
    """Read a file. External textures are looked up next to it."""
    path = Path(path)
    data = path.read_bytes()
    return read(data, fmt or _sniff(data) or path.suffix, path.parent)


def save(scene: Scene, path: str | os.PathLike, fmt: str | None = None) -> list[Path]:
    """Write a file; the format comes from the extension unless given. Returns every file
    written (OBJ also writes a .mtl and textures next to it)."""
    path = Path(path)
    fmt = (fmt or path.suffix).lower().lstrip(".")
    if fmt == "obj":
        from . import obj
        return obj.write_file(scene, path)
    path.write_bytes(write(scene, fmt))
    return [path]
