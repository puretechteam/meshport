"""meshport: read and write 3D scene files in pure Python.

    import meshport
    scene = meshport.load("character.fbx")
    meshport.save(scene, "character.glb")

or from a terminal: ``meshport character.fbx character.glb``.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.4.0"

from .scene import (
    Animation,
    Channel,
    Image,
    Light,
    Material,
    Mesh,
    MeshportError,
    Morph,
    Node,
    Scene,
    Skin,
)

FORMATS = {"fbx": "FBX", "glb": "glTF binary", "gltf": "glTF", "3ds": "Autodesk 3DS",
           "usd": "USD", "usda": "USD text", "usdc": "USD binary", "usdz": "USD package", "abc": "Alembic",
           "blend": "Blender (read only)", "obj": "Wavefront OBJ"}
_USD = ("usd", "usda", "usdc", "usdz")

__all__ = ["FORMATS", "Animation", "Channel", "Image", "Light", "Material", "Mesh", "MeshportError", "Morph", "Node", "Scene",
           "Skin", "__version__", "load", "read", "save", "write"]


def _module(fmt: str):
    fmt = fmt.lower().lstrip(".")
    if fmt == "fbx":
        from . import fbx
        return fbx
    if fmt in ("glb", "gltf"):
        from . import gltf
        return gltf
    if fmt == "3ds":
        from . import max3ds
        return max3ds
    if fmt in _USD:
        from . import usd
        return usd
    if fmt == "abc":
        from . import alembic
        return alembic
    if fmt == "blend":
        from . import blend
        return blend
    if fmt == "obj":
        from . import obj
        return obj
    raise MeshportError(f"meshport doesn't handle .{fmt} files (it handles {', '.join(FORMATS)})")


def _sniff(data: bytes) -> str | None:
    if data[:4] == b"glTF":
        return "glb"
    if data[:20] == b"Kaydara FBX Binary  ":
        return "fbx"
    if data[:2] == b"MM" and data[6:8] in (b"\x02\x00", b"==", b"\x00\xb0"):  # then version, editor or keyframer
        return "3ds"
    if data[:5] == b"Ogawa":
        return "abc"
    if data[:7] == b"BLENDER":
        return "blend"
    if data[:4] == b"\x28\xb5\x2f\xfd" or data[:2] == b"\x1f\x8b":  # a compressed .blend
        from .blend import _decompress
        try:
            head = _decompress(data[:1 << 16], 16)  # only the start: a gzip file may hold gigabytes
        except MeshportError:
            head = None  # (zstd without a decompressor installed, for one: reading it says so)
        if head is None or head[:7] == b"BLENDER":
            return "blend"
    from .usd import detect
    usd = detect(data)
    if usd:
        return usd
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b";") or b"FBXHeaderExtension" in head:
        return "fbx"
    if head.startswith(b"{") and b'"asset"' in data[:1 << 16]:
        return "gltf"
    from .obj import detect as obj
    if obj(data):
        return "obj"
    return None


def read(data: bytes, fmt: str | None = None, base_dir: str | os.PathLike | None = None) -> Scene:
    """Parse file bytes. The format is detected from the content when not given."""
    fmt = fmt or _sniff(data)
    if fmt is None:
        raise MeshportError("Couldn't tell what kind of 3D file this is")
    return _module(fmt).read(data, base_dir)


def write(scene: Scene, fmt: str, ascii: bool = False) -> bytes:
    """File bytes. 3DS and USD layers (other than .usdz) only name their textures: save()
    writes them next to the file. ascii=True writes FBX as text instead of binary."""
    fmt = fmt.lower().lstrip(".")
    if ascii and fmt != "fbx":
        raise MeshportError("Only FBX can be written as ASCII")
    if fmt == "obj":
        raise MeshportError("OBJ is a set of files (.obj, .mtl, textures): write it with save()")
    if fmt == "blend":
        raise MeshportError("meshport reads .blend files but can't write them")
    mod = _module(fmt)
    if fmt in ("glb", "gltf"):
        return mod.write(scene, binary=fmt == "glb")
    if fmt in _USD:
        return mod.write(scene, "usdc" if fmt == "usd" else fmt)
    if fmt == "fbx":
        return mod.write(scene, ascii)
    return mod.write(scene)


def load(path: str | os.PathLike, fmt: str | None = None) -> Scene:
    """Read a file. External textures are looked up next to it."""
    path = Path(path)
    data = path.read_bytes()
    return read(data, fmt or _sniff(data) or path.suffix, path.parent)


def save(scene: Scene, path: str | os.PathLike, fmt: str | None = None, ascii: bool = False) -> list[Path]:
    """Write a file; the format comes from the extension unless given. Returns every file
    written (OBJ also writes a .mtl and textures next to it, 3DS its textures, and USD other
    than .usdz its textures in a textures folder). ascii=True writes FBX as text."""
    path = Path(path)
    fmt = (fmt or path.suffix).lower().lstrip(".")
    if ascii and fmt != "fbx":
        raise MeshportError("Only FBX can be written as ASCII")
    if fmt == "obj":
        from . import obj
        return obj.write_file(scene, path)
    if fmt == "3ds":
        from . import max3ds
        return max3ds.write_file(scene, path)
    if fmt in _USD:
        from . import usd
        return usd.write_file(scene, path, "usdc" if fmt == "usd" else fmt)
    path.write_bytes(write(scene, fmt, ascii))
    return [path]
