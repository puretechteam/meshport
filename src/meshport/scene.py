"""The in-memory scene every format reads into and writes from.

Conventions (the same as glTF): right-handed, +Y up, +Z forward, 1 unit = 1 meter.
Readers convert into this; writers convert out of it.

Meshes keep polygons as they are (no forced triangulation). Every per-corner attribute
(normals, UVs, colors) is stored per *corner*, so a cube is 8 positions, 6 faces and 24
corners. UV coordinates use the FBX/OpenGL convention: v = 0 at the bottom of the image.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class MeshportError(Exception):
    """A file meshport can't read or write. The message is meant for end users."""


# What a damaged or hostile file can make the parsers raise. Readers turn these into
# MeshportError, so callers only ever have to catch one thing.
DAMAGED = (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError, RecursionError,
           MemoryError)

# Limits on what one file can make meshport build. Real models stay far below them; a hostile file
# that would otherwise take hours or all the memory gets a MeshportError instead.
MAX_NODES = 1_000_000   # nodes one file may make, instances and copies included
MAX_FRAMES = 100_000    # samples on one animation's frame grid (an hour at 24 fps is 86,400)


class Budget:
    """A count that may not pass its limit: spend() and need() raise MeshportError(message) past it."""

    def __init__(self, limit: int, message: str):
        self.left, self.message = limit, message

    def spend(self, n: int = 1) -> None:
        self.left -= n
        if self.left < 0:
            raise MeshportError(self.message)

    def need(self, n: int) -> None:
        """Check that n more would still fit, without spending them."""
        if n > self.left:
            raise MeshportError(self.message)


def node_budget() -> Budget:
    return Budget(MAX_NODES, f"This file makes too many objects (over {MAX_NODES:,}); it may be damaged or "
                             "made to stall the reader")


def frame_grid(first: float, last: float) -> np.ndarray | None:
    """Every whole frame number from first to last, or None when that's more than MAX_FRAMES."""
    lo, hi = np.ceil(first), np.floor(last)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo + 1 > MAX_FRAMES:
        return None
    return np.arange(lo, hi + 1)


def too_many_frames(count: float) -> MeshportError:
    return MeshportError(f"This animation is too long: {count:,.0f} frames, over meshport's limit of {MAX_FRAMES:,}")


def linked_path(name: str, folder: Path, root: Path | None = None) -> Path | None:
    """Where a file a model links to by name (a texture, a USD sublayer or reference) may be read
    from: name taken relative to folder (the linking file's own), if it ends up inside root (by
    default the folder above folder). Absolute, network and URL paths (file: included), and ones
    reaching further out, give None, so a model can only bring in files that sit near it."""
    n = name.replace("\\", "/")
    if not n or is_remote_path(n) or n.startswith("/") or ":" in n:
        return None
    try:
        folder = folder.resolve()
        path = (folder / n).resolve()
        return path if path.is_relative_to(root or folder.parent) else None
    except (OSError, ValueError):
        return None


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
class Morph:
    """A blend shape (glTF's morph target): offsets that move the mesh's positions, and its
    normals, in proportion to the target's weight."""
    name: str
    positions: np.ndarray               # (V, 3) offsets
    normals: np.ndarray | None = None   # (C, 3) offsets, per corner like the mesh's normals


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
    targets: list[Morph] = field(default_factory=list)
    target_weights: np.ndarray | None = None  # (M,) the targets' weights when nothing animates them

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
        for m in self.targets:
            if len(m.positions) != len(self.positions) or (m.normals is not None and len(m.normals) != c):
                raise MeshportError(f"Mesh {self.name!r}: blend shape {m.name!r} doesn't fit the mesh")
        if self.target_weights is not None and len(self.target_weights) != len(self.targets):
            raise MeshportError(f"Mesh {self.name!r}: it has a different number of blend shapes and weights")

    def morphed(self, weights: np.ndarray | None = None) -> np.ndarray:
        """The positions with the blend shapes applied at these weights (by default the mesh's own)."""
        weights = self.target_weights if weights is None else weights
        if not self.targets or weights is None:
            return self.positions
        offsets = np.stack([m.positions for m in self.targets])
        return self.positions + np.einsum("m,mvi->vi", np.asarray(weights, np.float64), offsets)

    def triangles(self) -> np.ndarray:
        """(T, 3) corner indices, triangulating every polygon (ear clipping for concave ones)."""
        from .geometry import triangulate
        return triangulate(self.positions, self.face_sizes, self.corners)

    def triangle_faces(self) -> np.ndarray:
        """For each row of triangles(): the polygon it came from."""
        return np.repeat(np.arange(len(self.face_sizes)), self.face_sizes - 2)


@dataclass(eq=False)
class Skin:
    """How a mesh bends with a skeleton, as glTF has it: each vertex follows up to k joints (nodes
    in the scene) by weight. A joint's inverse bind matrix takes a point of the mesh into that
    joint's space as it stood when the mesh was bound, so a posed point is the weighted sum of
    joint_world @ inverse_bind @ point. The skinned node's own transform plays no part then;
    formats without skins place the mesh by it as usual."""
    joints: list[Node]
    inverse_binds: np.ndarray   # (J, 4, 4)
    indices: np.ndarray         # (V, k) int64 into joints
    weights: np.ndarray         # (V, k) float64, each row summing to 1
    name: str = "skin"

    def validate(self, mesh: Mesh) -> None:
        j = len(self.joints)
        if self.inverse_binds.shape != (j, 4, 4) or self.indices.shape != self.weights.shape \
                or len(self.indices) != len(mesh.positions):
            raise MeshportError(f"Mesh {mesh.name!r}: its skin doesn't fit it")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= j):
            raise MeshportError(f"Mesh {mesh.name!r}: its skin points past its joints")

    def bind_space(self, worlds: dict) -> np.ndarray | None:
        """Where the mesh stood when it was bound, as its joints standing at worlds (node id ->
        world matrix) have it: joint_world @ inverse_bind, when every joint agrees (a rig at its bind
        pose), else None. Writers place a skinned mesh there, so importers that rebuild the bind
        from the nodes as they stand (Blender's) see the same rig."""
        found = None
        for joint, inverse in zip(self.joints, self.inverse_binds):
            if id(joint) not in worlds:
                return None
            here = worlds[id(joint)] @ inverse
            if found is None:
                found = here
            elif not np.allclose(here, found, atol=1e-4 * max(1.0, float(np.abs(found[:3, 3]).max()))):  # float32 binds
                return None
        return found

    def pose(self, points: np.ndarray, joint_worlds: np.ndarray) -> np.ndarray:
        """points (V, 3) moved by the joints standing at joint_worlds (J, 4, 4)."""
        m = joint_worlds @ self.inverse_binds
        out = np.zeros_like(points, dtype=np.float64)
        for k in range(self.indices.shape[1]):  # one influence at a time keeps memory to (V, 4, 4)
            mk = m[self.indices[:, k]]
            out += self.weights[:, k, None] * (np.einsum("vij,vj->vi", mk[:, :3, :3], points) + mk[:, :3, 3])
        return out


def pack_influences(rows: np.ndarray, joints: np.ndarray, weights: np.ndarray, count: int):
    """(indices, weights), each (count, k), from a list of (vertex, joint, weight): each vertex's
    influences side by side, the strongest first, zeros after."""
    order = np.lexsort((-weights, rows))
    rows, joints, weights = rows[order], joints[order], weights[order]
    per = np.bincount(rows, minlength=count)
    width = max(int(per.max(initial=0)), 1)
    slot = np.arange(len(rows)) - np.concatenate(([0], np.cumsum(per)[:-1]))[rows]
    indices, packed = np.zeros((count, width), np.int64), np.zeros((count, width))
    indices[rows, slot], packed[rows, slot] = joints, weights
    return indices, packed


@dataclass(eq=False)
class Light:
    """A light as glTF's KHR_lights_punctual has it, shining down its node's -Z axis: a point,
    a spot or the sun ("directional"). Linear RGB color; intensity in candela (point and spot)
    or lux (directional); range in meters (None: no limit); a spot's cone angles in radians
    from its axis, full brightness inside the inner one."""
    name: str
    kind: str = "point"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float | None = None
    inner_cone: float = 0.0
    outer_cone: float = math.pi / 4


@dataclass(eq=False)
class Node:
    name: str
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))  # local, column vectors
    mesh: Mesh | None = None
    materials: list[Material] = field(default_factory=list)  # slots for mesh.face_materials
    children: list[Node] = field(default_factory=list)
    skin: Skin | None = None    # bends mesh with a skeleton
    light: Light | None = None

    def walk(self, parent: np.ndarray | None = None):
        """Yield (node, world_matrix) for this node and everything under it, parents first. A loop
        rather than recursion, so no depth of nesting can overflow the stack."""
        todo = [(self, parent)]
        while todo:
            node, above = todo.pop()
            world = node.matrix if above is None else above @ node.matrix
            yield node, world
            todo.extend((c, world) for c in reversed(node.children))


def frame_rate(times: np.ndarray) -> float:
    """The common frame rate all the key times sit on, or 30 when they sit on none of them."""
    for fps in (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 100.0, 120.0):
        f = np.asarray(times) * fps
        if np.allclose(f, np.round(f), atol=1e-3):
            return fps
    return 30.0


def inbetween_weights(value, knots) -> np.ndarray:
    """A blend shape with in-between shapes, as FBX and USD have them: shapes standing at weights
    knots[0] < ... < knots[-1] (the last one the full shape; 0 is the mesh itself), blended linearly
    between neighbours, the first and last stretch carried on beyond. Each shape is a target of its
    own: this gives their weights (len(value), len(knots)) for the blend shape at each value."""
    value = np.atleast_1d(np.asarray(value, np.float64))
    k = np.concatenate(([0.0], np.asarray(knots, np.float64)))
    seg = np.clip(np.searchsorted(k, value, side="right") - 1, 0, len(k) - 2)
    u = (value - k[seg]) / (k[seg + 1] - k[seg])
    out = np.zeros((len(value), len(k) - 1))
    rows = np.arange(len(value))
    out[rows, seg] = u
    lower = seg > 0
    out[rows[lower], seg[lower] - 1] = 1 - u[lower]
    return out


def with_crossings(times: np.ndarray, values: np.ndarray, knots) -> np.ndarray:
    """times, plus each moment a value sampled at them (linear between) passes one of the knots,
    where a blend shape's in-between weights bend."""
    extra = []
    for knot in np.asarray(knots, np.float64)[:-1]:
        a, b = values[:-1] - knot, values[1:] - knot
        cross = np.flatnonzero(a * b < 0)
        extra.append(times[cross] + (times[cross + 1] - times[cross]) * a[cross] / (a[cross] - b[cross]))
    return np.unique(np.concatenate([times, *extra]))


PATHS = {"translation": 3, "rotation": 4, "scale": 3}  # what a channel can move of a node, and its width
WEIGHTS = "weights"  # a channel of the node's mesh's blend shape weights, one per target
INTERPOLATIONS = ("STEP", "LINEAR", "CUBICSPLINE")


@dataclass(eq=False)
class Channel:
    """One animated property of one node, as glTF stores it: key times in seconds and a value
    at each. Rotations are (x, y, z, w) quaternions. CUBICSPLINE values are (T, 3, width): each
    key's in-tangent, value and out-tangent."""
    node: Node
    path: str            # "translation", "rotation", "scale" or "weights"
    times: np.ndarray    # (T,) float64, increasing
    values: np.ndarray   # (T, width), or (T, 3, width) for CUBICSPLINE
    interpolation: str = "LINEAR"

    def sample(self, t) -> np.ndarray:
        """The value at time t (a number, or an array of times), held at the first and last
        keys outside them. LINEAR rotations are slerped and cubic ones renormalized."""
        t = np.asarray(t, np.float64)
        many = np.atleast_1d(t)
        times = self.times
        cubic = self.interpolation == "CUBICSPLINE"
        values = self.values[:, 1] if cubic else self.values
        k = np.clip(np.searchsorted(times, many, side="right") - 1, 0, len(times) - 1)
        if len(times) == 1 or self.interpolation == "STEP":
            out = values[k]
        else:
            k = np.minimum(k, len(times) - 2)
            span = times[k + 1] - times[k]
            u = np.where(span > 0, np.clip((many - times[k]) / np.where(span > 0, span, 1.0), 0.0, 1.0), 1.0)
            if cubic:  # Hermite, with tangents scaled by the span (glTF's definition)
                u2, u3 = u * u, u * u * u
                h = [2 * u3 - 3 * u2 + 1, (u3 - 2 * u2 + u) * span, -2 * u3 + 3 * u2, (u3 - u2) * span]
                out = (h[0][:, None] * values[k] + h[1][:, None] * self.values[k, 2]
                       + h[2][:, None] * values[k + 1] + h[3][:, None] * self.values[k + 1, 0])
                if self.path == "rotation":
                    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
            elif self.path == "rotation":
                from .geometry import slerp
                out = slerp(values[k], values[k + 1], u)
            else:
                out = values[k] + u[:, None] * (values[k + 1] - values[k])
        return out[0] if t.ndim == 0 else out


@dataclass(eq=False)
class Animation:
    name: str
    channels: list[Channel] = field(default_factory=list)

    @property
    def start(self) -> float:
        return min((float(c.times[0]) for c in self.channels), default=0.0)

    @property
    def end(self) -> float:
        return max((float(c.times[-1]) for c in self.channels), default=0.0)

    def local_matrix(self, node: Node, t: float) -> np.ndarray:
        """node's local matrix at time t: its animated parts sampled, the rest as it stands."""
        from .geometry import compose_trs, decompose_trs
        mine = [c for c in self.channels if c.node is node and c.path in PATHS]
        if not mine:
            return node.matrix
        trs = dict(zip(("translation", "rotation", "scale"), decompose_trs(node.matrix)))
        for c in mine:
            trs[c.path] = c.sample(t)
        return compose_trs(trs["translation"], trs["rotation"], trs["scale"])

    def weights(self, node: Node, t: float) -> np.ndarray | None:
        """The blend shape weights of node's mesh at time t, or None when this doesn't animate them."""
        mine = [c for c in self.channels if c.node is node and c.path == WEIGHTS]
        return mine[-1].sample(t) if mine else None

    def bake(self, rate: float) -> Animation:
        """The same motion as LINEAR keys every 1/rate seconds from start to end, for formats
        that can't hold the original curves."""
        start, end = self.start, self.end
        times = np.arange(start, end, 1.0 / rate) if end > start else np.array([start])
        if times[-1] < end:
            times = np.append(times, end)
        return Animation(self.name, [Channel(c.node, c.path, times.copy(), c.sample(times), "LINEAR")
                                     for c in self.channels])


@dataclass(eq=False)
class Scene:
    roots: list[Node] = field(default_factory=list)
    animations: list[Animation] = field(default_factory=list)

    def walk(self):
        for root in self.roots:
            yield from root.walk()

    def walk_at(self, t: float, animation: Animation | list[Animation] | None = None):
        """Yield (node, world_matrix) posed at time t by an animation, or by several played
        together (Blender exports each object's action as its own). Without one, the first
        animation poses the scene; without any, this is the same as walk()."""
        animation = self._playing(animation)
        todo = [(r, None) for r in reversed(self.roots)]
        while todo:
            node, parent = todo.pop()
            local = animation.local_matrix(node, t) if animation is not None else node.matrix
            world = local if parent is None else parent @ local
            yield node, world
            todo.extend((c, world) for c in reversed(node.children))

    def _playing(self, animation: Animation | list[Animation] | None) -> Animation | None:
        if isinstance(animation, list):
            animation = Animation("together", [c for a in animation for c in a.channels])
        return animation or (self.animations[0] if self.animations else None)

    def posed_points(self, t: float = 0.0, animation: Animation | list[Animation] | None = None):
        """Yield (node, world-space positions) for every mesh posed at time t, as walk_at() picks
        the animation: its blend shapes at their weights, then its skin (or its node's world
        matrix, without one)."""
        animation = self._playing(animation)
        placed = list(self.walk_at(t, animation))
        worlds = {id(n): w for n, w in placed}
        for node, world in placed:
            if node.mesh is None:
                continue
            weights = animation.weights(node, t) if animation is not None else None
            points = node.mesh.morphed(weights)
            if node.skin is not None:
                joints = np.array([worlds.get(id(j), np.eye(4)) for j in node.skin.joints]).reshape(-1, 4, 4)
                yield node, node.skin.pose(points, joints)
            else:
                yield node, points @ world[:3, :3].T + world[:3, 3]

    def deformed(self) -> Scene:
        """The scene with every skin and blend shape applied where it stands (unanimated), for
        formats that can't hold them: each such mesh becomes a copy whose points are already bent,
        in its node's space, so placing it by its node puts it where the skin does. Nodes are
        copied; node animations follow the copies, weights channels are dropped."""
        if not any(n.skin is not None or (n.mesh is not None and n.mesh.targets) for n, _ in self.walk()):
            return self
        from dataclasses import replace
        copies: dict[int, Node] = {}
        for node, _ in self.walk():
            copies[id(node)] = Node(node.name, node.matrix, node.mesh, node.materials)
        for node, _ in self.walk():
            copies[id(node)].children = [copies[id(c)] for c in node.children]
        roots = [copies[id(r)] for r in self.roots]
        posed = dict(self.posed_points(0.0, []))  # an empty list of animations: nothing moves
        for node, world in self.walk():
            if node.mesh is not None and (node.skin is not None or node.mesh.targets):
                points = posed[node] if node.skin is not None else node.mesh.morphed()
                if node.skin is not None:  # back into the node's space
                    inverse = np.linalg.inv(world)
                    points = points @ inverse[:3, :3].T + inverse[:3, 3]
                copies[id(node)].mesh = replace(node.mesh, positions=points, targets=[], target_weights=None)
        animations = [Animation(a.name, [replace(c, node=copies[id(c.node)]) for c in a.channels
                                         if c.path in PATHS and id(c.node) in copies]) for a in self.animations]
        return Scene(roots, [a for a in animations if a.channels])

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
