"""Blender armatures and shape keys -> skins and blend shapes.

An armature object's data keeps its bones as a tree, each with its rest matrix in the armature's
space (arm_mat); the object's pose keeps a channel per bone with its own location, rotation (in
the channel's rotation mode) and scale. A bone stands at its parent's pose, then its rest matrix
relative to its parent's, then its channel's transform. Bones become nodes under the armature
object's node. A bone can leave out its parent's rotation (it keeps its rest orientation) and its
scale (inherit scale None, or only the average of it, Average); worked out from files Blender saves,
these match Blender's poses exactly. Fix Shear and Aligned, which differ from Full only under a
parent that scales unevenly, are followed as Full.

A mesh object with an armature modifier (on in the viewport, deforming by vertex groups) gets a
skin: its vertex groups (named on the mesh from Blender 3.0, on the object before) whose names
are bones, weighted per vertex (MDeformVert), normalized as the modifier normalizes them. A vertex
in no bone's group stays with the mesh object, as in Blender. The modifier's envelopes, preserve
volume (dual quaternion) and strength aren't followed, and the mesh has to stay where it is
relative to the armature (it's usually parented to it).

A mesh's shape keys (relative ones) become blend shapes: each key's offsets from the key it's
relative to, scaled by its vertex group when it has one, at its current value. Muted keys are
left out.
"""

from __future__ import annotations

import re

import numpy as np

from .scene import Morph, Node

_BONE_PATH = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(\w+)$')
_KEY_PATH = re.compile(r'^key_blocks\["((?:[^"\\]|\\.)*)"\]\.value$')
_KEYBLOCK_MUTE = 1
_BONE_HINGE = 0x200  # Bone.flag: the bone doesn't inherit its parent's rotation
_SCALE_FULL, _SCALE_AVERAGE, _SCALE_NONE, _SCALE_NONE_LEGACY = 0, 2, 3, 4  # Bone.inherit_scale_mode
_ARM_DEF_VGROUP = 1
_CD_MDEFORMVERT = 2  # the vertex data layer type of vertex group weights
WEIGHT_BUDGET = 64_000_000   # vertex group memberships one mesh may have
SHAPE_BUDGET = 100_000_000   # shape key values (keys x points x 3) one mesh may have


def bone_curve(path: str) -> tuple[str, str] | None:
    """(bone name, property) for a pose bone F-curve's data path."""
    got = _BONE_PATH.match(path)
    return (_unescape(got.group(1)), got.group(2)) if got else None


def key_curve(path: str) -> str | None:
    """The shape key a value F-curve animates."""
    got = _KEY_PATH.match(path)
    return _unescape(got.group(1)) if got else None


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)


def _matrix(value) -> np.ndarray:
    """A float[4][4] as Blender stores it (column by column) as a column-vector matrix."""
    return np.asarray(value, np.float64).reshape(4, 4).T


class Rig:
    """An armature object's bones as nodes: names, rest matrices and pose channels."""

    def __init__(self, ob, arm):
        self.nodes: dict[str, Node] = {}
        self.rest: dict[str, np.ndarray] = {}      # bone -> its rest matrix in the armature's space
        self.relative: dict[str, np.ndarray] = {}  # bone -> its rest matrix in its parent's space
        self.channels: dict[str, object] = {}      # bone -> its pose channel
        self.parent: dict[str, str] = {}
        self.attached: dict[str, str] = {}         # bone -> the bone its node hangs under (others: the armature)
        self.options: dict[str, tuple[bool, int]] = {}  # bone -> (inherits rotation, scale mode), when not full
        self.order: list[str] = []                 # parents before children
        self.roots: list[Node] = []
        pose = ob.follow("pose", "bPose")
        if pose is not None:
            for ch in pose.listbase("chanbase", "bPoseChannel"):
                self.channels.setdefault(str(ch.get("name", "")), ch)
        self._add(arm.listbase("bonebase", "Bone"), None, 0)

    def _add(self, bones, parent: str | None, depth: int) -> None:
        if depth > 512:
            return
        for bone in bones:
            name = str(bone.get("name", "")) or "bone"
            if name in self.nodes:
                continue
            rest = _matrix(bone.get("arm_mat"))
            relative = rest if parent is None else np.linalg.inv(self.rest[parent]) @ rest
            self.rest[name], self.relative[name] = rest, relative
            hinge = bool(int(bone.get("flag", 0) or 0) & _BONE_HINGE)
            scale = int(bone.get("inherit_scale_mode", 0) or 0)
            if parent is not None and (hinge or scale in (_SCALE_AVERAGE, _SCALE_NONE, _SCALE_NONE_LEGACY)):
                self.options[name] = (not hinge, scale)
            node = Node(name)
            self.nodes[name] = node
            self.order.append(name)
            if parent is not None:
                self.parent[name] = parent
            # a bone that leaves out its parent's rotation or scale hangs from the armature itself: under a
            # parent that scales unevenly its matrix would need a shear, which a node's can't hold
            above = None if parent is None or name in self.options else parent
            if above is not None:
                self.attached[name] = above
            (self.roots if above is None else self.nodes[above].children).append(node)
            self._add(bone.listbase("childbase", "Bone"), name, depth + 1)
        if depth == 0:
            bases = {n: basis(ch) for n, ch in self.channels.items() if n in self.nodes}
            for name, local in self.locals(bases).items():
                self.nodes[name].matrix = local

    def poses(self, bases: dict) -> dict[str, np.ndarray]:
        """Each bone's pose in the armature's space, its channel's transform being bases[bone]."""
        out: dict[str, np.ndarray] = {}
        for name in self.order:
            own = bases.get(name, np.eye(4))
            if name not in self.parent:
                out[name] = self.rest[name] @ own
                continue
            above = out[self.parent[name]]
            placed = above @ self.relative[name]
            if name in self.options:  # the parent's rotation, its scale, or both left out
                rotates, scale = self.options[name]
                turn = _rotation(above[:3, :3] if rotates else self.rest[self.parent[name]][:3, :3])
                stretch = turn.T @ above[:3, :3] if rotates else _rotation(above[:3, :3]).T @ above[:3, :3]
                if scale in (_SCALE_NONE, _SCALE_NONE_LEGACY):
                    stretch = np.eye(3)
                elif scale == _SCALE_AVERAGE:
                    stretch = np.eye(3) * np.cbrt(abs(np.linalg.det(above[:3, :3])))
                placed = placed.copy()
                placed[:3, :3] = turn @ stretch @ self.relative[name][:3, :3]
            out[name] = placed @ own
        return out

    def locals(self, bases: dict) -> dict[str, np.ndarray]:
        """Each bone node's matrix (relative to the node it hangs under) for these channel transforms."""
        poses = self.poses(bases)
        return {n: poses[n] if n not in self.attached else np.linalg.inv(poses[self.attached[n]]) @ poses[n]
                for n in self.order}


def _rotation(m: np.ndarray) -> np.ndarray:
    """The rotation in a 3x3 matrix (its polar decomposition's)."""
    u, _, vt = np.linalg.svd(m)
    q = u @ vt
    return q if np.linalg.det(q) > 0 else u @ np.diag([1, 1, -1]) @ vt


def basis(ch, values: dict | None = None) -> np.ndarray:
    """A pose channel's own transform: location, rotation in its mode, scale. values overrides
    its fields (an animation's at some frame)."""
    from .blend import _EULER, _axis_angle, _quat
    from .geometry import euler, scaling, translation

    def get(name, default=None):
        return values[name] if values and name in values else ch.get(name, default)
    mode = int(ch.get("rotmode", 0))
    if mode > 0:
        rot = euler(np.degrees(np.asarray(get("eul"), np.float64)), _EULER.get(mode, "XYZ"))
    elif mode == -1:
        rot = _axis_angle(get("rotAxis"), float(get("rotAngle", 0.0)))
    else:
        rot = _quat(get("quat"))
    return translation(np.asarray(get("loc"), np.float64)) @ rot @ scaling(np.asarray(get("size"), np.float64))


# the pose channel fields each data path animates
BONE_FIELDS = {"location": "loc", "rotation_quaternion": "quat", "rotation_euler": "eul", "scale": "size"}


def armature_modifier(ob):
    """The object's first armature modifier that's on in the viewport and deforms by vertex groups."""
    from .blendmods import _REALTIME
    for md in ob.listbase("modifiers"):
        head = md.get("modifier")
        if (md.name == "ArmatureModifierData" and head is not None and int(head.get("mode", 0)) & _REALTIME
                and int(md.get("deformflag", _ARM_DEF_VGROUP)) & _ARM_DEF_VGROUP):
            return md
    return None


def group_names(ob, me) -> list[str]:
    groups = me.listbase("vertex_group_names", "bDeformGroup") if me.has("vertex_group_names") else []
    if not groups:
        groups = ob.listbase("defbase", "bDeformGroup")
    return [str(g.get("name", "")) for g in groups]


def vertex_weights(f, me, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(vertex, group, weight) for every vertex group membership of the mesh's vertices. They're
    on the mesh (dvert) or, as newer Blenders save them, a layer of its vertex data."""
    from .blend import _S
    from .scene import MeshportError
    address = me.get("dvert", 0)
    cd = me.get("vdata")
    if not address and cd is not None and "CustomDataLayer" in f.struct_index:
        got = f.at(cd.get("layers", 0), me.owner)
        size = f.struct_size["CustomDataLayer"]
        for i in range(min(int(cd.get("totlayer", 0) or 0), 256, (got[0].length - got[1]) // max(size, 1))
                       if got is not None else 0):
            layer = _S(f, "CustomDataLayer", got[0].offset + got[1] + i * f.struct_size["CustomDataLayer"], got[0].scope)
            if int(layer.get("type", -1)) == _CD_MDEFORMVERT:
                address = layer.get("data", 0)
                break
    got = f.at(address, me.owner)
    if got is None or "MDeformVert" not in f.struct_index:
        return None
    blk, at = got
    size = f.struct_size["MDeformVert"]
    if at + size * count > blk.length:
        return None
    rows, groups, weights = [], [], []
    wsize = f.struct_size.get("MDeformWeight", 8)
    lists: dict[tuple[int, int], tuple | None] = {}  # (address, count) -> its weights: vertices can share one
    left = WEIGHT_BUDGET
    for v in range(count):
        dv = _S(f, "MDeformVert", blk.offset + at + v * size, blk.scope)
        n = int(dv.get("totweight", 0) or 0)
        if n <= 0:
            continue
        key = (dv.get("dw", 0), n)
        if key not in lists:
            hit = f.at(key[0], blk.scope)
            if hit is None or n > 4096 or hit[1] + wsize * n > hit[0].length or wsize != 8:
                lists[key] = None
            else:
                raw = np.frombuffer(f.data, np.dtype([("def_nr", "<i4"), ("weight", "<f4")]), n, hit[0].offset + hit[1])
                lists[key] = (raw["def_nr"].astype(np.int64), raw["weight"].astype(np.float64))
        if lists[key] is None:
            continue
        left -= n
        if left < 0:
            raise MeshportError(f"This mesh has too many vertex group weights (over {WEIGHT_BUDGET:,})")
        rows.append(np.full(n, v))
        groups.append(lists[key][0])
        weights.append(lists[key][1])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(groups), np.concatenate(weights)


def shape_keys(f, ob, me, count: int) -> list[tuple[str, np.ndarray, float]]:
    """(name, offsets, value) for each of the mesh's relative shape keys but the reference one."""
    from .blend import _S
    from .scene import MeshportError
    key = None
    address = me.get("key", 0)
    if address:
        b = f.ids.get(address)
        key = _S(f, "Key", b.offset, b.index) if b is not None else None
    if key is None or int(key.get("type", 1)) != 1:  # absolute keys (KEY_NORMAL) aren't blend shapes
        return []
    blocks = list(key.listbase("block", "KeyBlock"))
    if len(blocks) * count * 3 > SHAPE_BUDGET:
        raise MeshportError(f"This mesh has too many shape key values (over {SHAPE_BUDGET:,})")
    data = []
    points: dict[int, np.ndarray] = {}  # address -> the points stored there: keys can share them
    for kb in blocks:
        n = int(kb.get("totelem", 0) or 0)
        got = f.at(kb.get("data", 0), key.owner)
        if n != count or got is None or got[1] + 12 * n > got[0].length:
            data.append(None)
            continue
        at = got[0].offset + got[1]
        if at not in points:
            points[at] = np.frombuffer(f.data, "<f4", 3 * n, at).reshape(n, 3).astype(np.float64)
        data.append(points[at])
    ref = key.get("refkey", 0)
    names = group_names(ob, me)
    weights = None
    out = []
    for i, kb in enumerate(blocks):
        if data[i] is None or (ref and kb.offset == _offset_of(f, ref, key)) or (not ref and i == 0):
            continue
        if int(kb.get("flag", 0)) & _KEYBLOCK_MUTE:
            continue
        rel = int(kb.get("relative", 0))
        base = data[rel] if 0 <= rel < len(data) and data[rel] is not None else None
        if base is None:
            continue
        offsets = data[i] - base
        group = str(kb.get("vgroup", "") or "")
        if group and group in names:  # the key moves each vertex as much as its weight in the group
            if weights is None:
                weights = vertex_weights(f, me, count)
            scale = np.zeros(count)
            if weights is not None:
                v, g, w = weights
                pick = g == names.index(group)
                scale[v[pick]] = w[pick]
            offsets = offsets * scale[:, None]
        out.append((str(kb.get("name", "")) or f"key{i}", offsets, float(kb.get("curval", 0.0) or 0.0)))
    return out


def _offset_of(f, address: int, owner) -> int | None:
    got = f.at(address, owner.owner)
    return None if got is None else got[0].offset + got[1]


def morphs(keys: list) -> tuple[list[Morph], np.ndarray]:
    return [Morph(name, offsets) for name, offsets, _ in keys], np.array([w for _, _, w in keys])
