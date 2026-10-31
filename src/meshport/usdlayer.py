"""The in-memory form of one USD layer, shared by the text (.usda) and binary (.usdc) codecs.

A layer is a set of specs keyed by path. Paths are kept as USD writes them: "/" for the root,
"/World/Chair" for prims, "/World/Chair.points" for properties, and "/Asset{lod=high}" for a
variant (its children continue without a slash: "/Asset{lod=high}Body").

Read specs hold plain Python values in `fields`: str for tokens, strings, asset paths and
specifiers; numbers; NumPy arrays; lists; ListOp, Reference and Payload; dicts for
dictionaries and time samples ({time: value}). The root's "subLayerOffsets" are (offset, scale)
pairs, one per sublayer. Specs being written also carry `typed`:
{field: (crate type, is_array, value)}, so both codecs know how to store each value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# spec kinds (SdfSpecType)
SPEC_ATTRIBUTE, SPEC_PRIM, SPEC_PSEUDO_ROOT, SPEC_RELATIONSHIP, SPEC_VARIANT, SPEC_VARIANT_SET = 1, 6, 7, 8, 10, 11


@dataclass(eq=False)
class Spec:
    kind: int
    fields: dict = field(default_factory=dict)
    typed: dict = field(default_factory=dict)


@dataclass(eq=False)
class Layer:
    specs: dict[str, Spec] = field(default_factory=dict)

    def get(self, path: str, name: str, default=None):
        spec = self.specs.get(path)
        return default if spec is None else spec.fields.get(name, default)


@dataclass
class ListOp:
    """An edit to a list: replace it (explicit) or add to and remove from what weaker layers say."""
    explicit: bool = False
    explicit_items: list = field(default_factory=list)
    added: list = field(default_factory=list)
    prepended: list = field(default_factory=list)
    appended: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    ordered: list = field(default_factory=list)

    def items(self) -> list:
        if self.explicit:
            return list(self.explicit_items)
        try:
            deleted = set(self.deleted)  # (a set: lists of many items stay fast)
        except TypeError:  # items that can't be hashed
            deleted = self.deleted
        return [x for x in self.prepended + self.added + self.appended if x not in deleted]


@dataclass(frozen=True)
class Reference:
    asset: str     # "" for a prim in the same layer stack
    prim: str      # "" for the referenced layer's default prim
    offset: float = 0.0  # its layer offset: a time t there is offset + scale * t here
    scale: float = 1.0


@dataclass(frozen=True)
class Payload:
    asset: str
    prim: str
    offset: float = 0.0
    scale: float = 1.0


def child_path(parent: str, name: str) -> str:
    if name.startswith("{") or parent.endswith("}"):
        return parent + name
    return "/" + name if parent == "/" else f"{parent}/{name}"


def property_path(prim: str, name: str) -> str:
    return f"{prim}.{name}"


def _cut(path: str) -> int:
    """Where the last element starts: after the last '/' or variant '}'."""
    return max(path.rfind("/"), path.rfind("}")) + 1


def is_property(path: str) -> bool:
    return "." in path[_cut(path):]


def parent_path(path: str) -> str:
    if is_property(path):
        return path[:path.rindex(".")]
    if path.endswith("}"):
        return path[:path.rindex("{")]
    cut = _cut(path)
    if path[cut - 1] == "}":
        return path[:cut]
    return path[:cut - 1] or "/"


def element_name(path: str) -> str:
    if is_property(path):
        return path[path.rindex(".") + 1:]
    if path.endswith("}"):
        return path[path.rindex("{"):]
    return path[_cut(path):]


def strip_variants(path: str) -> str:
    """'/A{v=x}B.points' -> '/A/B.points': the path a variant's contents have on the stage."""
    out, i = [], 0
    while i < len(path):
        c = path[i]
        if c == "{":
            end = path.index("}", i)
            i = end + 1
            if i < len(path) and path[i] not in "./":
                out.append("/")
            continue
        out.append(c)
        i += 1
    return "".join(out)
