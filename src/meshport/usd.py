"""USD (.usd, .usda, .usdc, .usdz) <-> Scene.

Reading composes the stage the way USD does for what a model needs: sublayers, inherits,
variant selections, references and payloads (to other files, inside a .usdz package, or within the
same file) and specializes, in USD's strength order, with the namespace mapping that comes with
them. Then:

- Xform ops of every kind (translate, rotate in any order, scale, orient, transform, pivots and
  their inverses, resetXformStack), and stage upAxis and metersPerUnit (USD's default is
  centimeters);
- Mesh prims with normals, every UV set and display colors in any interpolation, indexed
  primvars, holes, left-handed orientation, and GeomSubsets for per-face materials;
  Cube, Sphere, Cylinder, Cone, Capsule and Plane prims become meshes;
- point instancers: each instance becomes a node holding its prototype (meshes shared), placed by
  the instancer's positions, orientations and scales; hidden and deactivated instances are left out;
- UsdPreviewSurface materials (directly or through material and node-graph inputs) with their
  textures, and displayColor when no material is bound;
- invisible, inactive, class-only, guide and proxy prims are left out.

- animated transforms: every xform op's time samples, retimed through layer offsets on
  sublayers, references and payloads (and each layer's timeCodesPerSecond), become node channels
  on the stage's time line (seconds = time code / timeCodesPerSecond);
- UsdSkel: a Skeleton's joints become nodes under it, posed by its SkelAnimation (translations,
  rotations, scales, sampled or not) over its rest transforms; meshes bound to it get a skin
  (jointIndices and jointWeights, the mesh's own skel:joints order, geomBindTransform) and their
  BlendShapes (offsets, pointIndices, normalOffsets), weighted by the animation's
  blendShapeWeights, matched by name. In-between shapes are left out.

Animated points (a mesh whose points have time samples and whose faces don't change) play as
USD plays them: each later sample is a blend shape, weighted so the mesh blends linearly from
sample to sample, up to alembic.CACHE_BUDGET values.

Not handled: curves, points prims and other animated attributes. Classes are composed from the
layer stacks they're found in; relocates aren't read.

Writing produces a stage Apple's AR Quick Look and usdchecker accept: Y up, meters, a default
prim, UsdPreviewSurface materials with textures, binary (.usdc) layers inside .usdz packages.
An animated node gets translate, orient and scale ops with time samples (USD has one time line,
so the first animation is written). Skins and blend shapes are written as UsdSkel: the root prim
becomes a SkelRoot, each skin a Skeleton beside the scene's roots with a SkelAnimation (sampled
every frame while its joints move), and a mesh with blend shapes but no skin a Skeleton without
joints whose animation carries the weights.
"""

from __future__ import annotations

import io
import os
import re
import struct
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

from . import usda, usdcrate
from .geometry import (
    compose_trs,
    decompose_many,
    decompose_trs,
    euler,
    scaling,
    slerp,
    translation,
    unique_rows,
)
from .scene import (
    DAMAGED,
    WEIGHTS,
    Animation,
    Budget,
    Channel,
    Image,
    Material,
    Mesh,
    MeshportError,
    Morph,
    Node,
    Scene,
    Skin,
    frame_grid,
    frame_rate,
    inbetween_weights,
    linked_path,
    looks_like_image,
    node_budget,
    read_image_file,
    too_many_frames,
    with_crossings,
)
from .usdlayer import (
    SPEC_ATTRIBUTE,
    SPEC_PRIM,
    SPEC_PSEUDO_ROOT,
    SPEC_RELATIONSHIP,
    Layer,
    ListOp,
    Spec,
    child_path,
    element_name,
    parent_path,
    property_path,
    strip_variants,
)

MAX_ARC_DEPTH = 32       # references within references...
MAX_LAYERS = 512         # files one stage may pull in
MAX_SITES = 4_000_000    # opinions one stage may compose (a prim has one per layer and arc that speaks of it)
_GPRIMS = {"Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule", "Plane", "Cylinder_1", "Capsule_1"}
_SKIP_TYPES = {"Material", "Shader", "NodeGraph", "GeomSubset", "Camera", "DistantLight", "DomeLight",
               "SphereLight", "RectLight", "DiskLight", "CylinderLight", "SkelAnimation",
               "BlendShape", "BasisCurves", "NurbsCurves", "Points", "HermiteCurves"}


# ============================================================================ files and packages

def detect(data: bytes) -> str | None:
    if data[:8] == usdcrate.MAGIC:
        return "usdc"
    if data.lstrip(b"\xef\xbb\xbf")[:5] == b"#usda":
        return "usda"
    if data[:4] == b"PK\x03\x04":
        name_len = int.from_bytes(data[26:28], "little")
        if data[30:30 + name_len].lower().rsplit(b".", 1)[-1] in (b"usd", b"usda", b"usdc"):
            return "usdz"
    return None


def _parse_layer(data: bytes) -> Layer:
    kind = detect(data)
    if kind == "usdc":
        return usdcrate.read(data)
    if kind == "usda":
        return usda.read(data)
    raise MeshportError("This isn't a USD layer")


class _Source:
    """Where a layer came from, to find what it points at: a folder on disk, or a folder
    inside a .usdz package. Files on disk are only read from inside root: the top file's folder
    and the one above it (and below them), however the layers point from one to the next."""

    def __init__(self, base_dir: Path | None, package: _Package | None = None, inner_dir: str = "",
                 root: Path | None = None):
        self.base_dir, self.package, self.inner_dir = base_dir, package, inner_dir
        self.root = root if root is not None or base_dir is None else base_dir.resolve().parent

    def key(self, asset: str) -> str:
        if self.package is not None:
            return f"{id(self.package)}:{self._inner(asset)}"
        return str(self._disk(asset)) if self._disk(asset) else asset

    def _inner(self, asset: str) -> str:
        p = PurePosixPath(self.inner_dir) / asset.replace("\\", "/").lstrip("./") if not asset.startswith("/") \
            else PurePosixPath(asset.lstrip("/"))
        parts = []
        for part in p.parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part not in (".", ""):
                parts.append(part)
        return "/".join(parts)

    def _disk(self, asset: str) -> Path | None:
        return linked_path(asset, self.base_dir, self.root) if self.base_dir is not None else None

    def read(self, asset: str, max_size: int = 1 << 31) -> tuple[bytes, _Source] | None:
        """A file next to (or inside the same package as) this layer: (bytes, its own source)."""
        if "[" in asset and asset.endswith("]"):  # package[inner path]
            outer, inner = asset[:-1].split("[", 1)
            got = self.read(outer)
            if got is None or detect(got[0]) != "usdz":
                return None
            pkg = _Package(got[0])
            return _Source(None, pkg, "").read(inner)
        if self.package is not None:
            name = self._inner(asset)
            try:
                info = self.package.getinfo(name)
            except KeyError:
                return None
            if info.file_size > max_size:
                return None
            return self.package.read(name), _Source(None, self.package, str(PurePosixPath(name).parent))
        path = self._disk(asset)
        if path is None:
            return None
        try:
            if not path.is_file() or path.stat().st_size > max_size:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        if detect(data) == "usdz":
            pkg = _Package(data)
            root = pkg.root()
            return pkg.read(root), _Source(None, pkg, str(PurePosixPath(root).parent))
        return data, _Source(path.parent, root=self.root)

    def image(self, asset: str) -> bytes | None:
        """Texture bytes, if the file is really an image (see scene.read_image_file)."""
        if not asset:
            return None
        if self.package is not None or ("[" in asset and asset.endswith("]")):
            got = self.read(asset, 1 << 28)
            if got is None or not looks_like_image(got[0][:64], asset):
                return None
            return got[0]
        path = self._disk(asset)
        data = read_image_file(path) if path is not None else None
        if data is None and self.base_dir is not None:  # pointing at the artist's drive: try next to us
            name = asset.replace("\\", "/").rsplit("/", 1)[-1]
            for folder in ("", "textures/", "Textures/"):
                path = linked_path(folder + name, self.base_dir, self.root)
                data = read_image_file(path) if path is not None else None
                if data is not None:
                    break
        return data


class _Package:
    """A .usdz: a zip whose files are stored as they are (USDZ allows no compression), each read
    out once, and no more than 8 times the package's size (plus 64 MB) read out in all."""

    def __init__(self, data: bytes):
        try:
            self.zip = zipfile.ZipFile(io.BytesIO(data))
            infos = self.zip.infolist()
        except (zipfile.BadZipFile, *DAMAGED) as e:
            raise MeshportError(f"This USDZ package is damaged ({e})") from None
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED:
                raise MeshportError("This USDZ package has compressed files in it, which USDZ doesn't allow")
            if info.file_size != info.compress_size or info.header_offset + info.file_size > len(data):
                raise MeshportError("This USDZ package is damaged (a file in it has the wrong size)")
        self.files: dict[str, bytes] = {}
        self.budget = Budget(8 * len(data) + (64 << 20), "This USDZ package reads out too much data; it may be "
                                                          "damaged or made to fill the memory")

    def getinfo(self, name: str) -> zipfile.ZipInfo:
        return self.zip.getinfo(name)

    def read(self, name: str) -> bytes:
        if name not in self.files:
            self.budget.spend(self.zip.getinfo(name).file_size)
            try:
                self.files[name] = self.zip.read(name)
            except (zipfile.BadZipFile, *DAMAGED) as e:
                raise MeshportError(f"This USDZ package is damaged ({e})") from None
        return self.files[name]

    def root(self) -> str:
        for info in self.zip.infolist():  # the first layer in the package is its root
            if info.filename.lower().rsplit(".", 1)[-1] in ("usd", "usda", "usdc"):
                return info.filename
        raise MeshportError("This USDZ package has no USD layer in it")


# ============================================================================ composition

def time_codes_per_second(layer: Layer) -> float:
    """A layer's timeCodesPerSecond (or framesPerSecond, as older layers have it; USD's default is 24)."""
    for key in ("timeCodesPerSecond", "framesPerSecond"):
        v = layer.get("/", key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v)
    return 24.0


def _offset(pair) -> tuple[float, float]:
    try:
        o, sc = (float(v) for v in pair)
    except (TypeError, ValueError):
        return 0.0, 1.0
    return (o, sc) if np.isfinite(o) and np.isfinite(sc) and sc != 0 else (0.0, 1.0)


def _compose(outer: tuple[float, float], inner: tuple[float, float]) -> tuple[float, float]:
    """Time maps t -> offset + scale * t, inner applied first."""
    return outer[0] + outer[1] * inner[0], outer[1] * inner[1]


def _inverse(m: tuple[float, float]) -> tuple[float, float]:
    return -m[0] / m[1], 1.0 / m[1]


class _Site:
    """One opinion about a prim: a spec at `path` in `layer`, plus how paths written there map
    onto the stage (a reference to /Asset placed at /World/Chair maps /Asset -> /World/Chair),
    and how its times do: (offset, scale), a time t there being offset + scale * t on the stage."""
    __slots__ = ("dst_root", "layer", "path", "source", "src_root", "stack", "time")

    def __init__(self, layer, source, stack, path, src_root="/", dst_root="/", time=(0.0, 1.0)):
        self.layer, self.source, self.stack, self.path = layer, source, stack, path
        self.src_root, self.dst_root, self.time = src_root, dst_root, time

    def spec(self, path: str | None = None) -> Spec | None:
        return self.layer.specs.get(self.path if path is None else path)

    def field(self, name: str, default=None):
        spec = self.spec()
        return default if spec is None else spec.fields.get(name, default)

    def map(self, path: str) -> str:
        """A path written in this site's layer, as a stage path."""
        path = strip_variants(path)
        root = self.src_root
        if root == "/" or path == root or path.startswith((root + "/", root + ".")):
            return self.dst_root + path[len(root):] if root != "/" else path
        return path


class _Stage:
    def __init__(self, data: bytes, base_dir: Path | None):
        self.layers: dict[str, tuple[Layer, _Source]] = {}
        self.stacks: dict[str, list[tuple[Layer, _Source]]] = {}
        self.stack_times: dict[int, list[tuple[float, float]]] = {}  # per stack entry, onto the stack's root
        kind = detect(data)
        self.site_budget = Budget(MAX_SITES, f"This USD file composes too many opinions (over {MAX_SITES:,}); "
                                             "it may be damaged or made to stall the reader")
        if kind == "usdz":
            pkg = _Package(data)
            name = pkg.root()
            source = _Source(None, pkg, str(PurePosixPath(name).parent))
            root_data = pkg.read(name)
        else:
            source, root_data = _Source(base_dir), data
        root = _parse_layer(root_data)
        self.root_stack = self._stack(root, source, "<root>")
        self.root = root
        self._sites: dict[str, list[_Site]] = {}
        self._busy: set[str] = set()  # prims being composed: an inherit loop stops here
        self._cuts = 0  # how often a loop was stopped
        self._depth = 0

    # -- layer stacks

    def _stack(self, layer: Layer, source: _Source, key: str, seen: frozenset = frozenset()) -> list:
        """The layer and its sublayers, strongest first."""
        if key in self.stacks:
            return self.stacks[key]
        stack, times = [(layer, source)], [(0.0, 1.0)]
        in_stack = {id(layer)}  # a layer listed again (or reached again through another) is used once
        offsets = layer.get("/", "subLayerOffsets") or []
        for i, sub in enumerate(layer.get("/", "subLayers", []) or []):
            got = self._load(source, sub)
            if got is None:
                continue
            sub_layer, sub_source, sub_key = got
            if sub_key in seen or sub_key == key:
                continue
            sub_stack = self._stack(sub_layer, sub_source, sub_key, seen | {key})
            o, sc = _offset(offsets[i] if i < len(offsets) else None)
            to_here = (o, sc * time_codes_per_second(layer) / time_codes_per_second(sub_layer))
            for entry, t in zip(sub_stack, self.stack_times[id(sub_stack)]):
                if id(entry[0]) not in in_stack:
                    in_stack.add(id(entry[0]))
                    stack.append(entry)
                    times.append(_compose(to_here, t))
        self.stacks[key] = stack
        self.stack_times[id(stack)] = times
        return stack

    def _load(self, source: _Source, asset: str):
        key = source.key(asset)
        if key not in self.layers:
            if len(self.layers) >= MAX_LAYERS:
                return None
            got = source.read(asset)
            if got is None:
                self.layers[key] = None
            else:
                try:
                    self.layers[key] = (_parse_layer(got[0]), got[1])
                except MeshportError:
                    self.layers[key] = None
        if self.layers[key] is None:
            return None
        layer, src = self.layers[key]
        return layer, src, key

    # -- prims

    def sites(self, path: str) -> list[_Site]:
        if path in self._sites:
            return self._sites[path]
        if path == "/":
            out = [_Site(layer, src, self.root_stack, "/", time=t)
                   for (layer, src), t in zip(self.root_stack, self.stack_times[id(self.root_stack)])]
        else:
            name = element_name(path)
            direct = []
            for s in self.sites(parent_path(path)):
                child = child_path(s.path, name)
                if child in s.layer.specs:
                    direct.append(_Site(s.layer, s.source, s.stack, child, s.src_root, s.dst_root, s.time))
            specials: list[_Site] = []
            cuts = self._cuts
            self._busy.add(path)
            try:
                out = self._expand(direct, path, 0, None, specials, _visits(direct)) + specials
            finally:
                self._busy.discard(path)
            if self._cuts != cuts:  # an inherit loop was cut short on the way: right only for this
                return out          # caller (USD drops just the arc that closes the loop)
        self._sites[path] = out
        return out

    def _expand(self, sites: list[_Site], stage_path: str, depth: int,
                stronger: dict[str, str] | None = None, specials: list[_Site] | None = None,
                chain: frozenset = frozenset()) -> list[_Site]:
        """USD's strength order: local opinions, then inherited classes', the selected
        variants', references' and payloads'. Specialized classes are weakest of all, so they
        collect in specials, which the caller puts last. A variant selection made by a stronger
        opinion (the stage choosing "big" for a referenced asset whose own default is "small")
        wins over the weaker ones. chain holds the (layer, path) opinions on the way here: an arc
        back to one of them is a cycle, and is dropped, as USD drops it."""
        if depth > MAX_ARC_DEPTH or not sites:
            return sites
        self.site_budget.spend(len(sites))
        specials = [] if specials is None else specials
        out = list(sites)
        for s in sites:
            out += self._classes(s, "inheritPaths", stage_path, depth, stronger, specials, chain)
        selections: dict[str, str] = dict(stronger or {})
        set_names: list[str] = []
        for s in sites:
            for k, v in (s.field("variantSelection") or {}).items():
                selections.setdefault(k, v)
            op = s.field("variantSetNames")
            for n in (op.items() if isinstance(op, ListOp) else (op or [])):
                if n not in set_names:
                    set_names.append(n)
        for name in set_names:
            choice = selections.get(name)
            if not choice:
                continue
            chosen = []
            for s in sites:
                vpath = f"{s.path}{{{name}={choice}}}"
                if vpath in s.layer.specs:
                    chosen.append(_Site(s.layer, s.source, s.stack, vpath, s.src_root, s.dst_root, s.time))
            chosen = [c for c in chosen if (id(c.layer), c.path) not in chain]
            out += self._expand(chosen, stage_path, depth + 1, selections, specials, chain | _visits(chosen))
        for s in sites:
            arcs = []
            for key in ("references", "payload"):
                v = s.field(key)
                if isinstance(v, ListOp):
                    arcs += v.items()
                elif v is not None and hasattr(v, "asset"):
                    arcs.append(v)
            for arc in arcs:
                found = [t for t in self._arc_sites(s, arc, stage_path) if (id(t.layer), t.path) not in chain]
                out += self._expand(found, stage_path, depth + 1, selections, specials, chain | _visits(found))
        for s in sites:
            specials += self._classes(s, "specializes", stage_path, depth, selections, specials, chain)
        return out

    def _classes(self, site: _Site, key: str, stage_path: str, depth: int, selections, specials,
                 chain: frozenset = frozenset()) -> list[_Site]:
        """The opinions of the classes a site inherits or specializes, re-rooted onto this prim.
        A class path is looked up on the stage (where the root layer stack can add to a class
        a referenced asset uses) and, when the reference doesn't bring the class along, in the
        referenced layer stack itself."""
        op = site.field(key)
        out: list[_Site] = []
        for item in (op.items() if isinstance(op, ListOp) else []):
            path = strip_variants(str(item))
            mapped = site.map(path)
            if mapped in self._busy or mapped == stage_path:
                self._cuts += 1
                continue
            for t in self.sites(mapped):
                out.append(_Site(t.layer, t.source, t.stack, t.path, t.path, stage_path, t.time))
            if mapped == path and site.stack is not self.root_stack:
                base = _compose(site.time, _inverse(self._entry_time(site)))  # site.stack's root onto the stage
                own = [_Site(layer, src, site.stack, path, path, stage_path, _compose(base, t))
                       for (layer, src), t in zip(site.stack, self.stack_times[id(site.stack)])
                       if path in layer.specs and (id(layer), path) not in chain]
                out += self._expand(own, stage_path, depth + 1, selections, specials, chain | _visits(own))
        return out

    def _arc_sites(self, site: _Site, arc, stage_path: str) -> list[_Site]:
        timing = (getattr(arc, "offset", 0.0), getattr(arc, "scale", 1.0))
        if arc.asset:
            got = self._load(site.source, arc.asset)
            if got is None:
                return []
            layer, source, key = got
            stack = self._stack(layer, source, key)
            ratio = time_codes_per_second(site.layer) / time_codes_per_second(layer)
            base = _compose(site.time, (timing[0], timing[1] * ratio))
        else:  # within the same layer stack
            stack = site.stack
            base = _compose(_compose(site.time, _inverse(self._entry_time(site))), timing)
        target = arc.prim
        if not target:
            target = next((f"/{p}" for layer, _ in stack if (p := layer.get("/", "defaultPrim"))), None)
            if target is None:  # no default prim: the first root prim
                kids = stack[0][0].get("/", "primChildren") or []
                if not kids:
                    return []
                target = f"/{kids[0]}"
        out = []
        for (layer, source), t in zip(stack, self.stack_times[id(stack)]):
            if target in layer.specs:
                out.append(_Site(layer, source, stack, target, target, stage_path, _compose(base, t)))
        return out

    def _entry_time(self, site: _Site) -> tuple[float, float]:
        """Where the site's layer sits in its own layer stack's time."""
        for (layer, _), t in zip(site.stack, self.stack_times.get(id(site.stack), [])):
            if layer is site.layer:
                return t
        return (0.0, 1.0)

    def children(self, path: str) -> list[str]:
        names: dict[str, None] = {}  # in order, each once
        for s in self.sites(path):
            names.update(dict.fromkeys(s.field("primChildren") or []))
        return list(names)

    def prim_field(self, path: str, name: str, default=None):
        for s in self.sites(path):
            v = s.field(name)
            if v is not None:
                return v
        return default

    def attribute(self, path: str, name: str, with_site: bool = False):
        """The strongest value of an attribute (its first time sample if it's animated)."""
        for s in self.sites(path):
            spec = s.spec(property_path(s.path, name))
            if spec is None:
                continue
            f = spec.fields
            if "default" in f:
                return (f["default"], s) if with_site else f["default"]
            samples = f.get("timeSamples")
            if samples:
                v = samples[min(samples)]
                return (v, s) if with_site else v
        return (None, None) if with_site else None

    def samples(self, path: str, name: str) -> dict | None:
        """The strongest opinion's time samples, on the stage's time line ({time code: value}), or
        None when that opinion is a single value (or there's none). Within one spec, samples win
        over a default, as in USD."""
        for s in self.sites(path):
            spec = s.spec(property_path(s.path, name))
            if spec is None:
                continue
            got = spec.fields.get("timeSamples")
            if isinstance(got, dict) and got:
                o, sc = s.time
                return {o + sc * float(t): v for t, v in got.items() if v is not None} or None
            if "default" in spec.fields:
                return None
        return None

    def attribute_meta(self, path: str, name: str, key: str, default=None):
        for s in self.sites(path):
            spec = s.spec(property_path(s.path, name))
            if spec is not None and key in spec.fields:
                return spec.fields[key]
        return default

    def properties(self, path: str) -> list[str]:
        names: dict[str, None] = {}
        for s in self.sites(path):
            names.update(dict.fromkeys(s.field("properties") or []))
        return list(names)

    def targets(self, path: str, name: str, field: str = "targetPaths") -> list[str]:
        for s in self.sites(path):
            spec = s.spec(property_path(s.path, name))
            if spec is not None and isinstance(spec.fields.get(field), ListOp):
                items = spec.fields[field].items()
                if items or spec.fields[field].explicit:
                    return [s.map(t) for t in items]
        return []

    def connection_source(self, path: str, name: str):
        """For an attribute connected to another: (prim path, attribute name) it reads from."""
        got = self.targets(path, name, "connectionPaths")
        if not got:
            return None
        target = got[0]
        if "." not in target.rsplit("/", 1)[-1]:
            return None
        prim, _, prop = target.rpartition(".")
        return prim, prop

    def source_of(self, path: str, name: str) -> _Source | None:
        for s in self.sites(path):
            spec = s.spec(property_path(s.path, name))
            if spec is not None and ("default" in spec.fields or "timeSamples" in spec.fields):
                return s.source
        return None

    def defined(self, path: str) -> bool:
        """Some opinion defines it ("def"), and it isn't an abstract class."""
        specifiers = [s.field("specifier") for s in self.sites(path)]
        return "def" in specifiers and specifiers[0] != "class"


def _visits(sites: list[_Site]) -> frozenset:
    return frozenset((id(s.layer), s.path) for s in sites)


# ============================================================================ stage -> scene

def read(data: bytes, base_dir: str | os.PathLike | None = None) -> Scene:
    try:
        return _Builder(_Stage(data, Path(base_dir) if base_dir else None)).scene()
    except MeshportError:
        raise
    except (*DAMAGED, struct.error, zipfile.BadZipFile, zlib.error) as e:
        raise MeshportError(f"This USD file is damaged ({type(e).__name__}: {e})") from None


class _Builder:
    def __init__(self, stage: _Stage):
        self.stage = stage
        self.materials: dict[str, Material] = {}
        self.images: dict[tuple, Image | None] = {}
        self.uv_names: dict[str, str] = {}  # material path -> the UV set its textures read
        self.animated: list[tuple[Node, str, np.ndarray | None]] = []  # (node, prim, parent world if reset)
        self.point_caches: list[tuple[Node, np.ndarray]] = []  # meshes with animated points, and their time codes
        # skeleton prim -> (joint tokens, joint nodes, bind transforms or None, its SkelAnimation or None)
        self.skeletons: dict[str, tuple[list[str], list[Node], np.ndarray | None, str | None]] = {}
        self.skinned: list[tuple[Node, str]] = []  # meshes that may be bound to a skeleton
        self.joint_channels: list[Channel] = []
        self.shape_knots: dict[str, list[tuple[str, np.ndarray]]] = {}  # mesh -> its blend shapes' in-betweens
        self.nodes = node_budget()

    def scene(self) -> Scene:
        st = self.stage
        up = str(st.root.get("/", "upAxis", "Y") or "Y").upper()
        meters = st.root.get("/", "metersPerUnit", 0.01)
        meters = float(meters) if isinstance(meters, (int, float)) and meters > 0 else 0.01
        convert = scaling((meters,) * 3)
        if up == "Z":
            convert = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]]) @ convert
        roots = []
        for name in st.children("/"):
            node = self._node(child_path("/", name), np.eye(4), 0)
            if node is not None:
                node.matrix = convert @ node.matrix
                roots.append(node)
        scene = Scene(roots)
        if not scene.meshes:
            raise MeshportError("No 3D geometry found in this file")
        in_scene = {id(n) for n, _ in scene.walk()}
        root_ids = {id(r) for r in roots}
        channels = []
        tcps = time_codes_per_second(st.root)
        for node, path, reset_parent in self.animated:
            if id(node) in in_scene:
                try:
                    channels += self._channels(node, path, reset_parent, convert if id(node) in root_ids else None, tcps)
                except DAMAGED:
                    continue  # damaged samples lose this prim's animation, not the file
        channels += [c for c in self.joint_channels if id(c.node) in in_scene]
        for node, codes in self.point_caches:  # sample k is target k - 1 at full weight, linear between
            if id(node) in in_scene:
                weights = np.vstack([np.zeros(len(codes) - 1), np.eye(len(codes) - 1)])
                channels.append(Channel(node, WEIGHTS, codes / tcps, weights, "LINEAR"))
        for node, path in self.skinned:
            if id(node) not in in_scene:
                continue
            try:
                node.skin, weights = self._skin(node, path, tcps)
            except (*DAMAGED, MeshportError):
                node.skin, weights = None, None  # a damaged binding leaves the mesh where its node puts it
            if weights is not None:
                channels.append(weights)
        if channels:
            scene.animations = [Animation(str(st.root.get("/", "defaultPrim") or "") or "animation", channels)]
        return scene

    def _node(self, path: str, parent_world: np.ndarray, depth: int) -> Node | None:
        st = self.stage
        if depth > 512 or not st.sites(path) or not st.defined(path):
            return None
        type_name = st.prim_field(path, "typeName", "") or ""
        if type_name in _SKIP_TYPES or st.prim_field(path, "active") is False:
            return None
        if st.attribute(path, "visibility") == "invisible" or st.attribute(path, "purpose") in ("guide", "proxy"):
            return None
        local, reset = self._xform(path)
        if reset:  # ignores the parents' transforms
            local = np.linalg.inv(parent_world) @ local
        self.nodes.spend()
        node = Node(element_name(path), local)
        if self._animated_ops(path):
            self.animated.append((node, path, parent_world if reset else None))
        world = parent_world @ local
        if type_name in _GPRIMS:
            got = self._mesh(path, type_name)
            if got is not None:
                node.mesh, node.materials = got
                if type_name == "Mesh":
                    self.skinned.append((node, path))
                    self._point_samples(node, path)
        if type_name == "Skeleton":
            try:
                node.children = self._joints(path)
            except DAMAGED:
                node.children = []
            return node if node.children else None
        if type_name == "PointInstancer":  # its children are the prototypes: drawn only as instances
            node.children = self._instances(path, depth)
            return node if node.children else None
        for name in st.children(path):
            child = self._node(child_path(path, name), world, depth + 1)
            if child is not None:
                node.children.append(child)
        if node.mesh is None and not node.children:
            return None
        return node

    def _point_samples(self, node: Node, path: str) -> None:
        """A mesh whose points have time samples (and whose faces don't): the first sample as its
        points, each later one a blend shape that a weights channel plays at its time."""
        from .alembic import CACHE_BUDGET
        st, mesh = self.stage, node.mesh
        samples = st.samples(path, "points")
        if not samples or len(samples) < 2 or mesh.targets or st.samples(path, "faceVertexCounts") \
                or st.samples(path, "faceVertexIndices"):
            return
        times = sorted(samples)
        count = len(mesh.positions)
        if len(times) * count * 3 + len(times) * (len(times) - 1) > CACHE_BUDGET:
            return
        try:
            frames = [np.asarray(samples[t], np.float64).reshape(-1, 3) for t in times]
        except (TypeError, ValueError):
            return
        if any(len(f) != count for f in frames) or not all(np.isfinite(f).all() for f in frames):
            return
        mesh.positions = frames[0]
        mesh.targets = [Morph(f"sample{k}", f - frames[0]) for k, f in enumerate(frames[1:], 1)]
        mesh.target_weights = np.zeros(len(mesh.targets))
        self.point_caches.append((node, np.array(times, np.float64)))

    def _instances(self, path: str, depth: int) -> list[Node]:
        """A PointInstancer's instances: one node each, placed by the instancer's positions,
        orientations and scales, holding its prototype (whose own transform comes along, as in
        USD). Hidden (invisibleIds) and deactivated (inactiveIds) instances are left out. All
        instances of a prototype share its meshes."""
        st = self.stage
        protos = st.targets(path, "prototypes")
        index = st.attribute(path, "protoIndices")
        points = st.attribute(path, "positions")
        if not protos or index is None or points is None:
            return []
        index = np.asarray(index, np.int64).ravel()
        points = np.asarray(points, np.float64).reshape(-1, 3)
        orient = st.attribute(path, "orientationsf")
        orient = st.attribute(path, "orientations") if orient is None else orient
        orient = None if orient is None else np.asarray(orient, np.float64).reshape(-1, 4)
        scales = st.attribute(path, "scales")
        scales = None if scales is None else np.asarray(scales, np.float64).reshape(-1, 3)
        ids = st.attribute(path, "ids")
        ids = np.arange(len(index)) if ids is None or len(ids) != len(index) else np.asarray(ids, np.int64)
        off = {int(i) for i in np.asarray(st.attribute(path, "invisibleIds") or [], np.int64).ravel()}
        inactive = st.prim_field(path, "inactiveIds")
        off |= {int(i) for i in (inactive.items() if isinstance(inactive, ListOp) else inactive or [])}
        built: dict[int, Node | None] = {}
        out = []
        for i, (k, key) in enumerate(zip(index, ids)):
            if int(key) in off or not 0 <= k < len(protos) or i >= len(points):
                continue
            if k not in built:
                built[k] = self._node(protos[k], np.eye(4), depth + 1)
            proto = built[k]
            if proto is None:
                continue
            m = translation(points[i])
            if orient is not None and i < len(orient):
                m = m @ _op_matrix("orient", orient[i])
            if scales is not None and i < len(scales):
                m = m @ scaling(scales[i])
            self.nodes.spend()
            out.append(Node(f"{proto.name}_{i}", m, children=[_shared_copy(proto, self.nodes)]))
        return out

    # -- transforms

    def _xform(self, path: str) -> tuple[np.ndarray, bool]:
        st = self.stage
        order = st.attribute(path, "xformOpOrder") or []
        m = np.eye(4)
        reset = False
        for op in order:
            op = str(op)
            if op == "!resetXformStack!":
                reset, m = True, np.eye(4)
                continue
            invert = op.startswith("!invert!")
            name = op[8:] if invert else op
            v = st.attribute(path, name)
            if v is None:
                continue
            kind = name.split(":")[1] if ":" in name else name
            mat = _op_matrix(kind, v)
            if mat is None:
                continue
            m = m @ (np.linalg.inv(mat) if invert else mat)
        return m, reset

    def _animated_ops(self, path: str) -> dict:
        """{op name: its time samples} for the prim's xform ops that have more than one."""
        st = self.stage
        out = {}
        for op in st.attribute(path, "xformOpOrder") or []:
            name = str(op).removeprefix("!invert!")
            if name != "!resetXformStack!" and name not in out:
                got = st.samples(path, name)
                if got is not None and len(got) > 1:
                    out[name] = got
        return out

    def _xform_at(self, path: str, times: np.ndarray) -> np.ndarray:
        """(N, 4, 4) local matrices at stage time codes: each op's samples interpolated as USD
        does (held outside them, linear between, quaternions slerped)."""
        st = self.stage
        m = np.tile(np.eye(4), (len(times), 1, 1))
        for op in st.attribute(path, "xformOpOrder") or []:
            op = str(op)
            if op == "!resetXformStack!":
                m = np.tile(np.eye(4), (len(times), 1, 1))
                continue
            invert = op.startswith("!invert!")
            name = op[8:] if invert else op
            kind = name.split(":")[1] if ":" in name else name
            got = st.samples(path, name)
            if got is None:
                v = st.attribute(path, name)
                mats = None if v is None else _op_matrix(kind, v)
                mats = None if mats is None else np.broadcast_to(mats, m.shape)
            else:
                values = _interpolate(kind, got, times)
                mats = None if values is None else [_op_matrix(kind, v) for v in values]
                mats = None if mats is None or any(x is None for x in mats) else np.array(mats)
            if mats is not None:
                m = m @ (np.linalg.inv(mats) if invert else mats)
        return m

    def _channels(self, node: Node, path: str, reset_parent, convert, tcps: float) -> list[Channel]:
        """The prim's animated transform as translation, rotation and scale channels: posed at
        every sample time, and at every time code between unless only translations move (whose
        blend is exact anyway)."""
        ops = self._animated_ops(path)
        keys = np.unique(np.concatenate([np.array(sorted(v), np.float64) for v in ops.values()]))
        keys = keys[np.isfinite(keys)]
        if len(keys) < 2:
            return []
        times = keys
        moving = {n.split(":")[1] if ":" in n else n for n in ops}
        grid = frame_grid(keys[0], keys[-1]) if not moving <= {"translate"} else None
        if grid is not None:  # (too long a span: at the samples only)
            times = np.unique(np.concatenate([keys, grid]))
        m = self._xform_at(path, times)
        if reset_parent is not None:
            m = np.linalg.inv(reset_parent) @ m
        if convert is not None:
            m = convert @ m
        out = []
        for name, vals in zip(("translation", "rotation", "scale"), decompose_many(m)):
            if not np.allclose(vals, vals[0], atol=1e-9):
                out.append(Channel(node, name, times / tcps, vals, "LINEAR"))
        return out

    # -- skeletons (UsdSkel)

    def _joints(self, path: str) -> list[Node]:
        """A Skeleton's joints as nodes under it, nested as their paths are ("Hip/Knee" under "Hip"),
        each at its rest transform (restTransforms, or worked out from bindTransforms), or where
        the skeleton's animation first puts it; the animation's samples become channels."""
        st = self.stage
        tokens = [str(t) for t in st.attribute(path, "joints") or []]
        rest, binds = _matrices(st.attribute(path, "restTransforms"), len(tokens)), \
            _matrices(st.attribute(path, "bindTransforms"), len(tokens))
        nodes: list[Node] = []
        top: list[Node] = []
        index = {t: k for k, t in enumerate(tokens)}
        for k, token in enumerate(tokens):
            parent, up = None, token
            while "/" in up and parent is None:  # the nearest joint above it
                up = up.rpartition("/")[0]
                parent = index.get(up) if index.get(up, k) < k else None
            if rest is not None:
                local = rest[k]
            elif binds is not None:
                local = binds[k] if parent is None else np.linalg.inv(binds[parent]) @ binds[k]
            else:
                local = np.eye(4)
            node = Node(token.rsplit("/", 1)[-1], local)
            nodes.append(node)
            (top if parent is None else nodes[parent].children).append(node)
        anim = self._bound(path, "skel:animationSource")
        anim = anim if anim and st.prim_field(anim, "typeName") == "SkelAnimation" else None
        self.skeletons[path] = (tokens, nodes, binds, anim)
        if anim is not None:
            self._joint_animation(anim, tokens, nodes)
        return top

    def _joint_animation(self, anim: str, tokens: list[str], nodes: list[Node]) -> None:
        """A SkelAnimation's translations, rotations and scales (arrays over its own joint order):
        each joint stands where the first sample puts it, and moves with the samples (blended as
        USD blends them, linearly, rotations slerped)."""
        st = self.stage
        order = [str(t) for t in st.attribute(anim, "joints") or []]
        pairs = [(col, nodes[tokens.index(t)]) for col, t in enumerate(order) if t in tokens]
        tcps = time_codes_per_second(st.root)
        parts = {}
        for name, path, width in (("translations", "translation", 3), ("rotations", "rotation", 4), ("scales", "scale", 3)):
            first = st.attribute(anim, name)
            first = None if first is None else np.asarray(first, np.float64).reshape(-1, width)
            samples = st.samples(anim, name)
            keys = sorted(samples) if samples else []
            try:
                values = np.array([np.asarray(samples[t], np.float64).reshape(-1, width) for t in keys]) if keys else None
            except ValueError:
                values = None  # samples of different sizes
            parts[path] = first
            if values is None or len(keys) < 2 or values.shape[1] < len(order) or not np.isfinite(values).all():
                continue
            if path == "rotation":
                values = np.roll(values, -1, axis=2)  # (w, x, y, z) -> (x, y, z, w)
            times = np.array(keys, np.float64) / tcps
            for col, node in pairs:
                v = values[:, col]
                if not np.allclose(v, v[0], atol=1e-9):
                    self.joint_channels.append(Channel(node, path, times, v.copy(), "LINEAR"))
        for col, node in pairs:
            t, q, s = decompose_trs(node.matrix)
            got = parts["translation"]
            t = got[col] if got is not None and col < len(got) else t
            got = parts["rotation"]
            q = np.roll(got[col], -1) if got is not None and col < len(got) else q
            got = parts["scale"]
            s = got[col] if got is not None and col < len(got) else s
            node.matrix = compose_trs(t, q / (np.linalg.norm(q) or 1.0), s)

    def _bound(self, path: str, name: str) -> str | None:
        """A skel binding relationship's target: on the prim, or inherited from the prims above it."""
        st = self.stage
        while path and path != "/":
            got = st.targets(path, name)
            if got:
                return strip_variants(got[0])
            path = parent_path(path)
        return None

    def _skin(self, node: Node, path: str, tcps: float) -> tuple[Skin | None, Channel | None]:
        """A mesh's skin from its bound skeleton (jointIndices and jointWeights, vertex or constant,
        elementSize per point, over the mesh's own skel:joints order or the skeleton's), and the
        skeleton animation's blend shape weights matched to its targets by name. Each joint's inverse
        bind matrix is inv(bindTransform) @ geomBindTransform."""
        st = self.stage
        skel = self._bound(path, "skel:skeleton")
        if skel not in self.skeletons:
            return None, None
        tokens, joints, binds, anim = self.skeletons[skel]
        mesh = node.mesh
        weights = self._shape_weights(node, path, anim, tcps) if mesh.targets and anim else None
        idx, w = st.attribute(path, "primvars:skel:jointIndices"), st.attribute(path, "primvars:skel:jointWeights")
        if idx is None or w is None or binds is None:
            return None, weights
        idx, w = np.asarray(idx, np.int64).ravel(), np.asarray(w, np.float64).ravel()
        size = st.attribute_meta(path, "primvars:skel:jointIndices", "elementSize", 1)
        size = int(size) if isinstance(size, (int, float)) and size >= 1 else 1
        count = len(mesh.positions)
        if st.attribute_meta(path, "primvars:skel:jointIndices", "interpolation", "vertex") == "constant":
            idx, w = np.tile(idx[:size], count), np.tile(w[:size], count)  # every point bound alike
        if len(idx) != count * size or len(w) != len(idx):
            return None, weights
        local = [str(t) for t in st.attribute(path, "skel:joints") or []] or tokens
        if any(t not in tokens for t in local):
            return None, weights
        used = [tokens.index(t) for t in local]
        geom = st.attribute(path, "primvars:skel:geomBindTransform")
        geom = _op_matrix("transform", geom) if geom is not None else None
        geom = np.eye(4) if geom is None else geom
        idx, w = idx.reshape(count, size), w.reshape(count, size)
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
        idx = np.where(w > 0, idx, 0)
        total = w.sum(1, keepdims=True)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        w[total[:, 0] <= 0, 0] = 1.0
        skin = Skin([joints[j] for j in used], np.array([np.linalg.inv(binds[j]) @ geom for j in used]), idx, w,
                    element_name(skel))
        skin.validate(mesh)
        return skin, weights

    def _shape_weights(self, node: Node, path: str, anim: str, tcps: float) -> Channel | None:
        """The mesh's targets' weights from the skeleton's animation (its blendShapes name the
        mesh's blend shapes; one it doesn't name stays at 0), each blend shape's weight spread over
        its targets as inbetween_weights() blends in-between shapes. The first sample is the mesh's
        resting weight."""
        st = self.stage
        names = [str(t) for t in st.attribute(anim, "blendShapes") or []]
        shapes = self.shape_knots.get(path, [])
        cols = [names.index(n) if n in names else -1 for n, _ in shapes]
        if sum(len(k) for _, k in shapes) != len(node.mesh.targets) or all(c < 0 for c in cols):
            return None

        def levels(v) -> np.ndarray:  # each blend shape's weight
            v = np.asarray(v, np.float64).ravel()
            return np.array([v[c] if 0 <= c < len(v) else 0.0 for c in cols])

        def spread(values: np.ndarray) -> np.ndarray:  # (T, blend shapes) -> (T, targets)
            return np.hstack([inbetween_weights(values[:, k], knots) for k, (_, knots) in enumerate(shapes)])
        first = st.attribute(anim, "blendShapeWeights")
        if first is not None:
            node.mesh.target_weights = spread(levels(first)[None])[0]
        samples = st.samples(anim, "blendShapeWeights")
        if not samples or len(samples) < 2:
            return None
        keys = np.array(sorted(samples), np.float64)
        values = np.array([levels(samples[t]) for t in sorted(samples)])
        if not np.isfinite(values).all():
            return None
        times = keys
        for k, (_, knots) in enumerate(shapes):
            if len(knots) > 1:  # in-between shapes: a sample wherever the weight passes one
                times = with_crossings(times, np.interp(times, keys, values[:, k]), knots)
        values = np.stack([np.interp(times, keys, values[:, k]) for k in range(len(shapes))], 1)  # USD blends linearly
        return Channel(node, WEIGHTS, times / tcps, spread(values))

    def _blend_shapes(self, path: str, count: int, corners: np.ndarray, normals: bool) -> list[Morph]:
        """The mesh's skel:blendShapeTargets, named by skel:blendShapes: each BlendShape's offsets for
        the points it lists (pointIndices; all of them without), and its normalOffsets per corner
        when the mesh has normals. In-between shapes (inbetweens:NAME, at their weight) become
        targets of their own, before the full one; shape_knots[path] keeps where each stands."""
        st = self.stage
        names = [str(t) for t in st.attribute(path, "skel:blendShapes") or []]
        targets = st.targets(path, "skel:blendShapeTargets")
        out, knots = [], []

        def morph(name: str, target: str, prop: str, index: np.ndarray) -> Morph | None:
            offsets = st.attribute(target, prop)
            if offsets is None:
                return None
            offsets = np.asarray(offsets, np.float64).reshape(-1, 3)
            if len(index) != len(offsets):
                return None
            moved = np.zeros((count, 3))
            moved[index] = offsets
            shift = st.attribute(target, prop.replace("offsets", "normalOffsets") if prop == "offsets"
                                 else prop + ":normalOffsets") if normals else None
            turned = None
            if shift is not None and np.asarray(shift).size == offsets.size:
                turned = np.zeros((count, 3))
                turned[index] = np.asarray(shift, np.float64).reshape(-1, 3)
                turned = turned[corners]
            return Morph(name, moved, turned)
        for name, target in zip(names, targets):
            index = st.attribute(target, "pointIndices")
            index = np.arange(count) if index is None else np.asarray(index, np.int64).ravel()
            if len(index) and (index.min() < 0 or index.max() >= count):
                return []  # damaged: the mesh loads without blend shapes
            between = []
            for prop in st.properties(target):
                if prop.startswith("inbetweens:") and prop.count(":") == 1:
                    weight = st.attribute_meta(target, prop, "weight")
                    if isinstance(weight, (int, float)) and 0 < weight < 1:
                        between.append((float(weight), prop))
            between.sort()
            shapes = [morph(f"{name} {w * 100:g}%", target, prop, index) for w, prop in between]
            shapes.append(morph(name, target, "offsets", index))
            if any(m is None for m in shapes):
                return []
            out += shapes
            knots.append((name, np.array([w for w, _ in between] + [1.0])))
        self.shape_knots[path] = knots
        return out if len(knots) == len(names) else []

    # -- meshes

    def _mesh(self, path: str, type_name: str):
        st = self.stage
        if type_name != "Mesh":
            mesh = _implicit(type_name, st, path)
            if mesh is None:
                return None
            mat = self._bound_material(path) or self._display_material(path)
            return mesh, [mat] if mat else []
        points = st.attribute(path, "points")
        counts = st.attribute(path, "faceVertexCounts")
        indices = st.attribute(path, "faceVertexIndices")
        if points is None or counts is None or indices is None:
            return None
        points = np.asarray(points, np.float64).reshape(-1, 3)
        counts = np.asarray(counts, np.int64).ravel()
        indices = np.asarray(indices, np.int64).ravel()
        if not len(points) or not len(counts):
            return None
        if counts.min(initial=0) < 0 or int(counts.sum()) != len(indices):
            raise MeshportError(f"Mesh {path}: its face sizes don't add up to its index list")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
            raise MeshportError(f"Mesh {path}: a face points past its vertex list")
        n_faces = len(counts)
        face_of_corner = np.repeat(np.arange(n_faces), counts)

        slots, materials = self._face_materials(path, n_faces)
        normals = None
        for name in ("primvars:normals", "normals"):
            got = self._primvar(path, name, counts, indices, "vertex")
            if got is not None:
                normals = got
                break
        uv_names = self._uv_names(path)
        uvs = [u for u in (self._primvar(path, n, counts, indices, "vertex") for n in uv_names) if u is not None]
        uvs = [u[:, :2].astype(np.float64) for u in uvs if u.ndim == 2 and u.shape[1] >= 2]
        colors = self._colors(path, counts, indices)

        # faces kept: real polygons that aren't holes
        keep = counts >= 3
        holes = st.attribute(path, "holeIndices")
        if holes is not None and len(np.asarray(holes)):
            h = np.asarray(holes, np.int64)
            keep[h[(h >= 0) & (h < n_faces)]] = False
        corner_keep = keep[face_of_corner]
        sizes = counts[keep]
        corners = indices[corner_keep]
        normals = normals[corner_keep] if normals is not None else None
        uvs = [u[corner_keep] for u in uvs]
        colors = colors[corner_keep] if colors is not None else None
        slots = slots[keep] if slots is not None else None
        if not len(sizes):
            return None
        if st.attribute(path, "orientation") == "leftHanded":  # reverse each face's corners
            s0 = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            local = np.arange(len(corners)) - np.repeat(s0, sizes)
            flip = np.repeat(s0 + sizes - 1, sizes) - local
            corners = corners[flip]
            normals = normals[flip] if normals is not None else None
            uvs = [u[flip] for u in uvs]
            colors = colors[flip] if colors is not None else None
        mesh = Mesh(element_name(path), points, sizes, corners,
                    normals.astype(np.float64) if normals is not None else None, uvs,
                    colors, slots)
        try:
            mesh.targets = self._blend_shapes(path, len(points), corners, normals is not None)
        except DAMAGED:
            mesh.targets = []
        return mesh, materials

    def _primvar(self, path: str, name: str, counts, indices, default_interp: str):
        """A primvar expanded to one value per corner, or None."""
        st = self.stage
        values = st.attribute(path, name)
        if values is None:
            return None
        values = np.asarray(values)
        if values.dtype.kind not in "fiu" or not values.size:
            return None
        if values.ndim == 1:
            values = values[:, None]
        idx = st.attribute(path, name + ":indices")
        if idx is not None and len(np.asarray(idx)):
            idx = np.asarray(idx, np.int64)
            if idx.min() < 0 or idx.max() >= len(values):
                return None
            values = values[idx]
        interp = st.attribute_meta(path, name, "interpolation", default_interp)
        n_corners = len(indices)
        if interp == "constant":
            return np.repeat(values[:1], n_corners, 0)
        if interp == "uniform":
            return np.repeat(values, counts, 0) if len(values) == len(counts) else None
        if interp in ("vertex", "varying"):
            return values[indices] if len(values) > indices.max(initial=-1) else None
        if interp == "faceVarying":
            return values if len(values) == n_corners else None
        return None

    def _uv_names(self, path: str) -> list[str]:
        st = self.stage
        names = []
        for prop in st.properties(path):
            if not prop.startswith("primvars:") or prop.endswith(":indices") or prop.count(":") > 1:
                continue
            spec_type = st.attribute_meta(path, prop, "typeName", "")
            if spec_type.startswith("texCoord2") or (spec_type in ("float2[]", "double2[]", "half2[]")):
                names.append(prop)
        preferred = self._material_uv_name(path)
        names.sort(key=lambda n: (n != f"primvars:{preferred}", n != "primvars:st", n))
        return names

    def _material_uv_name(self, path: str) -> str:
        mat_path = self._binding(path)
        return self.uv_names.get(mat_path, "st") if mat_path else "st"

    def _colors(self, path: str, counts, indices):
        rgb = self._primvar(path, "primvars:displayColor", counts, indices, "constant")
        if rgb is None or self.stage.attribute_meta(path, "primvars:displayColor", "interpolation",
                                                    "constant") == "constant":
            return None  # constant: becomes the material color instead
        alpha = self._primvar(path, "primvars:displayOpacity", counts, indices, "constant")
        out = np.ones((len(indices), 4))
        out[:, :3] = rgb[:, :3]
        if alpha is not None:
            out[:, 3] = alpha[:, 0]
        return out

    # -- materials

    def _binding(self, path: str) -> str | None:
        """The material bound to a prim, directly or through an ancestor."""
        st = self.stage
        p = path
        while p != "/":
            for rel in ("material:binding", "material:binding:preview", "material:binding:full"):
                t = st.targets(p, rel)
                if t:
                    return t[0]
            p = parent_path(p)
        return None

    def _bound_material(self, path: str) -> Material | None:
        target = self._binding(path)
        return self._material(target) if target else None

    def _display_material(self, path: str) -> Material | None:
        st = self.stage
        color = st.attribute(path, "primvars:displayColor")
        if color is None or not np.asarray(color).size:
            return None
        if st.attribute_meta(path, "primvars:displayColor", "interpolation", "constant") != "constant":
            return None  # colors that vary are vertex colors (see _colors), not a tint
        rgb = np.asarray(color, np.float64).reshape(-1, 3)[0]
        alpha = st.attribute(path, "primvars:displayOpacity")
        a = float(np.asarray(alpha, np.float64).ravel()[0]) if alpha is not None and np.asarray(alpha).size else 1.0
        key = ("display", *np.round(rgb, 6), round(a, 6))
        if key not in self.materials:
            self.materials[key] = Material("displayColor", (*map(float, rgb), a),
                                           alpha_mode="BLEND" if a < 1 else "OPAQUE")
        return self.materials[key]

    def _face_materials(self, path: str, n_faces: int):
        st = self.stage
        base = self._bound_material(path) or self._display_material(path)
        materials = [base] if base else []
        slots = None
        for name in st.children(path):
            sub = child_path(path, name)
            if st.prim_field(sub, "typeName") != "GeomSubset" or not st.defined(sub):
                continue
            if (st.attribute(sub, "elementType") or "face") != "face":
                continue
            target = st.targets(sub, "material:binding") or st.targets(sub, "material:binding:preview")
            if not target:
                continue
            idx = st.attribute(sub, "indices")
            if idx is None:
                continue
            idx = np.asarray(idx, np.int64).ravel()
            idx = idx[(idx >= 0) & (idx < n_faces)]
            mat = self._material(target[0])
            if slots is None:
                slots = np.zeros(n_faces, np.int64)
                if not materials:
                    materials = [Material("default")]
            materials.append(mat)
            slots[idx] = len(materials) - 1
        return slots, materials

    def _material(self, path: str) -> Material:
        if path in self.materials:
            return self.materials[path]
        mat = Material(element_name(path))
        self.materials[path] = mat
        st = self.stage
        surface = None
        for out in ["outputs:surface"] + [p for p in st.properties(path)
                                          if p.startswith("outputs:") and p.endswith(":surface")]:
            src = st.connection_source(path, out)
            if src is not None:
                surface = self._follow(*src)
                if surface is not None:
                    break
        if surface is None:
            return mat
        shader = surface[0]
        if (st.attribute(shader, "info:id") or "") != "UsdPreviewSurface":
            return mat
        color, tex = self._input(shader, "diffuseColor")
        if color is not None:
            c = np.asarray(color, np.float64).ravel()
            mat.base_color = (*map(float, c[:3]), 1.0)
        elif tex is not None:
            mat.base_color = (1.0, 1.0, 1.0, 1.0)
        if tex is not None:
            mat.base_color_texture = self._image(*tex)
        opacity, op_tex = self._input(shader, "opacity")
        threshold, _ = self._input(shader, "opacityThreshold")
        if opacity is not None:
            mat.base_color = (*mat.base_color[:3], float(np.asarray(opacity).ravel()[0]))
        if threshold is not None and float(np.asarray(threshold).ravel()[0]) > 0:
            mat.alpha_mode = "MASK"
        elif mat.base_color[3] < 1 or op_tex is not None:
            mat.alpha_mode = "BLEND"
        for key, attr in (("metallic", "metallic"), ("roughness", "roughness")):
            v, _ = self._input(shader, key)
            if v is not None:
                setattr(mat, attr, float(np.asarray(v).ravel()[0]))
        emissive, em_tex = self._input(shader, "emissiveColor")
        if emissive is not None:
            mat.emissive = tuple(float(x) for x in np.asarray(emissive, np.float64).ravel()[:3])
        if em_tex is not None:
            mat.emissive_texture = self._image(*em_tex)
            if not any(mat.emissive):
                mat.emissive = (1.0, 1.0, 1.0)
        _, normal_tex = self._input(shader, "normal")
        if normal_tex is not None:
            mat.normal_texture = self._image(*normal_tex)
        return mat

    def _follow(self, prim: str, prop: str, depth: int = 0):
        """Where a connection really leads: through material and node-graph inputs and outputs
        to a shader's output. Returns (shader prim, output name) or None."""
        st = self.stage
        if depth > 32:
            return None
        nxt = st.connection_source(prim, prop)
        if nxt is not None:
            return self._follow(*nxt, depth + 1)
        if (st.prim_field(prim, "typeName") or "") == "Shader":
            return prim, prop
        return None

    def _input(self, shader: str, name: str):
        """(value, texture) for a shader input: a value, or a texture it's connected to."""
        st = self.stage
        attr = f"inputs:{name}"
        prim, prop = shader, attr
        for _ in range(32):
            src = st.connection_source(prim, prop)
            if src is None:
                return st.attribute(prim, prop), None
            prim, prop = src
            if (st.prim_field(prim, "typeName") or "") == "Shader":
                if (st.attribute(prim, "info:id") or "") == "UsdUVTexture":
                    file_value, site = st.attribute(prim, "inputs:file", with_site=True)
                    if file_value:
                        self._remember_uv(shader, prim)
                        # the texture's scale is the factor its colors are multiplied by
                        scale = st.attribute(prim, "inputs:scale")
                        return scale, (str(file_value), site.source if site else None)
                return None, None
        return None, None

    def _remember_uv(self, shader: str, texture: str) -> None:
        src = self.stage.connection_source(texture, "inputs:st")
        if src is None:
            return
        reader = self._follow(*src) or (src[0], src[1])
        varname = self.stage.attribute(reader[0], "inputs:varname")
        if varname:
            self.uv_names[parent_path(shader)] = str(varname)

    def _image(self, asset: str, source: _Source | None) -> Image | None:
        if source is None:
            return None
        key = (id(source.package) if source.package else str(source.base_dir), source.inner_dir, asset)
        if key not in self.images:
            data = source.image(asset)
            name = PurePosixPath(asset.replace("\\", "/").split("[")[-1].rstrip("]")).stem or "texture"
            self.images[key] = Image(name, data, asset) if data or asset else None
        return self.images[key]


def _matrices(value, count: int) -> np.ndarray | None:
    """matrix4d[] (row-vector, translation last) as (count, 4, 4) column-vector matrices, or None."""
    if value is None:
        return None
    a = np.asarray(value, np.float64)
    if a.size != count * 16 or not np.isfinite(a).all():
        return None
    return a.reshape(count, 4, 4).transpose(0, 2, 1)


def _shared_copy(node: Node, budget: Budget) -> Node:
    """A copy of a node tree whose copies use the same meshes and materials, each copy spent from budget."""
    copies = {}
    for n, _ in node.walk():
        budget.spend()
        copies[id(n)] = Node(n.name, n.matrix.copy(), n.mesh, n.materials)
    for n, _ in node.walk():
        copies[id(n)].children = [copies[id(c)] for c in n.children]
    return copies[id(node)]


def _interpolate(kind: str, samples: dict, times: np.ndarray) -> list | None:
    """An op's value at each time, from its samples: held before the first and after the last,
    linear between (quaternions, stored (w, x, y, z), slerped); values that aren't numbers hold."""
    keys = sorted(samples)
    try:
        vals = np.array([np.asarray(samples[k], np.float64).ravel() for k in keys])
    except (TypeError, ValueError):
        return None
    if vals.ndim != 2 or not np.isfinite(vals).all():
        return None
    keys = np.array(keys, np.float64)
    k = np.clip(np.searchsorted(keys, times, side="right") - 1, 0, len(keys) - 1)
    k1 = np.minimum(k + 1, len(keys) - 1)
    span = keys[k1] - keys[k]
    u = np.where(span > 0, np.clip((times - keys[k]) / np.where(span > 0, span, 1.0), 0.0, 1.0), 0.0)
    if kind == "orient" and vals.shape[1] == 4:
        xyzw = slerp(np.roll(vals[k], -1, axis=1), np.roll(vals[k1], -1, axis=1), u)
        out = np.roll(xyzw, 1, axis=1)
    else:
        out = vals[k] + u[:, None] * (vals[k1] - vals[k])
    return list(out)


def _op_matrix(kind: str, v) -> np.ndarray | None:
    a = np.asarray(v, np.float64)
    if kind == "transform":
        return a.reshape(4, 4).T if a.size == 16 else None
    if kind in ("translate", "pivot"):
        return translation(a.ravel()[:3])
    if kind == "scale":
        a = a.ravel()
        return scaling(a[:3] if a.size >= 3 else (a[0],) * 3)
    if kind.startswith("rotate"):
        axes = kind[6:]
        if len(axes) == 1:
            deg = [0.0, 0.0, 0.0]
            deg["XYZ".index(axes)] = float(a.ravel()[0])
            return euler(deg, "XYZ")
        if len(axes) == 3 and set(axes) == set("XYZ"):
            return euler(a.ravel()[:3], axes)
        return None
    if kind == "orient":
        w, x, y, z = a.ravel()[:4]
        n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        return m
    return None


# -- implicit shapes

def _implicit(type_name: str, st: _Stage, path: str) -> Mesh | None:
    def num(name, default):
        v = st.attribute(path, name)
        return float(np.asarray(v).ravel()[0]) if v is not None and np.asarray(v).size else default
    axis = str(st.attribute(path, "axis") or "Z")
    name = element_name(path)
    if type_name == "Cube":
        s = num("size", 2.0) / 2
        p = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], float)
        quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        return Mesh(name, p, np.full(6, 4), np.array(quads).ravel())
    if type_name == "Sphere":
        return _revolve(name, _sphere_profile(num("radius", 1.0)), "Y", round_cap=True)
    if type_name in ("Cylinder", "Cylinder_1"):
        r, h = num("radius", 1.0), num("height", 2.0)
        return _revolve(name, [(0, -h / 2), (r, -h / 2), (r, h / 2), (0, h / 2)], axis)
    if type_name == "Cone":
        r, h = num("radius", 1.0), num("height", 2.0)
        return _revolve(name, [(0, -h / 2), (r, -h / 2), (0, h / 2)], axis)
    if type_name in ("Capsule", "Capsule_1"):
        r, h = num("radius", 0.5), num("height", 1.0)
        prof = [(r * np.sin(t), -h / 2 - r * np.cos(t)) for t in np.linspace(0, np.pi / 2, 7)]
        prof += [(r * np.cos(t), h / 2 + r * np.sin(t)) for t in np.linspace(0, np.pi / 2, 7)]
        return _revolve(name, prof, axis)
    if type_name == "Plane":
        w, ln = num("width", 2.0) / 2, num("length", 2.0) / 2
        quad = {"X": [(0, -w, -ln), (0, w, -ln), (0, w, ln), (0, -w, ln)],
                "Y": [(-w, 0, ln), (w, 0, ln), (w, 0, -ln), (-w, 0, -ln)],
                "Z": [(-w, -ln, 0), (w, -ln, 0), (w, ln, 0), (-w, ln, 0)]}.get(axis.upper())
        if quad is None:
            return None
        return Mesh(name, np.array(quad, float), np.array([4]), np.arange(4))
    return None


def _sphere_profile(r: float) -> list:
    return [(r * np.sin(t), -r * np.cos(t)) for t in np.linspace(0, np.pi, 13)]


def _revolve(name: str, profile, axis: str, segments: int = 24, round_cap: bool = False) -> Mesh:
    """A surface of revolution: profile is (radius, height) points from bottom to top."""
    prof = np.array(profile, float)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pts, ids = [], []
    for r, h in prof:
        if r < 1e-12:
            ids.append([len(pts)] * segments)
            pts.append((0.0, h, 0.0))
        else:
            ids.append(list(range(len(pts), len(pts) + segments)))
            pts += [(r * np.cos(a), h, -r * np.sin(a)) for a in ang]
    faces = []
    for i in range(len(prof) - 1):
        for j in range(segments):
            k = (j + 1) % segments
            quad = [ids[i][j], ids[i][k], ids[i + 1][k], ids[i + 1][j]]
            quad = [q for n, q in enumerate(quad) if q not in quad[:n]]
            if len(quad) >= 3:
                faces.append(quad)
    p = np.array(pts)
    if axis.upper() == "Z":
        p = p[:, [0, 2, 1]] * [1, -1, 1]
    elif axis.upper() == "X":
        p = p[:, [1, 0, 2]] * [1, -1, 1]
    return Mesh(name, p, np.array([len(f) for f in faces]), np.array([c for f in faces for c in f]))


# ============================================================================ scene -> layer

_ID = re.compile(r"[^A-Za-z0-9_]")


def _prim_name(name: str, taken: set[str], fallback: str) -> str:
    base = _ID.sub("_", name or "") or fallback
    if base[0].isdigit():
        base = "_" + base
    out, i = base, 1
    while out in taken:
        out = f"{base}_{i}"
        i += 1
    taken.add(out)
    return out


class _LayerWriter:
    """Builds a layer for a scene. Textures are named textures/<name>.png and returned
    separately, to be packed into a .usdz or written next to the file."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.layer = Layer()
        self.textures: dict[int, str] = {}
        self.files: dict[str, bytes] = {}
        self.materials: dict[int, str] = {}
        self.mat_names: set[str] = set()
        self.looks: str | None = None  # the Scope holding the materials, made when first needed
        self.animation = scene.animations[0] if scene.animations else None
        self.moving: dict[int, list[Channel]] = {}
        self.weights: dict[int, Channel] = {}  # node -> its mesh's blend shape weights
        walked = list(scene.walk())
        self.rest = {id(n): w for n, w in walked}
        self.parent = {id(c): n for n, _ in walked for c in n.children}
        if self.animation is not None:
            for c in self.animation.channels:
                if c.path in ("translation", "rotation", "scale") and len(c.times):
                    self.moving.setdefault(id(c.node), []).append(c)
                elif (c.path == WEIGHTS and len(c.times) and c.node.mesh is not None
                      and c.values.shape[-1] == len(c.node.mesh.targets)):
                    self.weights[id(c.node)] = c
        times = [c.times for cs in self.moving.values() for c in cs] + [c.times for c in self.weights.values()]
        self.fps = frame_rate(np.concatenate(times)) if times else 24.0
        self.span = (min(float(t[0]) for t in times), max(float(t[-1]) for t in times)) if times else None
        # what UsdSkel needs a Skeleton for: each skin whose joints are all in the scene, and each mesh
        # with blend shapes (their weights live on a skeleton's animation)
        self.bound: dict[int, list[Node]] = {}  # skin (or the mesh's node, unskinned) -> nodes it binds
        self.skins: dict[int, Skin | None] = {}
        for n, _ in walked:
            if n.mesh is None or not len(n.mesh.corners):
                continue
            skin = n.skin if n.skin is not None and all(id(j) in self.rest for j in n.skin.joints) else None
            if skin is not None or n.mesh.targets:
                key = id(skin) if skin is not None else id(n)
                self.skins.setdefault(key, skin)
                self.bound.setdefault(key, []).append(n)
        self.mesh_prims: dict[int, str] = {}  # node -> its mesh prim
        # a skinned mesh's own transform plays no part in UsdSkel, but Blender's importer places the
        # mesh by its geomBindTransform, taken as relative to the mesh's parent prim: so a skinned
        # mesh is written where it was bound (as its joints have it), and one whose parents move it
        # is lifted to the SkelRoot, the skeleton's own space, where both readings agree
        self.placed: dict[int, np.ndarray] = {}  # node -> its local matrix as written
        self.lifted: list[Node] = []
        for n, _ in walked:
            skin = self.skins.get(id(n.skin)) if n.skin is not None else None
            if skin is not None and not n.children and id(n) not in self.moving:
                bind = skin.bind_space(self.rest)
                if bind is not None:
                    self.placed[id(n)] = bind
                    above = self.rest[id(self.parent[id(n)])] if id(n) in self.parent else np.eye(4)
                    if not np.allclose(above, np.eye(4)):
                        self.lifted.append(n)

    def spec(self, path: str, kind: int, **fields) -> Spec:
        spec = Spec(kind, {})
        self.layer.specs[path] = spec
        for k, v in fields.items():
            self.set(spec, k, v)
        return spec

    @staticmethod
    def set(spec: Spec, name: str, value, kind: int | None = None, array: bool = False) -> None:
        spec.fields[name] = value
        if kind is None:
            kind = {"specifier": usdcrate.SPECIFIER, "typeName": usdcrate.TOKEN, "variability": usdcrate.VARIABILITY,
                    "primChildren": usdcrate.TOKEN_VECTOR, "properties": usdcrate.TOKEN_VECTOR,
                    "defaultPrim": usdcrate.TOKEN, "upAxis": usdcrate.TOKEN, "metersPerUnit": usdcrate.DOUBLE,
                    "interpolation": usdcrate.TOKEN, "kind": usdcrate.TOKEN, "documentation": usdcrate.STRING,
                    "elementSize": usdcrate.INT, "timeCodesPerSecond": usdcrate.DOUBLE,
                    "framesPerSecond": usdcrate.DOUBLE, "startTimeCode": usdcrate.DOUBLE,
                    "endTimeCode": usdcrate.DOUBLE}[name]
        spec.typed[name] = (kind, array, value)

    def attr(self, prim: str, name: str, type_name: str, value, uniform: bool = False, **meta) -> None:
        path = property_path(prim, name)
        spec = self.spec(path, SPEC_ATTRIBUTE, typeName=type_name)
        if uniform:
            self.set(spec, "variability", "uniform")
        if value is not None:
            kind, array = _crate_type(type_name)
            v = value.items() if isinstance(value, ListOp) else value
            self.set(spec, "default", v, kind, array)
        for k, v in meta.items():
            self.set(spec, k, v)
        self._add_property(prim, name)

    def attr_samples(self, prim: str, name: str, type_name: str, samples: dict) -> None:
        """An attribute with time samples ({time code: value}) instead of one value."""
        path = property_path(prim, name)
        spec = self.spec(path, SPEC_ATTRIBUTE, typeName=type_name)
        kind, array = _crate_type(type_name)
        spec.fields["timeSamples"] = samples
        spec.typed["timeSamples"] = (usdcrate.TIME_SAMPLES, False, (kind, samples, array))
        self._add_property(prim, name)

    def connect(self, prim: str, name: str, type_name: str, target: str) -> None:
        path = property_path(prim, name)
        spec = self.layer.specs.get(path) or self.spec(path, SPEC_ATTRIBUTE, typeName=type_name)
        op = ListOp(explicit=True, explicit_items=[target])
        spec.fields["connectionPaths"] = op
        spec.typed["connectionPaths"] = (usdcrate.PATH_LISTOP, False, [target])
        self._add_property(prim, name)

    def rel(self, prim: str, name: str, target: str | list[str]) -> None:
        path = property_path(prim, name)
        spec = self.spec(path, SPEC_RELATIONSHIP)
        self.set(spec, "variability", "uniform")
        targets = [target] if isinstance(target, str) else list(target)
        spec.fields["targetPaths"] = ListOp(explicit=True, explicit_items=targets)
        spec.typed["targetPaths"] = (usdcrate.PATH_LISTOP, False, targets)
        self._add_property(prim, name)

    def api(self, prim: str, schema: str) -> None:
        spec = self.layer.specs[prim]
        got = spec.fields.get("apiSchemas")
        schemas = (got.prepended if isinstance(got, ListOp) else []) + [schema]
        self.set(spec, "apiSchemas", ListOp(prepended=schemas), usdcrate.TOKEN_LISTOP)

    def _add_property(self, prim: str, name: str) -> None:
        spec = self.layer.specs[prim]
        props = spec.fields.get("properties", [])
        if name not in props:
            self.set(spec, "properties", props + [name])

    def prim(self, parent: str, name: str, type_name: str) -> str:
        path = child_path(parent, name)
        self.spec(path, SPEC_PRIM, specifier="def", typeName=type_name)
        parent_spec = self.layer.specs[parent]
        if "primChildren" in parent_spec.fields:
            parent_spec.fields["primChildren"].append(name)  # (typed holds the same list)
        else:
            self.set(parent_spec, "primChildren", [name])
        return path

    def build(self) -> Layer:
        self.spec("/", SPEC_PSEUDO_ROOT, defaultPrim="Root", upAxis="Y", metersPerUnit=1.0,
                  documentation="Written by meshport")
        if self.span is not None:
            start, end = self.span
            spec = self.layer.specs["/"]
            for key, value in (("timeCodesPerSecond", self.fps), ("framesPerSecond", self.fps),
                               ("startTimeCode", start * self.fps), ("endTimeCode", end * self.fps)):
                self.set(spec, key, float(value))
        top = self.top = self.prim("/", "Root", "SkelRoot" if self.bound else "Xform")
        self.set(self.layer.specs[top], "kind", "component")
        taken: set[str] = set()
        for node in self.scene.roots:
            self._node(top, node, taken)
        for node in self.lifted:
            self._node(top, node, taken)
        for key, nodes in self.bound.items():
            self._skeleton(top, taken, self.skins[key], nodes)
        return self.layer

    def _node(self, parent: str, top: Node, taken: set[str]) -> None:
        """A prim for the node and each one under it, parents first: a loop rather than recursion,
        so no depth of nesting can overflow the stack."""
        todo = [(parent, top, taken)]
        while todo:
            parent, node, taken = todo.pop()
            if node in self.lifted and parent != self.top:
                continue  # written under the SkelRoot instead
            name = _prim_name(node.name, taken, "node")
            has_mesh = node.mesh is not None and len(node.mesh.corners) > 0
            if has_mesh and not node.children:
                path = self.prim(parent, name, "Mesh")
                self._xform(path, node)
                self._mesh(path, node.mesh, node.materials)
                self.mesh_prims[id(node)] = path
                continue
            path = self.prim(parent, name, "Xform")
            self._xform(path, node)
            kids: set[str] = set()
            if has_mesh:  # a mesh can't have mesh children in USD: it goes in a child of its own
                mesh_path = self.prim(path, _prim_name(node.mesh.name or "mesh", kids, "mesh"), "Mesh")
                self._mesh(mesh_path, node.mesh, node.materials)
                self.mesh_prims[id(node)] = mesh_path
            todo.extend((path, child, kids) for child in reversed(node.children))

    def _xform(self, path: str, node: Node) -> None:
        if id(node) in self.moving:
            self._animated_xform(path, node, self.moving[id(node)])
            return
        m = self.placed.get(id(node), node.matrix)
        if np.allclose(m, np.eye(4)):
            return
        self.attr(path, "xformOp:transform", "matrix4d", np.asarray(m, np.float64).T.copy())
        self.attr(path, "xformOpOrder", "token[]", ["xformOp:transform"], uniform=True)

    def _animated_xform(self, path: str, node: Node, channels: list[Channel]) -> None:
        """translate, orient and scale ops, with time samples where the node moves. USD blends
        translate and scale linearly and slerps orient, as glTF's LINEAR does, so linear keys stay
        as they are; steps and cubic splines get a sample every frame."""
        keys = np.unique(np.concatenate([c.times for c in channels]))
        times = self._times(keys, any(c.interpolation != "LINEAR" for c in channels))
        t0, q0, s0 = decompose_trs(node.matrix)
        values = {"translation": np.tile(t0, (len(times), 1)), "rotation": np.tile(q0, (len(times), 1)),
                  "scale": np.tile(s0, (len(times), 1))}
        for c in channels:  # as Animation.local_matrix: a later channel for the same property wins
            values[c.path] = c.sample(times)
        codes = [float(t * self.fps) for t in times]
        moved = {c.path for c in channels}
        for op, prop, type_name, convert in (
                ("xformOp:translate", "translation", "double3", lambda v: np.asarray(v, np.float64)),
                ("xformOp:orient", "rotation", "quatf", lambda v: np.roll(np.asarray(v, np.float32), 1)),  # (w, x, y, z)
                ("xformOp:scale", "scale", "float3", lambda v: np.asarray(v, np.float32))):
            if prop in moved:
                self.attr_samples(path, op, type_name, {t: convert(v) for t, v in zip(codes, values[prop])})
            else:
                self.attr(path, op, type_name, convert(values[prop][0]))
        self.attr(path, "xformOpOrder", "token[]", ["xformOp:translate", "xformOp:orient", "xformOp:scale"],
                  uniform=True)

    def _times(self, keys: np.ndarray, every_frame: bool) -> np.ndarray:
        if not every_frame:
            return keys
        grid = frame_grid(keys[0] * self.fps - 1e-6, keys[-1] * self.fps + 1e-6)
        if grid is None:
            raise too_many_frames((keys[-1] - keys[0]) * self.fps)
        times = np.unique(np.concatenate([keys, grid / self.fps]))
        return times[np.concatenate(([True], np.diff(times) > 1e-7))]

    def _skeleton(self, top: str, taken: set[str], skin: Skin | None, nodes: list[Node]) -> None:
        """A Skeleton and its SkelAnimation for a skin, binding the meshes it skins; or one without
        joints that only carries an unskinned mesh's blend shape weights (UsdSkel keeps them on a
        skeleton's animation). The skeleton stands at the SkelRoot, where the scene's roots are, so
        its space is the scene's: bindTransforms are the inverse bind matrices inverted, and the
        animation stands each joint where the scene does, relative to the joint above it, with a
        sample every frame while anything above it moves."""
        joints = skin.joints if skin is not None else []
        path = self.prim(top, _prim_name(skin.name if skin is not None else f"{nodes[0].name}_shapes", taken,
                                         "skeleton"), "Skeleton")
        self.api(path, "SkelBindingAPI")
        member = {id(j): k for k, j in enumerate(joints)}
        above = []  # each joint's nearest joint above it, or -1
        for j in joints:
            p = self.parent.get(id(j))
            while p is not None and id(p) not in member:
                p = self.parent.get(id(p))
            above.append(member[id(p)] if p is not None else -1)

        def depth(k: int) -> int:
            d = 0
            while above[k] >= 0 and d <= len(joints):
                k, d = above[k], d + 1
            return d
        order = sorted(range(len(joints)), key=depth)  # every joint after the one above it
        tokens: list[str] = [""] * len(joints)
        siblings: dict[int, set[str]] = {}
        for k in order:
            name = _prim_name(joints[k].name, siblings.setdefault(above[k], set()), "joint")
            tokens[k] = name if above[k] < 0 else f"{tokens[above[k]]}/{name}"

        def relative(worlds: dict) -> np.ndarray:
            return np.array([np.linalg.inv(worlds[id(joints[above[k]])]) @ worlds[id(joints[k])] if above[k] >= 0
                             else worlds[id(joints[k])] for k in order])
        anim = child_path(path, "Animation")
        if joints:
            self.attr(path, "joints", "token[]", [tokens[k] for k in order], uniform=True)
            # bound where its joints put the mesh: bindTransforms are then the joints at rest, as
            # importers that rebuild the bind from them expect (Blender's), and the math the same
            geom = skin.bind_space(self.rest)
            geom = np.eye(4) if geom is None else geom
            binds = np.array([geom @ np.linalg.pinv(skin.inverse_binds[k]) for k in order])
            self.attr(path, "bindTransforms", "matrix4d[]", binds.transpose(0, 2, 1).copy(), uniform=True)
            self.attr(path, "restTransforms", "matrix4d[]", relative(self.rest).transpose(0, 2, 1).copy(),
                      uniform=True)
        self.rel(path, "skel:animationSource", anim)
        self.prim(path, "Animation", "SkelAnimation")
        if joints:
            self.attr(anim, "joints", "token[]", [tokens[k] for k in order], uniform=True)
            moves = False
            for j in joints:  # a joint moves when it, or anything above it, does
                n = j
                while n is not None and id(n) not in self.moving:
                    n = self.parent.get(id(n))
                moves |= n is not None
            if moves:
                keys = np.unique(np.concatenate([c.times for cs in self.moving.values() for c in cs]))
                times = self._times(keys, True)
                worlds = [{id(n): w for n, w in self.scene.walk_at(t, self.animation)} for t in times]
                parts = decompose_many(np.array([relative(w) for w in worlds]).reshape(-1, 4, 4))
                t, q, s = (v.reshape(len(times), len(joints), -1) for v in parts)
                codes = [float(x * self.fps) for x in times]
                self.attr_samples(anim, "translations", "float3[]",
                                  {c: t[i].astype(np.float32) for i, c in enumerate(codes)})
                self.attr_samples(anim, "rotations", "quatf[]",
                                  {c: np.roll(q[i], 1, axis=1).astype(np.float32) for i, c in enumerate(codes)})
                self.attr_samples(anim, "scales", "half3[]",
                                  {c: s[i].astype(np.float16) for i, c in enumerate(codes)})
            else:
                t, q, s = decompose_many(relative(self.rest))
                self.attr(anim, "translations", "float3[]", t.astype(np.float32))
                self.attr(anim, "rotations", "quatf[]", np.roll(q, 1, axis=1).astype(np.float32))
                self.attr(anim, "scales", "half3[]", s.astype(np.float16))
        names: list[str] = []  # every bound mesh's targets, named apart
        shaped = [n for n in nodes if n.mesh.targets and id(n) in self.mesh_prims]
        for n in nodes:
            mesh_prim = self.mesh_prims.get(id(n))
            if mesh_prim is None:
                continue
            self.api(mesh_prim, "SkelBindingAPI")
            self.rel(mesh_prim, "skel:skeleton", path)
            if skin is not None:
                self.attr(mesh_prim, "skel:joints", "token[]", tokens, uniform=True)
                size = skin.indices.shape[1]
                self.attr(mesh_prim, "primvars:skel:jointIndices", "int[]", np.asarray(skin.indices, np.int32).ravel(),
                          interpolation="vertex", elementSize=size)
                self.attr(mesh_prim, "primvars:skel:jointWeights", "float[]",
                          np.asarray(skin.weights, np.float32).ravel(), interpolation="vertex", elementSize=size)
                self.attr(mesh_prim, "primvars:skel:geomBindTransform", "matrix4d", geom.T.copy())
            if n in shaped:
                taken_names = set(names)
                mine = [_prim_name(m.name, taken_names, "shape") for m in n.mesh.targets]
                names += mine
                kids = set(self.layer.specs[mesh_prim].fields.get("primChildren", []))
                shapes = []
                for m in n.mesh.targets:
                    moved = np.flatnonzero(np.abs(m.positions).max(1) > 0)
                    moved = moved if len(moved) else np.zeros(1, np.int64)  # a shape needs a point
                    shape = self.prim(mesh_prim, _prim_name(m.name, kids, "shape"), "BlendShape")
                    self.attr(shape, "offsets", "vector3f[]", np.asarray(m.positions[moved], np.float32), uniform=True)
                    self.attr(shape, "pointIndices", "int[]", moved.astype(np.int32), uniform=True)
                    shapes.append(shape)
                self.attr(mesh_prim, "skel:blendShapes", "token[]", mine, uniform=True)
                self.rel(mesh_prim, "skel:blendShapeTargets", shapes)
        if not names:
            return
        self.attr(anim, "blendShapes", "token[]", names, uniform=True)

        def weights_at(times: np.ndarray) -> np.ndarray:
            cols = []
            for n in shaped:
                c = self.weights.get(id(n))
                rest = n.mesh.target_weights if n.mesh.target_weights is not None else np.zeros(len(n.mesh.targets))
                cols.append(c.sample(times) if c is not None else np.tile(rest, (len(times), 1)))
            return np.hstack(cols)
        moving = [self.weights[id(n)] for n in shaped if id(n) in self.weights]
        if moving:  # USD blends weights linearly: steps and cubic splines get a sample every frame
            keys = np.unique(np.concatenate([c.times for c in moving]))
            times = self._times(keys, any(c.interpolation != "LINEAR" for c in moving))
            self.attr_samples(anim, "blendShapeWeights", "float[]",
                              {float(t * self.fps): w.astype(np.float32) for t, w in zip(times, weights_at(times))})
        else:
            self.attr(anim, "blendShapeWeights", "float[]", weights_at(np.zeros(1))[0].astype(np.float32))

    def _mesh(self, path: str, mesh: Mesh, materials: list[Material]) -> None:
        pts = np.asarray(mesh.positions, np.float32)
        self.attr(path, "extent", "float3[]", np.array([pts.min(0), pts.max(0)], np.float32))
        self.attr(path, "faceVertexCounts", "int[]", np.asarray(mesh.face_sizes, np.int32))
        self.attr(path, "faceVertexIndices", "int[]", np.asarray(mesh.corners, np.int32))
        self.attr(path, "points", "point3f[]", pts)
        if mesh.normals is not None:
            self.attr(path, "normals", "normal3f[]", np.asarray(mesh.normals, np.float32),
                      interpolation="faceVarying")
        for i, uv in enumerate(mesh.uvs):
            first, index = unique_rows(np.ascontiguousarray(uv, np.float32))
            name = "primvars:st" if i == 0 else f"primvars:st{i}"
            self.attr(path, name, "texCoord2f[]", np.asarray(uv, np.float32)[first], interpolation="faceVarying")
            self.attr(path, name + ":indices", "int[]", index.astype(np.int32))
        if mesh.colors is not None:
            c = np.asarray(mesh.colors, np.float32)
            self.attr(path, "primvars:displayColor", "color3f[]", c[:, :3], interpolation="faceVarying")
            if (c[:, 3] < 1).any():
                self.attr(path, "primvars:displayOpacity", "float[]", c[:, 3], interpolation="faceVarying")
        self.attr(path, "subdivisionScheme", "token", "none", uniform=True)
        if materials and any(m.double_sided for m in materials):
            self.attr(path, "doubleSided", "bool", True, uniform=True)
        if not materials:
            return
        slots = mesh.face_materials if mesh.face_materials is not None else np.zeros(len(mesh.face_sizes), np.int64)
        slots = np.clip(slots, 0, len(materials) - 1)
        used = np.unique(slots)
        self.set(self.layer.specs[path], "apiSchemas", ListOp(prepended=["MaterialBindingAPI"]),
                 usdcrate.TOKEN_LISTOP)
        if len(used) == 1:
            self.rel(path, "material:binding", self._material(materials[int(used[0])]))
            return
        self.attr(path, "subsetFamily:materialBind:familyType", "token", "nonOverlapping", uniform=True)
        kids: set[str] = set()
        for slot in used:
            mat = materials[int(slot)]
            sub = self.prim(path, _prim_name(f"{mat.name}_faces", kids, "faces"), "GeomSubset")
            self.set(self.layer.specs[sub], "apiSchemas", ListOp(prepended=["MaterialBindingAPI"]),
                     usdcrate.TOKEN_LISTOP)
            self.attr(sub, "elementType", "token", "face", uniform=True)
            self.attr(sub, "familyName", "token", "materialBind", uniform=True)
            self.attr(sub, "indices", "int[]", np.nonzero(slots == slot)[0].astype(np.int32))
            self.rel(sub, "material:binding", self._material(mat))

    def _material(self, mat: Material) -> str:
        if id(mat) in self.materials:
            return self.materials[id(mat)]
        if self.looks is None:
            self.looks = self.prim("/Root", "Materials", "Scope")
        path = self.prim(self.looks, _prim_name(mat.name, self.mat_names, "material"), "Material")
        self.materials[id(mat)] = path
        shader = self.prim(path, "PreviewSurface", "Shader")
        self.attr(shader, "info:id", "token", "UsdPreviewSurface", uniform=True)
        self.connect(path, "outputs:surface", "token", property_path(shader, "outputs:surface"))
        self.attr(shader, "outputs:surface", "token", None)
        reader = None

        def texture(image: Image, name: str, color_space: str, scale=None) -> str | None:
            nonlocal reader
            file = self._texture_file(image)
            if file is None:
                return None
            if reader is None:
                reader = self.prim(path, "TexCoordReader", "Shader")
                self.attr(reader, "info:id", "token", "UsdPrimvarReader_float2", uniform=True)
                self.attr(reader, "inputs:varname", "string", "st")
                self.attr(reader, "outputs:result", "float2", None)
            tex = self.prim(path, name, "Shader")
            self.attr(tex, "info:id", "token", "UsdUVTexture", uniform=True)
            self.attr(tex, "inputs:file", "asset", file)
            self.connect(tex, "inputs:st", "float2", property_path(reader, "outputs:result"))
            self.attr(tex, "inputs:wrapS", "token", "repeat")
            self.attr(tex, "inputs:wrapT", "token", "repeat")
            self.attr(tex, "inputs:sourceColorSpace", "token", color_space)
            if scale is not None:
                self.attr(tex, "inputs:scale", "float4", np.asarray(scale, np.float32))
            if color_space == "raw" and name == "NormalTexture":
                self.attr(tex, "inputs:bias", "float4", np.array([-1, -1, -1, 0], np.float32))
            self.attr(tex, "outputs:rgb", "float3", None)
            self.attr(tex, "outputs:a", "float", None)
            return tex

        r, g, b, a = (float(x) for x in mat.base_color)
        tex = texture(mat.base_color_texture, "BaseColorTexture", "sRGB",
                      (r, g, b, a)) if mat.base_color_texture is not None else None
        if tex is not None:
            self.connect(shader, "inputs:diffuseColor", "color3f", property_path(tex, "outputs:rgb"))
        else:
            self.attr(shader, "inputs:diffuseColor", "color3f", np.array([r, g, b], np.float32))
        if mat.alpha_mode == "BLEND" and tex is not None:
            self.connect(shader, "inputs:opacity", "float", property_path(tex, "outputs:a"))
        elif mat.alpha_mode != "OPAQUE" or a < 1:
            self.attr(shader, "inputs:opacity", "float", a)
        if mat.alpha_mode == "MASK":
            if tex is not None:
                self.connect(shader, "inputs:opacity", "float", property_path(tex, "outputs:a"))
            self.attr(shader, "inputs:opacityThreshold", "float", 0.5)
        self.attr(shader, "inputs:metallic", "float", float(mat.metallic))
        self.attr(shader, "inputs:roughness", "float", float(mat.roughness))
        em = np.asarray(mat.emissive, np.float32)
        em_tex = texture(mat.emissive_texture, "EmissiveTexture", "sRGB",
                         (*em, 1.0)) if mat.emissive_texture is not None else None
        if em_tex is not None:
            self.connect(shader, "inputs:emissiveColor", "color3f", property_path(em_tex, "outputs:rgb"))
        elif em.any():
            self.attr(shader, "inputs:emissiveColor", "color3f", em)
        nrm = texture(mat.normal_texture, "NormalTexture", "raw",
                      (2.0, 2.0, 2.0, 1.0)) if mat.normal_texture is not None else None
        if nrm is not None:
            self.connect(shader, "inputs:normal", "normal3f", property_path(nrm, "outputs:rgb"))
        return path

    def _texture_file(self, image: Image | None) -> str | None:
        if image is None or not image.data:
            return None
        if id(image) not in self.textures:
            data = image.data
            ext = {"image/png": "png", "image/jpeg": "jpg"}.get(image.mime or "")
            if ext is None:  # USDZ allows PNG and JPEG only
                from .gltf import _to_png
                data, _ = _to_png(data)
                ext = "png"
            if data is None:
                self.textures[id(image)] = None
                return None
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", image.name or "texture")[:60] or "texture"
            name, i = f"textures/{stem}.{ext}", 1
            while name in self.files:
                name = f"textures/{stem}_{i}.{ext}"
                i += 1
            self.files[name] = data
            self.textures[id(image)] = name
        return self.textures[id(image)]


def _crate_type(type_name: str) -> tuple[int, bool]:
    is_array = type_name.endswith("[]")
    base = type_name.removesuffix("[]")
    c = usdcrate
    table = {"bool": c.BOOL, "int": c.INT, "float": c.FLOAT, "double": c.DOUBLE, "token": c.TOKEN,
             "string": c.STRING, "asset": c.ASSET, "matrix4d": c.MATRIX4D, "float2": c.VEC2F, "float3": c.VEC3F,
             "float4": c.VEC4F, "double3": c.VEC3D, "point3f": c.VEC3F, "normal3f": c.VEC3F, "color3f": c.VEC3F,
             "texCoord2f": c.VEC2F, "vector3f": c.VEC3F, "quatf": c.QUATF, "half3": c.VEC3H}
    return table[base], is_array


def write(scene: Scene, fmt: str = "usdz") -> bytes:
    """.usdz bytes (everything inside), or a .usda/.usdc/.usd layer (textures go next to it:
    use save())."""
    writer = _LayerWriter(scene)
    layer = writer.build()
    if fmt == "usda":
        return usda.write(layer)
    data = usdcrate.write(layer)
    if fmt != "usdz":
        return data
    return pack_usdz([("model.usdc", data), *writer.files.items()])


def write_file(scene: Scene, path: str | os.PathLike, fmt: str) -> list[Path]:
    path = Path(path)
    if fmt == "usdz":
        path.write_bytes(write(scene, "usdz"))
        return [path]
    writer = _LayerWriter(scene)
    layer = writer.build()
    path.write_bytes(usda.write(layer) if fmt == "usda" else usdcrate.write(layer))
    written = [path]
    for name, data in writer.files.items():
        out = path.parent / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        written.append(out)
    return written


def pack_usdz(files: list[tuple[str, bytes]]) -> bytes:
    """A .usdz: an uncompressed zip whose files each start on a 64-byte boundary, root layer first."""
    out = bytearray()
    central = bytearray()
    for name, data in files:
        raw_name = name.encode("utf-8")
        crc = zlib.crc32(data) & 0xFFFFFFFF
        offset = len(out)
        pad = -(offset + 30 + len(raw_name) + 4) % 64
        extra = struct.pack("<HH", 0x1986, pad) + bytes(pad)  # padding, in an extra field readers skip
        out += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0x21, crc, len(data), len(data),
                           len(raw_name), len(extra)) + raw_name + extra
        out += data
        central += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, 0, 0x21, crc, len(data), len(data),
                               len(raw_name), 0, 0, 0, 0, 0, offset) + raw_name
    out += central
    out += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(files), len(files), len(central), len(out) - len(central), 0)
    return bytes(out)

