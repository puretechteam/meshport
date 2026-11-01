"""Wavefront OBJ (+ .mtl and texture files next to it), read and written.

Reading: each object ("o", or "g" in a file without objects) becomes a node with one mesh,
its materials taken from the .mtl files the OBJ names. Writing: OBJ has no hierarchy, so
every mesh is written in world space as its own object. Polygons stay polygons either way;
mirrored nodes get their winding flipped so faces keep pointing outward.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .fbx import _EXT, TextureFinder
from .geometry import transform_normals, transform_points, unique_rows
from .scene import (
    DAMAGED,
    Image,
    Material,
    Mesh,
    MeshportError,
    Node,
    Scene,
    linked_path,
    node_budget,
)

# ============================================================================ reading

MAX_MTL = 16 << 20  # bytes of one .mtl file read; real ones are a few KB
KEYWORDS = (b"v", b"vt", b"vn", b"f", b"o", b"g", b"s", b"usemtl", b"mtllib")
# texture options and how many values each takes (-o, -s and -t take one to three)
_MAP_OPTIONS = {"-blendu": 1, "-blendv": 1, "-bm": 1, "-boost": 1, "-cc": 1, "-clamp": 1, "-imfchan": 1,
                "-mm": 2, "-texres": 1, "-type": 1, "-o": 3, "-s": 3, "-t": 3}


def detect(data: bytes) -> bool:
    """Whether data starts like an OBJ file: its first statement is one of OBJ's keywords."""
    for line in data[:4096].lstrip(b"\xef\xbb\xbf").splitlines():
        words = line.split()
        if words and not words[0].startswith(b"#"):
            return words[0] in KEYWORDS
    return False


def read(data: bytes, base_dir: str | os.PathLike | None = None) -> Scene:
    """Parse OBJ bytes into a Scene. base_dir is where to look for the .mtl and texture files."""
    try:
        return _Reader(Path(base_dir) if base_dir else None).read(data)
    except DAMAGED as e:
        raise MeshportError(f"This OBJ file is damaged ({type(e).__name__}: {e})") from None


def read_file(path: str | os.PathLike) -> Scene:
    path = Path(path)
    return read(path.read_bytes(), path.parent)


def _text(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if "\\" in text:  # a backslash at the end of a line joins the next one to it
        text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    return text


class _Reader:
    def __init__(self, base_dir: Path | None):
        self.base_dir = base_dir
        self.finder = TextureFinder(base_dir)
        self.materials: dict[str, Material] = {}
        self.images: dict[str, Image | None] = {}

    def read(self, data: bytes) -> Scene:
        v_lines, vt_lines, vn_lines = [], [], []
        corners: list[str] = []
        sizes: list[int] = []
        # where the state changes: (first face, object, material, smoothing group, v, vt and vn so far)
        marks: list[tuple[int, ...]] = []
        changed = True
        objects: list[str] = []
        material_slots: dict[str | None, int] = {None: 0}
        obj = material = None
        smooth = 0
        has_o = False
        group = ""
        for line in _text(data).splitlines():
            head = line[:2]
            if head == "f ":
                parts = line.split()
                if len(parts) < 4:
                    continue  # a face needs three corners
                if obj is None:
                    objects.append(group)
                    obj, changed = len(objects) - 1, True
                if changed:
                    marks.append((len(sizes), obj, material_slots.setdefault(material, len(material_slots)),
                                  smooth, len(v_lines), len(vt_lines), len(vn_lines)))
                    changed = False
                corners.extend(parts[1:])
                sizes.append(len(parts) - 1)
                continue
            changed = True
            if head == "v ":
                v_lines.append(line)
            elif head == "vt":
                vt_lines.append(line)
            elif head == "vn":
                vn_lines.append(line)
            else:
                word, _, rest = line.strip().partition(" ")
                rest = rest.strip()
                if word == "o":
                    has_o = True
                    objects.append(rest)
                    obj = len(objects) - 1
                elif word == "g":
                    group = rest
                    if not has_o:
                        obj = None  # each group is an object of its own, until the file names objects
                elif word == "usemtl":
                    material = rest
                elif word == "s":
                    smooth = int(rest) % (1 << 62) if rest.isdigit() else 0 if rest.lower() in ("off", "") else 1
                elif word == "mtllib":
                    self._mtllib(rest)
        if not sizes:
            raise MeshportError("This OBJ file has no faces (meshport reads faces, not lines or points)")
        positions, colors = _vertices(_words(v_lines))
        uv_table, normal_table = _table(_words(vt_lines), 2), _table(_words(vn_lines), 3)
        v, vt, vn = _corner_indices(corners)
        sizes_a = np.array(sizes, np.int64)
        marks_a = np.array(marks, np.int64)
        per_face = np.repeat(marks_a[:, 1:], np.diff(np.append(marks_a[:, 0], len(sizes))), axis=0)
        f_object_a, f_material_a, f_smooth_a = per_face[:, 0], per_face[:, 1], per_face[:, 2]
        before = np.repeat(per_face[:, 3:], sizes_a, axis=0)
        v = _resolve(v, before[:, 0], len(positions), "vertex", required=True)
        vt = _resolve(vt, before[:, 1], len(uv_table), "texture coordinate") if vt is not None else None
        vn = _resolve(vn, before[:, 2], len(normal_table), "normal") if vn is not None else None
        slot_names = list(material_slots)
        order = np.argsort(f_object_a, kind="stable")
        starts = np.concatenate(([0], np.cumsum(sizes_a)[:-1]))
        bounds = np.searchsorted(f_object_a[order], np.arange(len(objects) + 1))
        roots = []
        budget = node_budget()
        for o in range(len(objects)):
            faces = order[bounds[o]:bounds[o + 1]]
            if not len(faces):
                continue
            budget.spend()
            fs = sizes_a[faces]
            first = np.concatenate(([0], np.cumsum(fs)[:-1]))
            cs = np.repeat(starts[faces] - first, fs) + np.arange(int(fs.sum()))
            roots.append(self._node(objects[o] or "mesh", fs, v[cs], positions, colors,
                                    None if vt is None else vt[cs], uv_table, None if vn is None else vn[cs],
                                    normal_table, f_material_a[faces], slot_names, f_smooth_a[faces]))
        return Scene(roots)

    def _node(self, name, sizes, v, positions, colors, vt, uv_table, vn, normal_table, slots, slot_names,
              smooth) -> Node:
        used, local = np.unique(v, return_inverse=True)
        mesh = Mesh(name, positions[used], sizes, local.astype(np.int64).ravel())
        if vt is not None and (vt >= 0).any():
            mesh.uvs = [np.where((vt >= 0)[:, None], uv_table[np.maximum(vt, 0)], 0.0)]
        if colors is not None:
            mesh.colors = colors[v]
        if vn is not None and (vn >= 0).all():
            mesh.normals = _unit(normal_table[vn])
        elif vn is not None and (vn >= 0).any() or smooth.any():
            mesh.normals = _shaded_normals(mesh, smooth)  # filled in from the smoothing groups
            if vn is not None:
                mesh.normals[vn >= 0] = _unit(normal_table[vn[vn >= 0]])
        materials = []
        if (slots > 0).any():  # slot 0: faces before any usemtl
            kinds, index = np.unique(slots, return_inverse=True)
            mesh.face_materials = index.astype(np.int64).ravel()
            materials = [self._material(slot_names[k]) for k in kinds]
        mesh.validate()
        return Node(name, mesh=mesh, materials=materials)

    def _material(self, name: str | None) -> Material:
        name = "default" if name is None else name
        if name not in self.materials:
            self.materials[name] = Material(name)  # used but in no .mtl: the default look, as in Blender
        return self.materials[name]

    def _mtllib(self, rest: str) -> None:
        if self.base_dir is None:
            return
        for names in ([rest], rest.split()):  # one name with spaces in it, or several names
            found = False
            for n in names:
                path = linked_path(n, self.base_dir)
                if path is None or path.suffix.lower() != ".mtl" or not path.is_file():
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read(MAX_MTL)
                except OSError:
                    continue
                self._parse_mtl(_text(data))
                found = True
            if found:
                return

    def _parse_mtl(self, text: str) -> None:
        found: list[tuple[str, dict[str, str]]] = []
        for line in text.splitlines():
            word, _, rest = line.strip().partition(" ")
            if word == "newmtl":
                found.append((rest.strip(), {}))
            elif found and rest.strip():
                found[-1][1][word.lower() if word.lower().startswith(("map_", "bump")) else word] = rest.strip()
        for name, values in found:
            self.materials[name] = self._make_material(name, values)

    def _make_material(self, name: str, values: dict[str, str]) -> Material:
        """A material from its statements, read the way Blender's importer reads them: roughness
        from Pr, else from Ns (0 to 1000); metallic from Pm, else from Ka when illum turns
        reflection on (3 to 8); a texture with no Kd is shown as it is."""
        def number(key, n=1, default=None):
            try:
                return _floats(values[key], n) if key in values else default
            except ValueError:  # an unreadable value: as if it weren't there
                return default

        mat = Material(name)
        kd = number("Kd", 3)
        alpha = number("d", 1, None)
        alpha = alpha[0] if alpha else 1.0 - number("Tr", 1, (0.0,))[0]
        texture = self._image(_map_file(values.get("map_kd", "")))
        rgb = kd or ((1.0, 1.0, 1.0) if texture is not None else mat.base_color[:3])
        mat.base_color = tuple(min(max(c, 0.0), 1.0) for c in (*rgb, alpha))
        mat.alpha_mode = "BLEND" if mat.base_color[3] < 1.0 else "OPAQUE"
        mat.base_color_texture = texture
        pr, ns = number("Pr"), number("Ns")
        if pr or ns:
            mat.roughness = pr[0] if pr else 1.0 - np.sqrt(min(max(ns[0], 0.0), 1000.0) / 1000.0)
        pm, ka, illum = number("Pm"), number("Ka", 3), number("illum")
        if pm:
            mat.metallic = pm[0]
        elif ka and illum and 3 <= illum[0] <= 8:
            mat.metallic = sum(ka) / 3
        mat.roughness, mat.metallic = (float(min(max(x, 0.0), 1.0)) for x in (mat.roughness, mat.metallic))
        mat.emissive = tuple(max(c, 0.0) for c in number("Ke", 3, mat.emissive))
        mat.emissive_texture = self._image(_map_file(values.get("map_ke", "")))
        mat.normal_texture = self._image(_map_file(values.get("norm") or values.get("map_bump")
                                                   or values.get("bump", "")))
        return mat

    def _image(self, name: str) -> Image | None:
        if not name:
            return None
        if name not in self.images:
            data = self.finder.find([name])
            stem = Path(name.replace("\\", "/")).stem or "texture"
            self.images[name] = Image(stem, data, name) if data is not None else None
        return self.images[name]


def _floats(text: str, n: int) -> tuple[float, ...]:
    values = [float(x) for x in text.split()[:n]]
    if not values or not np.isfinite(values).all():
        raise ValueError(text)
    return tuple(values + values[-1:] * (n - len(values)))  # "Kd 0.5" is grey


def _map_file(rest: str) -> str:
    """The file name in a texture statement, after its options (-s 1 1, -bm 0.5, ...)."""
    words = rest.split(" ")
    i = 0
    while i < len(words) and words[i] in _MAP_OPTIONS:
        option, take = words[i], _MAP_OPTIONS[words[i]]
        i += 1
        numeric = option in ("-o", "-s", "-t", "-mm", "-bm", "-boost", "-texres")
        while take and i < len(words) and (not numeric or _number(words[i])):
            i += 1
            take -= 1
    return " ".join(words[i:]).strip()


def _number(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


def _words(lines: list[str]) -> list[list[str]] | np.ndarray:
    """The words of each line: one array when every line has as many (the usual case, and much
    faster than splitting line by line), else a list per line."""
    if not lines:
        return []
    words = " ".join(lines).split()
    width = len(words) // len(lines)
    if width * len(lines) == len(words) and words[::width].count(words[0]) == len(lines):
        return np.array(words, dtype=object).reshape(len(lines), width)
    return [line.split() for line in lines]


def _vertices(rows: list[list[str]] | np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Positions, and colors when any vertex has them ("v x y z r g b"; the others are white)."""
    if not len(rows):
        raise MeshportError("This OBJ file is damaged (its faces have no vertices)")
    table = _numbers(rows, 1)
    if table.shape[1] < 3:
        raise MeshportError("This OBJ file is damaged (a vertex has fewer than three coordinates)")
    colors = None
    if table.shape[1] >= 6:  # sRGB, as Blender, ZBrush and MeshLab write them
        colors = np.ones((len(table), 4))
        c = np.clip(table[:, 3:6], 0.0, 1.0)
        colors[:, :3] = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        if isinstance(rows, list):
            colors[[len(r) < 7 for r in rows]] = 1.0
    return np.ascontiguousarray(table[:, :3]), colors


def _table(rows: list[list[str]] | np.ndarray, width: int) -> np.ndarray:
    """The first width numbers after each line's keyword ("vt u" without v gives v = 0)."""
    if not len(rows):
        return np.zeros((0, width))
    table = _numbers(rows, 0)
    if table.shape[1] < width:
        table = np.hstack([table, np.zeros((len(table), width - table.shape[1]))])
    return np.ascontiguousarray(table[:, :width])


def _numbers(rows: list[list[str]] | np.ndarray, pad: float) -> np.ndarray:
    """The numbers after each row's keyword, short rows padded with pad."""
    if isinstance(rows, np.ndarray):
        table = rows[:, 1:].astype(np.float64)
    else:
        width = max(len(r) for r in rows)
        table = np.array([r[1:] + [pad] * (width - len(r)) for r in rows], dtype=object).astype(np.float64)
    if not np.isfinite(table).all():  # "nan", "inf", "1e999": Python reads them, a model can't use them
        raise MeshportError("This OBJ file is damaged (a number in it isn't a real number)")
    return table


def _corner_indices(tokens: list[str]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Each corner's v, vt and vn numbers as written (0 where it has none)."""
    joined = " ".join(tokens)
    n, slashes = len(tokens), joined.count("/")
    try:
        if slashes == 0:
            return np.array(tokens, dtype=object).astype(np.int64), None, None
        doubles = joined.count("//")
        if slashes == n and doubles == 0:
            t = np.array(joined.replace("/", " ").split(), dtype=object).astype(np.int64).reshape(n, 2)
            return t[:, 0], t[:, 1], None
        if slashes == 2 * n and doubles in (0, n):
            t = np.array(joined.replace("//", "/0/").replace("/", " ").split(), dtype=object)
            t = t.astype(np.int64).reshape(n, 3)
            return t[:, 0], (t[:, 1] if doubles == 0 else None), t[:, 2]
    except ValueError:
        pass  # corners written different ways: one at a time
    t = np.zeros((n, 3), np.int64)
    for i, token in enumerate(tokens):
        for j, part in enumerate(token.split("/")[:3]):
            if part:
                t[i, j] = int(part)
    return t[:, 0], (t[:, 1] if t[:, 1].any() else None), (t[:, 2] if t[:, 2].any() else None)


def _resolve(index: np.ndarray, before: np.ndarray, count: int, what: str, required: bool = False) -> np.ndarray:
    """1-based numbers, or negative ones counting back from the last one so far -> 0-based (-1: none)."""
    out = np.where(index > 0, index - 1, np.where(index < 0, before + index, -1))
    if (out >= count).any() or ((index < 0) & (out < 0)).any() or required and (out < 0).any():
        raise MeshportError(f"This OBJ file is damaged (a face uses a {what} it doesn't have)")
    return out


def _unit(n: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 0)


def _shaded_normals(mesh: Mesh, smooth: np.ndarray) -> np.ndarray:
    """Per-corner normals from the smoothing groups, as Blender shades: a face in group 0 (s off)
    gets its own normal. Around each vertex, the faces of one group that meet along edges (a fan)
    share the average of their normals, each weighted by its angle at that vertex."""
    sizes, starts = mesh.face_sizes, mesh.face_starts
    face = np.repeat(np.arange(len(sizes)), sizes)
    c = np.arange(len(mesh.corners))
    nxt, prv = c + 1, c - 1
    nxt[starts + sizes - 1] = starts
    prv[starts] = starts + sizes - 1
    p = mesh.positions[mesh.corners]
    cross = np.cross(p, p[nxt])
    fn = _unit(np.stack([np.bincount(face, cross[:, k], len(sizes)) for k in range(3)], axis=1))
    a, b = _unit(p[nxt] - p), _unit(p[prv] - p)
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))
    group = smooth[face]
    fan = _fans(mesh.corners, nxt, group)
    summed = np.stack([np.bincount(fan, fn[face, k] * angle, len(c)) for k in range(3)], axis=1)
    return _unit(np.where((group > 0)[:, None], summed[fan], fn[face]))


def _fans(corners: np.ndarray, nxt: np.ndarray, group: np.ndarray) -> np.ndarray:
    """For each corner, the lowest corner of its fan: the corners at the same vertex whose faces
    meet along an edge used by exactly two faces of the same smoothing group."""
    lo, hi = np.minimum(corners, corners[nxt]), np.maximum(corners, corners[nxt])
    order = np.lexsort((hi, lo))
    same = (lo[order][1:] == lo[order][:-1]) & (hi[order][1:] == hi[order][:-1])
    alone = np.ones(len(order) + 1, bool)  # an edge used by exactly two half-edges
    alone[1:-1] = ~same
    pair = np.nonzero(same & alone[:-2] & alone[2:])[0] if len(order) > 1 else np.zeros(0, np.int64)
    x, y = order[pair], order[pair + 1]
    keep = (group[x] > 0) & (group[x] == group[y])
    x, y = x[keep], y[keep]
    # x runs u -> v; join the corners of x's face and y's face that sit at the same vertex
    yu = np.where(corners[y] == corners[x], y, nxt[y])
    yv = np.where(corners[y] == corners[x], nxt[y], y)
    a = np.concatenate([x, nxt[x]])
    b = np.concatenate([yu, yv])
    label = np.arange(len(corners))
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]
        if (new == label).all():
            return label
        label = new


# ============================================================================ writing


def write_file(scene: Scene, path: str | os.PathLike) -> list[Path]:
    """Write path (.obj), plus a .mtl and texture files when there are materials.
    Returns every file written. Skins and blend shapes are applied where they stand."""
    path = Path(path)
    scene = scene.deformed()
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
        points = transform_points(world, mesh.positions)
        if mesh.colors is not None:  # OBJ colors are per vertex and sRGB: each vertex's first corner's
            first = np.full(len(points), -1)
            first[mesh.corners[::-1]] = np.arange(len(mesh.corners))[::-1]
            c = np.clip(np.where(first[:, None] >= 0, mesh.colors[np.maximum(first, 0), :3], 1.0), 0.0, 1.0)
            points = np.hstack([points, np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)])
        lines.append(_rows("v", points))
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
    # Ns as Blender writes and reads it, plus the PBR extension's Pr and Pm, which win where they're read
    out = [f"newmtl {name}", f"Kd {r:.6g} {g:.6g} {b:.6g}", "Ka 0 0 0", "Ks 0.2 0.2 0.2",
           f"Ns {(1.0 - mat.roughness) ** 2 * 1000:.6g}", f"Pr {mat.roughness:.6g}", f"Pm {mat.metallic:.6g}",
           f"d {a:.6g}", "illum 2"]
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
