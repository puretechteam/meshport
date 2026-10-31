"""USD's text layer format (.usda): text <-> Layer.

Parses the whole grammar a layer can use: prims (def/over/class) with metadata, attributes with
values, connections and time samples, relationships, variant sets, list edits (prepend, append,
delete...), dictionaries, references, payloads and layer metadata. Big numeric arrays are read
in one go with NumPy rather than token by token.
"""

from __future__ import annotations

import re

import numpy as np

from .scene import MeshportError
from .usdlayer import (
    SPEC_ATTRIBUTE,
    SPEC_PRIM,
    SPEC_PSEUDO_ROOT,
    SPEC_RELATIONSHIP,
    SPEC_VARIANT,
    Layer,
    ListOp,
    Payload,
    Reference,
    Spec,
    child_path,
    property_path,
)

_TOKEN = re.compile(r'''
    (?P<skip>(?:\s+|\#[^\n]*)+)
  | (?P<str3>"""(?:[^"\\]|\\.|"(?!""))*"""|\'\'\'(?:[^'\\]|\\.|'(?!''))*\'\'\')
  | (?P<str>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
  | (?P<asset>@@@.*?@@@|@[^@\n]*@)
  | (?P<path><[^>\n]*>)
  | (?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
  | (?P<word>[A-Za-z_][\w:.]*(?:\[\])?)
  | (?P<punct>[()\[\]{}=,:;&-])
''', re.VERBOSE | re.DOTALL)

_LISTOP_WORDS = {"add": "added", "prepend": "prepended", "append": "appended", "delete": "deleted",
                 "reorder": "ordered"}
_FIELD_NAMES = {"variants": "variantSelection", "inherits": "inheritPaths", "variantSets": "variantSetNames",
                "doc": "documentation"}
_NUMERIC_BASE = {
    "bool": ("u1", 1), "uchar": ("u1", 1), "int": ("i4", 1), "uint": ("u4", 1), "int64": ("i8", 1),
    "uint64": ("u8", 1), "half": ("f2", 1), "float": ("f4", 1), "double": ("f8", 1), "timecode": ("f8", 1),
    "matrix2d": ("f8", 4), "matrix3d": ("f8", 9), "matrix4d": ("f8", 16), "frame4d": ("f8", 16),
    "quatd": ("f8", 4), "quatf": ("f4", 4), "quath": ("f2", 4),
}
for _base, _dt in (("double", "f8"), ("float", "f4"), ("half", "f2"), ("int", "i4")):
    for _n in (2, 3, 4):
        _NUMERIC_BASE[f"{_base}{_n}"] = (_dt, _n)
for _role in ("point", "normal", "vector", "color", "texCoord"):
    for _n in (2, 3, 4):
        for _s, _dt in (("d", "f8"), ("f", "f4"), ("h", "f2")):
            _NUMERIC_BASE[f"{_role}{_n}{_s}"] = (_dt, _n)
_STRINGY = {"string", "token", "asset", "pathExpression", "opaque", "group"}


def numeric_type(type_name: str) -> tuple[str, int] | None:
    """('f4', 3) for 'point3f[]': element dtype and components, or None for non-numeric types."""
    return _NUMERIC_BASE.get(type_name.removesuffix("[]"))


# ============================================================================ reading

class _Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self._peeked: tuple[str, str] | None = None

    def peek(self) -> tuple[str, str]:
        if self._peeked is None:
            self._peeked = self._scan()
        return self._peeked

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        self._peeked = None
        return tok

    def _scan(self) -> tuple[str, str]:
        while True:
            if self.pos >= len(self.text):
                return ("eof", "")
            m = _TOKEN.match(self.text, self.pos)
            if m is None:
                line = self.text.count("\n", 0, self.pos) + 1
                raise MeshportError(f"This USD text file has something unexpected on line {line}")
            self.pos = m.end()
            if m.lastgroup != "skip":
                return (m.lastgroup, m.group())

    def expect(self, text: str) -> None:
        _, got = self.next()
        if got != text:
            line = self.text.count("\n", 0, self.pos) + 1
            raise MeshportError(f"This USD text file is damaged near line {line} (expected {text!r}, found {got!r})")

    def accept(self, text: str) -> bool:
        if self.peek()[1] == text:
            self.next()
            return True
        return False

    def raw_until(self, close: str) -> str:
        """The raw text up to the next `close` (for a numeric array), consuming it."""
        assert self._peeked is None
        end = self.text.find(close, self.pos)
        if end < 0:
            raise MeshportError("This USD text file is cut short (an array never ends)")
        out = self.text[self.pos:end]
        self.pos = end + 1
        return out


def _unquote(s: str) -> str:
    body = s[3:-3] if s[:3] in ('"""', "'''") else s[1:-1]
    if "\\" not in body:
        return body
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "r": "\r"}.get(m.group(1), m.group(1)), body)


def _asset(s: str) -> str:
    return s[3:-3] if s.startswith("@@@") else s[1:-1]


class _Parser:
    def __init__(self, text: str):
        if not text.lstrip("\ufeff").startswith("#usda"):
            raise MeshportError("This isn't a USD text file")
        self.lex = _Lexer(text.lstrip("\ufeff"))
        self.layer = Layer()

    def parse(self) -> Layer:
        root = Spec(SPEC_PSEUDO_ROOT, {})
        self.layer.specs["/"] = root
        if self.lex.accept("("):
            self._metadata(root.fields, ")")
        children = []
        while self.lex.peek()[0] != "eof":
            children.append(self._prim("/"))
        root.fields["primChildren"] = children
        return self.layer

    # -- prims

    def _prim(self, parent: str) -> str:
        _, spec_word = self.lex.next()
        if spec_word not in ("def", "over", "class"):
            raise MeshportError(f"This USD text file is damaged (expected a prim, found {spec_word!r})")
        type_name = ""
        if self.lex.peek()[0] == "word":
            type_name = self.lex.next()[1]
        name = self._string()
        path = child_path(parent, name)
        spec = Spec(SPEC_PRIM, {"specifier": spec_word})
        if type_name:
            spec.fields["typeName"] = type_name
        self.layer.specs[path] = spec
        if self.lex.accept("("):
            self._metadata(spec.fields, ")")
        self.lex.expect("{")
        self._body(path, spec)
        return name

    def _body(self, path: str, spec: Spec) -> None:
        children, properties = [], {}  # (properties in order, each once)
        while not self.lex.accept("}"):
            kind, word = self.lex.peek()
            if kind == "eof":
                raise MeshportError("This USD text file is cut short (a prim never ends)")
            if word in ("def", "over", "class"):
                children.append(self._prim(path))
            elif word == "variantSet":
                self._variant_set(path, spec)
            elif word == "reorder":
                self.lex.next()
                self.lex.next()  # nameChildren / properties
                self.lex.expect("=")
                self._value()
            elif word == ";":
                self.lex.next()
            else:
                properties[self._property(path)] = None
        if children:
            spec.fields["primChildren"] = children
        if properties:
            spec.fields["properties"] = list(properties)

    def _variant_set(self, path: str, spec: Spec) -> None:
        self.lex.next()
        set_name = self._string()
        self.lex.expect("=")
        self.lex.expect("{")
        names = spec.fields.setdefault("variantSetNames", ListOp())
        if set_name not in names.items():
            names.prepended.append(set_name)
        while not self.lex.accept("}"):
            variant = self._string()
            vpath = f"{path}{{{set_name}={variant}}}"
            vspec = Spec(SPEC_VARIANT, {})
            self.layer.specs[vpath] = vspec
            if self.lex.accept("("):
                self._metadata(vspec.fields, ")")
            self.lex.expect("{")
            self._body(vpath, vspec)

    # -- properties

    def _property(self, prim: str) -> str:
        listop = None
        custom = variability = None
        while True:
            word = self.lex.peek()[1]
            if word in _LISTOP_WORDS:
                listop = _LISTOP_WORDS[self.lex.next()[1]]
            elif word == "custom":
                custom = self.lex.next() and True
            elif word in ("uniform", "varying", "config"):
                variability = self.lex.next()[1]
            else:
                break
        kind, type_name = self.lex.next()
        if kind != "word":
            raise MeshportError(f"This USD text file is damaged (unexpected {type_name!r})")
        if type_name == "rel":
            return self._relationship(prim, listop)
        if self.lex.accept("["):  # 'float []' with a space
            self.lex.expect("]")
            type_name += "[]"
        name = self.lex.next()[1]
        suffix = None
        for s in (".connect", ".timeSamples"):
            if name.endswith(s):
                name, suffix = name[:-len(s)], s
        path = property_path(prim, name)
        spec = self.layer.specs.get(path)
        if spec is None:
            spec = self.layer.specs[path] = Spec(SPEC_ATTRIBUTE, {"typeName": type_name})
        if custom:
            spec.fields["custom"] = True
        if variability == "uniform":
            spec.fields["variability"] = "uniform"
        if self.lex.accept("="):
            if suffix == ".connect":
                targets = self._paths()
                op = spec.fields.setdefault("connectionPaths", ListOp(explicit=listop is None))
                getattr(op, "explicit_items" if listop is None else listop).extend(targets)
            elif suffix == ".timeSamples":
                spec.fields["timeSamples"] = self._time_samples(type_name)
            else:
                spec.fields["default"] = self._typed(type_name)
        if self.lex.accept("("):
            self._metadata(spec.fields, ")")
        return name

    def _relationship(self, prim: str, listop: str | None) -> str:
        name = self.lex.next()[1]
        path = property_path(prim, name)
        spec = self.layer.specs.get(path) or Spec(SPEC_RELATIONSHIP, {})
        self.layer.specs[path] = spec
        if self.lex.accept("="):
            targets = self._paths()
            op = spec.fields.setdefault("targetPaths", ListOp(explicit=listop is None))
            getattr(op, "explicit_items" if listop is None else listop).extend(targets)
        if self.lex.accept("("):
            self._metadata(spec.fields, ")")
        return name

    def _paths(self) -> list[str]:
        v = self._value()
        if v is None:
            return []
        return [p for p in (v if isinstance(v, list) else [v]) if isinstance(p, _Path)]

    def _time_samples(self, type_name: str) -> dict:
        self.lex.expect("{")
        out = {}
        while not self.lex.accept("}"):
            t = float(self.lex.next()[1])
            self.lex.expect(":")
            out[t] = self._typed(type_name)
            self.lex.accept(",")
        return out

    # -- metadata

    def _metadata(self, fields: dict, close: str) -> None:
        while not self.lex.accept(close):
            kind, word = self.lex.next()
            if kind == "eof":
                raise MeshportError("This USD text file is cut short (metadata never ends)")
            if kind in ("str", "str3"):
                fields["documentation"] = _unquote(word)
                continue
            if word == ";":
                continue
            listop = None
            if word in _LISTOP_WORDS:
                listop = _LISTOP_WORDS[word]
                word = self.lex.next()[1]
            key = _FIELD_NAMES.get(word, word)
            if not self.lex.accept("="):
                continue
            value = self._value()
            if key in ("references", "payload", "inheritPaths", "specializes", "apiSchemas",
                       "variantSetNames", "subLayers"):
                items = value if isinstance(value, list) else ([] if value is None else [value])
                if key == "subLayers":
                    fields[key] = [str(i) for i in items]
                    fields["subLayerOffsets"] = [(getattr(i, "offset", 0.0), getattr(i, "scale", 1.0)) for i in items]
                    continue
                if key in ("references", "payload"):
                    items = [self._arc(i, key) for i in items]
                op = fields.get(key) if isinstance(fields.get(key), ListOp) else ListOp(explicit=listop is None)
                getattr(op, "explicit_items" if listop is None else listop).extend(items)
                fields[key] = op
            elif key == "variantSelection" and isinstance(value, dict):
                fields[key] = {k: str(v) for k, v in value.items()}
            else:
                fields[key] = value

    @staticmethod
    def _arc(item, key: str):
        cls = Reference if key == "references" else Payload
        timing = (getattr(item, "offset", 0.0), getattr(item, "scale", 1.0))
        if isinstance(item, _ArcValue):
            return cls(item.asset, item.prim, *timing)
        if isinstance(item, _Path):
            return cls("", str(item), *timing)
        return cls(str(item), "", *timing)

    # -- values

    def _string(self) -> str:
        kind, s = self.lex.next()
        if kind not in ("str", "str3"):
            raise MeshportError(f"This USD text file is damaged (expected a name, found {s!r})")
        return _unquote(s)

    def _typed(self, type_name: str):
        """A value of a declared attribute type; numeric arrays come back as NumPy arrays."""
        num = numeric_type(type_name)
        is_array = type_name.endswith("[]")
        if is_array and num is not None:
            if self.lex.peek()[1] == "None":
                self.lex.next()
                return None
            self.lex.expect("[")
            raw = self.lex.raw_until("]")
            flat = raw.replace("(", " ").replace(")", " ").replace(",", " ").split()
            dtype, comps = num
            try:
                arr = np.array(flat, dtype=np.float64)
            except ValueError:
                raise MeshportError("This USD text file has a damaged number list") from None
            arr = arr.astype(dtype)
            if comps > 1:
                if len(arr) % comps:
                    raise MeshportError("This USD text file has a damaged list of tuples")
                arr = arr.reshape(-1, comps)
            return arr
        v = self._value()
        if num is not None and v is not None:
            dtype, comps = num
            if isinstance(v, str):  # 'inf', 'nan', 'true'
                v = {"true": 1, "false": 0}.get(v, v)
            arr = np.array(v, dtype=np.float64).ravel()
            if comps == 1:
                return bool(arr[0]) if type_name == "bool" else (int(arr[0]) if dtype[0] in "iu" else float(arr[0]))
            return arr.reshape(4, 4) if comps == 16 else (arr.reshape(3, 3) if comps == 9 else arr)
        if type_name.removesuffix("[]") in _STRINGY and isinstance(v, (_Path, _ArcValue)):
            return str(v)
        return v

    def _value(self):
        kind, tok = self.lex.next()
        if kind == "num":
            return float(tok) if any(c in tok for c in ".eE") else int(tok)
        if kind in ("str", "str3"):
            return _unquote(tok)
        if kind == "asset":
            asset = _AssetValue(_asset(tok))
            if self.lex.peek()[0] == "path":
                asset = _ArcValue(asset, self.lex.next()[1][1:-1])
            if self.lex.peek()[1] == "(":  # a layer offset (kept) or custom data
                self.lex.next()
                _set_offset(asset, self._layer_offset())
            return asset
        if kind == "path":
            p = _Path(tok[1:-1])
            if self.lex.peek()[1] == "(":
                self.lex.next()
                _set_offset(p, self._layer_offset())
            return p
        if tok == "(":
            items = []
            while not self.lex.accept(")"):
                items.append(self._value())
                self.lex.accept(",")
            return tuple(items)
        if tok == "[":
            items = []
            while not self.lex.accept("]"):
                items.append(self._value())
                self.lex.accept(",")
            return items
        if tok == "{":
            return self._dictionary()
        if tok == "-":  # '- inf'
            v = self._value()
            return -v if isinstance(v, (int, float)) else v
        if kind == "word":
            if tok in ("None", "none"):
                return None
            if tok in ("true", "false"):
                return tok == "true"
            if tok in ("inf", "nan"):
                return float(tok)
            return tok
        raise MeshportError(f"This USD text file is damaged (unexpected {tok!r})")

    def _layer_offset(self) -> tuple[float, float]:
        """(offset = 10; scale = 2) after an asset or path, already past its "("."""
        md: dict = {}
        self._metadata(md, ")")
        num = (lambda v, d: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else d)
        return num(md.get("offset"), 0.0), num(md.get("scale"), 1.0)

    def _dictionary(self) -> dict:
        out = {}
        while not self.lex.accept("}"):
            kind, tok = self.lex.next()
            if tok in (";", ","):
                continue
            if kind == "eof":
                raise MeshportError("This USD text file is cut short (a dictionary never ends)")
            if kind == "word" and self.lex.peek()[1] != "=":  # a type, then the key
                if self.lex.accept("["):
                    self.lex.expect("]")
                kind, tok = self.lex.next()
            key = _unquote(tok) if kind in ("str", "str3") else tok
            if self.lex.accept(":"):  # time samples written as a dictionary
                pass
            else:
                self.lex.expect("=")
            out[key] = self._value()
        return out


def _set_offset(value, timing: tuple[float, float]) -> None:
    value.offset, value.scale = timing


class _Path(str):
    """A <path> value."""


class _AssetValue(str):
    """An @asset@ value."""


class _ArcValue(str):
    """@asset@</prim>: a reference or payload."""

    def __new__(cls, asset: str, prim: str):
        obj = super().__new__(cls, f"{asset}{prim}")
        obj.asset, obj.prim = str(asset), prim
        return obj


def read(data: bytes) -> Layer:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    try:
        return _Parser(text).parse()
    except MeshportError:
        raise
    except (ValueError, IndexError, KeyError, TypeError, OverflowError, RecursionError) as e:
        raise MeshportError(f"This USD text file is damaged ({type(e).__name__}: {e})") from None


# ============================================================================ writing

def _num(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if v in (float("inf"), float("-inf")):
        return "inf" if v > 0 else "-inf"
    r = repr(float(v))
    return r.removesuffix(".0")


def _fmt_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.dtype.kind in "iub":
        flat = [str(int(x)) for x in arr.ravel()]
    else:
        flat = [_num(float(x)) for x in arr.astype(np.float32 if arr.dtype == np.float32 else np.float64).ravel()]
    if arr.ndim == 1:
        return "[" + ", ".join(flat) + "]"
    n = arr.shape[1]
    return "[" + ", ".join("(" + ", ".join(flat[i:i + n]) + ")" for i in range(0, len(flat), n)) + "]"


def _fmt_scalar(type_name: str, v) -> str:
    base = type_name.removesuffix("[]")
    if type_name.endswith("[]"):
        if base in _STRINGY:
            q = (lambda s: f"@{s}@") if base == "asset" else _quote
            return "[" + ", ".join(q(s) for s in v) + "]"
        if base.startswith("matrix"):  # each matrix a tuple of rows
            d = int(base[6])
            return "[" + ", ".join(_fmt_scalar(base, m) for m in np.asarray(v, np.float64).reshape(-1, d, d)) + "]"
        return _fmt_array(np.asarray(v))
    if base in ("token", "string"):
        return _quote(v)
    if base == "asset":
        return f"@{v}@"
    if base == "bool":
        return "true" if v else "false"
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return str(int(v)) if base in ("int", "uint", "int64", "uint64", "uchar") else _num(float(v))
    if arr.ndim == 2:
        return "(" + ", ".join("(" + ", ".join(_num(x) for x in row) + ")" for row in arr) + ")"
    return "(" + ", ".join(_num(x) for x in arr) + ")"


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def write(layer: Layer) -> bytes:
    """Serialize a layer whose specs carry `fields` as written by meshport's scene writer."""
    out = ["#usda 1.0"]
    root = layer.specs["/"].fields
    meta = []
    for key in ("defaultPrim", "metersPerUnit", "upAxis", "documentation", "timeCodesPerSecond", "framesPerSecond",
                "startTimeCode", "endTimeCode"):
        if key in root:
            v = root[key]
            meta.append(f"    {'doc' if key == 'documentation' else key} = "
                        f"{_num(v) if isinstance(v, float) else _quote(str(v))}")
    if meta:
        out += ["(", *meta, ")"]
    out.append("")
    for name in root.get("primChildren", []):
        _write_prim(layer, child_path("/", name), out, 0)
        out.append("")
    return "\n".join(out).encode("utf-8")


def _write_prim(layer: Layer, top: str, out: list[str], depth: int) -> None:
    """The prim and everything under it. A loop rather than recursion, so no depth of nesting can
    overflow the stack: todo holds prims still to write and lines to put out after them."""
    todo: list[tuple[str, int] | str] = [(top, depth)]
    while todo:
        item = todo.pop()
        if isinstance(item, str):
            out.append(item)
            continue
        path, depth = item
        spec = layer.specs[path]
        f = spec.fields
        pad = "    " * depth
        type_name = f" {f['typeName']}" if f.get("typeName") else ""
        head = f"{pad}{f.get('specifier', 'def')}{type_name} {_quote(path.rsplit('/', 1)[-1])}"
        meta = []
        if "apiSchemas" in f:
            meta.append(f'{pad}    prepend apiSchemas = [{", ".join(_quote(s) for s in f["apiSchemas"].items())}]')
        if "kind" in f:
            meta.append(f"{pad}    kind = {_quote(f['kind'])}")
        out.append(head + (" (" if meta else ""))
        if meta:
            out += meta
            out.append(f"{pad})")
        out.append(f"{pad}{{")
        for name in f.get("properties", []):
            _write_property(layer, property_path(path, name), name, out, depth + 1)
        todo.append(f"{pad}}}")
        for i, name in reversed(list(enumerate(f.get("primChildren", [])))):
            todo.append((child_path(path, name), depth + 1))
            if i or f.get("properties"):
                todo.append("")


def _write_property(layer: Layer, path: str, name: str, out: list[str], depth: int) -> None:
    spec = layer.specs[path]
    f = spec.fields
    pad = "    " * depth
    uniform = "uniform " if f.get("variability") == "uniform" else ""
    if spec.kind == SPEC_RELATIONSHIP:
        targets = f.get("targetPaths", ListOp()).items()
        value = f"<{targets[0]}>" if len(targets) == 1 else "[" + ", ".join(f"<{t}>" for t in targets) + "]"
        out.append(f"{pad}rel {name} = {value}")
        return
    type_name = f["typeName"]
    meta = [f"{k} = {_quote(f[k])}" for k in ("interpolation",) if k in f]
    if "elementSize" in f:
        meta.append(f"elementSize = {int(f['elementSize'])}")
    tail = (" (\n" + "".join(f"{pad}    {m}\n" for m in meta) + f"{pad})") if meta else ""
    if "connectionPaths" in f:
        targets = f["connectionPaths"].items()
        out.append(f"{pad}{uniform}{type_name} {name}.connect = <{targets[0]}>")
    if "timeSamples" in f:
        out.append(f"{pad}{uniform}{type_name} {name}.timeSamples = {{")
        for t, v in sorted(f["timeSamples"].items()):
            out.append(f"{pad}    {_num(float(t))}: {_fmt_scalar(type_name, v)},")
        out.append(f"{pad}}}{tail}")
    elif "default" in f:
        out.append(f"{pad}{uniform}{type_name} {name} = {_fmt_scalar(type_name, f['default'])}{tail}")
    elif "connectionPaths" not in f:
        out.append(f"{pad}{uniform}{type_name} {name}{tail}")
