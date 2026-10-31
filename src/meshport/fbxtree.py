"""FBX as a tree of records, binary and ASCII, with no meaning attached yet.

Every FBX file, either flavor, is a list of records. A record has a name, a list of
property values and child records. fbx.py turns that tree into a Scene and back.

Value types when reading: int, float, bool, str, bytes, or a numpy array for array
properties. When writing, the property type is taken from the Python type, so pick it on
purpose:

    bool -> C    int -> I (int32)    np.int64 -> L    np.int16 -> Y
    float -> D   np.float32 -> F     str -> S         bytes -> R
    ndarray -> f/d/i/l/b by dtype (float32, float64, int32, int64, bool)
"""

from __future__ import annotations

import base64
import re
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from .scene import MeshportError

BINARY_MAGIC = b"Kaydara FBX Binary  \x00"


@dataclass(eq=False)
class Record:
    name: str
    props: list = field(default_factory=list)
    children: list[Record] = field(default_factory=list)

    def find(self, name: str) -> Record | None:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name: str) -> list[Record]:
        return [c for c in self.children if c.name == name]

    def value(self, name: str, default=None):
        """props[0] of the first child called name."""
        c = self.find(name)
        return c.props[0] if c is not None and c.props else default

    def array(self, name: str, dtype=np.float64) -> np.ndarray | None:
        c = self.find(name)
        return None if c is None else as_array(c.props, dtype)


def as_array(props: list, dtype=np.float64) -> np.ndarray:
    """A record's values as one array: binary/long ASCII arrays are one ndarray already,
    short ASCII lists come as separate numbers."""
    if len(props) == 1 and isinstance(props[0], np.ndarray):
        return props[0].astype(dtype, copy=False)
    vals = [p for p in props if isinstance(p, (int, float, np.ndarray))]
    if any(isinstance(v, np.ndarray) for v in vals):
        return np.concatenate([np.atleast_1d(v) for v in vals]).astype(dtype)
    return np.array(vals, dtype=dtype)


def is_binary(data: bytes) -> bool:
    return data[:21] == BINARY_MAGIC


def parse(data: bytes) -> tuple[int, list[Record]]:
    """(version, top-level records) for a binary or ASCII FBX file."""
    if is_binary(data) or BINARY_MAGIC.startswith(data[:21]) and len(data) >= 7:
        return _read_binary(data)
    head = data[:1024].lstrip()
    if head.startswith((b";", b"FBXHeaderExtension", b"\xef\xbb\xbf")) or b"FBXHeaderExtension" in data[:4096]:
        return _read_ascii(data.decode("utf-8", "replace"))
    raise MeshportError("This isn't an FBX file")


# ---- binary

class _Layout:
    """Struct formats for one byte order (byte 22 of the header is 1 in big-endian files)."""

    def __init__(self, endian: str, wide: bool):
        self.head = struct.Struct(endian + ("QQQB" if wide else "IIIB"))
        self.u32 = struct.Struct(endian + "I")
        self.array_head = struct.Struct(endian + "III")
        self.scalar = {ord(k): struct.Struct(endian + v) for k, v in
                       {"Y": "h", "C": "?", "I": "i", "F": "f", "D": "d", "L": "q"}.items()}
        self.array = {ord(k): np.dtype(endian + v) for k, v in
                      {"f": "f4", "d": "f8", "l": "i8", "i": "i4", "b": "u1"}.items()}


_U32 = struct.Struct("<I")
_ARRAY_HEAD = struct.Struct("<III")


def _read_binary(data: bytes) -> tuple[int, list[Record]]:
    if len(data) < 27:
        raise MeshportError("This FBX file is cut short")
    endian = ">" if data[22] == 1 else "<"
    version = struct.unpack_from(endian + "I", data, 23)[0]
    layout = _Layout(endian, version >= 7500)
    records, pos = [], 27
    try:
        while pos + layout.head.size <= len(data):
            rec, pos = _read_record(data, pos, layout)
            if rec is None:
                break
            records.append(rec)
    except (struct.error, zlib.error, IndexError, ValueError) as e:
        raise MeshportError(f"This FBX file is damaged ({e})") from None
    return version, records


def _read_record(data: bytes, pos: int, layout: _Layout):
    start = pos
    end, count, _, name_len = layout.head.unpack_from(data, pos)
    pos += layout.head.size
    if end == 0:
        return None, pos
    if end > len(data):
        raise ValueError("record runs past the end of the file")
    if end <= start:  # reading would go back over what it has read, forever
        raise ValueError("a record ends before it starts")
    name = data[pos:pos + name_len].decode("ascii", "replace")
    pos += name_len
    props = []
    for _ in range(count):
        t = data[pos]
        pos += 1
        if t in layout.scalar:
            s = layout.scalar[t]
            props.append(s.unpack_from(data, pos)[0])
            pos += s.size
        elif t in layout.array:
            n, encoding, size = layout.array_head.unpack_from(data, pos)
            pos += layout.array_head.size
            raw = data[pos:pos + size]
            pos += size
            if encoding == 1:  # never inflate past the declared size: a tiny file can't claim gigabytes
                raw = zlib.decompressobj().decompress(raw, n * layout.array[t].itemsize)
            arr = np.frombuffer(raw, dtype=layout.array[t], count=n)
            props.append(arr.astype(bool) if t == ord("b") else arr)
        elif t in (ord("S"), ord("R")):
            n = layout.u32.unpack_from(data, pos)[0]
            pos += 4
            raw = data[pos:pos + n]
            pos += n
            props.append(raw.decode("utf-8", "replace") if t == ord("S") else bytes(raw))
        else:
            raise ValueError(f"unknown property type {chr(t)!r}")
    children = []
    while pos < end:
        child, pos = _read_record(data, pos, layout)
        if child is None:
            break
        children.append(child)
    return Record(name, props, children), end


# Fixed values every FBX writer uses together; readers check that they match.
FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
CREATION_TIME = "1970-01-01 10:00:00:000"
_FOOTER_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOTER_END = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"


def write_binary(records: list[Record], version: int = 7400) -> bytes:
    head = struct.Struct("<QQQB" if version >= 7500 else "<IIIB")
    out = bytearray(BINARY_MAGIC + b"\x1a\x00" + _U32.pack(version))
    for rec in records:
        _write_record(out, rec, head)
    out += bytes(head.size)  # end of the top-level list
    out += _FOOTER_ID + bytes(4)
    pad = (16 - len(out) % 16) or 16
    out += bytes(pad) + _U32.pack(version) + bytes(120) + _FOOTER_END
    return bytes(out)


def _write_record(out: bytearray, rec: Record, head: struct.Struct) -> None:
    """Write rec and everything under it. A loop rather than recursion, so no depth of nesting
    can overflow the stack (Python 3.10 on Windows crashes outright at a few thousand levels)."""
    todo: list[tuple[Record, bool]] = [(rec, False)]
    open_records: list[tuple[int, int]] = []  # (header offset, property bytes) of records not yet closed
    while todo:
        r, closing = todo.pop()
        if closing:
            start, props_len = open_records.pop()
            if r.children or not r.props:
                out += bytes(head.size)
            head.pack_into(out, start, len(out), len(r.props), props_len, len(r.name.encode("ascii")))
            continue
        start = len(out)
        out += bytes(head.size) + r.name.encode("ascii")
        props_start = len(out)
        for p in r.props:
            _write_prop(out, p)
        open_records.append((start, len(out) - props_start))
        todo.append((r, True))
        todo.extend((child, False) for child in reversed(r.children))


_ARRAY_CODE = {np.dtype("float32"): b"f", np.dtype("float64"): b"d", np.dtype("int64"): b"l",
               np.dtype("int32"): b"i", np.dtype("bool"): b"b"}


def _write_prop(out: bytearray, p) -> None:
    if isinstance(p, (bool, np.bool_)):
        out += b"C" + struct.pack("<?", bool(p))
    elif isinstance(p, np.int16):
        out += b"Y" + struct.pack("<h", int(p))
    elif isinstance(p, np.int64):
        out += b"L" + struct.pack("<q", int(p))
    elif isinstance(p, int):
        out += b"I" + struct.pack("<i", p)
    elif isinstance(p, np.float32):
        out += b"F" + struct.pack("<f", float(p))
    elif isinstance(p, float):
        out += b"D" + struct.pack("<d", p)
    elif isinstance(p, str):
        raw = p.encode("utf-8")
        out += b"S" + _U32.pack(len(raw)) + raw
    elif isinstance(p, (bytes, bytearray)):
        out += b"R" + _U32.pack(len(p)) + p
    elif isinstance(p, np.ndarray):
        code = _ARRAY_CODE.get(p.dtype)
        if code is None:
            raise TypeError(f"can't write a {p.dtype} array")
        raw = np.ascontiguousarray(p, dtype=p.dtype.newbyteorder("<")).tobytes()
        encoding = 0
        if len(raw) > 128:
            raw, encoding = zlib.compress(raw, 1), 1
        out += code + _ARRAY_HEAD.pack(len(p), encoding, len(raw)) + raw
    else:
        raise TypeError(f"can't write {type(p).__name__} as an FBX property")


# ---- ASCII

_TOKEN = re.compile(r"""
    (?P<skip>\s+|;[^\n]*)
  | (?P<key>[A-Za-z_][\w|.\-]*)[ \t]*:
  | "(?P<str>[^"]*)"
  | (?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
  | \*(?P<count>\d+)
  | (?P<open>\{) | (?P<close>\}) | (?P<comma>,)
  | (?P<word>[^\s,{}";]+)
""", re.VERBOSE)
_VALUES_END = re.compile(r"[{}]|\n[ \t]*[A-Za-z_][\w|.\-]*[ \t]*:")
_NUMERIC = re.compile(r"[\s\d.,eE+\-]*")
_INT = re.compile(r"[-+]?\d+")


class _Count(int):
    """The *N before an ASCII array block."""


def _read_ascii(text: str) -> tuple[int, list[Record]]:
    version = 0
    m = re.search(r"FBXVersion:\s*(\d+)", text[:4096])
    if m:
        version = int(m.group(1))
    else:
        m = re.search(r"FBX (\d+)\.(\d+)\.(\d+)", text[:200])
        if m:
            version = int(m.group(1)) * 1000 + int(m.group(2)) * 100
    root = Record("")
    stack = [root]
    cur: Record | None = None
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if m is None:
            raise MeshportError(f"This FBX file is damaged (unexpected text at character {pos})")
        pos = m.end()
        kind = m.lastgroup
        if kind == "skip" or kind == "comma":
            continue
        if kind == "key":
            cur = Record(m.group("key"))
            stack[-1].children.append(cur)
            end = _VALUES_END.search(text, pos)
            end = end.start() if end else n
            if end - pos > 256 and _NUMERIC.fullmatch(text, pos, end):
                cur.props.append(_numbers(text[pos:end]))
                pos = end
        elif kind == "open":
            if cur is None:
                raise MeshportError("This FBX file is damaged (a block without a name)")
            stack.append(cur)
            cur = None
        elif kind == "close":
            if len(stack) == 1:
                raise MeshportError("This FBX file is damaged (too many closing braces)")
            done = stack.pop()
            if done.props and isinstance(done.props[0], _Count) and done.find("a") is not None:
                done.props = [as_array(done.find("a").props)]
                done.children = []
            cur = None
        elif cur is None:
            continue  # stray value outside a record: ignore
        elif kind == "str":
            cur.props.append(m.group("str").replace("&quot;", '"'))
        elif kind == "num":
            s = m.group("num")
            cur.props.append(int(s) if _INT.fullmatch(s) else float(s))
        elif kind == "count":
            cur.props.append(_Count(m.group("count")))
        else:
            cur.props.append(m.group("word"))
    return version, root.children


def _numbers(chunk: str) -> np.ndarray:
    chunk = chunk.strip().strip(",")
    if not chunk:
        return np.zeros(0)
    try:
        return np.array(chunk.replace("\n", "").replace("\r", "").split(","), dtype=np.float64)
    except ValueError as e:
        raise MeshportError(f"This FBX file is damaged ({e})") from None


def write_ascii(records: list[Record], version: int = 7400) -> str:
    """Readable dump, mainly for debugging and tests."""
    major, minor = divmod(version, 1000)
    lines = [f"; FBX {major}.{minor // 100}.0 project file", ""]

    def fmt(p):
        if isinstance(p, (bool, np.bool_)):
            return "T" if p else "F"
        if isinstance(p, str):
            if "\x00\x01" in p:
                name, cls = p.split("\x00\x01", 1)
                p = f"{cls}::{name}"
            return '"' + p.replace('"', "&quot;") + '"'
        if isinstance(p, (bytes, bytearray)):
            return '"' + base64.b64encode(p).decode() + '"'
        if isinstance(p, (float, np.floating)):
            return repr(float(p))
        return str(int(p))

    def emit(rec: Record, depth: int):
        pad = "\t" * depth
        arrays = [p for p in rec.props if isinstance(p, np.ndarray)]
        if arrays:
            arr = arrays[0]
            vals = ",".join(fmt(v) for v in (arr.tolist()))
            lines.append(f"{pad}{rec.name}: *{len(arr)} {{")
            lines.append(f"{pad}\ta: {vals}")
            lines.append(f"{pad}}}")
            return
        head = f"{pad}{rec.name}: " + ", ".join(fmt(p) for p in rec.props)
        if rec.children:
            lines.append(head.rstrip() + " {")
            for c in rec.children:
                emit(c, depth + 1)
            lines.append(f"{pad}}}")
        else:
            lines.append(head.rstrip())

    for rec in records:
        emit(rec, 0)
    return "\n".join(lines) + "\n"


def decode_content(props: list) -> bytes:
    """An embedded file (Video/Content): raw bytes in binary FBX, base64 text in ASCII."""
    for p in props:
        if isinstance(p, (bytes, bytearray)) and p:
            return bytes(p)
    text = "".join(p for p in props if isinstance(p, str))
    if not text:
        return b""
    try:
        return base64.b64decode(text)
    except ValueError:
        return b""
