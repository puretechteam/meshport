"""USD's binary layer format ("Crate", .usdc): bytes <-> Layer.

A crate file is a bootstrap header, a table of contents and six sections: TOKENS (every name
and string, LZ4-compressed), STRINGS (indices into the tokens), FIELDS (name + value pairs),
FIELDSETS (each spec's list of fields), PATHS (a compressed tree of every path) and SPECS (path,
fields and kind of each spec). Values are 64-bit "reps": a type, flags, and either the value
itself (small ones) or the file offset of its data. Integer tables are delta-encoded with
2-bit size codes, then LZ4-compressed ("Usd_IntegerCompression").

Written from the format as Pixar's open-source OpenUSD library (Apache 2.0) reads and writes it,
and checked against files written by that library.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter

import numpy as np

from .scene import Budget, MeshportError
from .usdlayer import (
    SPEC_ATTRIBUTE,
    SPEC_PRIM,
    SPEC_PSEUDO_ROOT,
    SPEC_RELATIONSHIP,
    Layer,
    ListOp,
    Payload,
    Reference,
    Spec,
    child_path,
    element_name,
    is_property,
    parent_path,
    property_path,
)

MAGIC = b"PXR-USDC"
WRITE_VERSION = (0, 8, 0)  # read by every USD release since 2020 (Blender, Apple, NVIDIA)

# value types (crateDataTypes.h)
(BOOL, UCHAR, INT, UINT, INT64, UINT64, HALF, FLOAT, DOUBLE, STRING, TOKEN, ASSET, MATRIX2D, MATRIX3D,
 MATRIX4D, QUATD, QUATF, QUATH, VEC2D, VEC2F, VEC2H, VEC2I, VEC3D, VEC3F, VEC3H, VEC3I, VEC4D, VEC4F,
 VEC4H, VEC4I, DICTIONARY, TOKEN_LISTOP, STRING_LISTOP, PATH_LISTOP, REFERENCE_LISTOP, INT_LISTOP,
 INT64_LISTOP, UINT_LISTOP, UINT64_LISTOP, PATH_VECTOR, TOKEN_VECTOR, SPECIFIER, PERMISSION, VARIABILITY,
 VARIANT_SELECTION_MAP, TIME_SAMPLES, PAYLOAD, DOUBLE_VECTOR, LAYER_OFFSET_VECTOR, STRING_VECTOR,
 VALUE_BLOCK, VALUE, UNREGISTERED_VALUE, UNREGISTERED_VALUE_LISTOP, PAYLOAD_LISTOP, TIME_CODE) = range(1, 57)

# element dtype and component count of numeric types
_NUMERIC = {
    BOOL: ("u1", 1), UCHAR: ("u1", 1), INT: ("<i4", 1), UINT: ("<u4", 1), INT64: ("<i8", 1), UINT64: ("<u8", 1),
    HALF: ("<f2", 1), FLOAT: ("<f4", 1), DOUBLE: ("<f8", 1), TIME_CODE: ("<f8", 1),
    MATRIX2D: ("<f8", 4), MATRIX3D: ("<f8", 9), MATRIX4D: ("<f8", 16),
    QUATD: ("<f8", 4), QUATF: ("<f4", 4), QUATH: ("<f2", 4),
    VEC2D: ("<f8", 2), VEC2F: ("<f4", 2), VEC2H: ("<f2", 2), VEC2I: ("<i4", 2),
    VEC3D: ("<f8", 3), VEC3F: ("<f4", 3), VEC3H: ("<f2", 3), VEC3I: ("<i4", 3),
    VEC4D: ("<f8", 4), VEC4F: ("<f4", 4), VEC4H: ("<f2", 4), VEC4I: ("<i4", 4),
}
_INTS = {INT: 4, UINT: 4, INT64: 8, UINT64: 8}
_INT_LISTOPS = {INT_LISTOP: "<i", INT64_LISTOP: "<q", UINT_LISTOP: "<I", UINT64_LISTOP: "<Q"}
_QUATS = {QUATD, QUATF, QUATH}  # stored (x, y, z, w); kept as (w, x, y, z), the way .usda writes them
_FLOATS = {HALF, FLOAT, DOUBLE}
_SPECIFIERS = ["def", "over", "class"]
_VARIABILITY = ["varying", "uniform"]
_MIN_COMPRESSED = 16  # arrays shorter than this are never compressed

_IS_ARRAY, _IS_INLINED, _IS_COMPRESSED = 1 << 63, 1 << 62, 1 << 61
_PAYLOAD_MASK = (1 << 48) - 1
LZ4_CHUNK = 0x7E000000  # LZ4's largest block (LZ4_MAX_INPUT_SIZE): bigger data is split, as USD splits it
DECODE_BUDGET = 256 << 20  # bytes of values one file may decode to, plus 16 times its own size


# ============================================================================ LZ4 and integer coding

def lz4_block_decompress(src: bytes, max_size: int) -> bytes:
    """Decode one LZ4 block into at most max_size bytes (USD often only knows an upper bound).
    LZ4 can't expand data more than 255 times, which caps what a damaged file can claim."""
    max_size = min(max_size, len(src) * 255 + 16)
    try:
        import lz4.block
        return lz4.block.decompress(src, uncompressed_size=max_size)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001 - lz4 raises its own error type
        raise MeshportError(f"This USD file is damaged (bad compressed data: {e})") from None
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        if len(out) + lit > max_size:
            raise MeshportError("This USD file is damaged (compressed data is too long)")
        out += src[i:i + lit]
        i += lit
        if i >= n:
            break
        offset = src[i] | src[i + 1] << 8
        i += 2
        length = token & 15
        if length == 15:
            while True:
                b = src[i]
                i += 1
                length += b
                if b != 255:
                    break
        length += 4
        if offset == 0 or offset > len(out) or len(out) + length > max_size:
            raise MeshportError("This USD file is damaged (bad compressed data)")
        start = len(out) - offset
        if offset >= length:
            out += out[start:start + length]
        else:  # the match overlaps what it copies: a repeating pattern
            pattern = bytes(out[start:])
            out += (pattern * (length // offset + 1))[:length]
    if len(out) > max_size:
        raise MeshportError("This USD file is damaged (compressed data is too long)")
    return bytes(out)


def fast_decompress(src: bytes, max_size: int) -> bytes:
    """USD's TfFastCompression: a chunk count byte, then one LZ4 block (or several chunks)."""
    if not src:
        raise MeshportError("This USD file is damaged (empty compressed data)")
    chunks = src[0]
    if chunks == 0:
        return lz4_block_decompress(src[1:], max_size)
    out, pos = bytearray(), 1
    for _ in range(chunks):
        n = int.from_bytes(src[pos:pos + 4], "little", signed=True)
        pos += 4
        if n < 0 or pos + n > len(src):
            raise MeshportError("This USD file is damaged (a compressed chunk is cut short)")
        out += lz4_block_decompress(src[pos:pos + n], min(max_size - len(out), LZ4_CHUNK))
        pos += n
    return bytes(out)


def fast_compress(data: bytes) -> bytes:
    """TfFastCompression framing around LZ4 blocks: one block (after a 0), or for data over
    LZ4_CHUNK the chunk count, then each chunk's size and block. Compressed with the lz4 package
    when it's installed, otherwise made of literals only (valid for every reader, just not smaller)."""
    try:
        import lz4.block

        def block(chunk: bytes) -> bytes:
            return lz4.block.compress(chunk, store_size=False)
    except ImportError:
        block = _literal_block
    if len(data) <= LZ4_CHUNK:
        return b"\0" + block(data)
    chunks = [block(data[i:i + LZ4_CHUNK]) for i in range(0, len(data), LZ4_CHUNK)]
    if len(chunks) > 127:
        raise MeshportError("This is too much data for one USD table")
    return bytes([len(chunks)]) + b"".join(struct.pack("<i", len(c)) + c for c in chunks)


def _literal_block(data: bytes) -> bytes:
    """An LZ4 block holding data as one run of literals."""
    n = len(data)
    if n < 15:
        return bytes([n << 4]) + data
    rest = n - 15
    return bytes([0xF0]) + b"\xff" * (rest // 255) + bytes([rest % 255]) + data


def decode_ints(data: bytes, count: int, width: int) -> np.ndarray:
    """Usd_IntegerCompression after LZ4: the common delta, 2-bit codes, then the deltas."""
    if count == 0:
        return np.zeros(0, np.int64)
    signed = {4: "<i4", 8: "<i8"}[width]
    common = int(np.frombuffer(data, signed, 1, 0)[0])
    pos = width
    ncode = (count * 2 + 7) // 8
    codes = (np.frombuffer(data, np.uint8, ncode, pos)[:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
    codes = codes.ravel()[:count]
    pos += ncode
    sizes = np.array([0, 1, 2, 4] if width == 4 else [0, 2, 4, 8])[codes]
    starts = pos + np.concatenate(([0], np.cumsum(sizes)[:-1]))
    if len(data) < pos + int(sizes.sum()):
        raise MeshportError("This USD file is damaged (an integer table is cut short)")
    raw = np.frombuffer(data, np.uint8)
    deltas = np.full(count, common, np.int64)
    for code, dtype in ((1, "<i1" if width == 4 else "<i2"), (2, "<i2" if width == 4 else "<i4"),
                        (3, "<i4" if width == 4 else "<i8")):
        sel = np.nonzero(codes == code)[0]
        if len(sel):
            w = np.dtype(dtype).itemsize
            gathered = raw[starts[sel][:, None] + np.arange(w)].copy()
            deltas[sel] = gathered.view(dtype).ravel()
    return np.cumsum(deltas)


def encode_ints(values, width: int = 4) -> bytes:
    """The writer's side of decode_ints, then fast_compress."""
    v = np.asarray(values, np.int64)
    if not len(v):
        return fast_compress(b"")
    deltas = np.diff(np.concatenate(([0], v)))
    if width == 4:  # 32-bit steps wrap, as in USD: 0xFFFFFFFF after 7 is a step of -8
        deltas = (deltas + (1 << 31)) % (1 << 32) - (1 << 31)
    common = Counter(deltas.tolist()).most_common(1)[0][0]
    widths = (1, 2, 4) if width == 4 else (2, 4, 8)
    codes = np.full(len(v), 3, np.uint8)
    for code in (2, 1):
        half = 1 << (8 * widths[code - 1] - 1)
        codes[(deltas >= -half) & (deltas < half)] = code
    codes[deltas == common] = 0
    packed = np.zeros((len(v) + 3) // 4 * 4, np.uint8)
    packed[:len(v)] = codes
    packed = packed.reshape(-1, 4)
    code_bytes = (packed[:, 0] | packed[:, 1] << 2 | packed[:, 2] << 4 | packed[:, 3] << 6).astype(np.uint8)
    sizes = np.array([0, *widths])[codes]
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    body = np.zeros(int(sizes.sum()), np.uint8)
    for code, w in zip((1, 2, 3), widths):
        sel = np.nonzero(codes == code)[0]
        if len(sel):
            raw = deltas[sel].astype(f"<i{w}").view(np.uint8).reshape(-1, w)
            body[starts[sel][:, None] + np.arange(w)] = raw
    head = np.array(common, "<i4" if width == 4 else "<i8").tobytes()
    return fast_compress(head + code_bytes.tobytes() + body.tobytes())


def _encoded_size(count: int, width: int) -> int:
    return width + (count * 2 + 7) // 8 + count * width


# ============================================================================ reading

class _Reader:
    def __init__(self, data: bytes):
        if data[:8] != MAGIC:
            raise MeshportError("This isn't a binary USD file")
        self.data = data
        self.values: dict[int, object] = {}  # rep -> its value: specs share fields, each is decoded once
        limit = DECODE_BUDGET + 16 * len(data)
        self.budget = Budget(limit, f"This USD file decodes to too much data (over {limit >> 20:,} MB); it may be "
                                    "damaged or made to fill the memory")
        self.version = tuple(data[8:11])
        if self.version < (0, 4, 0) or self.version[0] != 0:
            raise MeshportError(f"This USD file uses crate version {'.'.join(map(str, self.version))}, "
                                "which meshport can't read")
        toc = self._u64(16)
        if not 88 <= toc <= len(data) - 8:
            raise MeshportError("This USD file is damaged (its table of contents is missing)")
        count = self._u64(toc)
        if count > min(64, (len(data) - toc - 8) // 32):
            raise MeshportError("This USD file is damaged (its table of contents is cut short)")
        sections = {}
        for i in range(count):
            at = toc + 8 + 32 * i
            name = data[at:at + 16].split(b"\0", 1)[0].decode("ascii", "replace")
            start, size = self._i64(at + 16), self._i64(at + 24)
            if start < 0 or size < 0 or start + size > len(data):
                raise MeshportError(f"This USD file is damaged (section {name} is outside the file)")
            sections[name] = (start, size)
        self.sections = sections
        for name in ("TOKENS", "STRINGS", "FIELDS", "FIELDSETS", "PATHS", "SPECS"):
            if name not in sections:
                raise MeshportError(f"This USD file is damaged (no {name} section)")
        self._tokens()
        self._strings()
        self._fields()
        self._fieldsets()
        self._paths()
        self._specs()

    # -- primitives

    def _room(self, at: int, size: int) -> None:
        if at < 0 or at + size > len(self.data):
            raise MeshportError("This USD file is damaged (a value runs past its end)")

    def _u64(self, at: int) -> int:
        self._room(at, 8)
        return int.from_bytes(self.data[at:at + 8], "little")

    def _i64(self, at: int) -> int:
        self._room(at, 8)
        return int.from_bytes(self.data[at:at + 8], "little", signed=True)

    def _u32(self, at: int) -> int:
        self._room(at, 4)
        return int.from_bytes(self.data[at:at + 4], "little")

    def _entries(self, at: int, count: int, size: int) -> int:
        """A count read from the file, refused when that many entries of size bytes don't fit after at."""
        self._room(at, count * size)
        return count

    def _count_ok(self, count: int, per_byte: int = 4) -> int:
        """A count read from the file, refused when the file is too small to hold that many."""
        if count > len(self.data) * per_byte:
            raise MeshportError("This USD file is damaged (a table claims more entries than fit in it)")
        return count

    def _compressed_ints(self, at: int, count: int, width: int = 4) -> tuple[np.ndarray, int]:
        size = self._u64(at)
        if at + 8 + size > len(self.data):
            raise MeshportError("This USD file is damaged (compressed data runs past the end)")
        # an integer takes at least 2 bits before LZ4, which shrinks data at most 255 times
        self._count_ok(count, 4 * 255)
        raw = fast_decompress(self.data[at + 8:at + 8 + size], _encoded_size(count, width))
        return decode_ints(raw, count, width), at + 8 + size

    # -- sections

    def _tokens(self) -> None:
        at = self.sections["TOKENS"][0]
        count, size, csize = self._u64(at), self._u64(at + 8), self._u64(at + 16)
        if size > csize * 255 + 16 or at + 24 + csize > len(self.data):
            raise MeshportError("This USD file is damaged (its name table is cut short)")
        raw = fast_decompress(self.data[at + 24:at + 24 + csize], size)
        if len(raw) != size:
            raise MeshportError("This USD file is damaged (its name table has the wrong size)")
        names = raw.split(b"\0")
        if len(names) < count:
            raise MeshportError("This USD file is damaged (token table is cut short)")
        self.tokens = [t.decode("utf-8", "replace") for t in names[:count]]

    def _strings(self) -> None:
        at = self.sections["STRINGS"][0]
        n = self._count_ok(self._u64(at), 1) // 1
        self.strings = np.frombuffer(self.data, "<u4", n, at + 8).astype(np.int64)
        if len(self.strings) and self.strings.max() >= len(self.tokens):
            raise MeshportError("This USD file is damaged (a string points past the name table)")

    def _fields(self) -> None:
        at = self.sections["FIELDS"][0]
        n = self._u64(at)
        names, at = self._compressed_ints(at + 8, n)
        size = self._u64(at)
        if size > len(self.data) or n * 8 > size * 255 + 16:
            raise MeshportError("This USD file is damaged (its field table is cut short)")
        reps = fast_decompress(self.data[at + 8:at + 8 + size], n * 8)
        if len(reps) != n * 8:
            raise MeshportError("This USD file is damaged (its field table has the wrong size)")
        self.fields = list(zip(names.tolist(), np.frombuffer(reps, "<u8").tolist()))

    def _fieldsets(self) -> None:
        at = self.sections["FIELDSETS"][0]
        n = self._u64(at)
        self.fieldsets = self._compressed_ints(at + 8, n)[0].astype(np.uint32).tolist()

    def _paths(self) -> None:
        at = self.sections["PATHS"][0]
        total = self._u64(at)
        n = self._u64(at + 8)
        at += 16
        index, at = self._compressed_ints(at, n)
        element, at = self._compressed_ints(at, n)
        jumps, at = self._compressed_ints(at, n)
        # (Pixar's library can leave a slot or so unused, but not more slots than the file has bytes)
        if total > len(self.data) or (n and (index.min() < 0 or index.max() >= total)):
            raise MeshportError("This USD file is damaged (its path table doesn't add up)")
        index, element, jumps = index.tolist(), element.tolist(), jumps.tolist()
        paths: list[str | None] = [None] * total
        stack = [(0, None)]  # (encoded index, parent path): siblings waiting their turn
        steps = 0
        while stack:
            cur, parent = stack.pop()
            while True:
                steps += 1
                if steps > n + 1 or cur >= n:
                    raise MeshportError("This USD file is damaged (its path table loops)")
                this = cur
                cur += 1
                if parent is None:
                    path = "/"
                else:
                    tok = element[this]
                    name = self.tokens[abs(tok)]
                    path = property_path(parent, name) if tok < 0 else child_path(parent, name)
                paths[index[this]] = path
                jump = jumps[this]
                has_child, has_sibling = jump > 0 or jump == -1, jump >= 0
                if has_child and has_sibling:
                    stack.append((this + jump, parent))
                if has_child:
                    parent = path
                elif not has_sibling:
                    break
        self.paths = paths

    def _specs(self) -> None:
        at = self.sections["SPECS"][0]
        n = self._u64(at)
        paths, at = self._compressed_ints(at + 8, n)
        fsets, at = self._compressed_ints(at, n)
        kinds, at = self._compressed_ints(at, n)
        self.specs = list(zip(paths.tolist(), fsets.tolist(), kinds.tolist()))

    # -- values

    def layer(self) -> Layer:
        layer = Layer()
        for path_index, fset, kind in self.specs:
            path = self.paths[path_index]
            fields = {}
            i = fset
            while i < len(self.fieldsets) and self.fieldsets[i] != 0xFFFFFFFF:
                name_index, rep = self.fields[self.fieldsets[i]]
                name = self.tokens[name_index]
                try:
                    fields[name] = self.value(rep)
                except _Unsupported:
                    pass
                i += 1
            layer.specs[path] = Spec(kind, fields)
        return layer

    def token(self, i: int) -> str:
        return self.tokens[i]

    def string(self, i: int) -> str:
        return self.tokens[int(self.strings[i])]

    def value(self, rep: int):
        if rep not in self.values:  # (unsupported types raise each time, and aren't kept)
            self.values[rep] = self._value(rep)
        return self.values[rep]

    def _value(self, rep: int):
        kind = (rep >> 48) & 0xFF
        payload = rep & _PAYLOAD_MASK
        if rep & _IS_INLINED:
            return self._inlined(kind, payload, bool(rep & _IS_ARRAY))
        if rep & _IS_ARRAY:
            return self._array(kind, payload, bool(rep & _IS_COMPRESSED))
        return self._scalar(kind, payload)

    def _inlined(self, kind: int, payload: int, is_array: bool):
        if is_array:
            return np.zeros(0)  # an empty array
        raw = payload.to_bytes(8, "little")
        if kind == BOOL:
            return bool(payload)
        if kind == UCHAR:
            return payload & 0xFF
        if kind == UINT:
            return payload & 0xFFFFFFFF
        if kind == INT:
            return int.from_bytes(raw[:4], "little", signed=True)
        if kind == FLOAT:
            return struct.unpack("<f", raw[:4])[0]
        if kind in (DOUBLE, TIME_CODE):  # doubles that fit a float are stored as one
            return float(struct.unpack("<f", raw[:4])[0])
        if kind == HALF:
            return float(np.frombuffer(raw[:2], "<f2")[0])
        if kind in (TOKEN, ASSET):
            return self.tokens[payload]
        if kind == STRING:
            return self.string(payload)
        if kind == SPECIFIER:
            return _SPECIFIERS[payload] if payload < 3 else "def"
        if kind == VARIABILITY:
            return _VARIABILITY[payload] if payload < 2 else "varying"
        if kind == PERMISSION:
            return payload
        if kind in _NUMERIC and kind not in (INT64, UINT64):
            dtype, n = _NUMERIC[kind]
            small = np.frombuffer(raw, np.int8)
            if kind in (MATRIX2D, MATRIX3D, MATRIX4D):  # diagonal matrices: just the diagonal
                d = {MATRIX2D: 2, MATRIX3D: 3, MATRIX4D: 4}[kind]
                return np.diag(small[:d].astype(np.float64))
            return small[:n].astype(np.dtype(dtype).newbyteorder("="))
        if kind == DICTIONARY:
            return {}
        if kind in (VALUE_BLOCK, VALUE):
            return None
        if kind in (TOKEN_VECTOR, PATH_VECTOR, STRING_VECTOR, DOUBLE_VECTOR):
            return []
        raise _Unsupported(kind)

    def _count(self, at: int) -> tuple[int, int]:
        if self.version < (0, 7, 0):
            return self._u32(at), at + 4
        return self._u64(at), at + 8

    def _array(self, kind: int, at: int, compressed: bool):
        n, at = self._count(at)
        self._count_ok(n, 4 * 255)
        if kind in (TOKEN, ASSET):
            self.budget.spend(8 * n)
            return [self.tokens[i] for i in np.frombuffer(self.data, "<u4", n, at).tolist()]
        if kind == STRING:
            self.budget.spend(8 * n)
            return [self.string(i) for i in np.frombuffer(self.data, "<u4", n, at).tolist()]
        if kind not in _NUMERIC:
            raise _Unsupported(kind)
        dtype, comps = _NUMERIC[kind]
        self.budget.spend(n * comps * np.dtype(dtype).itemsize)  # before decoding (LZ4 can grow data 255 times)
        if compressed and n >= _MIN_COMPRESSED:
            if kind in _INTS and self.version >= (0, 5, 0):
                return self._compressed_ints(at, n, _INTS[kind])[0].astype(dtype)
            if kind in _FLOATS and self.version >= (0, 6, 0):
                code = self.data[at]
                if code == ord("i"):
                    return self._compressed_ints(at + 1, n)[0].astype(dtype)
                if code == ord("t"):
                    lut_n = self._u32(at + 1)
                    lut = np.frombuffer(self.data, dtype, lut_n, at + 5)
                    idx = self._compressed_ints(at + 5 + lut_n * lut.itemsize, n)[0]
                    if len(idx) and (idx.min() < 0 or idx.max() >= lut_n):
                        raise MeshportError("This USD file is damaged (a lookup table index is out of range)")
                    return lut[idx]
                raise MeshportError("This USD file is damaged (unknown array compression)")
        arr = np.frombuffer(self.data, dtype, n * comps, at)
        if kind in _QUATS:
            return np.roll(arr.reshape(n, 4), 1, axis=1)
        return arr.reshape(n, comps) if comps > 1 else arr

    def _vector(self, at: int, item) -> tuple[list, int]:
        n = self._entries(at + 8, self._u64(at), 4)
        at += 8
        out = []
        for _ in range(n):
            v, at = item(at)
            out.append(v)
        return out, at

    def _tok(self, at: int):
        return self.tokens[self._u32(at)], at + 4

    def _str(self, at: int):
        return self.string(self._u32(at)), at + 4

    def _path(self, at: int):
        return self.paths[self._u32(at)], at + 4

    def _scalar(self, kind: int, at: int):
        if kind in _NUMERIC:
            dtype, n = _NUMERIC[kind]
            arr = np.frombuffer(self.data, dtype, n, at)
            if kind in (MATRIX2D, MATRIX3D, MATRIX4D):
                d = {MATRIX2D: 2, MATRIX3D: 3, MATRIX4D: 4}[kind]
                return arr.reshape(d, d).astype(np.float64)
            if kind in _QUATS:
                arr = np.roll(arr, 1)
            return arr.astype(np.dtype(dtype).newbyteorder("=")) if n > 1 else arr[0].item()
        if kind == TOKEN_VECTOR:
            return self._vector(at, self._tok)[0]
        if kind == PATH_VECTOR:
            return self._vector(at, self._path)[0]
        if kind == STRING_VECTOR:
            return self._vector(at, self._str)[0]
        if kind == DOUBLE_VECTOR:
            n = self._entries(at + 8, self._u64(at), 8)
            return np.frombuffer(self.data, "<f8", n, at + 8).tolist()
        if kind == LAYER_OFFSET_VECTOR:
            n = self._entries(at + 8, self._u64(at), 16)
            return np.frombuffer(self.data, "<f8", 2 * n, at + 8).reshape(n, 2).tolist()
        if kind == TOKEN_LISTOP:
            return self._listop(at, self._tok)
        if kind == STRING_LISTOP:
            return self._listop(at, self._str)
        if kind == PATH_LISTOP:
            return self._listop(at, self._path)
        if kind == REFERENCE_LISTOP:
            return self._listop(at, self._reference)
        if kind == PAYLOAD_LISTOP:
            return self._listop(at, self._payload)
        if kind in _INT_LISTOPS:  # a point instancer's inactiveIds, for one
            fmt = _INT_LISTOPS[kind]
            size = struct.calcsize(fmt)
            return self._listop(at, lambda a: (struct.unpack_from(fmt, self.data, a)[0], a + size))
        if kind == PAYLOAD:
            return self._payload(at)[0]
        if kind == VARIANT_SELECTION_MAP:
            n = self._entries(at + 8, self._u64(at), 8)
            at += 8
            out = {}
            for _ in range(n):
                k, at = self._str(at)
                v, at = self._str(at)
                out[k] = v
            return out
        if kind == DICTIONARY:
            return self._dictionary(at)[0]
        if kind == TIME_SAMPLES:
            return self._time_samples(at)
        raise _Unsupported(kind)

    def _listop(self, at: int, item) -> ListOp:
        header = self.data[at]
        at += 1
        op = ListOp(explicit=bool(header & 1))
        for bit, name in ((2, "explicit_items"), (4, "added"), (32, "prepended"), (64, "appended"),
                          (8, "deleted"), (16, "ordered")):
            if header & bit:
                items, at = self._vector(at, item)
                setattr(op, name, items)
        return op

    def _layer_offset(self, at: int) -> tuple[tuple[float, float], int]:
        offset, scale = struct.unpack_from("<dd", self.data, at)
        return (offset, scale), at + 16

    def _reference(self, at: int):
        asset, at = self._str(at)
        prim, at = self._path(at)
        timing, at = self._layer_offset(at)
        _, at = self._dictionary(at)  # custom data
        return Reference(asset, prim, *timing), at

    def _payload(self, at: int):
        asset, at = self._str(at)
        prim, at = self._path(at)
        timing = (0.0, 1.0)
        if self.version >= (0, 8, 0):
            timing, at = self._layer_offset(at)
        return Payload(asset, prim, *timing), at

    def _recursive_value(self, at: int):
        """A value stored out of line: an offset (relative to itself) to its rep, which follows
        the value's own data. Reading continues after the rep."""
        rep_at = at + self._i64(at)
        if not at < rep_at <= len(self.data) - 8:
            raise MeshportError("This USD file is damaged (a value points outside the file)")
        try:
            value = self.value(self._u64(rep_at))
        except _Unsupported:
            value = None
        return value, rep_at + 8

    def _dictionary(self, at: int):
        n = self._entries(at + 8, self._u64(at), 12)  # a key and a value's offset at the least
        at += 8
        out = {}
        for _ in range(n):
            key, at = self._str(at)
            out[key], at = self._recursive_value(at)
        return out, at

    def _time_samples(self, at: int) -> dict:
        times, at = self._recursive_value(at)
        values_at = at + self._i64(at)
        n = self._entries(values_at + 8, self._u64(values_at), 8)
        reps = np.frombuffer(self.data, "<u8", n, values_at + 8).tolist()
        out = {}
        for t, rep in zip(times or [], reps):
            try:
                out[float(t)] = self.value(rep)
            except _Unsupported:
                pass
        return out


class _Unsupported(Exception):
    """A value type meshport doesn't need (splines, unregistered values...): skipped."""


def read(data: bytes) -> Layer:
    try:
        return _Reader(data).layer()
    except MeshportError:
        raise
    except (ValueError, IndexError, KeyError, TypeError, OverflowError, struct.error, MemoryError,
            RecursionError) as e:
        raise MeshportError(f"This USD file is damaged ({type(e).__name__}: {e})") from None


# ============================================================================ writing

class _Writer:
    def __init__(self):
        self.out = bytearray(MAGIC + bytes([*WRITE_VERSION, 0, 0, 0, 0, 0]) + bytes(8) + bytes(64))
        self.tokens: dict[str, int] = {}
        self.strings: dict[str, int] = {}
        self.paths: dict[str, int] = {}
        self.fields: dict[tuple[int, int], int] = {}
        self.stored: dict[tuple, int] = {}  # identical arrays are stored once, as USD does

    def token(self, t: str) -> int:
        if t not in self.tokens:
            self.tokens[t] = len(self.tokens)
        return self.tokens[t]

    def string(self, s: str) -> int:
        if s not in self.strings:
            self.token(s)
            self.strings[s] = len(self.strings)
        return self.strings[s]

    def path(self, p: str) -> int:
        return self.paths[p]

    def _align(self, n: int = 8) -> None:
        self.out += bytes(-len(self.out) % n)

    def _at(self) -> int:
        self._align()
        return len(self.out)

    def rep(self, kind: int, payload: int, array=False, inlined=False) -> int:
        return (array << 63) | (inlined << 62) | (kind << 48) | (payload & _PAYLOAD_MASK)

    # -- values: (type name in the layer, Python value) -> rep

    def value(self, v, kind: int) -> int:
        if kind in (TOKEN, ASSET):
            return self.rep(kind, self.token(v), inlined=True)
        if kind == STRING:
            return self.rep(kind, self.string(v), inlined=True)
        if kind == BOOL:
            return self.rep(kind, int(bool(v)), inlined=True)
        if kind == INT:
            return self.rep(kind, int(v) & 0xFFFFFFFF, inlined=True)
        if kind == FLOAT:
            return self.rep(kind, int.from_bytes(struct.pack("<f", v), "little"), inlined=True)
        if kind == DOUBLE:
            if float(np.float32(v)) == float(v):
                return self.rep(kind, int.from_bytes(struct.pack("<f", v), "little"), inlined=True)
            at = self._at()
            self.out += struct.pack("<d", v)
            return self.rep(kind, at)
        if kind == SPECIFIER:
            return self.rep(kind, _SPECIFIERS.index(v), inlined=True)
        if kind == VARIABILITY:
            return self.rep(kind, _VARIABILITY.index(v), inlined=True)
        if kind == TOKEN_VECTOR:
            at = self._at()
            self.out += struct.pack("<Q", len(v)) + np.array([self.token(t) for t in v], "<u4").tobytes()
            return self.rep(kind, at)
        if kind == PATH_LISTOP:
            at = self._at()
            self.out += bytes([1 | 2]) + struct.pack("<Q", len(v)) + \
                np.array([self.path(p) for p in v], "<u4").tobytes()  # explicit list
            return self.rep(kind, at)
        if kind == TOKEN_LISTOP:  # prepended, as USD writes apiSchemas
            items = v.items() if isinstance(v, ListOp) else list(v)
            at = self._at()
            self.out += bytes([32]) + struct.pack("<Q", len(items)) + \
                np.array([self.token(t) for t in items], "<u4").tobytes()
            return self.rep(kind, at)
        if kind in _NUMERIC:
            dtype = _NUMERIC[kind][0]
            arr = np.asarray(v, dtype)
            arr = np.ascontiguousarray(np.roll(arr, -1) if kind in _QUATS else arr)
            at = self._at()
            self.out += arr.tobytes()
            return self.rep(kind, at)
        if kind == DOUBLE_VECTOR:
            at = self._at()
            self.out += struct.pack("<Q", len(v)) + np.asarray(v, "<f8").tobytes()
            return self.rep(kind, at)
        if kind == TIME_SAMPLES:
            # the times (a double vector) and the values come first; then, where the rep points: an
            # offset to the times' rep, that rep, an offset to the values' reps, their count and them
            value_kind, samples, *array = v  # (kind, samples) or (kind, samples, values are arrays)
            times = sorted(samples)
            times_rep = self.value([float(t) for t in times], DOUBLE_VECTOR)
            store = self.array if array and array[0] else self.value
            reps = [store(samples[t], value_kind) for t in times]
            at = self._at()
            self.out += struct.pack("<qQqQ", 8, times_rep, 8, len(reps)) + np.array(reps, "<u8").tobytes()
            return self.rep(kind, at)
        raise MeshportError(f"meshport can't write USD value type {kind}")

    def array(self, v, kind: int) -> int:
        if kind in (TOKEN, ASSET):
            body = struct.pack("<Q", len(v)) + np.array([self.token(t) for t in v], "<u4").tobytes()
            return self._store(kind, body, False)
        dtype, comps = _NUMERIC[kind]
        arr = np.asarray(v, dtype).reshape(-1, comps)
        if kind in _QUATS:
            arr = np.roll(arr, -1, axis=1)  # stored (x, y, z, w)
        arr = np.ascontiguousarray(arr).reshape(-1)
        if kind in _INTS and len(arr) >= _MIN_COMPRESSED:  # index lists shrink a lot this way
            key = (kind, len(arr), hashlib.blake2b(arr.tobytes(), digest_size=16).digest())
            if key in self.stored:
                return self.stored[key]
            packed = encode_ints(arr, _INTS[kind])
            rep = self._store(kind, struct.pack("<QQ", len(arr), len(packed)) + packed, True)
            self.stored[key] = rep
            return rep
        return self._store(kind, struct.pack("<Q", len(arr) // comps) + arr.tobytes(), False)

    def _store(self, kind: int, body: bytes, compressed: bool) -> int:
        key = (kind, compressed, len(body), hashlib.blake2b(body, digest_size=16).digest())
        if key not in self.stored:
            at = self._at()
            self.out += body
            self.stored[key] = self.rep(kind, at, array=True) | (_IS_COMPRESSED if compressed else 0)
        return self.stored[key]

    # -- the whole layer

    def write(self, layer: Layer) -> bytes:
        specs = layer.specs
        if "/" not in specs:
            raise MeshportError("A USD layer needs its root")
        order = _tree_order(specs)
        for p in order:
            self.paths[p] = len(self.paths)
        for p in order:  # every field's name and value, in spec order
            spec = specs[p]
            spec.field_ids = []
            for name, (kind, is_array, v) in spec.typed.items():
                rep = self.array(v, kind) if is_array else self.value(v, kind)
                key = (self.token(name), rep)
                if key not in self.fields:
                    self.fields[key] = len(self.fields)
                spec.field_ids.append(self.fields[key])
        for p in order[1:]:  # the path table names its elements by token: they go in the table too
            self.token(element_name(p))
        sections = {}

        def section(name: str, body: bytes) -> None:
            at = self._at()
            self.out += body
            sections[name] = (at, len(body))

        tokens = "\0".join(self.tokens).encode("utf-8") + b"\0"
        packed = fast_compress(tokens)
        section("TOKENS", struct.pack("<QQQ", len(self.tokens), len(tokens), len(packed)) + packed)
        section("STRINGS", struct.pack("<Q", len(self.strings))
                + np.array([self.tokens[s] for s in self.strings], "<u4").tobytes())
        names = [k[0] for k in self.fields]
        reps = fast_compress(np.array([k[1] for k in self.fields], "<u8").tobytes())
        section("FIELDS", struct.pack("<Q", len(names)) + _ints(names) + struct.pack("<Q", len(reps)) + reps)
        fieldsets, starts = [], []
        for p in order:
            starts.append(len(fieldsets))
            fieldsets += specs[p].field_ids + [0xFFFFFFFF]
        section("FIELDSETS", struct.pack("<Q", len(fieldsets)) + _ints(fieldsets))
        section("PATHS", self._paths_section(order))
        kinds = [specs[p].kind for p in order]
        section("SPECS", struct.pack("<Q", len(order)) + _ints([self.paths[p] for p in order])
                + _ints(starts) + _ints(kinds))
        toc = self._at()
        self.out += struct.pack("<Q", len(sections))
        for name, (at, size) in sections.items():
            self.out += name.encode("ascii").ljust(16, b"\0") + struct.pack("<qq", at, size)
        self.out[16:24] = struct.pack("<q", toc)
        return bytes(self.out)

    def _paths_section(self, order: list[str]) -> bytes:
        """The path tree, depth first: each entry's element name (negative for properties)
        and a jump: -2 leaf, -1 child next, 0 sibling next, >0 both (the sibling that far on)."""
        children: dict[str, list[str]] = {}
        for p in order[1:]:
            children.setdefault(parent_path(p), []).append(p)
        index, element, jumps = [], [], []
        stack = [("/", False, None)]  # (path, has a sibling after it, entry whose jump it fixes)
        while stack:
            p, has_sibling, fix = stack.pop()
            if fix is not None:  # p's subtree is done: the entry before it can point here
                jumps[fix] = len(index) - fix
            me = len(index)
            index.append(self.paths[p])
            name = element_name(p) if p != "/" else ""
            element.append(0 if p == "/" else (-self.token(name) if is_property(p) else self.token(name)))
            kids = children.get(p, [])
            jumps.append(-1 if kids else (0 if has_sibling else -2))
            if kids and has_sibling:
                # p's next sibling (on top of the stack) comes after p's whole subtree: once it's
                # reached, p's jump becomes the distance to it
                sibling, its_sibling, _ = stack[-1]
                stack[-1] = (sibling, its_sibling, me)
            stack.extend((k, i < len(kids) - 1, None) for i, k in reversed(list(enumerate(kids))))
        return struct.pack("<QQ", len(order), len(index)) + _ints(index) + _ints(element) + _ints(jumps)


def _ints(values) -> bytes:
    packed = encode_ints(values)
    return struct.pack("<Q", len(packed)) + packed


def _tree_order(specs: dict) -> list[str]:
    """Root first, then each spec after its parent (insertion order otherwise)."""
    order = ["/"]
    seen = {"/"}
    for p in list(specs):
        chain = []
        q = p
        while q not in seen:
            chain.append(q)
            q = parent_path(q)
        for c in reversed(chain):
            seen.add(c)
            order.append(c)
    for c in order:
        if c not in specs:  # a parent nobody wrote: an empty "over"
            specs[c] = Spec(SPEC_ATTRIBUTE if is_property(c) else SPEC_PRIM)
    return order


def write(layer: Layer) -> bytes:
    return _Writer().write(layer)


__all__ = ["SPEC_ATTRIBUTE", "SPEC_PRIM", "SPEC_PSEUDO_ROOT", "SPEC_RELATIONSHIP", "read", "write"]
