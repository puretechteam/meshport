"""Hostile files built in code: each would hang, blow up memory, read outside the model's folder or
read wrongly without its fix. Readers may only ever raise MeshportError, within seconds."""

import gzip
import io
import struct
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pytest
from blendfile import (
    IDENTITY,
    BlendWriter,
    collection,
    cube_mesh,
    mesh_object,
    scene_file,
)
from conftest import cube, nodes_by_name, png

import meshport
from meshport import (
    Animation,
    Channel,
    Mesh,
    MeshportError,
    Node,
    Scene,
    alembic,
    fbx,
    usdcrate,
)
from meshport.fbxtree import parse, write_binary
from meshport.usdlayer import SPEC_ATTRIBUTE, SPEC_PRIM, SPEC_PSEUDO_ROOT, Layer, Spec

TESTS = Path(__file__).parent


def bounded(code: str, *args, timeout: float = 30) -> tuple[int, str]:
    """Run code in a fresh Python (a hang can't take the test run with it): its exit code (3 when it
    ended in a MeshportError, which it prints) and its output."""
    wrapped = ("import sys\nimport meshport\nfrom meshport import MeshportError\ntry:\n"
               + textwrap.indent(textwrap.dedent(code), "    ")
               + "\nexcept MeshportError as e:\n    print('MeshportError:', e)\n    sys.exit(3)\n")
    proc = subprocess.Popen([sys.executable, "-c", wrapped, *map(str, args)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=TESTS)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        pytest.fail(f"still running after {timeout} s")
    return proc.returncode, out


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a Python and what it started (a virtual environment's python.exe starts the real one)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
    proc.kill()
    proc.communicate()


def tri_mesh(name="tri") -> Mesh:
    return Mesh(name, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), np.array([3]), np.arange(3))


# ============================================================================ FBX

def _fbx_head(version=7400) -> bytes:
    return b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", version)


def test_fbx_record_ending_before_it_starts(tmp_path):
    head = struct.Struct("<IIIB")
    child = head.pack(27 + head.size + 1, 0, 0, 1) + b"c"  # ends where its parent began: reading went backwards
    parent = head.pack(27 + 2 * head.size + 2 + 64, 0, 0, 1) + b"p" + child
    path = tmp_path / "loop.fbx"
    path.write_bytes(_fbx_head() + parent + bytes(64))
    code, out = bounded("meshport.load(sys.argv[1])", path, timeout=10)
    assert code == 3 and "damaged" in out, out


def _rewrite(records: list, version: int) -> bytes:
    """Parsed records written back: ints as 64-bit, since parsing doesn't keep which were (ids are)."""
    todo = list(records)
    while todo:
        rec = todo.pop()
        todo += rec.children
        rec.props = [np.int64(p) if type(p) is int else p for p in rec.props]
    return write_binary(records, version)


def _fbx_texture(scene, name: str) -> bytes:
    """FBX bytes whose texture is named exactly so (the writer keeps only the file name)."""
    from test_damaged import _with_texture
    version, records = parse(fbx.write(_with_texture(scene, "wood.png")))
    todo = list(records)
    while todo:
        rec = todo.pop()
        todo += rec.children
        if rec.name in ("FileName", "RelativeFilename", "Filename") and rec.props == ["wood.png"]:
            rec.props = [name]
        elif rec.name == "P" and rec.props[:1] == ["Path"]:
            rec.props[-1] = name
    return _rewrite(records, version)


def test_fbx_texture_is_never_taken_from_the_working_directory(tmp_path, scene, monkeypatch):
    (tmp_path / "wood.png").write_bytes(png())
    monkeypatch.chdir(tmp_path)
    back = meshport.read(_fbx_texture(scene, "wood.png"), "fbx")  # no folder given
    assert nodes_by_name(back)["box"][0].materials[0].base_color_texture.data is None


@pytest.mark.parametrize("how", ["absolute", "two levels up", "texture folder two levels up"])
def test_fbx_texture_outside_the_model_folder_and_the_one_above(tmp_path, scene, how):
    model = tmp_path / "pack" / "models" / "m.fbx"  # may read from pack/models and pack (and below them)
    model.parent.mkdir(parents=True)
    far = tmp_path / ("Textures" if how.startswith("texture") else "") / "wood.png"
    far.parent.mkdir(exist_ok=True)
    far.write_bytes(png())
    name = {"absolute": far.as_posix(), "two levels up": "../../wood.png"}.get(how, "wood.png")
    model.write_bytes(_fbx_texture(scene, name))
    assert nodes_by_name(meshport.load(model))["box"][0].materials[0].base_color_texture.data is None


@pytest.mark.parametrize("how", ["sibling folder", "absolute inside", "texture folder one level up"])
def test_fbx_texture_inside_the_folder_rule_still_loads(tmp_path, scene, how):
    model = tmp_path / "pack" / "models" / "m.fbx"
    model.parent.mkdir(parents=True)
    tex = tmp_path / "pack" / ("Textures" if how.startswith("texture") else "maps") / "wood.png"
    tex.parent.mkdir()
    tex.write_bytes(png())
    name = {"sibling folder": "../maps/wood.png", "absolute inside": tex.as_posix()}.get(how, "C:/artist/wood.png")
    model.write_bytes(_fbx_texture(scene, name))
    assert nodes_by_name(meshport.load(model))["box"][0].materials[0].base_color_texture.data == png()


def _spinning(span: float, interpolation="LINEAR") -> Scene:
    node = Node("box", mesh=cube())
    q = np.array([[0, 0, 0, 1.0], [0, 0, np.sin(1.0), np.cos(1.0)]])
    return Scene([node], [Animation("spin", [Channel(node, "rotation", np.array([0.0, span]), q, interpolation)])])


def test_fbx_frame_grid_is_limited():
    version, records = parse(fbx.write(_spinning(1.0)))
    for rec in records:
        for curve in (r for r in rec.children if r.name == "AnimationCurve"):
            key = curve.find("KeyTime")
            key.props[0] = key.props[0] * 20_000  # the same keys over 20,000 seconds: 480,000 frames
    back = meshport.read(_rewrite(records, version), "fbx")
    times = [len(c.times) for c in back.animations[0].channels]
    assert times and max(times) <= 100_000


@pytest.mark.parametrize("fmt", ["fbx", "usdz", "abc"])
def test_writers_refuse_a_frame_grid_over_the_limit(fmt):
    code, out = bounded(f"""
        import numpy as np
        from meshport import Animation, Channel, Mesh, Node, Scene
        node = Node("box", mesh=Mesh("t", np.eye(3), np.array([3]), np.arange(3)))
        q = np.array([[0, 0, 0, 1.0], [0, 0, 0.6, 0.8]])
        a = Animation("spin", [Channel(node, "rotation", np.array([0.0, 1e6]), q, "STEP")])
        meshport.write(Scene([node], [a]), "{fmt}")
    """, timeout=30)
    assert code == 3 and "frames" in out, out


def test_fbx_curve_attribute_counts_dont_allocate_their_product():
    code, out = bounded("""
        import numpy as np
        from meshport.fbxanim import LINEAR, TICKS, Curve
        from meshport.fbxtree import Record
        n = 100_000
        rec = Record("AnimationCurve", children=[
            Record("KeyTime", [np.arange(n, dtype=np.int64) * TICKS]), Record("KeyValueFloat", [np.zeros(n, np.float32)]),
            Record("KeyAttrFlags", [np.full(n, LINEAR, np.int32)]), Record("KeyAttrDataFloat", [np.zeros(4 * n, np.float32)]),
            Record("KeyAttrRefCount", [np.full(n, 10**9, np.int32)])])  # 100,000 x 100,000 taken at their word
        curve = Curve.from_record(rec)
        print(len(curve.times), bool((curve.flags == LINEAR).all()))
    """, timeout=10)
    assert code == 0 and "100000 True" in out, out


# ============================================================================ USD binary (crate)

class _Forged:
    """A field value that writes its own bytes into the crate and gives its own rep."""

    def __init__(self, kind: int, body: bytes = b"", payload: int | None = None, **flags):
        self.kind, self.body, self.payload, self.flags = kind, body, payload, flags

    def rep(self, w) -> int:
        at = w._at()
        w.out += self.body
        return w.rep(self.kind, at if self.payload is None else self.payload, **self.flags)


def _crate(fields: dict, prims: dict | None = None) -> bytes:
    """A crate whose root spec has these fields ({name: _Forged or (kind, is_array, value)}), and prims
    {path: {property: (kind, is_array, value)}}."""
    w = usdcrate._Writer()
    real = w.value
    w.value = lambda v, kind: v.rep(w) if isinstance(v, _Forged) else real(v, kind)
    layer = Layer({"/": Spec(SPEC_PSEUDO_ROOT)})
    for name, v in fields.items():
        layer.specs["/"].typed[name] = (0, False, v) if isinstance(v, _Forged) else v
    for path, props in (prims or {}).items():
        layer.specs[path] = Spec(SPEC_PRIM, typed={"specifier": (usdcrate.SPECIFIER, False, "def")})
        for name, v in props.items():
            layer.specs[f"{path}.{name}"] = Spec(SPEC_ATTRIBUTE, typed={"default": v})
    return w.write(layer)


def _section(data: bytes, name: str) -> int:
    toc = int.from_bytes(data[16:24], "little")
    for i in range(int.from_bytes(data[toc:toc + 8], "little")):
        at = toc + 8 + 32 * i
        if data[at:at + 16].rstrip(b"\0") == name.encode():
            return int.from_bytes(data[at + 16:at + 24], "little")
    raise KeyError(name)


def test_usdc_value_past_the_end_is_damaged():
    data = _crate({"variantSelection": _Forged(usdcrate.VARIANT_SELECTION_MAP, payload=1 << 40)})
    with pytest.raises(MeshportError, match="damaged"):
        usdcrate.read(data)


def test_usdc_huge_variant_map_count(tmp_path):
    # the map's count sits in zeros that run to the end of the file, so its entries read as string 0
    data = _crate({"documentation": (usdcrate.STRING, False, "x"),
                   "variantSelection": _Forged(usdcrate.VARIANT_SELECTION_MAP, payload=1 << 20)})
    path = tmp_path / "m.usdc"
    path.write_bytes(data + bytes((1 << 20) - len(data)) + struct.pack("<Q", 1 << 62) + bytes(64))
    code, out = bounded("from meshport import usdcrate\nusdcrate.read(open(sys.argv[1], 'rb').read())", path,
                        timeout=10)
    assert code == 3 and "damaged" in out, out


def test_usdc_path_count_is_checked():
    data = bytearray(_crate({"documentation": (usdcrate.STRING, False, "x")}))
    at = _section(data, "PATHS")
    data[at:at + 8] = struct.pack("<Q", 20_000_000)  # 20 million paths, one stored
    with pytest.raises(MeshportError, match="damaged"):
        usdcrate.read(bytes(data))


def test_usdc_shared_value_is_decoded_once(monkeypatch):
    arr = np.arange(30_000, dtype=np.float32).reshape(-1, 3)
    data = _crate({}, {f"/P{i}": {"points": (usdcrate.VEC3F, True, arr)} for i in range(1000)})
    calls = []
    real = usdcrate._Reader._array
    monkeypatch.setattr(usdcrate._Reader, "_array", lambda self, *a: calls.append(a) or real(self, *a))
    layer = usdcrate.read(data)
    np.testing.assert_array_equal(layer.specs["/P999.points"].fields["default"], arr)
    assert len(calls) == 1


def _lz4_zeros(k: int) -> bytes:
    """An LZ4 block of k zero bytes: one literal zero, a match repeating it, five literal zeros to end."""
    ml = k - 6 - 4
    out = bytearray([(1 << 4) | min(ml, 15), 0]) + struct.pack("<H", 1)
    if ml >= 15:
        out += b"\xff" * ((ml - 15) // 255) + bytes([(ml - 15) % 255])
    return bytes(out) + bytes([5 << 4]) + bytes(5)


class _ForgedArray(_Forged):
    def rep(self, w) -> int:
        at = w._at()
        w.out += self.body
        return w.rep(self.kind, at, array=True) | usdcrate._IS_COMPRESSED


def _zero_ints(n: int) -> _Forged:
    """A compressed int array of n zeros, a few kilobytes on disk."""
    packed = b"\0" + _lz4_zeros(4 + (2 * n + 7) // 8)
    return _ForgedArray(usdcrate.INT, struct.pack("<QQ", n, len(packed)) + packed)


def test_usdc_decoding_has_a_budget():
    # 40 arrays of 4 million ints from 200 kB: 640 MB once decoded
    data = _crate({f"a{i}": _zero_ints(4_000_000) for i in range(40)})
    assert len(data) < 300_000
    with pytest.raises(MeshportError, match="too much"):
        usdcrate.read(data)


def test_usdc_inlined_uchar_keeps_its_byte():
    layer = usdcrate.read(_crate({"x": _Forged(usdcrate.UCHAR, payload=0x1FF, inlined=True)}))
    assert layer.specs["/"].fields["x"] == 0xFF


# ---- LZ4: python-lz4 and meshport's own decoder agree, and big data is split into chunks

def _without_lz4(monkeypatch):
    monkeypatch.setitem(sys.modules, "lz4", None)
    monkeypatch.setitem(sys.modules, "lz4.block", None)


@pytest.mark.parametrize("decoder", ["python-lz4", "meshport"])
def test_lz4_block_sizes(monkeypatch, decoder):
    lz4 = pytest.importorskip("lz4.block")
    data = (b"meshport " * 2000) + bytes(range(256)) * 10
    block = lz4.compress(data, store_size=False)
    if decoder == "meshport":
        _without_lz4(monkeypatch)
    assert usdcrate.lz4_block_decompress(block, len(data)) == data  # the exact size
    assert usdcrate.lz4_block_decompress(block, len(data) + 1000) == data  # only an upper bound
    with pytest.raises(MeshportError):
        usdcrate.lz4_block_decompress(block, len(data) - 1)
    assert usdcrate.fast_decompress(b"\0" + block, len(data)) == data


def test_usdc_written_with_python_lz4_reads_either_way(monkeypatch):
    pytest.importorskip("lz4.block")
    layer = Layer({"/": Spec(SPEC_PSEUDO_ROOT, typed={"documentation": (usdcrate.STRING, False, "x" * 5000)})})
    layer.specs["/M"] = Spec(SPEC_PRIM, typed={"specifier": (usdcrate.SPECIFIER, False, "def")})
    idx = np.tile(np.arange(4, dtype=np.int32), 50_000)
    layer.specs["/M.faceVertexIndices"] = Spec(SPEC_ATTRIBUTE, typed={"default": (usdcrate.INT, True, idx)})
    with_lz4 = usdcrate.write(layer)
    _without_lz4(monkeypatch)
    plain = usdcrate.write(layer)
    assert len(with_lz4) < len(plain) / 4  # really compressed
    for data in (with_lz4, plain):
        back = usdcrate.read(data)
        np.testing.assert_array_equal(back.specs["/M.faceVertexIndices"].fields["default"], idx)
        assert back.specs["/"].fields["documentation"] == "x" * 5000


@pytest.mark.parametrize("decoder", ["python-lz4", "meshport"])
def test_lz4_big_data_goes_in_chunks(monkeypatch, decoder):
    if decoder == "meshport":
        _without_lz4(monkeypatch)
    else:
        pytest.importorskip("lz4.block")
    monkeypatch.setattr(usdcrate, "LZ4_CHUNK", 1000)
    data = bytes(range(256)) * 20
    packed = usdcrate.fast_compress(data)
    assert packed[0] == 6  # 5120 bytes in 1000-byte chunks
    assert usdcrate.fast_decompress(packed, len(data)) == data


# ============================================================================ USD composition

USD_MESH = """def Mesh "{name}"
{{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}}
"""


def _usda(body: str, meta: str = "") -> str:
    return f"#usda 1.0\n({meta})\n\n{body}"


def test_usd_sublayers_listed_many_times(tmp_path):
    names = "abcdef"
    for i, name in enumerate(names):  # each layer lists the next one 20 times: 3.4 million layers if trusted
        sub = ", ".join([f"@{names[i + 1]}.usda@"] * 20) if i + 1 < len(names) else ""
        (tmp_path / f"{name}.usda").write_text(_usda(USD_MESH.format(name="M") if not sub else "",
                                                     f"subLayers = [{sub}]" if sub else ""))
    code, out = bounded("print(len(meshport.load(sys.argv[1]).meshes))", tmp_path / "a.usda", timeout=15)
    assert code == 0 and out.strip() == "1", out


def test_usd_references_fanning_out(tmp_path):
    prims = [f'def Xform "P{k}" (\n    references = [{", ".join([f"</P{k + 1}>"] * 8)}]\n)\n{{\n}}\n'
             for k in range(12)]  # 8 ** 12 opinions if each reference is followed separately
    (tmp_path / "fan.usda").write_text(_usda("".join(prims) + USD_MESH.format(name="P12")))
    code, out = bounded("from meshport import usd\nusd.MAX_SITES = 200_000\nmeshport.load(sys.argv[1])",
                        tmp_path / "fan.usda", timeout=20)  # (the default 4 million takes 15 s)
    assert code == 3 and "too many" in out, out


def test_usd_reference_cycle_with_fan_out(tmp_path):
    body = ('def Xform "A" (\n    references = [</B>, </B>]\n)\n{\n' + textwrap.indent(USD_MESH.format(name="M"), "    ")
            + '}\n\ndef Xform "B" (\n    references = [</A>, </A>]\n)\n{\n}\n')
    (tmp_path / "cycle.usda").write_text(_usda(body))
    code, out = bounded("s = meshport.load(sys.argv[1])\nprint(sorted(n.name for n, _ in s.walk()))",
                        tmp_path / "cycle.usda", timeout=15)
    assert code == 0 and "'A', 'B', 'M', 'M'" in out, out


def test_usd_point_instancer_copies_have_a_budget(tmp_path):
    kids = "".join(textwrap.indent(USD_MESH.format(name=f"c{i}"), "        ") for i in range(20))
    n = 2000  # 2000 instances of a 21-node prototype: 42,000 nodes
    body = (f'def PointInstancer "I"\n{{\n    rel prototypes = </I/Proto>\n'
            f'    int[] protoIndices = [{", ".join(["0"] * n)}]\n'
            f'    point3f[] positions = [{", ".join(f"({i}, 0, 0)" for i in range(n))}]\n'
            f'    def Xform "Proto"\n    {{\n{kids}    }}\n}}\n')
    (tmp_path / "inst.usda").write_text(_usda(body))
    code, out = bounded("meshport.scene.MAX_NODES = 10_000\nmeshport.load(sys.argv[1])", tmp_path / "inst.usda")
    assert code == 3 and "too many" in out, out


def _usdz(files: dict, compression=zipfile.ZIP_STORED) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return out.getvalue()


def test_usdz_with_compressed_files_is_refused():
    data = _usdz({"model.usda": _usda(USD_MESH.format(name="M"))}, zipfile.ZIP_DEFLATED)
    with pytest.raises(MeshportError, match="compressed"):
        meshport.read(data, "usdz")


def _textured_usda(files: list[str]) -> str:
    """A mesh whose faces each bind a material reading one of these texture files."""
    n = len(files)
    subsets = "".join(f'    def GeomSubset "s{i}"\n    {{\n        uniform token elementType = "face"\n'
                      f'        int[] indices = [0]\n        rel material:binding = </Mat{i}>\n    }}\n'
                      for i in range(n))
    mats = "".join(f'def Material "Mat{i}"\n{{\n    token outputs:surface.connect = </Mat{i}/S.outputs:surface>\n'
                   f'    def Shader "S"\n    {{\n        uniform token info:id = "UsdPreviewSurface"\n'
                   f'        color3f inputs:diffuseColor.connect = </Mat{i}/T.outputs:rgb>\n'
                   f'        token outputs:surface\n    }}\n'
                   f'    def Shader "T"\n    {{\n        uniform token info:id = "UsdUVTexture"\n'
                   f'        asset inputs:file = @{f}@\n        float3 outputs:rgb\n    }}\n}}\n'
                   for i, f in enumerate(files))
    mesh = USD_MESH.format(name="M").replace("int[] faceVertexCounts", 'rel material:binding = </Mat0>\n'
                                             '    int[] faceVertexCounts').rstrip("}\n") + "\n" + subsets + "}\n"
    return _usda(mesh + mats)


def test_usdz_texture_spelled_many_ways_is_read_once(monkeypatch):
    image = b"\x89PNG\r\n\x1a\n" + bytes(2_000_000)
    spellings = ["t.png"] + ["/".join(["a", ".."] * k) + "/t.png" for k in range(1, 300)]
    data = _usdz({"model.usda": _textured_usda(spellings), "t.png": image})
    reads = []
    real = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, name, *a: reads.append(name) or real(self, name, *a))
    meshport.read(data, "usdz")
    assert reads.count("t.png") == 1


def test_usdz_member_sizes_must_match():
    data = bytearray(_usdz({"model.usda": _usda(USD_MESH.format(name="M"))}))
    central = data.rindex(b"PK\x01\x02")
    data[central + 24:central + 28] = struct.pack("<I", 1 << 30)  # claims 1 GB, stores a few hundred bytes
    with pytest.raises(MeshportError):
        meshport.read(bytes(data), "usdz")


# ---- linked files: the model's own folder and the one above it, nothing further out

def _usd_tree(tmp_path) -> Path:
    root = tmp_path / "pack" / "scenes" / "root.usda"
    root.parent.mkdir(parents=True)
    (tmp_path / "evil.usda").write_text(_usda(USD_MESH.format(name="Evil")))
    (tmp_path / "pack" / "ok.usda").write_text(_usda(USD_MESH.format(name="Ok")))
    return root


@pytest.mark.parametrize("how", ["absolute", "two levels up", "file url"])
def test_usd_sublayer_outside_the_folder_rule_is_not_read(tmp_path, how):
    root = _usd_tree(tmp_path)
    evil = tmp_path / "evil.usda"
    name = {"absolute": evil.as_posix(), "two levels up": "../../evil.usda", "file url": evil.as_uri()}[how]
    root.write_text(_usda(USD_MESH.format(name="Root"), f"subLayers = [@{name}@, @../ok.usda@]"))
    names = {n.name for n, _ in meshport.load(root).walk()}
    assert names == {"Root", "Ok"}


def test_usd_reference_chain_cant_walk_outward(tmp_path):
    root = _usd_tree(tmp_path)
    (tmp_path / "pack" / "x").mkdir()
    (tmp_path / "pack" / "x" / "step.usda").write_text(  # each step one level up from the last
        _usda('def "S" (\n    references = @../../evil.usda@\n)\n{\n}\n', 'defaultPrim = "S"'))
    root.write_text(_usda(USD_MESH.format(name="Root") + 'def "R" (\n    references = @../x/step.usda@\n)\n{\n}\n'))
    names = {n.name for n, _ in meshport.load(root).walk()}
    assert names == {"Root"}  # R would hold the far file's mesh


@pytest.mark.parametrize("how", ["absolute", "two levels up"])
def test_usd_texture_outside_the_folder_rule_is_not_read(tmp_path, how):
    root = tmp_path / "pack" / "scenes" / "root.usda"
    root.parent.mkdir(parents=True)
    (tmp_path / "far.png").write_bytes(png())
    (tmp_path / "pack" / "near.png").write_bytes(png((0, 255, 0)))
    name = (tmp_path / "far.png").as_posix() if how == "absolute" else "../../far.png"
    root.write_text(_textured_usda([name, "../near.png"]))
    mats = meshport.load(root).roots[0].materials
    textures = {m.name: m.base_color_texture.data for m in mats if m.base_color_texture is not None}
    assert textures["Mat0"] is None and textures["Mat1"] == png((0, 255, 0))


# ============================================================================ .blend

def _cube_scene(**ob) -> bytes:
    w = BlendWriter()
    return scene_file(w, [mesh_object(w, "Cube", cube_mesh(w), **ob)])


def test_blend_gzip_over_a_megabyte_loads(tmp_path):
    w = BlendWriter()
    ob = mesh_object(w, "Cube", cube_mesh(w))
    w.raw(np.random.default_rng(0).integers(0, 256, 1_500_000, dtype=np.uint8).tobytes())  # incompressible
    path = tmp_path / "big.blend"
    path.write_bytes(gzip.compress(scene_file(w, [ob])))
    assert path.stat().st_size > 1 << 20
    assert meshport.load(path).roots[0].name == "Cube"


def test_blend_sniffing_inflates_only_the_start():
    import tracemalloc
    z = __import__("zlib").compressobj(9, wbits=31)
    bomb = z.compress(b"BLENDER-v280") + b"".join(z.compress(bytes(1 << 20)) for _ in range(200)) + z.flush()
    tracemalloc.start()
    try:
        assert meshport._sniff(bomb) == "blend"
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 8 << 20, peak  # 200 MB inside


@pytest.mark.parametrize("kind", ["gzip", "zstd"])
def test_blend_decompression_has_a_cap(monkeypatch, kind):
    from meshport import blend
    monkeypatch.setattr(blend, "MAX_INFLATED", 10 << 20)
    raw = b"BLENDER-v280" + bytes(50 << 20)
    if kind == "gzip":
        data = gzip.compress(raw, 1)
    else:
        zstd = pytest.importorskip("compression.zstd" if sys.version_info >= (3, 14) else "backports.zstd")
        data = zstd.compress(raw)
    with pytest.raises(MeshportError, match="too big"):
        blend.read(data)


def test_blend_collection_instances_have_a_node_budget(tmp_path):
    w = BlendWriter()
    inner = collection(w, [mesh_object(w, "Cube", cube_mesh(w))])
    for level in range(5):  # each level shows the one below 8 times: 32,768 cubes
        empties = [w.block(b"OB\0\0", "Object", {"id": {"name": f"OBe{level}_{i}"}, "type": 0, "rotmode": 1,
                                                 "transflag": 1 << 8, "instance_collection": inner,
                                                 "size": [1, 1, 1], "parentinv": IDENTITY}) for i in range(8)]
        inner = collection(w, empties, name=f"c{level}")
    top = w.block(b"OB\0\0", "Object", {"id": {"name": "OBtop"}, "type": 0, "rotmode": 1, "transflag": 1 << 8,
                                        "instance_collection": inner, "size": [1, 1, 1]})
    path = tmp_path / "inst.blend"
    path.write_bytes(scene_file(w, [top]))
    code, out = bounded("meshport.scene.MAX_NODES = 5000\nmeshport.load(sys.argv[1])", path)
    assert code == 3 and "too many" in out, out


def test_blend_layer_count_is_clamped_to_its_block():
    structs = {**__import__("blendfile").STRUCTS,
               "CustomData": [("CustomDataLayer", "*layers"), ("int", "totlayer")],
               "CustomDataLayer": [("int", "type"), ("int", "pad"), ("char", "name[64]"), ("void", "*data")]}
    structs["Mesh"] = structs["Mesh"] + [("CustomData", "ldata")]
    w = BlendWriter(structs)
    uv = w.raw(np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1)).astype("<f4").tobytes(), "float", 48)
    layers = w.block(b"DATA", "CustomDataLayer", {"type": 49, "name": "UVMap", "data": uv})
    me = cube_mesh(w, ldata={"layers": layers, "totlayer": 1_000_000_000})  # one layer stored
    data = scene_file(w, [mesh_object(w, "Cube", me)])
    assert len(meshport.read(data, "blend").roots[0].mesh.uvs) == 1


def _rigged(verts: int, weights_each: int, keys: int = 0) -> bytes:
    """A mesh of `verts` points with an armature: every point in the same `weights_each` groups (all
    pointing at one weight list), and `keys` shape keys all pointing at one block of points."""
    w = BlendWriter()
    p = np.zeros((verts, 3), np.float32)
    p[:3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    mv = w.raw(p.tobytes(), "MVert", verts)
    polys = w.block(b"DATA", "MPoly", {"loopstart": 0, "totloop": 3})
    loops = w.block(b"DATA", "MLoop", [{"v": 0}, {"v": 1}, {"v": 2}])
    dw = w.block(b"DATA", "MDeformWeight", [{"def_nr": 0, "weight": 1.0}] * weights_each)
    dvert = w.block(b"DATA", "MDeformVert", [{"dw": dw, "totweight": weights_each}] * verts)
    group = w.block(b"DATA", "bDeformGroup", {"name": "Bone"})
    key = 0
    if keys:
        block = w.raw((p + 1).tobytes(), "float", 3 * verts)
        ref = w.raw(p.tobytes(), "float", 3 * verts)
        addrs = [w.address() for _ in range(keys + 1)]
        for i, a in enumerate(addrs):
            w.block(b"DATA", "KeyBlock", {"next": addrs[i + 1] if i < keys else 0, "prev": addrs[i - 1] if i else 0,
                                          "totelem": verts, "data": ref if i == 0 else block,
                                          "name": f"k{i}", "curval": 0.5}, a)
        key = w.block(b"KE\0\0", "Key", {"id": {"name": "KEKey"}, "type": 1, "block": {"first": addrs[0],
                                                                                    "last": addrs[-1]}})
    me = w.block(b"ME\0\0", "Mesh", {"id": {"name": "MEm"}, "totvert": verts, "totpoly": 1, "totloop": 3,
                                     "mvert": mv, "mpoly": polys, "mloop": loops, "dvert": dvert, "key": key,
                                     "vertex_group_names": {"first": group, "last": group}})
    bone = w.block(b"DATA", "Bone", {"name": "Bone", "arm_mat": IDENTITY})
    arm = w.block(b"AR\0\0", "bArmature", {"id": {"name": "ARArm"}, "bonebase": {"first": bone, "last": bone}})
    chan = w.block(b"DATA", "bPoseChannel", {"name": "Bone", "size": [1, 1, 1], "quat": [1, 0, 0, 0]})
    pose = w.block(b"DATA", "bPose", {"chanbase": {"first": chan, "last": chan}})
    rig = w.block(b"OB\0\0", "Object", {"id": {"name": "OBRig"}, "type": 25, "rotmode": 1, "data": arm,
                                        "size": [1, 1, 1], "pose": pose})
    md = w.block(b"DATA", "ArmatureModifierData", {"modifier": {"mode": 1, "name": "Armature"}, "deformflag": 1,
                                                   "object": rig})
    ob = mesh_object(w, "Skinned", me, modifiers={"first": md, "last": md})
    return scene_file(w, [rig, ob])


def test_blend_rig_reads():
    node = nodes_by_name(meshport.read(_rigged(10, 1, 2), "blend"))["Skinned"][0]
    assert node.skin is not None and len(node.mesh.targets) == 2


def test_blend_vertex_weights_have_a_budget(tmp_path):
    path = tmp_path / "weights.blend"
    path.write_bytes(_rigged(2000, 4096))  # 8 million memberships through one shared list
    code, out = bounded("from meshport import blendskin\nblendskin.WEIGHT_BUDGET = 1_000_000\n"
                        "s = meshport.load(sys.argv[1])\n"
                        "print([n.skin is None for n, _ in s.walk() if n.name == 'Skinned'])", path)
    assert code == 0 and "[True]" in out, out


def test_blend_shape_keys_have_a_budget(tmp_path):
    path = tmp_path / "keys.blend"
    path.write_bytes(_rigged(1000, 1, 2000))  # 2000 keys x 1000 points, all one block
    code, out = bounded("from meshport import blendskin\nblendskin.SHAPE_BUDGET = 1_000_000\n"
                        "s = meshport.load(sys.argv[1])\n"
                        "print([len(n.mesh.targets) for n, _ in s.walk() if n.name == 'Skinned'])", path)
    assert code == 0 and "[0]" in out, out


def test_blend_modifier_output_has_a_budget(tmp_path):
    w = BlendWriter()
    array, sub = w.address(), w.address()
    w.block(b"DATA", "ArrayModifierData", {"modifier": {"next": sub, "mode": 1, "name": "Array"}, "count": 1000,
                                           "offset_type": 2, "scale": [1, 0, 0]}, array)
    w.block(b"DATA", "SubsurfModifierData", {"modifier": {"prev": array, "mode": 1, "name": "Subsurf"},
                                             "levels": 3}, sub)
    path = tmp_path / "mods.blend"
    path.write_bytes(scene_file(w, [mesh_object(w, "Cube", cube_mesh(w), modifiers={"first": array, "last": sub})]))
    code, out = bounded("from meshport import blendmods\nblendmods.MAX_GENERATED = 100_000\nmeshport.load(sys.argv[1])",
                        path)
    assert code == 3 and "too big" in out, out


# ============================================================================ Alembic

def _moving_box(frames: int = 2) -> Scene:
    node = Node("box", mesh=cube())
    times = np.arange(frames) / 24.0
    return Scene([node], [Animation("a", [Channel(node, "translation", times, np.c_[times, times * 0, times * 0])])])


def test_alembic_sample_count_beyond_what_is_stored(tmp_path, monkeypatch):
    real = alembic._Writer.scalar_samples
    monkeypatch.setattr(alembic._Writer, "scalar_samples",
                        lambda self, *a: real(self, *a)[:6] + ((50_000_000, a[-1]),))  # claims 50 million
    path = tmp_path / "claims.abc"
    path.write_bytes(alembic.write(_moving_box()))
    code, out = bounded("s = meshport.load(sys.argv[1])\nprint(max(len(c.times) for c in s.animations[0].channels))",
                        path, timeout=20)
    assert code == 0 and int(out.split()[-1]) <= 2, out


def test_alembic_point_cache_weights_are_capped():
    node = Node("box", mesh=cube())
    node.mesh.targets = [meshport.Morph("up", np.tile([0, 1.0, 0], (8, 1)))]
    times = np.arange(6000) / 24.0  # a point sample every frame for 6000 frames
    scene = Scene([node], [Animation("a", [Channel(node, "weights", times, (np.sin(times) ** 2)[:, None])])])
    back = alembic.read(alembic.write(scene))
    weights = [c for a in back.animations for c in a.channels if c.path == "weights"]
    assert sum(c.values.size for c in weights) <= alembic.CACHE_BUDGET


def test_alembic_shared_object_groups_have_a_node_budget(tmp_path):
    w = alembic._Writer()

    def xform():
        return w.compound(".xform", [w.scalar(".inherits", [1], "bool"), w.scalar(".ops", [0x30], "u1"),
                                     w.scalar(".vals", np.eye(4).ravel(), "<f8")], alembic._XFORM_PROP)
    group = w.object([xform()], [w._mesh("boxShape", cube(), [])])
    for _ in range(14):  # each object holds the one below twice: 16,384 boxes
        group = w.object([xform()], [("a", alembic._XFORM, group), ("b", alembic._XFORM, group)])
    path = tmp_path / "dag.abc"
    path.write_bytes(w.finish([("top", alembic._XFORM, group)]))
    code, out = bounded("meshport.scene.MAX_NODES = 5000\nmeshport.load(sys.argv[1])", path)
    assert code == 3 and "too many" in out, out


# ============================================================================ robustness

def _deep(depth: int) -> Scene:
    leaf = Node("leaf", mesh=cube())
    leaf.mesh.targets = [meshport.Morph("up", np.tile([0, 1.0, 0], (8, 1)))]
    node = leaf
    for i in range(depth):
        node = Node(f"n{i}", children=[node])
    return Scene([node])


def test_deep_scene_walks_without_recursion():
    scene = _deep(5000)
    assert len(list(scene.walk())) == 5001
    assert len(list(scene.walk_at(0.0))) == 5001
    assert len(scene.deformed().meshes) == 1


@pytest.mark.parametrize("fmt", ["usda", "usdc", "usdz", "abc"])
def test_deep_scene_writes(fmt):
    assert meshport.write(_deep(3000), fmt)


def test_gltf_accessor_values_have_a_budget(monkeypatch):
    import json

    from meshport import gltf
    monkeypatch.setattr(gltf, "MAX_ACCESSOR_VALUES", 1000)
    doc = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
           "accessors": [{"componentType": 5126, "count": 10_000, "type": "VEC3"}]}  # no data: zeros
    with pytest.raises(MeshportError, match="too much"):
        meshport.read(json.dumps(doc).encode(), "gltf")


def test_big_concave_faces_are_not_ear_clipped_forever(tmp_path):
    code, out = bounded("""
        import numpy as np
        from meshport.geometry import triangulate
        n = 3000
        top = [(i, 1 + (i % 2) * 3) for i in range(n - 2)]  # a zigzag edge: concave everywhere
        p = np.array([[x, y, 0] for x, y in [(0, 0), (n - 3, 0)] + top[::-1]], float)
        sizes = np.full(20, n)
        corners = np.tile(np.arange(n), 20)
        print(len(triangulate(p, sizes, corners)))
    """, timeout=20)
    assert code == 0 and out.split()[-1] == str(20 * 2998), out



def _shared_gltf(depth: int) -> bytes:
    """A glTF where node i lists node i+1 twice as a child (glTF forbids several parents), so the
    tree under the root has 2**depth placements of the last node, which holds a triangle."""
    import json
    doc = json.loads(meshport.write(Scene([Node("tri", np.eye(4), mesh=tri_mesh())]), "gltf"))
    leaf = doc["nodes"][0]
    doc["nodes"] = [{"name": f"n{i}", "children": [i + 1, i + 1]} for i in range(depth)] + [leaf]
    return json.dumps(doc).encode()


def test_gltf_shared_nodes_have_a_node_budget(tmp_path):
    path = tmp_path / "fan.gltf"
    path.write_bytes(_shared_gltf(40))  # 2**40 placements: building them never ended
    code, out = bounded("meshport.load(sys.argv[1])", path, timeout=10)
    assert code == 3 and "too many objects" in out, out


def test_gltf_node_with_two_parents_still_loads():
    scene = meshport.read(_shared_gltf(3), "gltf")
    assert sum(n.mesh is not None for n, _ in scene.walk()) == 8


class _FakeFont:
    """The VFont and Curve fields _font_data reads."""

    def __init__(self, name: str):
        self.fields = {"name": name, "vfont": 1, "packedfile": 0}

    def get(self, key, default=None):
        return self.fields.get(key, default)

    def follow(self, key, struct):
        return self


def _font_file(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\x00\x01\x00\x00" + path.name.encode()
    path.write_bytes(data)
    return data


@pytest.mark.parametrize("how", ["absolute", "relative"])
def test_blend_fonts_follow_the_folder_rule(tmp_path, monkeypatch, how):
    from meshport import blendcurve
    monkeypatch.setattr(blendcurve, "_font_dirs", list)
    far = _font_file(tmp_path / "far" / "secret.ttf")
    model = tmp_path / "a" / "b"
    model.mkdir(parents=True)
    name = str(tmp_path / "far" / "secret.ttf") if how == "absolute" else "//../../far/secret.ttf"
    data, builtin = blendcurve._font_data(_FakeFont(name), model)
    assert data != far and builtin


def test_blend_fonts_inside_the_folder_rule_still_load(tmp_path, monkeypatch):
    from meshport import blendcurve
    monkeypatch.setattr(blendcurve, "_font_dirs", list)
    model = tmp_path / "a" / "b"
    near = _font_file(model / "fonts" / "near.ttf")
    up = _font_file(tmp_path / "a" / "up.ttf")
    beside = _font_file(model / "moved.ttf")
    for name, want in [("//fonts/near.ttf", near), ("//../up.ttf", up), (r"C:\gone\moved.ttf", beside)]:
        assert blendcurve._font_data(_FakeFont(name), model) == (want, False), name


def test_blend_system_fonts_are_found_by_name(tmp_path, monkeypatch):
    from meshport import blendcurve
    system = tmp_path / "Fonts"
    font = _font_file(system / "arialbd.ttf")
    monkeypatch.setattr(blendcurve, "_font_dirs", lambda: [system])
    model = tmp_path / "model"
    model.mkdir()
    # a font from the machine the file was made on: only its name is used, in the system's font folders
    assert blendcurve._font_data(_FakeFont(r"D:\Windows\Fonts\arialbd.ttf"), model) == (font, False)
    assert blendcurve._font_data(_FakeFont("/elsewhere/../arialbd.ttf"), None) == (font, False)


def test_blend_fonts_never_open_network_paths(tmp_path, monkeypatch):
    from meshport import blendcurve
    opened = []
    monkeypatch.setattr(blendcurve, "_font_dirs", list)
    monkeypatch.setattr(blendcurve, "_read_font", lambda p: opened.append(str(p)))
    for name in [r"\\server\share\a.ttf", "////server/share/a.ttf", r"//\\server\share\a.ttf"]:
        blendcurve._font_data(_FakeFont(name), tmp_path)
    assert opened and all(p.startswith(str(tmp_path.resolve().parent)) for p in opened), opened
