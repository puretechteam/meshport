"""3DS files checked against Autodesk's own 3DS reader, run through Unity.

Unity imports .3ds with Autodesk's FBX SDK (libfbxsdk.dll, whose "3D Studio 3DS" reader is
built on Autodesk's 3DS File Toolkit). Set MESHPORT_UNITY to Unity.exe to run these; the
first run creates a Unity project (a few minutes), later runs reuse it. With
MESHPORT_3DS_CORPUS set as well, every .3ds file in that folder, and meshport's re-saved copy
of it, is checked too.

    MESHPORT_UNITY="C:/Program Files/Unity/Hub/Editor/6000.3.13f1/Editor/Unity.exe" pytest tests/test_autodesk_3ds.py
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from test_3ds import _keyed_file, _kf_node, _mesh_object

import meshport
from meshport import Mesh, Node, Scene
from meshport.geometry import compose

UNITY = os.environ.get("MESHPORT_UNITY")
CORPUS = os.environ.get("MESHPORT_3DS_CORPUS")
pytestmark = pytest.mark.skipif(not UNITY or not Path(UNITY).is_file(), reason="set MESHPORT_UNITY to Unity.exe")

_DUMP_CS = r"""
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

public class Keep3dsAsIs : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        var mi = (ModelImporter)assetImporter;
        mi.globalScale = 1f;
        mi.useFileScale = false;
        mi.importNormals = ModelImporterNormals.Import;
        mi.weldVertices = false;
        mi.optimizeMeshVertices = false;
        mi.optimizeMeshPolygons = false;
        mi.meshCompression = ModelImporterMeshCompression.Off;
        mi.isReadable = true;
    }
}

public static class Dump3ds
{
    static string F(float v) => v.ToString("R", CultureInfo.InvariantCulture);
    static string V(Vector3 v) => "[" + F(v.x) + "," + F(v.y) + "," + F(v.z) + "]";

    public static void Run()
    {
        AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
        var sb = new StringBuilder();
        foreach (string guid in AssetDatabase.FindAssets("t:Model", new[] { "Assets/Models" }))
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            var inst = Object.Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>(path));
            foreach (var mf in inst.GetComponentsInChildren<MeshFilter>(true))
            {
                var m = mf.sharedMesh;
                var t = mf.transform;
                var toWorld = t.localToWorldMatrix.inverse.transpose;
                var line = new StringBuilder("{\"file\":\"" + path + "\",\"triangles\":" + (m.triangles.Length / 3));
                line.Append(",\"positions\":[");
                var verts = m.vertices;
                for (int i = 0; i < verts.Length; i++) line.Append((i > 0 ? "," : "") + V(t.TransformPoint(verts[i])));
                line.Append("],\"normals\":[");
                var normals = m.normals;
                for (int i = 0; i < normals.Length; i++) line.Append((i > 0 ? "," : "") + V(toWorld.MultiplyVector(normals[i]).normalized));
                line.Append("],\"uvs\":[");
                var uvs = m.uv;
                for (int i = 0; i < uvs.Length; i++) line.Append((i > 0 ? "," : "") + "[" + F(uvs[i].x) + "," + F(uvs[i].y) + "]");
                line.Append("]}\n");
                sb.Append(line);
            }
            Object.DestroyImmediate(inst);
        }
        File.WriteAllText(System.Environment.GetEnvironmentVariable("DUMP3DS_OUT"), sb.ToString());
    }
}
"""


def _run_unity(inputs: dict[str, Path]) -> dict[str, list[dict]]:
    """Import inputs ({name: folder of files}) with Unity; returns each file's meshes."""
    project = Path(os.environ.get("MESHPORT_UNITY_PROJECT") or Path(tempfile.gettempdir()) / "meshport-unity-3ds")
    if not (project / "ProjectSettings").is_dir():
        subprocess.run([UNITY, "-batchmode", "-nographics", "-createProject", str(project), "-quit",
                        "-logFile", str(project.parent / "meshport-unity-create.log")], check=True, timeout=1800)
    (project / "Assets/Editor").mkdir(parents=True, exist_ok=True)
    (project / "Assets/Editor/Dump3ds.cs").write_text(_DUMP_CS)
    models = project / "Assets/Models"
    shutil.rmtree(models, ignore_errors=True)
    for name, folder in inputs.items():
        shutil.copytree(folder, models / name)
    out = project / "dump.jsonl"
    out.unlink(missing_ok=True)
    subprocess.run([UNITY, "-batchmode", "-nographics", "-projectPath", str(project), "-executeMethod", "Dump3ds.Run",
                    "-quit", "-logFile", str(project / "dump.log")], check=True, timeout=1800,
                   env={**os.environ, "DUMP3DS_OUT": str(out)})
    meshes = defaultdict(list)
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        meshes[rec["file"].removeprefix("Assets/Models/")].append(rec)
    return meshes


def _autodesk(meshes: list[dict]):
    """Points, (point, normal) rows and (point, uv) rows, in meshport's axes (Unity mirrors x)."""
    mirror = np.array([-1.0, 1, 1])
    pts = np.concatenate([np.array(m["positions"]) * mirror for m in meshes])
    pn = np.concatenate([np.hstack([np.array(m["positions"]) * mirror, np.array(m["normals"]) * mirror])
                         for m in meshes if m["normals"]])
    puv = [np.hstack([np.array(m["positions"]) * mirror, np.array(m["uvs"])]) for m in meshes if m["uvs"]]
    return pts, pn, np.concatenate(puv) if puv else None


def _ours(scene: Scene, skip=lambda node: False):
    pts, pn, puv = [], [], []
    for node, world in scene.walk():
        m = node.mesh
        if m is None or skip(node):
            continue
        t = m.triangles().ravel()
        p = m.positions[m.corners[t]] @ world[:3, :3].T + world[:3, 3]
        pts.append(p)
        if m.normals is not None:
            n = m.normals[t] @ np.linalg.inv(world[:3, :3])
            pn.append(np.hstack([p, n / np.linalg.norm(n, axis=1, keepdims=True)]))
        if m.uvs:
            puv.append(np.hstack([p, m.uvs[0][t]]))
    return (np.concatenate(pts), np.concatenate(pn) if pn else None, np.concatenate(puv) if puv else None)


def _check(name: str, scene: Scene, meshes: list[dict], skip=lambda node: False):
    from scipy.spatial import cKDTree
    ours, ours_pn, ours_uv = _ours(scene, skip)
    theirs, theirs_pn, theirs_uv = _autodesk(meshes)
    size = max(np.ptp(ours, 0).max(), 1.0)
    assert sum(m["triangles"] for m in meshes) >= sum(len(n.mesh.triangles()) for n, _ in scene.walk()
                                                       if n.mesh is not None and not skip(n)), name
    # every one of our points is where Autodesk has one
    assert cKDTree(theirs).query(ours)[0].max() < 1e-5 * size, name
    if ours_pn is not None:  # normals: smoothing groups read the same way
        tree = cKDTree(theirs_pn[:, :3])
        for row in ours_pn:
            near = tree.query_ball_point(row[:3], 1e-5 * size)
            assert max(theirs_pn[i, 3:] @ row[3:] for i in near) > np.cos(np.radians(1)), name
    if ours_uv is not None:
        assert cKDTree(theirs_uv).query(ours_uv)[0].max() < 1e-4 * size, name


def _probes() -> dict[str, bytes]:
    """Keyframer cases whose stored points are elsewhere and which say they were saved on
    frame 10, so only the keys (frame 0) give the pose."""
    import struct

    from test_3ds import _chunk

    from meshport import max3ds
    far = compose((100, 100, 100))
    shape = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], float)
    saved_later = [_chunk(max3ds.KF_SEGMENT, struct.pack("<II", 0, 100)), _chunk(max3ds.KF_CURRENT, struct.pack("<I", 1600))]
    return {
        "rotkey": _keyed_file([_mesh_object("box", far, shape)],
                              [*saved_later, _kf_node("box", 0, pos=(3, 0, 0), rot=(np.radians(30), 0, 0, 1))]),
        "pivot": _keyed_file([_mesh_object("box", far, shape)],
                             [*saved_later, _kf_node("box", 0, pivot=(1, 0, 0), rot=(np.radians(90), 0, 0, 1))]),
        "parent": _keyed_file([_mesh_object("arm", far, shape), _mesh_object("hand", far, shape)],
                              [*saved_later, _kf_node("arm", 0, pos=(5, 0, 0), rot=(np.radians(90), 0, 0, 1), scale=(1, 2, 1)),
                               _kf_node("hand", 1, parent=0, pos=(0, 3, 0))]),
    }


def _big_mesh() -> Scene:
    rng = np.random.default_rng(5)
    n = 30_000  # separate triangles: 90,000 vertices, so meshport splits it into two objects
    return Scene([Node("big", compose((1, 2, 3), (10, 20, 30)), Mesh("big", rng.random((3 * n, 3)), np.full(n, 3),
                                                                     np.arange(3 * n)))])


@pytest.fixture(scope="module")
def autodesk(tmp_path_factory, request):
    pytest.importorskip("scipy")
    work = tmp_path_factory.mktemp("autodesk")
    ours = work / "ours"
    ours.mkdir()
    scene = request.getfixturevalue("scene")  # conftest's rotated, mirrored, textured scene
    meshport.save(scene, ours / "scene.3ds")
    meshport.save(_big_mesh(), ours / "big.3ds")
    probes = work / "probes"
    probes.mkdir()
    for name, data in _probes().items():
        (probes / f"{name}.3ds").write_bytes(data)
    inputs = {"ours": ours, "probes": probes}
    if CORPUS:
        real, resaved = work / "real", work / "resaved"
        real.mkdir()
        resaved.mkdir()
        for f in Path(CORPUS).rglob("*"):
            if f.is_file() and not (real / f.name).exists():
                shutil.copy(f, real / f.name)
        for f in list(real.glob("*")):
            if f.suffix.lower() == ".3ds":
                meshport.save(meshport.load(f), resaved / f.name)
        inputs.update(real=real, resaved=resaved)
    return work, scene, _run_unity(inputs)


@pytest.fixture(scope="module")
def scene():
    from conftest import scene as make
    return make.__wrapped__()


def test_autodesk_reads_our_files(autodesk):
    work, scene, meshes = autodesk
    _check("scene.3ds", scene, meshes["ours/scene.3ds"])
    big = meshport.load(work / "ours/big.3ds")
    assert len([n for n, _ in big.walk() if n.mesh is not None]) == 2 == len(meshes["ours/big.3ds"])
    _check("big.3ds", _big_mesh(), meshes["ours/big.3ds"])


@pytest.mark.parametrize("probe", ["rotkey", "pivot", "parent"])
def test_autodesk_reads_the_keyframer_like_we_do(autodesk, probe):
    work, _, meshes = autodesk
    _check(probe, meshport.load(work / f"probes/{probe}.3ds"), meshes[f"probes/{probe}.3ds"])


def test_autodesk_reads_real_files_like_we_do(autodesk):
    if not CORPUS:
        pytest.skip("set MESHPORT_3DS_CORPUS to a folder of .3ds files")
    work, _, meshes = autodesk
    names = sorted(f.name for f in (work / "real").glob("*") if f.suffix.lower() == ".3ds")
    assert names
    for name in names:
        scene = meshport.load(work / "real" / name)
        cameras = _camera_names(work / "real" / name)
        # Autodesk's reader ignores a camera's aim when placing what hangs from it, where 3ds Max
        # (going by the geometry it stores) applies it, so those objects are left out here.
        under_camera = {id(c) for n, _ in scene.walk() if n.mesh is None and n.name in cameras for c in n.children}
        _check(f"real/{name}", scene, meshes[f"real/{name}"], lambda node, skip=under_camera: id(node) in skip)
        _check(f"resaved/{name}", meshport.load(work / "resaved" / name), meshes[f"resaved/{name}"])


def _camera_names(path: Path) -> set[str]:
    from meshport import max3ds
    reader = max3ds._Reader(path.read_bytes(), None)
    reader.scene()
    return set(reader.cameras)
