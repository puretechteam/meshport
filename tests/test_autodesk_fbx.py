"""FBX animation, skins and blend shapes checked against Autodesk's FBX SDK, run through Unity.

Unity imports .fbx with Autodesk's FBX SDK and, with "resample curves" on, has the SDK evaluate
every animated transform at each frame. Sampling those clips gives Autodesk's pose of every
frame: for probe files with cubic keys (plain and weighted), constant keys (both kinds) and a
model with pivots, offsets and pre/post rotation in ZXY order, and for meshport's own files.
Skinned meshes are baked (bones and blend shape weights as the clip leaves them) at every frame.
Set MESHPORT_UNITY to Unity.exe to run these (the Unity project is shared with
test_autodesk_3ds.py; the first run creates it, which takes a few minutes).

    MESHPORT_UNITY="C:/Program Files/Unity/Hub/Editor/6000.3.13f1/Editor/Unity.exe" pytest tests/test_autodesk_fbx.py
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import cube
from test_animation import animated
from test_skinning import DATA, inbetween_fbx, rigged

import meshport
from meshport import Animation, Channel, Node, Scene, fbx
from meshport.fbxanim import CONSTANT, CUBIC, THIRDS
from meshport.fbxtree import Record, write_binary
from meshport.geometry import compose

UNITY = os.environ.get("MESHPORT_UNITY")
pytestmark = pytest.mark.skipif(not UNITY or not Path(UNITY).is_file(), reason="set MESHPORT_UNITY to Unity.exe")

_DUMP_CS = r"""
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

public class KeepFbxAnimation : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        if (!assetPath.StartsWith("Assets/Anim/")) return;
        var mi = (ModelImporter)assetImporter;
        mi.globalScale = 1f;
        mi.useFileScale = false;
        mi.importAnimation = true;
        mi.animationType = ModelImporterAnimationType.Legacy;
        mi.resampleCurves = true;  // the FBX SDK evaluates every frame
        mi.animationCompression = ModelImporterAnimationCompression.Off;
        mi.isReadable = true;
    }
}

public static class DumpFbxAnimation
{
    static string F(float v) => v.ToString("R", CultureInfo.InvariantCulture);

    public static void Run()
    {
        AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
        var sb = new StringBuilder();
        foreach (string guid in AssetDatabase.FindAssets("t:Model", new[] { "Assets/Anim" }))
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            var go = Object.Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>(path));
            var clips = AssetDatabase.LoadAllAssetsAtPath(path).OfType<AnimationClip>().Where(c => !c.name.StartsWith("__preview__")).ToList();
            sb.Append("{\"file\":\"" + Path.GetFileName(path) + "\",\"clips\":" + clips.Count + "}\n");
            foreach (var clip in clips)
            {
                int frames = Mathf.RoundToInt(clip.length * clip.frameRate);
                for (int f = 0; f <= frames; f++)
                {
                    clip.SampleAnimation(go, f / clip.frameRate);
                    var line = new StringBuilder("{\"file\":\"" + Path.GetFileName(path) + "\",\"clip\":\"" + clip.name +
                                                 "\",\"fps\":" + F(clip.frameRate) + ",\"frame\":" + f + ",\"meshes\":{");
                    bool first = true;
                    foreach (var mf in go.GetComponentsInChildren<MeshFilter>(true))
                    {
                        line.Append((first ? "" : ",") + "\"" + mf.name + "\":[");
                        first = false;
                        var verts = mf.sharedMesh.vertices;
                        for (int i = 0; i < verts.Length; i++)
                        {
                            var p = mf.transform.TransformPoint(verts[i]);
                            line.Append((i > 0 ? "," : "") + "[" + F(p.x) + "," + F(p.y) + "," + F(p.z) + "]");
                        }
                        line.Append("]");
                    }
                    var baked = new Mesh();
                    foreach (var smr in go.GetComponentsInChildren<SkinnedMeshRenderer>(true))
                    {
                        // bones and blend shapes as the clip left them: with bones, in the renderer's place with its
                        // scale baked in; blend shapes alone, in the renderer's own space
                        bool boned = smr.bones.Length > 0;
                        smr.BakeMesh(baked, boned);
                        var place = boned ? Matrix4x4.TRS(smr.transform.position, smr.transform.rotation, Vector3.one)
                                          : smr.transform.localToWorldMatrix;
                        line.Append((first ? "" : ",") + "\"" + smr.name + "\":[");
                        first = false;
                        var verts = baked.vertices;
                        for (int i = 0; i < verts.Length; i++)
                        {
                            var p = place.MultiplyPoint3x4(verts[i]);
                            line.Append((i > 0 ? "," : "") + "[" + F(p.x) + "," + F(p.y) + "," + F(p.z) + "]");
                        }
                        line.Append("]");
                    }
                    sb.Append(line.Append("}}\n"));
                }
            }
            Object.DestroyImmediate(go);
        }
        File.WriteAllText(System.Environment.GetEnvironmentVariable("DUMPFBX_OUT"), sb.ToString());
    }
}
"""


def _run_unity(folder: Path) -> dict:
    """{file: {frame: (fps, {mesh name: world points in meshport's axes})}} for the first clip of each file."""
    project = Path(os.environ.get("MESHPORT_UNITY_PROJECT") or Path(tempfile.gettempdir()) / "meshport-unity-3ds")
    if not (project / "ProjectSettings").is_dir():
        subprocess.run([UNITY, "-batchmode", "-nographics", "-createProject", str(project), "-quit",
                        "-logFile", str(project.parent / "meshport-unity-create.log")], check=True, timeout=1800)
    (project / "Assets/Editor").mkdir(parents=True, exist_ok=True)
    (project / "Assets/Editor/DumpFbxAnimation.cs").write_text(_DUMP_CS)
    anim = project / "Assets/Anim"
    shutil.rmtree(anim, ignore_errors=True)
    shutil.copytree(folder, anim)
    out = project / "dump-fbx.jsonl"
    out.unlink(missing_ok=True)
    subprocess.run([UNITY, "-batchmode", "-nographics", "-projectPath", str(project), "-executeMethod",
                    "DumpFbxAnimation.Run", "-quit", "-logFile", str(project / "dump-fbx.log")], check=True,
                   timeout=1800, env={**os.environ, "DUMPFBX_OUT": str(out)})
    mirror = np.array([-1.0, 1, 1])  # Unity is left-handed: x is mirrored
    poses: dict = {}
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        if "frame" not in rec:  # a file's clip count
            continue
        clips = poses.setdefault(rec["file"], {})
        clip = clips.setdefault(rec["clip"], {})
        clip[rec["frame"]] = (rec["fps"], {k: np.array(v) * mirror for k, v in rec["meshes"].items()})
    return {f: next(iter(c.values())) for f, c in poses.items()}


# ---- probe files: meshport's FBX with curves and transform properties edited by hand

def _records(scene: Scene) -> list[Record]:
    return fbx._Writer(scene).records()


def _objects(records: list[Record], name: str) -> list[Record]:
    return next(r for r in records if r.name == "Objects").find_all(name)


def _set_keys(curve: Record, values, flags, data) -> None:
    for name, value in (("KeyValueFloat", np.asarray(values, np.float32)), ("KeyAttrFlags", np.asarray(flags, np.int32)),
                        ("KeyAttrDataFloat", np.asarray(data, np.float32).ravel()),
                        ("KeyAttrRefCount", np.ones(len(flags), np.int32))):
        curve.find(name).props = [value]


def _anchor() -> Node:
    """A second, still object: with only one, Unity folds it into the imported asset itself."""
    return Node("anchor", compose((0, -5, 0)), cube())


def _weights(right: float, next_left: float) -> np.float32:
    return np.array([round(right * 9999) | round(next_left * 9999) << 16], np.int32).view(np.float32)[0]


def _keys_probe() -> bytes:
    """x: cubic keys with their own slopes, y: cubic keys with weighted handles, z: a constant
    key and a constant-next key."""
    box = Node("box", mesh=cube())
    times = np.array([0.0, 0.5, 1.0])
    scene = Scene([box, _anchor()], [Animation("Take 001", [Channel(box, "translation", times, np.zeros((3, 3)))])])
    records = _records(scene)
    x, y, z = _objects(records, "AnimationCurve")
    user = CUBIC | 0x400  # user tangents: the stored slopes
    slopes = [4.0, -2.0, 1.0]  # each key's slope, on both sides
    _set_keys(x, [0, 1, 3], [user] * 3, [[slopes[k], slopes[min(k + 1, 2)], THIRDS, 0] for k in range(3)])
    weighted = user | 0x01000000 | 0x02000000
    _set_keys(y, [0, 2, 1], [weighted] * 3, [[3.0, -1.0, _weights(0.6, 0.15), 0], [-1.0, 5.0, _weights(0.2, 0.5), 0],
                                             [5.0, 0, THIRDS, 0]])
    _set_keys(z, [1, 2, 4], [CONSTANT, CONSTANT | 0x100, CONSTANT], [[0, 0, THIRDS, 0]] * 3)
    return write_binary(records)


# Unity converts pivots into plain transforms after the FBX SDK has read a file, and gets that wrong
# when a post rotation meets a rotation pivot, a scaling offset or a scaling pivot (even where the
# scale is 1, so the scaling pivot can't count; Blender reads those files as meshport does). So post
# rotation has a probe of its own, without them.
PROBES = {
    "pivots": {"RotationOffset": (0.2, 0, 0), "RotationPivot": (1, 0.5, 0), "ScalingOffset": (0, 0, 0.3),
               "ScalingPivot": (0, 1, 0), "PreRotation": (10, 20, 30)},
    "post": {"RotationOffset": (0.2, 0, 0), "PreRotation": (10, 20, 30), "PostRotation": (5, -15, 40)},
    "inactive": {"PreRotation": (10, 20, 30), "PostRotation": (5, -15, 40)},  # with RotationActive off
}


def _transform_probe(props: dict, active: bool) -> bytes:
    """Rotation and scale keys on a model with some of FBX's transform properties around them,
    turning in ZXY order."""
    box = Node("box", compose((1, 2, 0)), cube())
    times = np.array([0.0, 0.5, 1.0])
    turns = np.array([[0, 0, 0], [0.3, 0.5, 0.1], [0.6, 0.2, 0.9]])
    rot = np.array([fbx.decompose_trs(compose((0, 0, 0), np.degrees(a)))[1] for a in turns])
    scene = Scene([box, _anchor()], [Animation("Take 001", [Channel(box, "rotation", times, rot),
                                                 Channel(box, "scale", times, np.array([[1, 1, 1], [2, 1, 1], [1, 1, 3.0]]))])])
    records = _records(scene)
    p70 = _objects(records, "Model")[0].find("Properties70")
    for name, value in props.items():
        p70.children.append(fbx._p_vec(name, "Vector3D", "Vector", "", value))
    p70.children += [fbx._p("RotationActive", "bool", "", "", int(active)), fbx._p("RotationOrder", "enum", "", "", 4)]
    return write_binary(records)


def _total_one_probe() -> bytes:
    """rigged() with every vertex's weights halved and its clusters in Total1 mode: the rest of
    each vertex should stay with the mesh's own node."""
    scene = rigged()
    scene.roots[1].skin.weights = scene.roots[1].skin.weights * 0.5
    records = _records(scene)
    for r in _objects(records, "Deformer"):
        if r.props[2] == "Cluster":
            r.children.append(Record("Mode", ["Total1"]))
    return write_binary(records)


@pytest.fixture(scope="module")
def autodesk(tmp_path_factory):
    pytest.importorskip("scipy")
    work = tmp_path_factory.mktemp("autodesk_fbx")
    (work / "keys.fbx").write_bytes(_keys_probe())
    for name, props in PROBES.items():
        (work / f"{name}.fbx").write_bytes(_transform_probe(props, active=name != "inactive"))
    meshport.save(animated(), work / "ours.fbx")
    meshport.save(rigged(), work / "rigged.fbx")
    meshport.save(meshport.load(DATA / "skin.blend"), work / "blender_skin.fbx")
    (work / "total1.fbx").write_bytes(_total_one_probe())
    (work / "inbetween.fbx").write_bytes(inbetween_fbx())
    return work, _run_unity(work)


def _compare(scene: Scene, frames: dict, name: str) -> float:
    from scipy.spatial import cKDTree
    worst = 0.0
    for frame, (fps, meshes) in frames.items():
        ours = {n.name: n.mesh.positions @ w[:3, :3].T + w[:3, 3] for n, w in scene.walk_at(frame / fps, scene.animations)
                if n.mesh is not None}
        for mesh, points in meshes.items():
            worst = max(worst, cKDTree(ours[mesh]).query(points)[0].max())
    return worst


@pytest.mark.parametrize("probe", ["keys", *PROBES])
def test_we_read_fbx_curves_like_autodesk(autodesk, probe):
    work, poses = autodesk
    frames = poses[f"{probe}.fbx"]
    assert len(frames) > 20
    assert _compare(meshport.load(work / f"{probe}.fbx"), frames, probe) < 1e-4


def test_autodesk_reads_our_animation(autodesk):
    _, poses = autodesk
    frames = poses["ours.fbx"]
    assert len(frames) == 49  # 2 seconds at 24 frames per second
    assert _compare(animated(), frames, "ours") < 1e-4


def _compare_bent(scene: Scene, frames: dict, name: str) -> float:
    """The farthest any of Autodesk's baked points is from meshport's posed points of the mesh,
    and back. Unity's clip starts at the animation's first key."""
    from scipy.spatial import cKDTree
    start = min(a.start for a in scene.animations)
    worst = 0.0
    for frame, (fps, meshes) in frames.items():
        ours = {n.name: p for n, p in scene.posed_points(start + frame / fps, scene.animations)}[name]
        theirs = meshes[name] if name in meshes or len(meshes) != 1 else next(iter(meshes.values()))  # a lone
        # object takes the file's name in Unity
        worst = max(worst, cKDTree(ours).query(theirs)[0].max(), cKDTree(theirs).query(ours)[0].max())
    return worst


@pytest.mark.parametrize("probe", ["rigged", "blender_skin", "inbetween"])
def test_autodesk_bends_our_skins_and_blend_shapes(autodesk, probe):
    """meshport's skinned FBX, and a blend shape channel with an in-between shape (Autodesk's SDK
    blends between the half and full shapes as meshport reads them)."""
    work, poses = autodesk
    frames = poses[f"{probe}.fbx"]
    assert len(frames) > 20
    scene = meshport.load(work / f"{probe}.fbx")
    body = next(n for n, _ in scene.walk() if n.skin is not None or (n.mesh is not None and n.mesh.targets))
    assert _compare_bent(scene, frames, body.name) < 1e-4


def test_unity_normalizes_total1_weights(autodesk):
    """Unity scales every vertex's bone weights up to 1 as it imports, whatever the clusters' mode,
    so it bends the Total1 probe as rigged() bends: it can't show what Autodesk's SDK does with the
    rest of each vertex (meshport leaves it with the mesh's node)."""
    _, poses = autodesk
    assert _compare_bent(rigged(), poses["total1.fbx"], "Body") < 1e-4
