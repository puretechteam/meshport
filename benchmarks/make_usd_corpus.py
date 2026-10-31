"""Run with Pixar's USD (pip install usd-core): write a folder of USD files that use the parts of
the format readers most often get wrong, for compare_usd.py to check meshport against.

    python benchmarks/make_usd_corpus.py OUT_DIR

Each file comes in text and binary form, and binary ones also in older crate versions.
"""

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt


def grid_mesh(stage, path, n=24, lift=0.0, integer=False):
    m = UsdGeom.Mesh.Define(stage, path)
    pts = [(i if integer else i * 0.37, (round(math.sin(i * 0.3) * 4) if integer else math.sin(i * 0.3 + j * 0.2)) + lift,
            j if integer else j * 0.41) for i in range(n) for j in range(n)]
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4] * ((n - 1) ** 2))
    m.CreateFaceVertexIndicesAttr([v for i in range(n - 1) for j in range(n - 1)
                                   for v in (i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j)])
    return m


def textured_material(stage, path, texture):
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("uvset")
    tex = UsdShade.Shader.Define(stage, f"{path}/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    # through a material interface input, the way many exporters do it
    mat_in = mat.CreateInput("tint", Sdf.ValueTypeNames.Float)
    mat_in.Set(0.25)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(mat_in)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def composition(out: Path):
    """Inherits, specializes and point instancers."""
    # an asset whose root inherits a class kept outside it, in the asset's own file
    asset = Usd.Stage.CreateNew(str(out / "asset_class.usda"))
    UsdGeom.SetStageMetersPerUnit(asset, 1)
    cls = asset.CreateClassPrim("/_class_Prop")
    UsdGeom.Xformable(cls).AddTranslateOp().Set(Gf.Vec3d(0, 2, 0))
    grid_mesh(asset, "/_class_Prop/FromClass", 4, 1)
    prop = UsdGeom.Xform.Define(asset, "/Prop")
    prop.GetPrim().GetInherits().AddInherit("/_class_Prop")
    grid_mesh(asset, "/Prop/Own", 3)
    asset.SetDefaultPrim(prop.GetPrim())
    asset.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(out / "composition.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # a class in this file: inherited, with a local opinion stronger than the class's
    tree = stage.CreateClassPrim("/_class_Tree")
    UsdGeom.Xformable(tree).AddTranslateOp().Set(Gf.Vec3d(0, 0, 7))
    grid_mesh(stage, "/_class_Tree/Leaves", 5, 3)
    for i, name in enumerate(("TreeA", "TreeB")):
        t = UsdGeom.Xform.Define(stage, f"/World/{name}")
        t.GetPrim().GetInherits().AddInherit("/_class_Tree")
        if i:
            t.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
                Gf.Vec3d(20, 0, 0))  # stronger than the class's (0, 0, 7); the op order comes from the class
    # the stage adds to the class the referenced asset inherits: an implied inherit
    stage.CreateClassPrim("/_class_Prop")
    grid_mesh(stage, "/_class_Prop/FromStage", 3, 5)
    for i in range(2):
        p = UsdGeom.Xform.Define(stage, f"/World/Prop{i}")
        p.GetPrim().GetReferences().AddReference("./asset_class.usda")
        UsdGeom.XformCommonAPI(p).SetTranslate(Gf.Vec3d(-10 * (i + 1), 0, 0))
    # specializes: weaker than a reference, so the referenced asset's own opinions win
    base = stage.CreateClassPrim("/_base")
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3d(0, 50, 0))
    grid_mesh(stage, "/_base/Detail", 3, 1)
    spec = UsdGeom.Xform.Define(stage, "/World/Spec")
    spec.GetPrim().GetSpecializes().AddSpecialize("/_base")
    spec_ref = stage.DefinePrim("/World/SpecRef")
    spec_ref.GetSpecializes().AddSpecialize("/_base")
    spec_ref.GetReferences().AddReference("./asset_class.usda")
    # an inherit loop: A and B inherit each other (Pixar reports it and carries on)
    a = UsdGeom.Xform.Define(stage, "/World/LoopA")
    b = UsdGeom.Xform.Define(stage, "/World/LoopB")
    a.GetPrim().GetInherits().AddInherit("/World/LoopB")
    b.GetPrim().GetInherits().AddInherit("/World/LoopA")
    grid_mesh(stage, "/World/LoopA/G", 3)
    # a point instancer: two prototypes (one moved itself, one a referenced asset), six instances,
    # one hidden and one deactivated
    inst = UsdGeom.PointInstancer.Define(stage, "/World/Forest")
    UsdGeom.XformCommonAPI(inst).SetTranslate(Gf.Vec3d(0, 0, -30))
    cone = grid_mesh(stage, "/World/Forest/Protos/Cone", 4, 2)
    UsdGeom.XformCommonAPI(cone).SetTranslate(Gf.Vec3d(0, 1, 0))
    rock = stage.DefinePrim("/World/Forest/Protos/Rock")
    rock.GetReferences().AddReference("./asset_class.usda")
    inst.CreatePrototypesRel().SetTargets([cone.GetPath(), rock.GetPath()])
    inst.CreateProtoIndicesAttr([0, 1, 0, 1, 0, 0])
    inst.CreatePositionsAttr([(i * 12.0, 0, i * 3.0) for i in range(6)])
    inst.CreateOrientationsAttr([Gf.Quath(math.cos(i * 0.4), 0, math.sin(i * 0.4), 0) for i in range(6)])
    inst.CreateScalesAttr([(1 + i * 0.25, 1, 1 - i * 0.1) for i in range(6)])
    inst.CreateIdsAttr([10, 11, 12, 13, 14, 15])
    inst.CreateInvisibleIdsAttr([12])
    inst.DeactivateId(14)
    stage.GetRootLayer().Save()


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    composition(out)
    for src in ("composition", "asset_class"):
        Sdf.Layer.FindOrOpen(str(out / f"{src}.usda")).Export(str(out / f"{src}.usdc"))
    tex = out / "checker.png"
    tex.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000002000000020802000000fdd49a73000000164944415478"
        "9c63f8cfc0f01f0c18fe33303000002f0405fd8b45e3a10000000049454e44ae426082"))

    # an asset with its own variants, used by reference and by payload
    asset = Usd.Stage.CreateNew(str(out / "asset.usda"))
    UsdGeom.SetStageMetersPerUnit(asset, 1)
    root = UsdGeom.Xform.Define(asset, "/Asset")
    asset.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    vset = root.GetPrim().GetVariantSets().AddVariantSet("shape")
    for name, size in (("big", 3.0), ("small", 1.0)):
        vset.AddVariant(name)
        vset.SetVariantSelection(name)
        with vset.GetVariantEditContext():
            body = grid_mesh(asset, "/Asset/Body", 6, size)
            UsdGeom.XformCommonAPI(body).SetScale(Gf.Vec3f(size, 1, 1))
    vset.SetVariantSelection("small")
    m = UsdGeom.Mesh.Define(asset, "/Asset/Seat")
    m.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("uvset", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)])
    mat = textured_material(asset, "/Asset/Looks/Checker", "./checker.png")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    asset.GetRootLayer().Save()

    # the main scene: Z up, centimeters, references, payloads, a sublayer, instancing
    sub = Usd.Stage.CreateNew(str(out / "sublayer.usda"))
    over = sub.OverridePrim("/World/Moved")
    UsdGeom.Xformable(over).AddTranslateOp().Set(Gf.Vec3d(0, 0, 50))
    sub.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(out / "scene.usda"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().subLayerPaths.append("./sublayer.usda")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for i in range(3):
        p = UsdGeom.Xform.Define(stage, f"/World/Chair{i}")
        p.GetPrim().GetReferences().AddReference("./asset.usda")
        p.GetPrim().SetInstanceable(i > 0)
        UsdGeom.XformCommonAPI(p).SetTranslate(Gf.Vec3d(i * 150, 0, 0))
        UsdGeom.XformCommonAPI(p).SetRotate(Gf.Vec3f(0, 0, i * 30), UsdGeom.XformCommonAPI.RotationOrderZYX)
    p.GetPrim().GetVariantSet("shape").SetVariantSelection("big")
    lazy = UsdGeom.Xform.Define(stage, "/World/Lazy")
    lazy.GetPrim().GetPayloads().AddPayload("./asset.usda", "/Asset")
    UsdGeom.Xform.Define(stage, "/World/Moved")
    grid_mesh(stage, "/World/Moved/Grid", 10)
    # every primvar interpolation, indexed primvars, left-handed, holes
    odd = grid_mesh(stage, "/World/Odd", 5, 2)
    odd.CreateOrientationAttr("leftHanded")
    odd.CreateHoleIndicesAttr([0, 5])
    pv = UsdGeom.PrimvarsAPI(odd)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, "uniform").Set([(1, 0, 0)] * 16)
    st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    st.SetIndices(Vt.IntArray([0, 1, 2, 3] * 16))
    odd.CreateNormalsAttr([(0, 1, 0)] * 25)
    odd.SetNormalsInterpolation("vertex")
    # transform ops of many kinds
    ops = UsdGeom.Xform.Define(stage, "/World/Ops")
    ops.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    ops.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(5, 0, 0))
    ops.AddRotateYXZOp().Set(Gf.Vec3f(10, 50, 70))
    ops.AddOrientOp().Set(Gf.Quatf(0.9238795, 0.3826834, 0, 0))
    ops.AddScaleOp().Set(Gf.Vec3f(1, 2, 3))
    ops.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
    UsdGeom.Cube.Define(stage, "/World/Ops/Box").CreateSizeAttr(20)
    m4 = UsdGeom.Xform.Define(stage, "/World/Matrix")
    m4.AddTransformOp().Set(Gf.Matrix4d(Gf.Rotation(Gf.Vec3d(1, 1, 0), 33), Gf.Vec3d(-40, 3, 9)))
    grid_mesh(stage, "/World/Matrix/G", 4)
    reset = UsdGeom.Xform.Define(stage, "/World/Matrix/Reset")
    reset.SetResetXformStack(True)
    reset.AddTranslateOp().Set(Gf.Vec3d(0, -100, 0))
    grid_mesh(stage, "/World/Matrix/Reset/G", 3)
    # hidden, guide and inactive things stay out
    UsdGeom.Imageable(grid_mesh(stage, "/World/Hidden", 3)).MakeInvisible()
    UsdGeom.Imageable(grid_mesh(stage, "/World/Guide", 3)).CreatePurposeAttr("guide")
    grid_mesh(stage, "/World/Off", 3).GetPrim().SetActive(False)
    # animated: first sample
    anim = UsdGeom.Xform.Define(stage, "/World/Anim")
    op = anim.AddTranslateOp()
    op.Set(Gf.Vec3d(-300, 0, 0), 10)
    op.Set(Gf.Vec3d(-200, 0, 0), 20)
    grid_mesh(stage, "/World/Anim/G", 3)
    # big integer and repetitive float arrays: the crate compresses these ('i' and 't')
    grid_mesh(stage, "/World/Integer", 40, integer=True)
    flat = grid_mesh(stage, "/World/Flat", 30, 9)
    flat.CreateNormalsAttr([(0, 1, 0), (0, 0, 1), (1, 0, 0)] * 300)
    flat.SetNormalsInterpolation("vertex")
    stage.GetRootLayer().Save()

    # binary copies, the newest crate version and older ones
    for src in ("scene", "asset", "sublayer"):
        Sdf.Layer.FindOrOpen(str(out / f"{src}.usda")).Export(str(out / f"{src}.usdc"))
    text = (out / "scene.usda").read_text()
    (out / "scene_bin.usda").write_text(text.replace("./asset.usda", "./asset.usdc").replace(
        "./sublayer.usda", "./sublayer.usdc"))
    Sdf.Layer.FindOrOpen(str(out / "scene_bin.usda")).Export(str(out / "scene_bin.usdc"))
    (out / "scene_bin.usda").unlink()
    for version in ("0.7.0", "0.8.0", "0.9.0", "0.10.0"):
        env = {**os.environ, "USD_WRITE_NEW_USDC_FILES_AS_VERSION": version}
        subprocess.run([sys.executable, "-c",
                        "import sys; from pxr import Sdf; Sdf.Layer.FindOrOpen(sys.argv[1]).Export(sys.argv[2])",
                        str(out / "scene.usda"), str(out / f"scene_v{version}.usdc")], env=env, check=True)

    # packages: everything in one .usdz (references and texture inside), and ARKit's own packager
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(out / "scene.usda")), str(out / "scene.usdz"))
    UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(str(out / "asset.usda")), str(out / "asset_arkit.usdz"))
    shutil.copy(out / "checker.png", out / "unused.png")
    print("corpus written to", out)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
