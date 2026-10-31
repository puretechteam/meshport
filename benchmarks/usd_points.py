"""Run with Pixar's USD (pip install usd-core): print each mesh's world-space points, as Pixar's
library composes the stage, converted to meshport's convention (meters, Y up). JSON on stdout.

    python benchmarks/usd_points.py model.usdz [--check]

--check also runs every validator Pixar's library has (USDZ package rules included) and reports
what they find.
"""

import json
import sys

import numpy as np
from pxr import Usd, UsdGeom, UsdValidation


def main(path: str, check: bool) -> None:
    stage = Usd.Stage.Open(path)
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    convert = np.diag([meters] * 3)
    if up == "Z":
        convert = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]) @ convert
    time = Usd.TimeCode.EarliestTime()
    cache = UsdGeom.XformCache(time)
    out = {}

    def points(prim):
        if prim.IsA(UsdGeom.Cube):  # a cube's corners
            s = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            return [(x, y, z) for x in (-s, s) for y in (-s, s) for z in (-s, s)]
        if prim.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(prim).GetPointsAttr().Get(time)
        return None

    def add(key, pts, m):  # m: row vectors
        p = np.array(pts, float)
        out[key] = ((np.hstack([p, np.ones((len(p), 1))]) @ m)[:, :3] @ convert.T).round(6).tolist()

    def drawn(prim):
        img = UsdGeom.Imageable(prim)
        return img.ComputeVisibility() != UsdGeom.Tokens.invisible and img.ComputePurpose() not in ("guide", "proxy")

    instancers = []
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in it:
        if prim.IsA(UsdGeom.PointInstancer):
            instancers.append(prim)
            it.PruneChildren()  # prototypes are drawn only as instances
            continue
        pts = points(prim)
        if pts and drawn(prim):
            add(str(prim.GetPath()), pts, np.array(cache.GetLocalToWorldTransform(prim), float))
    for prim in instancers:
        if not drawn(prim):
            continue
        inst = UsdGeom.PointInstancer(prim)
        xforms = inst.ComputeInstanceTransformsAtTime(time, time, UsdGeom.PointInstancer.IncludeProtoXform,
                                                      UsdGeom.PointInstancer.IgnoreMask)
        mask = inst.ComputeMaskAtTime(time)
        protos = inst.GetPrototypesRel().GetTargets()
        indices = inst.GetProtoIndicesAttr().Get(time)
        base = np.array(cache.GetLocalToWorldTransform(prim), float)
        for i, (k, x) in enumerate(zip(indices, xforms)):
            if mask and not mask[i]:
                continue
            proto = stage.GetPrimAtPath(protos[k])
            for p in Usd.PrimRange(proto, Usd.TraverseInstanceProxies()):
                pts = points(p)
                if pts and drawn(p):
                    rel, _ = cache.ComputeRelativeTransform(p, proto)  # below the prototype root
                    add(f"{prim.GetPath()}[{i}]{p.GetPath()}", pts,
                        np.array(rel, float) @ np.array(x, float) @ base)
    result = {"meshes": out}
    if check:
        registry = UsdValidation.ValidationRegistry()
        context = UsdValidation.ValidationContext(registry.GetOrLoadAllValidators())
        result["problems"] = [f"{e.GetName()}: {e.GetMessage()}" for e in context.Validate(stage)]
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1], "--check" in sys.argv)
