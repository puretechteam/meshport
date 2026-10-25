import io
import shutil
from pathlib import Path

import numpy as np
import pytest

from meshport import Image, Material, Mesh, Node, Scene
from meshport.geometry import compose


def png(color=(255, 0, 0), size=4) -> bytes:
    from PIL import Image as PILImage
    out = io.BytesIO()
    PILImage.new("RGB", (size, size), color).save(out, "PNG")
    return out.getvalue()


def cube() -> Mesh:
    p = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    corners = np.array(quads).ravel()
    sizes = np.full(6, 4)
    normals = np.repeat(np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float), 4, 0)
    uv0 = np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1)).astype(float)
    uv1 = uv0 * 0.5 + 0.25
    colors = np.tile([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0.5]], (6, 1)).astype(float)
    return Mesh("cube", p, sizes, corners, normals, [uv0, uv1], colors, np.array([0, 0, 0, 1, 1, 1]))


def l_shape() -> Mesh:
    """A concave hexagon: fanning it from corner 0 would put a triangle outside the shape."""
    p = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], float)
    corners = np.array([3, 4, 5, 0, 1, 2])  # start at the reflex corner
    return Mesh("ell", p, np.array([6]), corners)


@pytest.fixture
def scene() -> Scene:
    red = Material("red", (1, 1, 1, 1), Image("checker", png()), roughness=0.4)
    blue = Material("blue", (0.1, 0.2, 0.9, 0.5), alpha_mode="BLEND", emissive=(0.1, 0, 0))
    box = cube()
    child = Node("box", compose((0, 2, 0), (10, 20, 30), (1, 2, 1)), box, [red, blue])
    twin = Node("box_mirror", compose((3, 0, 0), (0, 0, 0), (-1, 1, 1)), box, [red, blue])
    ell = Node("ell", compose((0, 0, -3)), l_shape(), [blue])
    root = Node("root", compose((1, 0, 0), (0, 45, 0)), children=[child, twin, ell])
    return Scene([root])


def world_triangles(scene: Scene) -> np.ndarray:
    """Every triangle in world space, as an order-independent sorted (T, 9) array."""
    out = []
    for node, world in scene.walk():
        if node.mesh is None:
            continue
        m = node.mesh
        pts = m.positions @ world[:3, :3].T + world[:3, 3]
        for tri in pts[m.corners[m.triangles()]].round(5) + 0.0:  # + 0.0 folds -0.0 into 0.0
            out.append(tuple(sorted(map(tuple, tri))))
    return sorted(out)


def nodes_by_name(scene: Scene) -> dict:
    return {n.name: (n, w) for n, w in scene.walk()}


BLENDER = shutil.which("blender") or next(
    (str(p) for p in sorted(Path("C:/Program Files/Blender Foundation").glob("*/blender.exe"), reverse=True)), None)
