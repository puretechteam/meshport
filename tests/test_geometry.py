import numpy as np
import pytest
from conftest import l_shape

from meshport.geometry import compose, decompose, euler, triangulate


def _area(p, tris):
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2


def test_concave_polygon_is_ear_clipped():
    mesh = l_shape()
    tris = mesh.corners[triangulate(mesh.positions, mesh.face_sizes, mesh.corners)]
    assert len(tris) == 4
    assert _area(mesh.positions, tris) == pytest.approx(3.0)  # an L of three unit squares
    normals = np.cross(mesh.positions[tris[:, 1]] - mesh.positions[tris[:, 0]],
                       mesh.positions[tris[:, 2]] - mesh.positions[tris[:, 0]])
    assert (normals[:, 2] > 0).all()  # same winding as the polygon


def test_convex_faces_are_fanned_in_order():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 2, 2]], float)
    tris = triangulate(p, np.array([4, 3]), np.array([0, 1, 2, 3, 0, 1, 4]))
    np.testing.assert_array_equal(tris, [[0, 1, 2], [0, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize("r", [(10, 20, 30), (-170, 45, 5), (0, 90, 0), (33, -90, 12)])
def test_decompose_inverts_compose(r):
    m = compose((1, 2, 3), r, (2, 0.5, 3))
    t, rr, s = decompose(m)
    np.testing.assert_allclose(compose(t, rr, s), m, atol=1e-9)


def test_euler_order():
    x90z90 = euler((90, 0, 90), "XYZ") @ [1, 0, 0, 1]
    np.testing.assert_allclose(x90z90[:3], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose((euler((90, 0, 90), "ZYX") @ [1, 0, 0, 1])[:3], [0, 0, 1], atol=1e-12)
