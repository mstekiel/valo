"""valo.math: matrices in math convention, reversed-Z projections, instance matrices and their decomposition."""

import numpy as np
import pytest

from valo.math import instance_matrices, pack_mat4, perspective_reversed_z, translate


def test_instance_matrices_point_z_along_axis_with_given_length():
    origins = np.array([[0, 0, 0], [1, 2, 3], [-1, 0, 0], [0, 0, 5], [2, 2, 2]], float)
    vectors = np.array([[1, 0, 0], [0, 2, 0], [0, 0, -3], [0, 0, 4], [1, 1, 1]], float)
    length = np.linalg.norm(vectors, axis=1)
    m = instance_matrices(origins, np.c_[0.1 * np.ones(5), 0.1 * np.ones(5), length], z_axis=vectors)
    tips = np.einsum("nij,j->ni", m, [0, 0, 1, 1])[:, :3]
    assert np.allclose(tips, origins + vectors, atol=1e-5)
    rot = m[:, :3, :3] / np.array([0.1, 0.1, 1])[None, :] / np.c_[np.ones(5), np.ones(5), length][:, None, :]
    assert np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    assert np.allclose(np.linalg.det(rot), 1.0, atol=1e-5)


def test_pack_mat4_puts_translation_where_wgsl_expects_it():
    """WGSL mat4x4f is column-major: column 3 (translation) is floats 12..14."""
    assert np.allclose(pack_mat4(translate(1, 2, 3)).ravel()[12:15], [1, 2, 3])


def test_reversed_z_depth_is_one_at_near_plane():
    p = perspective_reversed_z(40.0, 1.0, near=0.1) @ np.array([0, 0, -0.1, 1.0])
    assert p[2] / p[3] == pytest.approx(1.0)


def test_orthographic_reversed_z_maps_near_to_one_and_far_to_zero():
    from valo.math import orthographic_reversed_z

    m = orthographic_reversed_z(4.0, 2.0, 1.0, 9.0)
    for z, depth in ((-1.0, 1.0), (-9.0, 0.0)):
        p = m @ np.array([0, 0, z, 1.0])
        assert p[2] / p[3] == pytest.approx(depth)
    assert (m @ np.array([4.0, 2.0, -5, 1]))[:2] == pytest.approx([1, 1])  # half width = height·aspect/2


def test_decompose_returns_what_instance_matrices_built():
    from valo.math import axis_angle, decompose, euler_xyz, rotate
    from valo.meshes.shapefile import transform_matrix

    m = instance_matrices([[1.0, -2.0, 3.0]], [[0.5, 2.0, 1.5]], z_axis=[[1.0, 1.0, 0.0]])[0]
    t, r, s = decompose(m)
    assert t == pytest.approx([1, -2, 3]) and s == pytest.approx([0.5, 2.0, 1.5])
    assert r @ [0, 0, 1] == pytest.approx(np.array([1, 1, 0]) / np.sqrt(2), abs=1e-6)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)

    axis, angle = axis_angle(rotate((1, 2, 2), np.radians(70))[:3, :3])
    assert axis == pytest.approx(np.array([1, 2, 2]) / 3) and angle == pytest.approx(70)
    assert axis_angle(np.eye(3))[1] == 0.0
    for ax in ((0, 1, 0), (1, -1, 0), (1, 2, -2)):  # 180° takes another code path
        axis, angle = axis_angle(rotate(ax, np.pi)[:3, :3])
        assert abs(axis @ np.array(ax)) == pytest.approx(np.linalg.norm(ax)) and angle == pytest.approx(180, abs=1e-4)

    tm = transform_matrix({"rotate_deg": [10.0, -25.0, 140.0]})
    assert euler_xyz(decompose(tm)[1]) == pytest.approx([10, -25, 140])
