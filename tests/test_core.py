"""Fast tests: pure Python/numpy, no GPU."""

import numpy as np
import pytest

from valo import meshes
from valo.geometry import instance_matrices, pack_mat4, perspective_reversed_z, scale, translate
from valo.materials import ALL_SPECS, SCHEMA, linear_to_srgb, srgb_to_linear
from valo.undo import Command, UndoStack

R, L = 0.25, 1.0  # capsule defaults
CLOSED_VOLUMES = {  # analytic volume of each closed standard mesh with default arguments
    "sphere": np.pi / 6, "icosphere": np.pi / 6, "cube": 1.0, "cylinder": np.pi / 4, "cone": np.pi / 12,
    "torus": 2 * np.pi**2 * 0.35 * 0.15**2, "capsule": np.pi * R**2 * (L - 2 * R) + 4 / 3 * np.pi * R**3,
    "arrow": np.pi * 0.15**2 * 0.65 + np.pi * 0.5**2 * 0.35 / 3,
}


def _triangles(mesh):
    """Corner positions p0, p1, p2 (each (T, 3)) and the mean vertex normal (T, 3) of every triangle."""
    p, n, t = mesh.positions.astype(float), mesh.normals.astype(float), mesh.indices.reshape(-1, 3)
    return p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], n[t].mean(1)


@pytest.mark.parametrize("name", list(meshes.STANDARD))
def test_standard_meshes_are_well_formed_and_wound_outward(name):
    mesh = meshes.STANDARD[name]()
    assert mesh.vertices.dtype == np.float32 and mesh.vertices.shape[1] == 6
    assert mesh.indices.dtype == np.uint32 and mesh.indices.max() < len(mesh.vertices)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    p0, p1, p2, n = _triangles(mesh)
    face = np.cross(p1 - p0, p2 - p0)
    real = np.linalg.norm(face, axis=1) > 1e-9  # repeated profile points make zero-area seams
    assert (np.einsum("ij,ij->i", face, n)[real] > 0).all(), "triangles wound against their normals"


@pytest.mark.parametrize("name", list(CLOSED_VOLUMES))
def test_closed_meshes_enclose_their_analytic_volume(name):
    p0, p1, p2, _ = _triangles(meshes.STANDARD[name]())
    volume = np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6  # divergence theorem; > 0 when outward
    assert volume == pytest.approx(CLOSED_VOLUMES[name], rel=0.05)  # inscribed tessellation is a bit smaller


def test_plane_is_double_sided():
    n = meshes.plane().normals
    assert {tuple(x) for x in n} == {(0, 0, 1), (0, 0, -1)}


def test_transform_keeps_normals_perpendicular_under_non_uniform_scale():
    ellipsoid = meshes.transform(meshes.sphere(16, 32), scale(1, 1, 3))
    p, n = ellipsoid.positions, ellipsoid.normals
    grad = p / np.array([1, 1, 9])  # gradient of x² + y² + (z/3)²
    assert np.allclose(n, grad / np.linalg.norm(grad, axis=1, keepdims=True), atol=1e-4)


def test_merge_offsets_indices():
    a, b = meshes.cube(), meshes.plane()
    m = meshes.merge(a, b)
    assert len(m.vertices) == len(a.vertices) + len(b.vertices)
    assert m.indices[len(a.indices):].min() == len(a.vertices)


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


def test_schema_slots_fit_the_material_struct_without_overlap():
    for model, specs in SCHEMA.items():
        used = [i for s in specs for i in range(s.offset, s.offset + s.size)]
        assert len(used) == len(set(used)), model
        assert max(used) < 16, model
    assert {s.key for s in ALL_SPECS} >= {s.key for specs in SCHEMA.values() for s in specs}


def test_srgb_round_trip():
    for c in (0.0, 0.002, 0.2, 0.5, 1.0):
        assert srgb_to_linear(linear_to_srgb(c)) == pytest.approx(c, abs=1e-9)


def test_undo_merges_rapid_edits_of_same_key():
    value = {"x": 0}
    stack = UndoStack()

    def edit(new):
        old, value["x"] = value["x"], new
        stack.push(Command("x", lambda: value.update(x=old), lambda: value.update(x=new), merge_key=("x",)))

    for v in (1, 2, 3):  # like a slider drag
        edit(v)
    assert len(stack.done) == 1
    stack.undo()
    assert value["x"] == 0
    stack.redo()
    assert value["x"] == 3
