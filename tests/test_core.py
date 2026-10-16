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


@pytest.mark.parametrize("name", list(meshes.STANDARD))
def test_standard_meshes_are_well_formed_and_wound_outward(name):
    mesh = meshes.STANDARD[name]()
    assert mesh.vertices.dtype == np.float32 and mesh.vertices.shape[1] == 6
    assert mesh.indices.dtype == np.uint32 and mesh.indices.max() < len(mesh.vertices)
    report = meshes.check(mesh)
    assert report.ok, report.lines()


@pytest.mark.parametrize("name", list(CLOSED_VOLUMES))
def test_closed_meshes_are_watertight_with_their_analytic_volume(name):
    report = meshes.check(meshes.STANDARD[name]())
    assert report.watertight
    assert report.volume == pytest.approx(CLOSED_VOLUMES[name], rel=0.05)  # inscribed tessellation is a bit smaller


def test_plane_is_double_sided_and_open():
    plane = meshes.plane()
    assert {tuple(x) for x in plane.normals} == {(0, 0, 1), (0, 0, -1)}
    assert not meshes.check(plane).watertight


def test_check_flags_inside_out_and_bad_normals():
    cube = meshes.cube()
    inside_out = meshes.MeshData(cube.vertices, cube.indices.reshape(-1, 3)[:, ::-1].ravel().copy())
    assert meshes.check(inside_out).flipped == 12
    assert meshes.check(inside_out).volume == pytest.approx(-1.0)
    long_normals = meshes.MeshData(cube.vertices * np.r_[1, 1, 1, 2, 2, 2].astype(np.float32), cube.indices)
    assert meshes.check(long_normals).bad_normals == 24
    nan_normals = meshes.MeshData(cube.vertices * np.r_[1, 1, 1, np.nan, 1, 1].astype(np.float32), cube.indices)
    assert meshes.check(nan_normals).bad_normals == 24


def test_revolve_open_profile_ending_on_a_repeated_point_has_unit_normals():
    assert meshes.check(meshes.revolve([(0, 0), (0.5, 0), (0.5, 0), (0.5, 1), (0.5, 1)], 16)).ok


def test_check_sees_a_hole():
    cube = meshes.cube()
    assert not meshes.check(meshes.MeshData(cube.vertices, cube.indices[:-3])).watertight


# -- builders


def test_revolve_computes_normals_with_hard_edges_at_repeated_points():
    h, r = 1.0, 0.5
    cyl = meshes.revolve([(0, 0), (r, 0), (r, 0), (r, h), (r, h), (0, h)], segments=64)
    report = meshes.check(cyl)
    assert report.ok and report.watertight
    assert report.volume == pytest.approx(np.pi * r**2 * h, rel=0.01)
    side = np.abs(cyl.positions[:, 2] - 0.5) < 0.4  # no side vertex is near a cap
    assert np.allclose(cyl.normals[side, 2], 0, atol=1e-6)
    assert np.allclose(meshes.revolve([(0, 0), (r, 0), (r, 0), (r, h), (r, h), (0, h)], 64).vertices, cyl.vertices)


def test_parametric_sphere_is_outward_and_closed_and_flip_reverses_it():
    def p(u, v):
        th, ph = np.pi * u, 2 * np.pi * v
        return 0.5 * np.sin(th) * np.cos(ph), 0.5 * np.sin(th) * np.sin(ph), 0.5 * np.cos(th)

    report = meshes.check(meshes.parametric(p, 32, 64))
    assert report.ok and report.watertight
    assert report.volume == pytest.approx(np.pi / 6, rel=0.02)
    flipped = meshes.check(meshes.parametric(p, 32, 64, flip=True))
    assert flipped.ok and flipped.volume == pytest.approx(-report.volume)


def test_heightfield_normals_follow_the_gradient():
    field = meshes.heightfield(lambda x, y: 0.3 * x + 0.1 * y**2, size=2.0, n=16)
    p, n = field.positions, field.normals
    expected = np.c_[-0.3 * np.ones(len(p)), -0.2 * p[:, 1], np.ones(len(p))]
    assert np.allclose(n, expected / np.linalg.norm(expected, axis=1, keepdims=True), atol=1e-3)
    assert meshes.check(field).ok


# -- sculpt files and the library

SCULPT = '''"""A test sculpt."""
import numpy as np  # comment kept

from valo.meshes import revolve

RANGES = {"height": (0.1, 4.0)}


def mesh(height: float = 1.0, radius=0.5, segments: int = 24, capped: bool = True):
    # profile comment kept
    top = [(radius, height), (0, height)] if capped else [(radius, height)]
    return revolve([(0, 0), (radius, 0), (radius, 0), (radius, height), *top], segments)
'''


def test_sculpt_reads_parameters_kinds_and_ranges(tmp_path):
    from valo.sculpt import Sculpt

    (tmp_path / "s.py").write_text(SCULPT, encoding="utf-8")
    s = Sculpt.load(tmp_path / "s.py")
    assert [(p.name, p.kind, p.default) for p in s.params] == [
        ("height", "float", 1.0), ("radius", "float", 0.5), ("segments", "int", 24), ("capped", "bool", True)]
    assert (s.params[0].lo, s.params[0].hi) == (0.1, 4.0)  # from RANGES
    assert (s.params[1].lo, s.params[1].hi) == (0.0, 1.0)  # 0 .. 2 × default
    assert s.material is None and s.transform["scale"] == [1.0, 1.0, 1.0]


def test_sculpt_save_round_trip_keeps_comments_and_bakes_transform(tmp_path):
    from valo.sculpt import Sculpt

    (tmp_path / "s.py").write_text(SCULPT, encoding="utf-8")
    s = Sculpt.load(tmp_path / "s.py")
    params = {"height": 2.5, "radius": 0.25, "segments": 40, "capped": False}
    material = {"shading": "toon", "base_color": [0.1, 0.2, 0.3], "bands": 4.0}
    tf = {"translate": [0.0, 0.0, 1.0], "rotate_deg": [90.0, 0.0, 0.0], "scale": [2.0, 2.0, 2.0]}
    saved = Sculpt.load(s.save(tmp_path / "lib" / "tuned.py", params, material, tf))

    assert saved.defaults() == params
    assert saved.material == material and saved.transform == tf
    assert np.array_equal(saved.build().vertices, s.build(params, tf).vertices)
    text = saved.source
    assert "# comment kept" in text and "# profile comment kept" in text and 'RANGES = {"height": (0.1, 4.0)}' in text
    tip = saved.build({"capped": True})  # baked rotation turns +Z (the height) into −Y
    assert tip.positions[:, 1].min() == pytest.approx(-2 * 2.5, abs=1e-5)

    again = Sculpt.load(saved.save(tmp_path / "lib" / "tuned.py", {"height": 1.5}, material, tf))
    assert again.defaults()["height"] == 1.5 and again.source.count("MATERIAL =") == 1


def test_sculpt_errors_are_clear(tmp_path):
    from valo.sculpt import Sculpt

    (tmp_path / "a.py").write_text("def mesh(n):\n    pass\n", encoding="utf-8")
    with pytest.raises(TypeError, match="needs a default"):
        Sculpt.load(tmp_path / "a.py")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="no mesh"):
        Sculpt.load(tmp_path / "b.py")


def _library_names():
    from valo import library

    return library.names()


@pytest.mark.parametrize("name", _library_names())
def test_library_meshes_pass_the_checks(name):
    from valo import library

    report = meshes.check(library.mesh(name))
    assert report.ok, report.lines()


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
