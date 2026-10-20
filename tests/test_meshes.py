"""Mesh tests (pure numpy): primitives, builders, check(), sculpt files and the library."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from valo import meshes
from valo.geometry import scale

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
    from valo.meshes import Sculpt

    (tmp_path / "s.py").write_text(SCULPT, encoding="utf-8")
    s = Sculpt.load(tmp_path / "s.py")
    assert [(p.name, p.kind, p.default) for p in s.params] == [
        ("height", "float", 1.0), ("radius", "float", 0.5), ("segments", "int", 24), ("capped", "bool", True)]
    assert (s.params[0].lo, s.params[0].hi) == (0.1, 4.0)  # from RANGES
    assert (s.params[1].lo, s.params[1].hi) == (0.0, 1.0)  # 0 .. 2 × default
    assert s.material is None and s.transform["scale"] == [1.0, 1.0, 1.0]


def test_sculpt_save_round_trip_keeps_comments_and_bakes_transform(tmp_path):
    from valo.meshes import Sculpt

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
    from valo.meshes import Sculpt

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




@pytest.mark.parametrize("path", sorted((Path(__file__).parents[1] / "examples" / "sculpts").glob("*.py")),
                         ids=lambda p: p.stem)
def test_starter_sculpts_build_clean_meshes(path):
    from valo.meshes import Sculpt

    report = meshes.check(Sculpt.load(path).build())
    assert report.ok and report.triangles > 0, report.lines()


def test_library_material_comes_from_the_saved_preset(tmp_path, monkeypatch):
    from valo import rc
    from valo.meshes import library

    assert library.directory() == library.PROJECT_DIR / "data" / "procedural_meshes"  # the default
    monkeypatch.setattr(rc.studio, "procedural_meshes", str(tmp_path))
    (tmp_path / "blob.py").write_text(SCULPT.replace('RANGES = {"height": (0.1, 4.0)}',
                                                     'MATERIAL = {"shading": "toon", "bands": 5.0}'), encoding="utf-8")
    (tmp_path / "plain.py").write_text(SCULPT, encoding="utf-8")
    made = []
    renderer = SimpleNamespace(material=lambda name, shading, **params: made.append((name, shading, params)))
    library.material(renderer, "blob")
    library.material(renderer, "plain")
    assert made == [("blob", "toon", {"bands": 5.0}), ("plain", "pbr", {})]
    assert library.names() == ["blob", "plain"]
    with pytest.raises(KeyError):
        library.load("missing")


@pytest.mark.parametrize("name", list(meshes.STANDARD))
def test_every_primitive_has_editable_parameters_with_sane_ranges(name):
    from valo.meshes.params import shape_params

    params = shape_params(meshes.STANDARD[name])
    assert params and all(p.kind in ("float", "int") and p.lo <= p.default <= p.hi for p in params)
    assert all(p.lo >= 3 for p in params if p.kind == "int" and p.name in ("segments", "n_lat", "n_lon", "nu", "nv"))


def test_shape_params_need_defaults_and_simple_types():
    from valo.meshes.params import shape_params

    def ok(a: float = 1.0, n: int = 4, flag: bool = True, untyped=0.5):
        pass

    assert [(p.name, p.kind) for p in shape_params(ok)] == [("a", "float"), ("n", "int"), ("flag", "bool"),
                                                             ("untyped", "float")]
    with pytest.raises(TypeError, match="needs a default"):
        shape_params(lambda x: x)
    with pytest.raises(TypeError, match="unsupported"):
        shape_params(lambda s="text": s)


def test_transform_matrix_is_scale_then_rotate_then_translate():
    from valo.geometry import transform_matrix
    from valo.meshes.sculpt import transform_matrix as from_dict

    m = transform_matrix((1, 2, 3), (0, 0, 90), (2, 1, 1))
    assert np.allclose(m @ [1, 0, 0, 1], [1, 4, 3, 1])  # x scaled to 2, turned onto +y, moved
    t = {"translate": [1, 2, 3], "rotate_deg": [10, 20, 30], "scale": [1, 2, 3]}
    assert np.allclose(from_dict(t), transform_matrix(t["translate"], t["rotate_deg"], t["scale"]))


def test_tube_along_a_straight_line_is_a_capped_cylinder():
    m = meshes.tube([(0, 0, 0), (0, 0, 2.0)], radius=0.5, segments=64)
    report = meshes.check(m)
    assert report.ok and report.watertight
    polygon = 64 * np.sin(2 * np.pi / 64) / (2 * np.pi)  # area of the 64-gon / area of the circle
    assert report.volume == pytest.approx(np.pi * 0.25 * 2.0 * polygon, rel=1e-6)
    side = m.positions[np.abs(m.normals[:, 2]) < 0.5]
    assert np.allclose(np.linalg.norm(side[:, :2], axis=1), 0.5, atol=1e-6)


@pytest.mark.parametrize("closed", [False, True])
def test_tube_follows_a_curve_without_twisting(closed):
    a = np.linspace(0, 2 * np.pi * (1 if closed else 3), 240, endpoint=not closed)
    path = np.c_[np.cos(a), np.sin(a), 0 * a if closed else 0.15 * a]  # a ring, or a spring of 3 turns
    m = meshes.tube(path, radius=0.1, segments=24, closed=closed)
    report = meshes.check(m)
    assert report.ok and report.watertight
    if closed:  # a torus: 2π²Rr², up to the polygons
        assert report.volume == pytest.approx(2 * np.pi**2 * 0.01, rel=0.02)
    rows = len(path) + closed  # a loop repeats its first ring
    rings = m.positions[: 25 * rows].reshape(25, rows, 3)  # side vertices: angle × path point
    centre = path if not closed else np.r_[path, path[:1]]
    assert np.allclose(np.linalg.norm(rings - centre, axis=-1), 0.1, atol=1e-5)  # every ring around its point
    first = rings[0] - centre  # the seam line (angle 0) turns smoothly: no twist jumps along the path
    turn = np.einsum("ij,ij->i", first[1:], first[:-1]) / 0.01
    assert turn.min() > 0.95


def test_tube_radius_can_vary_and_bad_paths_are_explained():
    m = meshes.tube([(0, 0, 0), (0, 0, 1), (0, 0, 2)], radius=[0.2, 0.4, 0.2], segments=32)
    assert meshes.check(m).ok
    with pytest.raises(ValueError, match="at least 2"):
        meshes.tube([(0, 0, 0)])
    with pytest.raises(ValueError, match="repeats a point"):
        meshes.tube([(0, 0, 0), (0, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="radius"):
        meshes.tube([(0, 0, 0), (1, 0, 0)], radius=0)
