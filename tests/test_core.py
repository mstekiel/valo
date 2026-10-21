"""Fast tests: pure Python/numpy, no GPU."""

import numpy as np
import pytest

from valo.geometry import instance_matrices, pack_mat4, perspective_reversed_z, translate
from valo.materials import ALL_SPECS, SCHEMA, linear_to_srgb, srgb_to_linear
from valo.undo import Command, UndoStack


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


def test_orthographic_reversed_z_maps_near_to_one_and_far_to_zero():
    from valo.geometry import orthographic_reversed_z

    m = orthographic_reversed_z(4.0, 2.0, 1.0, 9.0)
    for z, depth in ((-1.0, 1.0), (-9.0, 0.0)):
        p = m @ np.array([0, 0, z, 1.0])
        assert p[2] / p[3] == pytest.approx(depth)
    assert (m @ np.array([4.0, 2.0, -5, 1]))[:2] == pytest.approx([1, 1])  # half width = height·aspect/2


def test_switching_projection_keeps_the_target_plane_the_same_size():
    from valo.render import OrbitCamera

    cam = OrbitCamera(distance=10.0)
    point = np.asarray(cam.target) + np.cross(cam.forward(), (0, 0, 1)) * 1.5  # beside the target, on its plane

    def ndc():
        vp, _ = cam.matrices(1.5)
        p = vp @ np.r_[point, 1.0]
        return p[:2] / p[3]

    persp = ndc()
    cam.set_projection("orthographic")
    assert ndc() == pytest.approx(persp, abs=1e-5)
    cam.zoom(2.0)
    cam.set_projection("perspective")
    assert cam.distance == pytest.approx(20.0)


def test_headlight_follows_the_camera_and_fixed_light_does_not():
    from valo.lighting import Lighting
    from valo.render import OrbitCamera

    cam = OrbitCamera()
    head, fixed = Lighting("headlight", 0.0, 0.0), Lighting("fixed", 30.0, 40.0)
    assert head.direction(cam) == pytest.approx(-cam.forward())  # azimuth 0, elevation 0: from the viewer
    before = fixed.direction(cam)
    cam.azimuth += 120
    assert head.direction(cam) == pytest.approx(-cam.forward())
    assert fixed.direction(cam) == pytest.approx(before)


@pytest.mark.parametrize("start", ["headlight", "fixed"])
def test_switching_light_mode_keeps_the_direction(start):
    from valo.lighting import MODES, Lighting
    from valo.render import OrbitCamera

    cam = OrbitCamera(azimuth=37.0, elevation=-12.0)
    light = Lighting(start, -50.0, 25.0)
    d = light.direction(cam)
    light.set_mode(next(m for m in MODES if m != start), cam)
    assert light.direction(cam) == pytest.approx(d, abs=1e-9)


def test_camera_changes_keep_the_scene_size():
    from valo.render import OrbitCamera

    radius = 4.75

    def outline_share(cam):  # visible height share of the bounding sphere's outline
        if cam.projection == "orthographic":
            return 2 * radius / cam.ortho_height
        return np.tan(np.arcsin(radius / cam.distance)) / np.tan(np.radians(cam.fovy / 2))

    cam = OrbitCamera(distance=15.0)
    share = outline_share(cam)
    for change in (lambda: cam.set_fov(60, radius), lambda: cam.set_fov(20, radius),
                   lambda: cam.set_projection("orthographic", radius),
                   lambda: cam.set_projection("perspective", radius)):
        change()
        assert outline_share(cam) == pytest.approx(share)
    cam.set_fov(35, radius)
    assert cam.distance == pytest.approx(15.0)  # back to the start: same distance


# -- config


def test_rc_validates_types_ranges_choices_and_names():
    from valo.config import RC

    rc = RC()
    rc["camera.fov"] = 40
    assert rc.camera.fov == 40.0 and isinstance(rc.camera.fov, float)
    for key, bad in (("camera.fov", 200), ("camera.fov", "wide"), ("lighting.mode", "spot"),
                     ("highlight.selected_color", (1, 1)), ("highlight.enabled", 1), ("picking.double_click_ms", 2.5)):
        with pytest.raises(ValueError, match=key.split(".")[1]):
            rc[key] = bad
    with pytest.raises(KeyError, match="did you mean 'fov'"):
        rc["camera.fovv"] = 1
    with pytest.raises(AttributeError):
        rc.camera = None


def test_rc_files_load_in_order_and_round_trip(tmp_path, monkeypatch):
    from valo import config

    user, project, env = tmp_path / "user.toml", tmp_path / "valorc.toml", tmp_path / "env.toml"
    user.write_text('[camera]\nfov = 40\norbit_speed = 0.5\n', encoding="utf-8")
    project.write_text('[camera]\nfov = 45\n[keys]\nfit_all = "h"\n', encoding="utf-8")
    env.write_text("[highlight]\nback_alpha = 0.1\n", encoding="utf-8")
    monkeypatch.setattr(config, "USER_FILE", user)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VALO_RC", str(env))
    rc = config.RC()
    assert rc.load_files() == [user, project, env]
    assert (rc.camera.fov, rc.camera.orbit_speed, rc.keys.fit_all, rc.highlight.back_alpha) == (45.0, 0.5, "h", 0.1)

    saved = config.RC()
    saved.load(rc.save(tmp_path / "out.toml"))  # only the changed keys are written
    assert dict(saved.items()) == dict(rc.items())
    full = tmp_path / "full.toml"
    full.write_text(config.RC().to_toml(), encoding="utf-8")  # the commented file with every key
    loaded = config.RC()
    loaded.load(full)
    assert dict(loaded.items()) == dict(config.RC().items())

    (tmp_path / "bad.toml").write_text("[camera]\nzoom = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.toml.*did you mean 'zoom_speed'"):
        config.RC().load(tmp_path / "bad.toml")


def test_rc_context_restores_everything():
    from valo import rc, rc_context

    before = dict(rc.items())
    with rc_context({"camera.fov": 60, "highlight": {"back_alpha": 1.0}}):
        assert rc.camera.fov == 60.0 and rc.highlight.back_alpha == 1.0
        rc.ui.scale = 2.0
    assert dict(rc.items()) == before


def test_lighting_defaults_come_from_rc():
    from valo import Lighting, rc_context

    with rc_context({"lighting.mode": "fixed", "lighting.ambient": 0.7}):
        light = Lighting()
    assert (light.mode, light.ambient) == ("fixed", 0.7)
    assert Lighting().mode == "headlight"


def test_valo_config_writes_a_commented_defaults_file(tmp_path):
    from valo.config import RC, main

    path = tmp_path / "valorc.toml"
    main(["--write-defaults", str(path)])
    text = path.read_text(encoding="utf-8")
    assert "[camera]" in text and "orbit_speed = 0.4" in text and "deg/px" in text
    loaded = RC()
    loaded.load(path)
    assert dict(loaded.items()) == dict(RC().items())
    with pytest.raises(SystemExit):
        main(["--write-defaults", str(path)])  # never overwrites


def test_decompose_returns_what_instance_matrices_built():
    from valo.geometry import axis_angle, decompose, euler_xyz, rotate
    from valo.meshes.sculpt import transform_matrix

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


def test_pan_moves_the_target_in_the_view_plane():
    from valo.render import OrbitCamera

    cam = OrbitCamera(azimuth=-90.0, elevation=0.0)  # looking along +Y: right = +X, up = +Z
    cam.pan(1.0, 2.0)
    assert cam.target == pytest.approx((-1.0, 0.0, 2.0))  # the scene follows the cursor
    assert cam.world_per_pixel(500) == pytest.approx(cam._height_at_target() / 500)


def test_center_on_animates_to_the_goal_and_ends_exactly():
    from valo.render import OrbitCamera

    cam = OrbitCamera(distance=10.0)
    cam.center_on((1.0, 2.0, 3.0), radius=1.0, duration=0.5)
    assert cam.animating and cam.target == (0.0, 0.0, 0.0)
    cam.step(0.25)
    assert 0 < cam.target[0] < 1.0 and cam.fit_distance(1.0) < cam.distance < 10.0
    cam.step(0.3)
    assert not cam.animating
    assert cam.target == pytest.approx((1, 2, 3)) and cam.distance == pytest.approx(cam.fit_distance(1.0))
    cam.center_on((0.0, 0.0, 0.0), duration=0)  # 0 jumps
    assert cam.target == (0.0, 0.0, 0.0) and not cam.animating


def test_example_valorc_loads_with_and_without_its_commented_defaults(tmp_path):
    import re
    from pathlib import Path

    from valo.config import RC

    example = Path(__file__).parents[1] / "examples" / "valorc.toml"
    rc = RC()
    rc.load(example)  # the active example values
    assert rc.camera.orbit_speed == 0.25 and rc.lighting.mode == "fixed" and rc.keys.fit_all == "h"
    text = example.read_text(encoding="utf-8")
    everything = re.sub(r"(?m)^# (\w+ = )", r"\1", text)  # uncomment the "# key = value" default lines too
    (tmp_path / "all.toml").write_text(everything, encoding="utf-8")
    full = RC()
    full.load(tmp_path / "all.toml")  # every documented key exists and its value is valid
    documented = set(re.findall(r"(?m)^\[(\w+)\]", text)), set(re.findall(r"(?m)^#? ?(\w+) = ", text))
    assert documented[0] == {k.split(".")[0] for k, _ in RC().items()}
    assert documented[1] == {k.split(".")[1] for k, _ in RC().items()}  # no key left undocumented


def test_py_literal_writes_readable_valid_python():
    import ast

    from valo.offline import py_literal

    data = {"scene": {"n": [6, 6, 6], "name": 'say "hi"', "materials": [{"name": "M", "rgb": [0.1, 0.2, 0.3]}] * 4},
            "image": {"transparent": True, "background": None, "samples": 2}}
    text = py_literal(data)
    assert ast.literal_eval(text) == data
    assert '"n": [6, 6, 6],' in text and '\n    "image": ' in text and max(map(len, text.splitlines())) <= 100


def test_box_layout_projects_the_12_edges_and_marks_the_back_ones():
    from valo import OrbitCamera
    from valo.ui import box_layout

    cam = OrbitCamera(azimuth=-60.0, elevation=25.0, distance=6.0)  # looking at the origin from front-right, above
    edges = box_layout(cam, np.eye(4), (0, 0, 800, 600))
    assert len(edges) == 12 and sum(back for *_, back in edges) == 3  # the 3 edges at the hidden corner
    xs = [p[0] for s, e, _ in edges for p in (s, e)]
    assert 0 < min(xs) < 400 < max(xs) < 800  # around the middle of the view
    cam.projection = "orthographic"
    assert sum(back for *_, back in box_layout(cam, np.eye(4), (0, 0, 800, 600))) == 3
    cam.projection = "perspective"

    behind = np.eye(4)
    behind[:3, 3] = cam.eye() - 3 * cam.forward()  # a unit box behind the camera: nothing to draw
    assert box_layout(cam, behind, (0, 0, 800, 600)) == []
    around = np.diag([20.0, 20.0, 20.0, 1.0])  # the camera inside a big box: edges crossing the near side are cut
    edges = box_layout(cam, around, (0, 0, 800, 600))
    assert 0 < len(edges) <= 12 and all(np.isfinite(p).all() for s, e, _ in edges for p in (s, e))
