"""Fast tests: pure Python/numpy, no GPU."""

import numpy as np
import pytest

from valo.gpu.material import ALL_SPECS, SCHEMA, linear_to_srgb, srgb_to_linear
from valo.ui.undo import Command, UndoStack


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


def navigator(**scene):
    from types import SimpleNamespace

    from valo import Camera
    from valo.ui.navigation import Navigator

    viewer = SimpleNamespace(camera=Camera(distance=10.0), scene=SimpleNamespace(**scene), selection=None,
                             viewport=(0, 0, 800, 600))
    return Navigator(viewer), viewer.camera


def test_center_on_eases_to_the_goal_and_ends_exactly():
    nav, cam = navigator()
    nav.center_on((1.0, 2.0, 3.0), radius=1.0, duration=0.5)
    assert nav.animating and cam.target == (0.0, 0.0, 0.0)
    nav.step(0.25)
    assert cam.target[0] == pytest.approx(0.5) and cam.fit_distance(1.0) < cam.distance < 10.0  # half way
    nav.step(0.3)
    assert not nav.animating
    assert cam.target == pytest.approx((1, 2, 3)) and cam.distance == pytest.approx(cam.fit_distance(1.0))
    nav.center_on((0.0, 0.0, 0.0), duration=0)  # 0 jumps
    assert cam.target == (0.0, 0.0, 0.0) and not nav.animating


def test_navigation_keys_presets_and_fitting_move_the_camera():
    nav, cam = navigator(center=lambda: np.array([4.0, 0, 0]), bounding_radius=lambda: 1.0)
    nav.fit_all()  # eased onto the scene's centre, sized to its radius
    nav.step(10.0)
    assert cam.target == pytest.approx((4.0, 0.0, 0.0)) and cam.distance < 10.0
    nav.set_view("+X")
    nav.toggle_projection()
    assert (cam.azimuth, cam.projection) == (0.0, "orthographic")
    assert nav.key("5") and cam.projection == "perspective"
    assert nav.key("ArrowLeft") and cam.azimuth == pytest.approx(345.0)
    target = cam.target
    assert nav.key("ArrowUp", ("Shift",)) and cam.target != target  # Shift+arrow pans
    assert nav.key("7") and cam.elevation == pytest.approx(89.99)
    assert not nav.key("q")  # not a navigation key


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
    from valo import Camera
    from valo.ui import box_layout

    cam = Camera(azimuth=-60.0, elevation=25.0, distance=6.0)  # looking at the origin from front-right, above
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
