"""valo.rc: typed, validated settings; TOML files in order; rc_context; valo-config; the example valorc.toml."""

import pytest


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
