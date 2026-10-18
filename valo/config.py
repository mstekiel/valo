"""Global configuration: ``valo.rc``, typed sections of defaults and preferences, overridable from TOML files.

    from valo import rc, rc_context
    rc.camera.orbit_speed            # attribute access
    rc["highlight.strength"] = 0.5   # dotted access, validated
    with rc_context({"camera.fov": 50}): ...

Load order: the defaults below → ``~/.valo/valorc.toml`` → ``./valorc.toml`` → the file in ``$VALO_RC``.
A file holds only the keys you change, grouped by section (``[camera]``, ``[keys]``, ...). Write a fully
commented file with every key: ``valo-config --write-defaults [path]``.

``rc`` holds defaults and preferences; live state stays on the objects (``viewer.camera``, ``viewer.lighting``).
"""

from __future__ import annotations

import copy
import difflib
import os
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path

USER_FILE = Path.home() / ".valo" / "valorc.toml"


def opt(default, label: str, lo=None, hi=None, unit: str = "", choices: tuple = (), help: str = ""):
    """A config field: its default plus the metadata used for validation, the docs and the Preferences window."""
    return field(default=default, metadata={"label": label, "lo": lo, "hi": hi, "unit": unit,
                                            "choices": choices, "help": help})


class Section:
    """Base of a config section: every assignment is checked against the field's type, range and choices."""

    def __setattr__(self, name, value):
        spec = {f.name: f for f in fields(self)}.get(name)
        if spec is None:
            raise AttributeError(_unknown(name, [f.name for f in fields(self)], f"[{self.NAME}]"))
        super().__setattr__(name, _check(f"{self.NAME}.{name}", spec, value))


def _check(key: str, spec, value):
    default, meta = spec.default, spec.metadata
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false, not {value!r}")
    elif isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number, not {value!r}")
        value = type(default)(value) if isinstance(default, float) or float(value).is_integer() else value
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError(f"{key} must be a whole number, not {value!r}")
        if (meta["lo"] is not None and value < meta["lo"]) or (meta["hi"] is not None and value > meta["hi"]):
            raise ValueError(f"{key} = {value} is outside [{meta['lo']}, {meta['hi']}]")
    elif isinstance(default, tuple):
        if not isinstance(value, (tuple, list)) or len(value) != len(default):
            raise ValueError(f"{key} must be a list of {len(default)} numbers, not {value!r}")
        value = tuple(type(d)(v) for d, v in zip(default, value, strict=True))
        if meta["lo"] is not None and any(not meta["lo"] <= v <= meta["hi"] for v in value):
            raise ValueError(f"{key} values must be in [{meta['lo']}, {meta['hi']}]")
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string, not {value!r}")
        if meta["choices"] and value not in meta["choices"]:
            raise ValueError(f"{key} must be one of {list(meta['choices'])}, not {value!r}")
    return value


def _unknown(name: str, known: list[str], where: str) -> str:
    hint = difflib.get_close_matches(name, known, n=1, cutoff=0.5)
    return f"unknown key {name!r} in {where}" + (f" (did you mean {hint[0]!r}?)" if hint else "")


@dataclass
class CameraConfig(Section):
    """Mouse and keyboard navigation, and the starting camera."""

    NAME = "camera"
    orbit_speed: float = opt(0.4, "Orbit speed", 0.02, 3.0, "deg/px", help="left-drag rotation per pixel")
    pan_speed: float = opt(1.0, "Pan speed", 0.1, 5.0, "x", help="1 = the target plane follows the cursor")
    zoom_speed: float = opt(1.0, "Zoom speed", 0.1, 5.0, "x", help="wheel zoom per notch, relative")
    fov: float = opt(35.0, "Field of view", 10.0, 90.0, "deg", help="starting perspective field of view")
    orbit_step: float = opt(15.0, "Orbit step", 1.0, 90.0, "deg", help="one arrow-key press")
    auto_rotate_speed: float = opt(24.0, "Auto-rotate speed", 1.0, 360.0, "deg/s")
    center_duration: float = opt(0.25, "Centring animation", 0.0, 2.0, "s", help="0 = jump")


@dataclass
class LightingConfig(Section):
    """Starting light and global ambient (colours are linear RGB)."""

    NAME = "lighting"
    mode: str = opt("headlight", "Light mode", choices=("headlight", "fixed"))
    azimuth: float = opt(-30.0, "Azimuth", -180.0, 180.0, "deg")
    elevation: float = opt(30.0, "Elevation", -90.0, 90.0, "deg")
    color: tuple = opt((1.0, 1.0, 1.0), "Light colour", 0.0, 1.0, "linear RGB")
    intensity: float = opt(1.0, "Intensity", 0.0, 10.0)
    sky: tuple = opt((0.55, 0.62, 0.72), "Ambient sky", 0.0, 1.0, "linear RGB")
    ground: tuple = opt((0.10, 0.09, 0.08), "Ambient ground", 0.0, 1.0, "linear RGB")
    ambient: float = opt(0.3, "Ambient intensity", 0.0, 5.0)


@dataclass
class RenderConfig(Section):
    """Scene background and the defaults of offline rendering (valo.render, File > Render image)."""

    NAME = "render"
    background: tuple = opt((0.02, 0.022, 0.026), "Background", 0.0, 1.0, "linear RGB", help="behind the scene")
    size: tuple = opt((1600, 1200), "Image size", 16, 16384, "px", help="offline render width, height")
    samples: int = opt(2, "Supersampling", 1, 4, "x", help="render n times larger and downscale: smoother edges")
    transparent: bool = opt(False, "Transparent background", help="offline PNGs get alpha 0 behind the scene")


@dataclass
class HighlightConfig(Section):
    """Tint of the hovered and selected object."""

    NAME = "highlight"
    enabled: bool = opt(True, "Highlight hovered object")
    color: tuple = opt((1.0, 0.8, 0.25), "Colour", 0.0, 1.0, "linear RGB")
    strength: float = opt(0.35, "Hover strength", 0.0, 2.0)
    selected_strength: float = opt(0.6, "Selection strength", 0.0, 2.0)


@dataclass
class PickingConfig(Section):
    """Object lookup under the mouse."""

    NAME = "picking"
    enabled: bool = opt(True, "Pick objects under the mouse")
    double_click_ms: int = opt(300, "Double-click time", 100, 1000, "ms")


@dataclass
class UIConfig(Section):
    """Look and layout of the panels."""

    NAME = "ui"
    font_size: float = opt(16.0, "Font size", 10.0, 32.0, "px")
    scale: float = opt(1.0, "UI scale", 0.5, 3.0, "x")
    accent: tuple = opt((66, 133, 244), "Accent colour", 0, 255, "sRGB 0-255")
    right_column: float = opt(0.3, "Right column width", 0.1, 0.7, "share")
    bottom_share: float = opt(0.42, "Lower dock height", 0.1, 0.9, "share")
    window_background: tuple = opt((0.12, 0.12, 0.13), "Window background", 0.0, 1.0, "sRGB 0-1",
                                   help="around the 3D view, behind the panels")


@dataclass
class KeysConfig(Section):
    """Keyboard shortcuts in the 3D view: a key name ("f", "Home", "Escape", "ArrowLeft", "1", ...)."""

    NAME = "keys"
    frame_selection: str = opt("f", "Frame selection")
    fit_all: str = opt("Home", "Fit everything")
    view_front: str = opt("1", "Front view (Ctrl: back)")
    view_right: str = opt("3", "Right view (Ctrl: left)")
    view_top: str = opt("7", "Top view (Ctrl: bottom)")
    toggle_ortho: str = opt("5", "Toggle orthographic")
    deselect: str = opt("Escape", "Deselect")
    orbit_left: str = opt("ArrowLeft", "Orbit left (Shift: pan)")
    orbit_right: str = opt("ArrowRight", "Orbit right (Shift: pan)")
    orbit_up: str = opt("ArrowUp", "Orbit up (Shift: pan)")
    orbit_down: str = opt("ArrowDown", "Orbit down (Shift: pan)")


@dataclass
class SculptorConfig(Section):
    """MeshSculptor."""

    NAME = "sculptor"
    poll_s: float = opt(0.3, "File poll interval", 0.05, 5.0, "s")
    max_normal_arrows: int = opt(2000, "Max normal arrows", 10, 100000)


SECTIONS = {cls.NAME: cls for cls in (CameraConfig, LightingConfig, RenderConfig, HighlightConfig, PickingConfig,
                                      UIConfig, KeysConfig, SculptorConfig)}


class RC:
    """All config sections: ``rc.<section>.<key>`` or ``rc["section.key"]``."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Back to the built-in defaults."""
        for name, cls in SECTIONS.items():
            object.__setattr__(self, name, cls())

    def __setattr__(self, name, value):
        raise AttributeError("set keys, not sections: rc.camera.fov = 40 or rc['camera.fov'] = 40")

    def _split(self, key: str) -> tuple[Section, str]:
        section, _, name = key.partition(".")
        if section not in SECTIONS:
            raise KeyError(_unknown(section, list(SECTIONS), "rc"))
        return getattr(self, section), name

    def __getitem__(self, key: str):
        sec, name = self._split(key)
        if name not in {f.name for f in fields(sec)}:
            raise KeyError(_unknown(name, [f.name for f in fields(sec)], f"[{sec.NAME}]"))
        return getattr(sec, name)

    def __setitem__(self, key: str, value) -> None:
        sec, name = self._split(key)
        try:
            setattr(sec, name, value)
        except AttributeError as exc:
            raise KeyError(str(exc)) from None

    def items(self):
        """All (``"section.key"``, value) pairs."""
        for name in SECTIONS:
            sec = getattr(self, name)
            for f in fields(sec):
                yield f"{name}.{f.name}", getattr(sec, f.name)

    def update(self, values: dict) -> None:
        """Set several keys: ``{"camera.fov": 40}`` or nested ``{"camera": {"fov": 40}}``."""
        for key, value in values.items():
            if isinstance(value, dict):
                if key not in SECTIONS:
                    raise KeyError(_unknown(key, list(SECTIONS), "rc"))
                self.update({f"{key}.{k}": v for k, v in value.items()})
            else:
                self[key] = value

    def load(self, path) -> None:
        """Apply the keys of a TOML file (errors name the file)."""
        try:
            with open(path, "rb") as fh:
                self.update(tomllib.load(fh))
        except (KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"{path}: {exc.args[0] if exc.args else exc}") from None

    def load_files(self) -> list[Path]:
        """Apply the user file, the project file and ``$VALO_RC`` (each if it exists); returns the files read."""
        candidates = (USER_FILE, Path.cwd() / "valorc.toml", os.environ.get("VALO_RC"))
        found = [p for p in candidates if p and Path(p).is_file()]
        for p in found:
            self.load(p)
        return [Path(p) for p in found]

    def changed(self) -> dict:
        """The keys whose value differs from the default, as ``{"section.key": value}``."""
        default = RC()
        return {k: v for k, v in self.items() if default[k] != v}

    def to_toml(self, only_changed: bool = False) -> str:
        """The config as TOML; ``only_changed`` writes only non-default keys, else every key with its help."""
        keep = self.changed() if only_changed else dict(self.items())
        out = [] if only_changed else ["# valo configuration. Delete any line to use the default.", ""]
        for name, cls in SECTIONS.items():
            sec = getattr(self, name)
            rows = [f for f in fields(sec) if f"{name}.{f.name}" in keep]
            if not rows:
                continue
            out.append(f"[{name}]" + ("" if only_changed else f"  # {cls.__doc__}"))
            for f in rows:
                line = f"{f.name} = {_toml_value(getattr(sec, f.name))}"
                if not only_changed:
                    m = f.metadata
                    bits = [m["label"], m["unit"], f"{m['lo']}..{m['hi']}" if m["lo"] is not None else "",
                            "/".join(m["choices"]), m["help"]]
                    line = f"{line:40s} # " + ", ".join(b for b in bits if b)
                out.append(line)
            out.append("")
        return "\n".join(out)

    def save(self, path=USER_FILE) -> Path:
        """Write the non-default keys to ``path`` (the user file by default)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml(only_changed=True), encoding="utf-8")
        return path


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return repr(v)


@contextmanager
def rc_context(values: dict | None = None):
    """Temporarily change config keys (restored on exit), e.g. in tests: ``with rc_context({"camera.fov": 50}):``."""
    saved = {name: copy.copy(getattr(rc, name)) for name in SECTIONS}
    try:
        rc.update(values or {})
        yield rc
    finally:
        for name, sec in saved.items():
            object.__setattr__(rc, name, sec)


rc = RC()
rc.load_files()


def main(argv=None) -> None:
    """``valo-config``: print the active config, or ``--write-defaults [path]`` for a commented file of every key."""
    import argparse

    ap = argparse.ArgumentParser(prog="valo-config", description=main.__doc__)
    ap.add_argument("--write-defaults", nargs="?", const=str(USER_FILE), metavar="PATH",
                    help=f"write every key with its default and help (default path: {USER_FILE})")
    args = ap.parse_args(argv)
    if args.write_defaults:
        path = Path(args.write_defaults)
        if path.exists():
            ap.error(f"{path} exists; remove it first")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RC().to_toml(), encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(rc.to_toml())
