"""Materials: the shading models' parameter schema and the observable Material (a GPU uniform buffer). The schema
drives both validation and the generated UI panel.

Add a shading model: its parameter list here in SCHEMA and its fragment shader in ``shaders.FRAGMENT_WGSL``.
Nothing else needs to change.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import wgpu

__all__ = ["ALL_SPECS", "AMBIENT", "BANDS", "BASE", "EMISSIVE", "METALLIC", "Material", "ParamSpec", "ROUGHNESS",
           "SCHEMA", "SHADINGS", "SHININESS", "SPEC_COLOR", "linear_to_srgb", "srgb_to_linear"]


@dataclass(frozen=True)
class ParamSpec:
    """One editable material parameter: its widget kind, slot in the uniform buffer, default and range."""
    key: str
    label: str
    kind: str  # "color" | "float" | "int"  → decides the widget
    offset: int  # float index in the Material struct
    default: tuple
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False

    @property
    def size(self):
        """Number of floats the parameter occupies (3 for a colour, else 1)."""
        return 3 if self.kind == "color" else 1


BASE = ParamSpec("base_color", "Base colour", "color", 0, (0.8, 0.8, 0.8))
SPEC_COLOR = ParamSpec("specular_color", "Specular colour", "color", 4, (0.6, 0.6, 0.6))
SHININESS = ParamSpec("shininess", "Shininess", "float", 7, (48.0,), 1.0, 1024.0, log=True)
METALLIC = ParamSpec("metallic", "Metallic", "float", 8, (0.0,))
ROUGHNESS = ParamSpec("roughness", "Roughness", "float", 9, (0.4,), 0.04, 1.0)
AMBIENT = ParamSpec("ambient", "Ambient", "float", 10, (1.0,))  # × the scene's global ambient light
BANDS = ParamSpec("bands", "Bands", "int", 11, (3,), 1, 8)
EMISSIVE = ParamSpec("emissive", "Emissive", "color", 12, (0.0, 0.0, 0.0))
ALL_SPECS = [BASE, SPEC_COLOR, SHININESS, METALLIC, ROUGHNESS, AMBIENT, BANDS, EMISSIVE]

SCHEMA = {
    "unlit": [BASE],
    "lambert": [BASE, AMBIENT, EMISSIVE],
    "blinn_phong": [BASE, SPEC_COLOR, SHININESS, AMBIENT, EMISSIVE],
    "pbr": [BASE, METALLIC, ROUGHNESS, AMBIENT, EMISSIVE],
    "toon": [BASE, BANDS, AMBIENT, EMISSIVE],
}
SHADINGS = list(SCHEMA)


class Material:
    """Observable shading parameters backed by one 64-byte uniform buffer + bind group."""

    def __init__(self, device, layout, name: str, shading: str, **params):
        self.device, self.layout, self.name, self.shading = device, layout, name, shading
        self.data = np.zeros(16, np.float32)
        for s in ALL_SPECS:
            self.data[s.offset: s.offset + s.size] = s.default
        self.buffer = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bind_group = device.create_bind_group(
            layout=layout, entries=[{"binding": 0, "resource": {"buffer": self.buffer}}])
        self.listeners: list[Callable] = []
        for k, v in params.items():
            self.set(k, v, notify=False)
        self.dirty = True

    def copy(self, name: str) -> Material:
        """Same shading and parameters, its own GPU buffer, no listeners."""
        m = Material(self.device, self.layout, name, self.shading)
        m.data[:] = self.data
        return m

    @staticmethod
    def spec(key: str) -> ParamSpec:
        """The ParamSpec for parameter ``key``."""
        return next(s for s in ALL_SPECS if s.key == key)

    def get(self, key: str):
        """Current value of ``key``: a float, an RGB tuple, or the shading name for ``"shading"``."""
        if key == "shading":
            return self.shading
        s = self.spec(key)
        v = self.data[s.offset: s.offset + s.size]
        return tuple(float(x) for x in v) if s.kind == "color" else float(v[0])

    def set(self, key: str, value, notify: bool = True) -> None:
        """Set ``key`` (clamped to its range) and notify listeners if the value changed."""
        old = self.get(key)
        if key == "shading":
            if value not in SCHEMA:
                raise ValueError(f"unknown shading model {value!r}")
            self.shading = value
        else:
            s = self.spec(key)
            arr = np.clip(np.atleast_1d(np.asarray(value, np.float32)), s.lo, s.hi)
            self.data[s.offset: s.offset + s.size] = np.round(arr) if s.kind == "int" else arr
            self.dirty = True
        new = self.get(key)
        if notify and new != old:
            for cb in list(self.listeners):
                cb(self, key, old, new)

    def flush(self) -> None:
        """Upload the parameters to the GPU if they changed since the last flush."""
        if self.dirty:
            self.device.queue.write_buffer(self.buffer, 0, self.data)
            self.dirty = False

    def to_dict(self, digits: int = 4) -> dict:
        """Shading + the parameters it uses (rounded), e.g. ``renderer.material(name, **d)`` rebuilds it."""
        def fmt(v):
            return [round(x, digits) for x in v] if isinstance(v, tuple) else round(v, digits)
        return {"shading": self.shading, **{s.key: fmt(self.get(s.key)) for s in SCHEMA[self.shading]}}

    def to_json(self) -> str:
        """The material as pretty-printed JSON (name, shading and its parameters)."""
        items = {"name": self.name, **self.to_dict(digits=3)}
        return "{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in items.items()) + "\n}"


# colour conversion: materials store linear RGB, UIs show sRGB


def linear_to_srgb(c):
    """Convert one linear colour channel to sRGB (0–1)."""
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def srgb_to_linear(c):
    """Convert one sRGB colour channel (0–1) to linear."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
