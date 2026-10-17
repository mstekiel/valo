"""Materials: shading models (WGSL fragment shaders), their parameter schema, and the observable
Material. The schema drives both validation and the generated UI panel.

Add a shading model: add a WGSL entry to FRAGMENT_WGSL and its parameter list to SCHEMA.
Nothing else needs to change.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import wgpu

FRAGMENT_WGSL = {
    "unlit": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f { return vec4f(material.base_color.rgb, 1.0); }""",
    "lambert": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
    let s = surface(in);
    let d = max(dot(s.n, s.l), 0.0);
    let amb = material.pbr.z;
    let light = frame.light_color.rgb;
    return vec4f(material.base_color.rgb * (amb + (1.0 - amb) * d * light) + material.emissive.rgb, 1.0);
}""",
    "blinn_phong": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
    let s = surface(in);
    let d = max(dot(s.n, s.l), 0.0);
    let spec = select(0.0, pow(max(dot(s.n, normalize(s.l + s.v)), 0.0), material.specular.w), d > 0.0);
    let amb = material.pbr.z;
    let light = frame.light_color.rgb;
    return vec4f(material.base_color.rgb * (amb + (1.0 - amb) * d * light)
                 + material.specular.rgb * spec * frame.light_color.rgb + material.emissive.rgb, 1.0);
}""",
    "pbr": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {   // metallic-roughness Cook-Torrance (GGX)
    let s = surface(in);
    let base = material.base_color.rgb;
    let metallic = material.pbr.x;
    let rough = max(material.pbr.y, 0.04);
    let h = normalize(s.l + s.v);
    let ndl = max(dot(s.n, s.l), 0.0);
    let ndv = max(dot(s.n, s.v), 1e-4);
    let ndh = max(dot(s.n, h), 0.0);
    let vdh = max(dot(s.v, h), 0.0);
    let a2 = pow(rough, 4.0);
    let dd = a2 / (PI * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
    let f0 = mix(vec3f(0.04), base, metallic);
    let f = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
    let spec = dd * g * f / max(4.0 * ndv * ndl, 1e-4);
    let direct = ((1.0 - f) * (1.0 - metallic) * base / PI + spec) * 3.0 * ndl * frame.light_color.rgb;
    let fr = f0 + (max(vec3f(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
    let env = environment(reflect(-s.v, s.n)) * fr * (1.0 - 0.75 * rough);
    let ambient = material.pbr.z * environment(s.n) * base * (1.0 - metallic);
    return vec4f(direct + env + ambient + material.emissive.rgb, 1.0);
}""",
    "toon": """
@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
    let s = surface(in);
    let bands = max(material.pbr.w, 1.0);
    let d = floor(max(dot(s.n, s.l), 0.0) * bands + 0.5) / bands;
    let rim = smoothstep(0.55, 0.75, 1.0 - max(dot(s.n, s.v), 0.0)) * 0.35;
    let amb = material.pbr.z;
    let light = frame.light_color.rgb;
    return vec4f(material.base_color.rgb * (amb + (1.0 - amb) * d * light) + vec3f(rim) + material.emissive.rgb, 1.0);
}""",
}


@dataclass(frozen=True)
class ParamSpec:
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
        return 3 if self.kind == "color" else 1


BASE = ParamSpec("base_color", "Base colour", "color", 0, (0.8, 0.8, 0.8))
SPEC_COLOR = ParamSpec("specular_color", "Specular colour", "color", 4, (0.6, 0.6, 0.6))
SHININESS = ParamSpec("shininess", "Shininess", "float", 7, (48.0,), 1.0, 1024.0, log=True)
METALLIC = ParamSpec("metallic", "Metallic", "float", 8, (0.0,))
ROUGHNESS = ParamSpec("roughness", "Roughness", "float", 9, (0.4,), 0.04, 1.0)
AMBIENT = ParamSpec("ambient", "Ambient", "float", 10, (0.15,))
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
        return next(s for s in ALL_SPECS if s.key == key)

    def get(self, key: str):
        if key == "shading":
            return self.shading
        s = self.spec(key)
        v = self.data[s.offset: s.offset + s.size]
        return tuple(float(x) for x in v) if s.kind == "color" else float(v[0])

    def set(self, key: str, value, notify: bool = True) -> None:
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
        if self.dirty:
            self.device.queue.write_buffer(self.buffer, 0, self.data)
            self.dirty = False

    def to_dict(self, digits: int = 4) -> dict:
        """Shading + the parameters it uses (rounded), e.g. ``renderer.material(name, **d)`` rebuilds it."""
        def fmt(v):
            return [round(x, digits) for x in v] if isinstance(v, tuple) else round(v, digits)
        return {"shading": self.shading, **{s.key: fmt(self.get(s.key)) for s in SCHEMA[self.shading]}}

    def to_json(self) -> str:
        items = {"name": self.name, **self.to_dict(digits=3)}
        return "{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in items.items()) + "\n}"


# colour conversion: materials store linear RGB, UIs show sRGB


def linear_to_srgb(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
