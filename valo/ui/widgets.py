"""Shared widgets: Blender-like number sliders (``slider``), the schema-driven material editor and an sRGB colour
picker for linear colours.

A slider is a bar filled up to its value: drag sideways to change it (in ``step`` increments when it has one;
Shift for fine control, ignoring the step), double-click or Ctrl-click to type a value, Esc while dragging to put the
old value back. Dragging stays inside the soft range; a typed value may go beyond it, as far as the hard limits.
Right-click: Reset to default, Copy / Paste value and, where allowed, the range and step of that slider (view
state, kept per slider key in ``editor.slider_ranges``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from imgui_bundle import imgui
from imgui_bundle.imgui import internal

from ..gpu.material import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .style import ICON


@dataclass(frozen=True)
class SliderSpec:
    """How a slider moves: the soft range ``lo``..``hi`` (dragging stays inside), ``step`` (drag increments; None:
    smooth), ``default`` (right-click > Reset), hard limits for typed values (None: unlimited), a logarithmic scale,
    the display format, whole numbers, and whether the right-click menu may change range and step."""

    lo: float
    hi: float
    default: float | None = None
    step: float | None = None
    hard_lo: float | None = None
    hard_hi: float | None = None
    log: bool = False
    fmt: str = "%.3g"
    integer: bool = False
    editable: bool = True


def snap(value: float, step: float | None, origin: float = 0.0) -> float:
    """``value`` rounded to the nearest ``origin + k * step`` (unchanged without a step)."""
    if not step:
        return value
    return round(origin + round((value - origin) / step) * step, 10)


def limit(value: float, spec: SliderSpec) -> float | int:
    """``value`` clamped to the hard limits of ``spec`` (an int for integer sliders)."""
    if spec.hard_lo is not None:
        value = max(value, spec.hard_lo)
    if spec.hard_hi is not None:
        value = min(value, spec.hard_hi)
    return int(round(value)) if spec.integer else float(value)


def parse_number(text: str) -> float | None:
    """The number in ``text`` (e.g. pasted from the clipboard), or None if it isn't one."""
    try:
        value = float(text.strip())
    except (ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


def fill_fraction(value: float, spec: SliderSpec) -> float:
    """How much of the bar is filled for ``value``: 0..1 across the soft range (log scale for log sliders)."""
    lo, hi, v = spec.lo, spec.hi, value
    if spec.log and lo > 0 and hi > lo and v > 0:
        lo, hi, v = math.log(lo), math.log(hi), math.log(v)
    return min(max((v - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.0


class Widgets:
    """Editor part: the slider widget, the material editor (``material_section``) and the colour picker."""

    @property
    def slider_ranges(self) -> dict[str, dict]:
        """Range / step overrides made in sliders' right-click menus: slider key → {"lo", "hi", "step"}."""
        return self.__dict__.setdefault("_slider_ranges", {})

    def set_slider_range(self, key: str, **changes) -> None:
        """Change the range or step of slider ``key`` (``lo=``, ``hi=``, ``step=``; ``step=0`` for smooth); no
        keywords: back to the slider's own range. View state, so not an undo step."""
        if not changes:
            self.slider_ranges.pop(key, None)
            return
        unknown = set(changes) - {"lo", "hi", "step"}
        if unknown:
            raise KeyError(f"unknown slider setting {sorted(unknown)} (use lo, hi, step)")
        merged = {**self.slider_ranges.get(key, {}), **changes}
        if "step" in merged and not merged["step"]:
            merged["step"] = None
        if merged.get("lo") is not None and merged.get("hi") is not None and merged["lo"] >= merged["hi"]:
            raise ValueError(f"slider range needs lo < hi, not {merged['lo']} .. {merged['hi']}")
        self.slider_ranges[key] = merged

    def slider_spec(self, key: str, spec: SliderSpec) -> SliderSpec:
        """``spec`` with the range / step changes made for slider ``key``."""
        return replace(spec, **self.slider_ranges.get(key, {}))

    def slider(self, key: str, label: str, value, spec: SliderSpec) -> tuple[bool, float | int]:
        """A Blender-like slider for ``value`` (see the module docstring); ``key`` names it for its range changes.
        Returns (changed, new value)."""
        spec = self.slider_spec(key, spec)
        io = imgui.get_io()
        pos, width, height = imgui.get_cursor_screen_pos(), imgui.calc_item_width(), imgui.get_frame_height()
        rounding = imgui.get_style().frame_rounding
        draw = imgui.get_window_draw_list()
        bar = imgui.get_style_color_vec4(imgui.Col_.slider_grab)
        draw.add_rect_filled(pos, imgui.ImVec2(pos.x + width, pos.y + height),
                             imgui.get_color_u32(imgui.Col_.frame_bg), rounding)
        draw.add_rect_filled(pos, imgui.ImVec2(pos.x + width * fill_fraction(value, spec), pos.y + height),
                             imgui.get_color_u32(imgui.ImVec4(bar.x, bar.y, bar.z, 0.45)), rounding)
        for col in (imgui.Col_.frame_bg, imgui.Col_.frame_bg_active):
            imgui.push_style_color(col, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(1, 1, 1, 0.06))
        speed = (spec.hi - spec.lo) / max(width, 1.0)  # one bar width of mouse travel covers the range
        if io.key_shift:
            speed *= 0.01  # ImGui speeds Shift-drags up 10x: this makes them 10x finer instead
        flags = imgui.SliderFlags_.logarithmic if spec.log else 0
        if spec.integer:
            changed, new = imgui.drag_int(label, int(value), max(speed, 0.02), int(spec.lo), int(spec.hi), spec.fmt,
                                          flags)
        else:
            changed, new = imgui.drag_float(label, float(value), speed, spec.lo, spec.hi, spec.fmt, flags)
        imgui.pop_style_color(3)
        if imgui.is_item_activated():
            self._slider_start = (key, value)
        start = getattr(self, "_slider_start", (None, None))
        dragging = imgui.is_item_active() and start[0] == key and imgui.is_mouse_down(0)
        if dragging and spec.step and not spec.log and not io.key_shift:
            # whole steps from the total mouse travel (snapping each frame's value would creep)
            travel = imgui.get_mouse_drag_delta(0, 0.0).x * (spec.hi - spec.lo) / max(width, 1.0)
            new = min(max(snap(start[1] + travel, spec.step, spec.lo), spec.lo), spec.hi)
            changed = new != value
        if imgui.is_item_active() and start[0] == key and imgui.is_key_pressed(imgui.Key.escape, False):
            new, changed = start[1], True
            internal.clear_active_id()
        picked = self._slider_menu(key, spec, value)
        if picked is not None:
            new, changed = picked, True
        return changed, limit(new, spec) if changed else value

    def _slider_menu(self, key: str, spec: SliderSpec, value) -> float | None:
        """The right-click menu of the slider just drawn; returns the value it picked (reset, paste) or None."""
        if not imgui.begin_popup_context_item(f"##menu {key}"):
            return None
        result = None
        if spec.default is not None and imgui.menu_item_simple(f"{ICON.ARROW_ROTATE_LEFT} Reset to default"):
            result = spec.default
        if imgui.menu_item_simple(f"{ICON.COPY} Copy value"):
            imgui.set_clipboard_text(repr(value))
        pasted = parse_number(imgui.get_clipboard_text() or "")
        if imgui.menu_item(f"{ICON.PASTE} Paste value", "", False, pasted is not None)[0]:
            result = pasted
        if spec.editable:
            imgui.separator_text("Range")
            imgui.set_next_item_width(imgui.get_font_size() * 8)
            changed, lo = imgui.input_float("Min", float(spec.lo), 0.0, 0.0, "%.6g")
            if changed and lo < spec.hi:
                self.set_slider_range(key, lo=lo)
            imgui.set_next_item_width(imgui.get_font_size() * 8)
            changed, hi = imgui.input_float("Max", float(spec.hi), 0.0, 0.0, "%.6g")
            if changed and hi > spec.lo:
                self.set_slider_range(key, hi=hi)
            imgui.set_next_item_width(imgui.get_font_size() * 8)
            changed, step = imgui.input_float("Step", float(spec.step or 0.0), 0.0, 0.0, "%.6g")
            if changed and step >= 0:
                self.set_slider_range(key, step=step)
            if key in self.slider_ranges and imgui.menu_item_simple("Reset range"):
                self.set_slider_range(key)
        imgui.end_popup()
        return result

    def material_section(self, m: Material) -> None:
        """Shading picker and schema-driven parameter widgets for material ``m``."""
        changed, si = imgui.combo("Shading", SHADINGS.index(m.shading), SHADINGS)
        if changed:
            self.set_param(m, "shading", SHADINGS[si])

        imgui.separator_text(f"Material parameters ({m.shading})")
        for s in SCHEMA[m.shading]:  # the schema decides the widgets
            if s.kind == "color":
                srgb = [linear_to_srgb(c) for c in m.get(s.key)]
                changed, col = imgui.color_edit3(s.label, srgb, imgui.ColorEditFlags_.picker_hue_wheel)
                if changed:
                    self.set_param(m, s.key, [srgb_to_linear(c) for c in col])
                continue
            integer = s.kind != "float"
            spec = SliderSpec(s.lo, s.hi, s.default[0], 1 if integer else None, hard_lo=s.lo if s.lo >= 0 else None,
                              log=s.log, fmt="%.0f" if s.log or integer else "%.3f", integer=integer)
            changed, v = self.slider(f"material.{s.key}", s.label, m.get(s.key), spec)
            if changed:
                self.set_param(m, s.key, v)
        if imgui.button(f"{ICON.ARROW_ROTATE_LEFT} Reset material parameters"):
            self.reset(m)

    def _color(self, label: str, rgb) -> tuple[bool, list[float]]:
        """sRGB colour swatch for a linear colour."""
        changed, col = imgui.color_edit3(label, [linear_to_srgb(c) for c in rgb], imgui.ColorEditFlags_.no_inputs)
        return changed, [srgb_to_linear(c) for c in col]
