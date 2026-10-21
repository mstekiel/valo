"""Shared widgets: the schema-driven material editor and an sRGB colour picker for linear colours."""

from __future__ import annotations

from imgui_bundle import imgui

from ..gpu.material import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .style import ICON


class Widgets:
    """Editor part: the material editor widget (``material_section``) and the colour picker it uses."""

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
            elif s.kind == "float":
                fl = imgui.SliderFlags_.logarithmic if s.log else 0
                changed, v = imgui.slider_float(s.label, m.get(s.key), s.lo, s.hi, "%.0f" if s.log else "%.3f", fl)
                if changed:
                    self.set_param(m, s.key, v)
            else:
                changed, v = imgui.slider_int(s.label, int(m.get(s.key)), int(s.lo), int(s.hi))
                if changed:
                    self.set_param(m, s.key, v)
        if imgui.button(f"{ICON.ARROW_ROTATE_LEFT} Reset material parameters"):
            self.reset(m)

    def _color(self, label: str, rgb) -> tuple[bool, list[float]]:
        """sRGB colour swatch for a linear colour."""
        changed, col = imgui.color_edit3(label, [linear_to_srgb(c) for c in rgb], imgui.ColorEditFlags_.no_inputs)
        return changed, [srgb_to_linear(c) for c in col]
