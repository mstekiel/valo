"""The Preferences window: every ``valo.rc`` setting as a widget, saved to the user's valorc.toml."""

from __future__ import annotations

from dataclasses import fields

from imgui_bundle import imgui

from ...config import SECTIONS, USER_FILE, rc
from ...materials import linear_to_srgb, srgb_to_linear
from ..style import ICON, apply_style


def _pref_widget(f, value) -> tuple[bool, object]:
    """One generated widget for a config field: checkbox, slider, colour, combo or text."""
    m, label = f.metadata, f"{f.metadata['label']}##{f.name}"
    unit = f" {m['unit']}" if m["unit"] and m["unit"] != "x" else ""
    if isinstance(value, bool):
        return imgui.checkbox(label, value)
    if isinstance(value, int):
        return imgui.slider_int(label, value, m["lo"], m["hi"], f"%d{unit}")
    if isinstance(value, float):
        return imgui.slider_float(label, value, m["lo"], m["hi"], f"%.3g{unit}")
    if isinstance(value, tuple) and len(value) == 3:
        if m["hi"] == 255:  # sRGB 0-255
            changed, col = imgui.color_edit3(label, [c / 255 for c in value], imgui.ColorEditFlags_.no_inputs)
            return changed, tuple(round(c * 255) for c in col)
        changed, col = imgui.color_edit3(label, [linear_to_srgb(c) for c in value], imgui.ColorEditFlags_.no_inputs)
        return changed, tuple(srgb_to_linear(c) for c in col)
    if m["choices"]:
        changed, i = imgui.combo(label, list(m["choices"]).index(value), list(m["choices"]))
        return changed, m["choices"][i]
    return imgui.input_text(label, value)


class PreferencesWindow:
    """Editor part: the Preferences window and the UI scale."""

    def set_preference(self, key: str, value) -> None:
        """Set config key ``key`` (e.g. "camera.orbit_speed") and apply it where it takes effect immediately."""
        rc[key] = value
        if key in ("ui.scale", "ui.accent"):
            self.set_ui_scale(rc.ui.scale)

    def preferences_window(self) -> None:
        """Every ``rc`` key, with widgets generated from the config schema, plus Save and Reset."""
        imgui.push_text_wrap_pos(0.0)
        imgui.text_disabled("Defaults and preferences; changes apply now. Font size applies on restart.")
        imgui.pop_text_wrap_pos()
        if imgui.button(f"{ICON.FLOPPY_DISK} Save to {USER_FILE.name}"):
            self.status_message = f"saved {rc.save()}"
        imgui.same_line()
        if imgui.button(f"{ICON.ARROW_ROTATE_LEFT} Reset all"):
            rc.reset()
            self.set_ui_scale(rc.ui.scale)
        if getattr(self, "status_message", ""):
            imgui.text_disabled(self.status_message)
        for name in SECTIONS:
            section = getattr(rc, name)
            if not imgui.collapsing_header(f"{SECTIONS[name].__doc__.split(':')[0].rstrip('.')}  ##{name}"):
                continue
            for f in fields(section):
                changed, value = _pref_widget(f, getattr(section, f.name))
                if changed:
                    self.set_preference(f"{name}.{f.name}", value)
                if f.metadata["help"] and imgui.is_item_hovered():
                    imgui.set_tooltip(f.metadata["help"])

    def set_ui_scale(self, scale: float) -> None:
        """Scale text and spacing of the whole UI."""
        self.ui_scale = scale
        apply_style(scale)
