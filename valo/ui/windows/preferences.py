"""The Preferences window: every ``valo.rc`` setting as a widget, saved to the user's valorc.toml."""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal

from imgui_bundle import imgui

from ...config import SECTIONS, USER_FILE, rc
from ...gpu.material import linear_to_srgb, srgb_to_linear
from ..style import ICON, apply_style
from ..widgets import SliderSpec

ON_RELEASE = ("ui.scale", "ui.font_size")  # applied when the slider is released, not while dragging


def _pref_widget(editor, key: str, f, value) -> tuple[bool, object]:
    """One generated widget for config field ``f`` (``key`` = "section.name"): checkbox, slider, colour, combo or
    text. Sliders keep to the field's range and step (their right-click menu resets to the default)."""
    m, label = f.metadata, f"{f.metadata['label']}##{f.name}"
    unit = f" {m['unit']}" if m["unit"] and m["unit"] != "x" else ""
    if isinstance(value, bool):
        return imgui.checkbox(label, value)
    if isinstance(value, (int, float)) and m["lo"] is not None:
        integer = isinstance(value, int)
        digits = max(0, -Decimal(str(m["step"])).as_tuple().exponent) if m["step"] else None
        fmt = f"%d{unit}" if integer else (f"%.{digits}f{unit}" if m["step"] else f"%.3g{unit}")
        spec = SliderSpec(m["lo"], m["hi"], f.default, m["step"] or (1 if integer else None), m["lo"], m["hi"],
                          fmt=fmt, integer=integer, editable=False)
        return editor.slider(f"preferences.{key}", label, value, spec)
    if isinstance(value, tuple) and len(value) == 3:
        if m["hi"] == 255:  # sRGB 0-255
            changed, col = imgui.color_edit3(label, [c / 255 for c in value], imgui.ColorEditFlags_.no_inputs)
            return changed, tuple(round(c * 255) for c in col)
        changed, col = imgui.color_edit3(label, [linear_to_srgb(c) for c in value], imgui.ColorEditFlags_.no_inputs)
        return changed, tuple(srgb_to_linear(c) for c in col)
    if isinstance(value, tuple) and len(value) == 2:  # e.g. an image size
        changed, pair = imgui.drag_int2(label, list(value), 1.0, m["lo"], m["hi"], f"%d{unit}")
        return changed, tuple(pair)
    if m["choices"]:
        changed, i = imgui.combo(label, list(m["choices"]).index(value), list(m["choices"]))
        return changed, m["choices"][i]
    if isinstance(value, str):
        return imgui.input_text(label, value)
    imgui.text_disabled(f"{m['label']}: {value!r}")  # no widget for this type: shown, not editable
    return False, value


class PreferencesWindow:
    """Editor part: the Preferences window and the UI scale."""

    def set_preference(self, key: str, value) -> None:
        """Set config key ``key`` (e.g. "camera.orbit_speed") and apply it where it takes effect immediately."""
        rc[key] = value
        if key in ("ui.scale", "ui.accent", "ui.font_size"):
            self.set_ui_scale(rc.ui.scale)

    def preferences_window(self) -> None:
        """Every ``rc`` key, with widgets generated from the config schema, plus Save and Reset."""
        imgui.push_text_wrap_pos(0.0)
        imgui.text_disabled("Defaults and preferences; changes apply now (font size and UI scale when you release "
                            "the slider).")
        imgui.pop_text_wrap_pos()
        if imgui.button(f"{ICON.FLOPPY_DISK} Save to {USER_FILE.name}"):
            path = rc.save()
            self.notify(f"preferences saved to {path.name}", detail=str(path))
        imgui.same_line()
        if imgui.button(f"{ICON.ARROW_ROTATE_LEFT} Reset all"):
            rc.reset()
            self.set_ui_scale(rc.ui.scale)
        imgui.same_line()
        expanded = getattr(self, "preferences_expanded", None)
        if imgui.button("Collapse all" if expanded else "Expand all"):
            self.preferences_expanded = not expanded
        for name in SECTIONS:
            section = getattr(rc, name)
            if expanded is not None:
                imgui.set_next_item_open(expanded, imgui.Cond_.always)
            if not imgui.collapsing_header(f"{SECTIONS[name].__doc__.split(':')[0].rstrip('.')}  ##{name}"):
                continue
            for f in fields(section):
                key = f"{name}.{f.name}"
                if key in ON_RELEASE:  # they resize the slider itself: apply when it is let go
                    dragged = self.__dict__.setdefault("_pref_dragged", {})
                    changed, value = _pref_widget(self, key, f, dragged.get(key, getattr(section, f.name)))
                    if changed:
                        dragged[key] = value
                    if imgui.is_item_deactivated_after_edit() and key in dragged:
                        self.set_preference(key, dragged.pop(key))
                    elif not imgui.is_item_active():
                        dragged.pop(key, None)
                else:
                    changed, value = _pref_widget(self, key, f, getattr(section, f.name))
                    if changed:
                        self.set_preference(key, value)
                if f.metadata["help"] and imgui.is_item_hovered():
                    imgui.set_tooltip(f.metadata["help"])

    def set_ui_scale(self, scale: float) -> None:
        """Scale text and spacing of the whole UI; applied at the start of the next frame (with the font size)."""
        self.ui_scale = scale
        self._restyle = True

    def apply_pending_style(self) -> None:
        """Apply a UI scale / font size / accent change made during the last frame (called before any window)."""
        if self.__dict__.pop("_restyle", False):
            apply_style(self.ui_scale)
