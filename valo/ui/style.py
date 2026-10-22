"""Look of the Dear ImGui panels: fonts (a vector UI font + merged icons + a monospace font) and the theme.

The Viewer calls ``load_fonts`` once. ImGui 1.92 rasterises TTF fonts at the real pixel density, so text
stays crisp on any monitor (the default ProggyClean bitmap font is what made the panel look like pixel art).
Use icons in labels with ``from valo.ui.style import ICON`` → ``f"{ICON.FLOPPY_DISK} Save"``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imgui_bundle
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from ..config import rc

FONT_DIR = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
UI_FONT = FONT_DIR / "Roboto" / "Roboto-Regular.ttf"
ICON_FONT = FONT_DIR / "Font_Awesome_6_Free-Solid-900.otf"
MONO_FONT = FONT_DIR / "Inconsolata-Medium.ttf"

# ICON.FLOPPY_DISK == fa.ICON_FA_FLOPPY_DISK, etc.
ICON = SimpleNamespace(**{k.removeprefix("ICON_FA_"): v for k, v in vars(fa).items() if k.startswith("ICON_FA_")})

def load_fonts(size: float | None = None) -> None:
    """Into the current ImGui context: Roboto with Font Awesome icons merged in (the default) + Inconsolata."""
    size = size or rc.ui.font_size
    io = imgui.get_io()
    ui = io.fonts.add_font_from_file_ttf(str(UI_FONT), size)
    icons = imgui.ImFontConfig()
    icons.merge_mode = True
    icons.glyph_min_advance_x = size  # monospaced icons line up in buttons and menus
    icons.glyph_offset = imgui.ImVec2(0, 1)
    io.fonts.add_font_from_file_ttf(str(ICON_FONT), size * 0.85, icons)
    io.fonts.add_font_from_file_ttf(str(MONO_FONT), size)
    io.font_default = ui
    imgui.get_style().font_size_base = size


def mono_font() -> imgui.ImFont:
    """The monospace font of the current context, for ``imgui.push_font(mono_font(), 0.0)``."""
    return imgui.get_io().fonts.fonts[1]  # [0] Roboto (+ merged icons), [1] Inconsolata


# Theme: neutral greys, one blue accent. sRGB 0–255; alpha 0–1.
def _palette(accent) -> dict:
    """The theme's colours, with ``accent`` (sRGB 0-255) for highlights."""
    a = tuple(accent)
    return {
        "text": (225, 227, 230), "text_disabled": (128, 132, 138),
        "window_bg": (30, 31, 34), "child_bg": (30, 31, 34), "popup_bg": (36, 37, 41),
        "menu_bar_bg": (24, 25, 27), "title_bg": (24, 25, 27), "title_bg_active": (24, 25, 27),
        "title_bg_collapsed": (24, 25, 27), "border": (52, 54, 58), "separator": (52, 54, 58),
        "frame_bg": (44, 46, 50), "frame_bg_hovered": (54, 57, 62), "frame_bg_active": (62, 66, 72),
        "button": (50, 53, 58), "button_hovered": (62, 66, 72), "button_active": a,
        "header": (50, 53, 58), "header_hovered": (62, 66, 72), "header_active": a,
        "check_mark": a, "slider_grab": a, "slider_grab_active": (110, 160, 250),
        "tab": (30, 31, 34), "tab_hovered": (62, 66, 72), "tab_selected": (44, 46, 50),
        "tab_selected_overline": a, "tab_dimmed": (30, 31, 34), "tab_dimmed_selected": (40, 41, 45),
        "scrollbar_bg": (30, 31, 34), "scrollbar_grab": (58, 61, 66), "scrollbar_grab_hovered": (74, 78, 84),
        "scrollbar_grab_active": (90, 94, 100), "resize_grip": (0, 0, 0, 0), "docking_preview": (*a, 0.5),
        "docking_empty_bg": (0, 0, 0, 0), "text_selected_bg": (*a, 0.35), "nav_cursor": a,
    }


def apply_style(scale: float = 1.0) -> None:
    """valo's theme for the current context: rounded, roomy, dark neutral greys with one accent colour.

    ``scale`` multiplies text and spacing together (on top of the monitor's pixel density), and the text size is
    ``rc.ui.font_size``; call it again to change them. Call it between frames (the Editor does it at the start of
    its ``gui()``): changing fonts while windows are drawing leaves them half laid out.
    """
    style = imgui.get_style()
    for key, rgba in _palette(rc.ui.accent).items():
        r, g, b, *a = rgba
        style.set_color_(getattr(imgui.Col_, key), imgui.ImVec4(r / 255, g / 255, b / 255, a[0] if a else 1.0))

    def v(x, y):
        return imgui.ImVec2(x * scale, y * scale)

    style.window_padding, style.frame_padding, style.cell_padding = v(10, 10), v(8, 5), v(6, 4)
    style.item_spacing, style.item_inner_spacing = v(8, 6), v(6, 4)
    style.separator_text_padding, style.window_min_size = v(0, 4), v(64, 48)
    style.window_rounding = style.child_rounding = style.popup_rounding = 6.0 * scale
    style.frame_rounding = style.tab_rounding = 4.0 * scale
    style.grab_rounding, style.scrollbar_rounding = 3.0 * scale, 6.0 * scale
    style.scrollbar_size, style.grab_min_size, style.indent_spacing = 12.0 * scale, 10.0 * scale, 18.0 * scale
    style.window_border_size, style.frame_border_size, style.popup_border_size = 1.0, 0.0, 1.0
    style.separator_text_border_size = 1.0
    style.window_title_align = imgui.ImVec2(0.0, 0.5)
    style.font_scale_main = scale
    # ImGui 1.92 rasterises fonts at any size (no reload); NewFrame copies the "next frame" size into the base
    style.font_size_base = style._next_frame_font_size_base = rc.ui.font_size
