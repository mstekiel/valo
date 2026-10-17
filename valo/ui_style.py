"""Look of the Dear ImGui panels: fonts (a vector UI font + merged icons + a monospace font) and the theme.

The Viewer calls ``load_fonts`` once. ImGui 1.92 rasterises TTF fonts at the real pixel density, so text
stays crisp on any monitor (the default ProggyClean bitmap font is what made the panel look like pixel art).
Use icons in labels with ``from valo.ui_style import ICON`` → ``f"{ICON.FLOPPY_DISK} Save"``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imgui_bundle
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

FONT_DIR = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
UI_FONT = FONT_DIR / "Roboto" / "Roboto-Regular.ttf"
ICON_FONT = FONT_DIR / "Font_Awesome_6_Free-Solid-900.otf"
MONO_FONT = FONT_DIR / "Inconsolata-Medium.ttf"
FONT_SIZE = 16.0  # logical pixels

# ICON.FLOPPY_DISK == fa.ICON_FA_FLOPPY_DISK, etc.
ICON = SimpleNamespace(**{k.removeprefix("ICON_FA_"): v for k, v in vars(fa).items() if k.startswith("ICON_FA_")})

def load_fonts(size: float = FONT_SIZE) -> None:
    """Into the current ImGui context: Roboto with Font Awesome icons merged in (the default) + Inconsolata."""
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


def set_scale(scale: float) -> None:
    """UI scale on top of the monitor's pixel density (1.0 = 16 px text): text and spacing together."""
    style = imgui.get_style()
    style.font_scale_main = scale
