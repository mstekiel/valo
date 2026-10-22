"""The native window around the app: its icon and, on Windows 11, the colour of its title bar.

Both apply only to a real glfw window (a Viewer opened without a canvas); offscreen canvases have no title bar, and
other systems draw the title bar themselves, so there these calls do nothing and return False.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from PIL import Image

from ..config import rc

ICON_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"  # shipped inside the package
DEFAULT_ICON = ICON_DIR / "valo.png"
ICON_SIZES = (16, 32, 48, 256)  # px; the window system picks the best one for the title bar and the taskbar
_DWMWA_USE_IMMERSIVE_DARK_MODE, _DWMWA_CAPTION_COLOR, _DWMWA_TEXT_COLOR = 20, 35, 36


def colorref(rgb) -> int:
    """A Windows COLORREF (0x00BBGGRR) from an sRGB 0-255 colour."""
    r, g, b = (int(round(c)) & 0xFF for c in rgb)
    return r | g << 8 | b << 16


def icon_images(path=DEFAULT_ICON) -> list[Image.Image]:
    """The icon at ``path`` as RGBA images in every size of ``ICON_SIZES``."""
    image = Image.open(path).convert("RGBA")
    return [image.resize((s, s), Image.LANCZOS) for s in ICON_SIZES]


def _glfw_window(canvas):
    """The glfw window of a rendercanvas glfw canvas, or None (offscreen canvases have none)."""
    return getattr(canvas, "_window", None)


def set_icon(canvas, path=DEFAULT_ICON) -> bool:
    """Give the window of ``canvas`` the icon at ``path``; False when there is no native window."""
    window = _glfw_window(canvas)
    if window is None:
        return False
    import glfw

    glfw.set_window_icon(window, len(ICON_SIZES), icon_images(path))
    return True


def set_title_bar(canvas, background=None, text=None) -> bool:
    """Colour the window's title bar (sRGB 0-255; default ``rc.ui.title_bar`` / ``rc.ui.title_text``) through
    Windows' DWM, dark mode on. True when Windows took it (Windows 11); False elsewhere or without a window."""
    window = _glfw_window(canvas)
    if window is None or sys.platform != "win32":
        return False
    import glfw

    hwnd = glfw.get_win32_window(window)
    dwm = ctypes.windll.dwmapi
    results = []
    for attribute, value in ((_DWMWA_USE_IMMERSIVE_DARK_MODE, 1),
                             (_DWMWA_CAPTION_COLOR, colorref(background or rc.ui.title_bar)),
                             (_DWMWA_TEXT_COLOR, colorref(text or rc.ui.title_text))):
        v = ctypes.c_int(value)
        results.append(dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(v), ctypes.sizeof(v)))
    return all(r == 0 for r in results)
