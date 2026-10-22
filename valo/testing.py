"""Test helpers for valo and the apps built on it (like ``numpy.testing``): an offscreen Viewer, where a world point
shows up on screen, pointer / key / menu events, and image comparison. It is also a pytest plugin
(``-p valo.testing``, set in pyproject.toml): the fixtures ``viewer`` and ``tmp_library``, and tests start from the
default config.

    viewer.show(scene, editor)
    before = viewer.snapshot()
    click(viewer, (1.5, 0, 0))                 # select whatever is drawn at that world point
    assert changed(before, viewer.snapshot()).any()

Screen positions follow the Viewer's conventions: pointer events use logical window pixels, ``Renderer.pick`` uses
framebuffer pixels of the window, and ``Viewer.snapshot()`` images are the scene region only.
"""

from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # the helpers work without pytest; only the plugin part below needs it
    pytest = None


def offscreen_viewer(size: tuple[int, int] = (900, 560)):
    """A Viewer on an offscreen canvas of ``size`` logical pixels (pixel ratio 1), or None without a GPU adapter."""
    from rendercanvas.offscreen import RenderCanvas

    from .ui.viewer import Viewer

    try:
        return Viewer(RenderCanvas(size=size, pixel_ratio=1))
    except Exception:  # no adapter on this machine
        return None


def _ndc(viewer, world) -> tuple[float, float]:
    x0, y0, w, h = viewer.viewport
    clip = viewer.camera.matrices(w / h)[0] @ np.r_[np.asarray(world, float), 1.0]
    return clip[0] / clip[3], clip[1] / clip[3]


def framebuffer_point(viewer, world) -> tuple[float, float]:
    """Where world point ``world`` is drawn, in framebuffer pixels of the window (what ``Renderer.pick`` takes)."""
    x0, y0, w, h = viewer.viewport
    nx, ny = _ndc(viewer, world)
    return x0 + (nx + 1) / 2 * w, y0 + (1 - ny) / 2 * h


def screen_point(viewer, world) -> tuple[float, float]:
    """Where world point ``world`` is drawn, in logical window pixels (what pointer events take)."""
    x, y = framebuffer_point(viewer, world)
    r = viewer.canvas.get_pixel_ratio()
    return x / r, y / r


def image_point(viewer, world) -> tuple[int, int]:
    """(column, row) of world point ``world`` in a ``viewer.snapshot()`` image (the scene region)."""
    x0, y0, w, h = viewer.viewport
    nx, ny = _ndc(viewer, world)
    return int((nx + 1) / 2 * w), int((1 - ny) / 2 * h)


def hover(viewer, world):
    """Move the pointer over world point ``world`` and draw (which picks under it); returns ``viewer.hover``."""
    x, y = screen_point(viewer, world)
    viewer.on_pointer({"event_type": "pointer_move", "x": x, "y": y})
    viewer.snapshot()
    return viewer.hover


def click(viewer, world, button: int = 1) -> None:
    """Hover world point ``world``, then press and release the mouse button there (a click: select, or centre when
    it follows another click soon)."""
    hover(viewer, world)
    x, y = screen_point(viewer, world)
    for kind in ("pointer_down", "pointer_up"):
        viewer.on_pointer({"event_type": kind, "button": button, "x": x, "y": y, "modifiers": ()})


def press(viewer, key: str, *modifiers: str) -> None:
    """A key press as the window sends it, e.g. ``press(viewer, "ArrowUp", "Shift")``."""
    viewer.on_key({"event_type": "key_down", "key": key, "modifiers": modifiers})


def release_pointer(viewer) -> None:
    """Forget the pointer, hover and last click (so the next test starts without them)."""
    viewer.pointer, viewer.hover, viewer._last_click = None, None, None


def open_every_menu(viewer) -> int:
    """Click along the menu bar, closing each menu again by clicking into the scene; returns how many clicks opened
    a menu. Errors in the menus end up in ``viewer.gui_error`` (reset here first)."""
    from imgui_bundle import imgui

    viewer.snapshot()  # lays out the dock: the scene rect is known
    x, y, w, h = viewer.editor.scene_rect
    opened, viewer.gui_error = 0, None
    for mx in range(4, 240, 6):
        viewer.ui_click(mx, 10)
        imgui.set_current_context(viewer.imgui.imgui_context)
        opened += imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
        viewer.ui_click(x + w / 2, y + h / 2)
    return opened


def changed(before, after, threshold: int = 3) -> np.ndarray:
    """Per-pixel mask (h, w) of where two images differ by more than ``threshold`` levels in any channel."""
    return np.abs(np.asarray(after, int) - np.asarray(before, int)).max(-1) > threshold


# -- the pytest plugin (loaded with ``-p valo.testing``)

def pytest_configure(config):
    """Tests run with the default config: forget what ``~/.valo/valorc.toml`` or ``./valorc.toml`` set at import."""
    from .config import rc

    rc.reset()


if pytest is not None:

    @pytest.fixture(scope="module")
    def viewer(request):
        """An offscreen Viewer shared by the tests of one module (its size: the module's ``VIEWER_SIZE``, default
        900 x 560). Skips the test without a GPU adapter."""
        v = offscreen_viewer(getattr(request.module, "VIEWER_SIZE", (900, 560)))
        if v is None:
            pytest.skip("no GPU adapter")
        return v

    @pytest.fixture
    def tmp_library(tmp_path, monkeypatch):
        """The procedural mesh library redirected to empty temporary folders (never the real ones): the user folder
        (returned) and the built-in one (``library.BUILTIN_DIR``)."""
        from .config import rc
        from .meshes import library

        folder = tmp_path / "library"
        monkeypatch.setattr(library, "BUILTIN_DIR", tmp_path / "builtin")
        monkeypatch.setattr(rc.studio, "procedural_meshes", str(folder))
        return folder
