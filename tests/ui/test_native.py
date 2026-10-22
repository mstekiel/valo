"""valo.ui.native: the window icon and the title bar colour (no window offscreen: they report False)."""

from types import SimpleNamespace

from valo.ui.native import DEFAULT_ICON, ICON_SIZES, colorref, icon_images, set_icon, set_title_bar


def test_colorref_packs_srgb_as_windows_bgr():
    assert colorref((24, 25, 27)) == 0x1B1918 and colorref((255, 0, 0)) == 0x0000FF


def test_the_default_icon_ships_with_valo_in_every_size():
    images = icon_images()
    assert DEFAULT_ICON.is_file() and [im.size for im in images] == [(s, s) for s in ICON_SIZES]
    assert all(im.mode == "RGBA" for im in images) and images[-1].getextrema()[3][0] == 0  # transparent corners


def test_without_a_native_window_nothing_happens():
    offscreen = SimpleNamespace()
    assert set_icon(offscreen) is False and set_title_bar(offscreen) is False
