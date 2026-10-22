"""valo.ui.viewer: the Viewer offscreen: fonts and style, the scene in the central dock node, mouse and keys, selection,
Info rows, and recovering from an error in a window."""

from types import SimpleNamespace

import numpy as np
import pytest

from valo.testing import click, open_every_menu, press

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


def test_vector_fonts_with_icons_are_loaded(viewer):
    from imgui_bundle import imgui

    from valo.ui.style import ICON, mono_font

    imgui.set_current_context(viewer.imgui.imgui_context)
    io = imgui.get_io()
    assert len(io.fonts.fonts) == 2 and io.font_default is io.fonts.fonts[0]
    assert io.font_default.is_glyph_in_font(ord(ICON.FLOPPY_DISK))  # icons merged into the UI font
    assert mono_font() is io.fonts.fonts[1]


def test_style_scale_is_absolute_not_compounding(viewer):
    from imgui_bundle import imgui

    from valo.ui.style import apply_style

    imgui.set_current_context(viewer.imgui.imgui_context)
    style = imgui.get_style()
    apply_style(1.5)
    apply_style(1.5)
    assert (style.frame_padding.x, style.font_scale_main) == (12.0, 1.5)
    apply_style()
    assert style.frame_padding.x == 8.0


def test_scene_fills_the_central_dock_node_and_windows_dock_around_it(viewer):
    from imgui_bundle import imgui

    from valo import Editor

    class Two(Editor):
        def windows(self):
            return {"A": lambda: imgui.text("a"), "B": lambda: imgui.text("b")}

    editor = Two()
    viewer.show(SimpleNamespace(batches=list), editor)
    viewer.snapshot()
    x, y, w, h = viewer.viewport
    assert x == 0 and 0 < y < 60  # below the menu bar
    assert 0.6 < w / viewer.canvas.get_physical_size()[0] < 0.8  # right column takes the rest
    assert y + h == viewer.canvas.get_physical_size()[1]
    assert set(editor.visible) == {"A", "B", "Info", "Camera", "Lighting", "History", "Console"}
    assert all(editor.visible.values())
    assert viewer.layout_file is None  # offscreen canvas: never writes a layout file


def test_layout_is_saved_on_close(tmp_path):
    from rendercanvas.offscreen import RenderCanvas

    from valo import Editor, Viewer

    ini = tmp_path / "sub" / "layout.ini"
    v = Viewer(RenderCanvas(size=(400, 300), pixel_ratio=1), layout_file=ini)
    v.show(SimpleNamespace(batches=list), Editor())
    v.canvas.draw()
    assert not ini.exists()  # ImGui itself waits a few seconds before saving
    v.canvas.close()
    assert "[Window][History]" in ini.read_text()


# -- every shared Editor action: changes something, and undo restores state and image exactly


def test_mouse_orbits_and_zooms_only_over_the_scene(viewer):
    from valo import Camera, Editor

    viewer.camera = cam = Camera()
    viewer.show(SimpleNamespace(batches=list), Editor())
    viewer.snapshot()  # lays out the dock, so the scene rect is known
    x, y, w, h = viewer.editor.scene_rect
    inside, outside = (x + w / 2, y + h / 2), (x + w + 20, y + h / 2)  # outside: over the right dock column

    def drag(start, dx):
        viewer.on_pointer({"event_type": "pointer_down", "button": 1, "x": start[0], "y": start[1]})
        viewer.on_pointer({"event_type": "pointer_move", "x": start[0] + dx, "y": start[1] + 10})
        viewer.on_pointer({"event_type": "pointer_up", "x": start[0] + dx, "y": start[1] + 10})

    az, el, dist = cam.azimuth, cam.elevation, cam.distance
    drag(outside, 50)
    viewer.on_pointer({"event_type": "wheel", "dy": 100, "x": outside[0], "y": outside[1]})
    assert (cam.azimuth, cam.elevation, cam.distance) == (az, el, dist)
    drag(inside, 50)
    assert cam.azimuth == pytest.approx(az - 20) and cam.elevation == pytest.approx(el + 4)  # 0.4 deg/px
    viewer.on_pointer({"event_type": "wheel", "dy": 100, "x": inside[0], "y": inside[1]})
    assert cam.distance > dist
    cam.set_projection("orthographic")
    height = cam.ortho_height
    viewer.on_pointer({"event_type": "wheel", "dy": -100, "x": inside[0], "y": inside[1]})
    assert cam.ortho_height < height  # orthographic zoom changes the visible height


def test_orbit_speed_comes_from_rc(viewer):
    from valo import Camera, Editor, rc_context

    viewer.camera = cam = Camera()
    viewer.show(SimpleNamespace(batches=list), Editor())
    viewer.snapshot()
    x, y, w, h = viewer.editor.scene_rect
    with rc_context({"camera.orbit_speed": 1.0}):
        az = cam.azimuth
        viewer.on_pointer({"event_type": "pointer_down", "button": 1, "x": x + w / 2, "y": y + h / 2})
        viewer.on_pointer({"event_type": "pointer_move", "x": x + w / 2 + 30, "y": y + h / 2})
        viewer.on_pointer({"event_type": "pointer_up", "x": x + w / 2 + 30, "y": y + h / 2})
    assert cam.azimuth == pytest.approx(az - 30)


def test_info_rows_show_how_the_instance_was_placed(viewer):
    from valo import instance_matrices, meshes
    from valo.gpu import Pick
    from valo.ui import pick_info

    m = viewer.renderer.material("Steel", "pbr")
    arrows = viewer.renderer.instances(meshes.arrow(), m, label="arrows", info=lambda i: {"vector": i})
    arrows.set(instance_matrices([[0, 0, 0], [1.0, 2.0, 3.0]], [[1, 1, 1], [0.5, 0.5, 2.0]],
                                 z_axis=[[0, 0, 1], [1, 0, 0]]))
    rows = pick_info(Pick(0, 1, arrows, np.array([1.0, 2.0, 4.0])), SimpleNamespace(batches=list))
    assert rows["Object"] == "arrows" and rows["Material"] == "Steel (pbr)" and rows["Instance"] == "2 of 2"
    assert rows["Translation"] == "(1.000, 2.000, 3.000)" and rows["Scale"] == "(0.500, 0.500, 2.000)"
    assert rows["Rotation"] == "90.0 deg about (0.00, 1.00, 0.00)"  # +Z turned onto +X
    assert rows["vector"] == "1"
    described = SimpleNamespace(batches=list, describe=lambda pick: {"site": "(0, 0, 0)"})
    assert pick_info(Pick(0, 0, arrows, np.zeros(3)), described)["site"] == "(0, 0, 0)"


def test_click_selects_keys_navigate_and_double_click_centres(viewer):
    from valo import Camera, Editor, rc_context

    m = viewer.renderer.material("M", "lambert")
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.5, 0, 0], [1.5, 0, 0]], 0.5)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()], bounding_radius=lambda: 2.0), Editor())
    viewer.camera = cam = Camera(azimuth=-90.0, elevation=0.0, distance=8.0)
    viewer.snapshot()

    def key(name, *mods):
        press(viewer, name, *mods)

    def click_ball():
        click(viewer, [1.5, 0, 0])

    with rc_context({"camera.center_duration": 0.0}):
        click_ball()
        assert viewer.selection.instance_index == 1
        key("f")  # frame the selection: centre on that sphere, zoomed to fit it
        assert cam.target == pytest.approx((1.5, 0.0, 0.0)) and cam.distance < 8.0
        key("Home")  # fit everything again
        assert cam.target == pytest.approx((0.0, 0.0, 0.0))
        key("Escape")
        assert viewer.selection is None
        key("7")
        assert cam.elevation == pytest.approx(89.99)  # top view
        key("5")
        assert cam.projection == "orthographic"
        az = cam.azimuth
        key("ArrowLeft")
        assert cam.azimuth == pytest.approx((az - 15) % 360)
        target = cam.target
        key("ArrowUp", "Shift")
        assert cam.target != target  # Shift+arrow pans
        assert len(viewer.editor.undo_stack.done) == 0  # navigation is never recorded (like Blender)

    viewer.camera = cam = Camera(azimuth=-90.0, elevation=0.0, distance=8.0)
    with rc_context({"camera.center_duration": 0.0}):
        click_ball()
        click_ball()  # a double-click centres on the point under the pointer
        assert cam.target[0] == pytest.approx(1.5, abs=0.1) and cam.target[1] < 0  # front of the sphere
    viewer.selection = None


def test_an_error_in_a_window_is_reported_and_later_frames_still_work(viewer):
    from valo import Editor

    class Broken(Editor):
        fail = True

        def windows(self):
            return {"Broken": self.broken}

        def broken(self):
            if self.fail:
                raise RuntimeError("broken widget")

    editor = Broken()
    viewer.gui_error = None
    viewer.show(SimpleNamespace(batches=list), editor)
    viewer.snapshot()
    assert "broken widget" in viewer.gui_error
    editor.fail, viewer.gui_error = False, None
    viewer.snapshot()  # ImGui recovered: no "Missing End()" from here on
    assert viewer.gui_error is None and open_every_menu(viewer) >= 3
