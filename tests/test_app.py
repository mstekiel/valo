"""GPU tests: the engine's Viewer rendered offscreen (software Vulkan is fine). Skipped without an adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from valo.testing import click, open_every_menu, press

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


def test_spheres_render_and_material_edits_undo_exactly(viewer):
    from valo import Editor

    m = viewer.renderer.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]], [0.3, 0.5, 0.7])
    editor = Editor()
    editor.watch(m)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()], bounding_radius=lambda: 2.5), editor)
    before = viewer.snapshot()
    assert before.std() > 5, "viewport is blank"

    editor.set_param(m, "shading", "toon")
    assert np.abs(viewer.snapshot().astype(int) - before).max() > 0
    editor.undo_stack.undo()
    assert np.array_equal(viewer.snapshot(), before)


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
    assert set(editor.visible) == {"A", "B", "Info", "Camera", "Lighting", "History"} and all(editor.visible.values())
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
EDITOR_ACTIONS = {  # name → (action(editor, material), changes the image)
    "light mode": (lambda ed, m: ed.set_light_mode("fixed"), False),  # switching keeps the light where it is
    "light direction": (lambda ed, m: ed.set_light("azimuth", 90.0), True),
    "light colour": (lambda ed, m: ed.set_light("color", (1.0, 0.3, 0.1)), True),
    "light intensity": (lambda ed, m: ed.set_light("intensity", 2.0), True),
    "ambient intensity": (lambda ed, m: ed.set_light("ambient", 1.2), True),
    "sky colour": (lambda ed, m: ed.set_light("sky", (1.0, 0.0, 0.0)), True),
    "flat ambient": (lambda ed, m: ed.set_flat_ambient((0.0, 0.0, 1.0)), True),
    "flat ambient switch": (lambda ed, m: ed.set_ambient_flat(True), True),  # ground = sky
    "material parameter": (lambda ed, m: ed.set_param(m, "base_color", (0.0, 1.0, 0.0)), True),
    "material reset": (lambda ed, m: ed.reset(m), True),
}


@pytest.mark.parametrize("name", list(EDITOR_ACTIONS))
def test_editor_action_undoes_exactly(viewer, name):
    import copy

    from valo import Camera, Editor, Lighting, meshes

    action, visible = EDITOR_ACTIONS[name]
    m = viewer.renderer.material("Shiny", "blinn_phong", base_color=(0.7, 0.3, 0.2), shininess=200.0)
    cube = viewer.renderer.instances(meshes.cube(), m)
    cube.set(np.eye(4)[None])
    viewer.camera, viewer.lighting = Camera(), Lighting()
    editor = Editor()
    editor.watch(m)
    viewer.show(SimpleNamespace(batches=lambda: [cube.batch()], bounding_radius=lambda: 0.9), editor)
    viewer.camera.distance = 6.0  # not fitted, so "fit view" has something to do

    def state():
        return copy.copy(viewer.camera), copy.copy(viewer.lighting), m.data.copy()

    before, image = state(), viewer.snapshot()
    action(editor, m)
    after = state()
    assert not (after[0] == before[0] and after[1] == before[1] and np.array_equal(after[2], before[2]))
    assert (np.abs(viewer.snapshot().astype(int) - image).max() > 10) == visible
    assert len(editor.undo_stack.done) == 1
    editor.undo_stack.undo()
    restored = state()
    assert restored[0] == before[0] and restored[1] == before[1] and np.array_equal(restored[2], before[2])
    assert np.array_equal(viewer.snapshot(), image)
    editor.undo_stack.redo()
    assert state()[0] == after[0] and state()[1] == after[1]
    viewer.lighting = Lighting()


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


def test_reset_layout_shows_closed_windows_and_ui_scale_resizes(viewer):
    from imgui_bundle import imgui

    from valo import Editor

    editor = Editor()
    viewer.show(SimpleNamespace(batches=list), editor)
    viewer.snapshot()
    editor.visible["History"] = False
    viewer.snapshot()
    editor.reset_layout()
    viewer.snapshot()
    assert editor.visible["History"]
    editor.set_ui_scale(1.5)
    imgui.set_current_context(viewer.imgui.imgui_context)
    assert imgui.get_style().font_scale_main == 1.5 and editor.ui_scale == 1.5
    editor.set_ui_scale(1.0)


def test_flat_ambient_can_be_switched_off_again(viewer):
    from valo import Editor, Lighting

    viewer.lighting = light = Lighting(ground=(0.3, 0.1, 0.0))
    editor = Editor()
    viewer.show(SimpleNamespace(batches=list), editor)
    editor.set_ambient_flat(True)
    assert light.ground == light.sky
    editor.set_ambient_flat(False)
    assert light.ground == (0.3, 0.1, 0.0)  # the ground colour from before Flat comes back
    viewer.lighting = light = Lighting(sky=(0.5, 0.5, 0.5), ground=(0.5, 0.5, 0.5))  # started flat
    editor.set_ambient_flat(False)
    assert light.ground != light.sky
    viewer.lighting = Lighting()


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


def test_preferences_window_edits_rc_live(viewer):
    from imgui_bundle import imgui

    from valo import Editor, rc, rc_context

    editor = Editor()
    viewer.show(SimpleNamespace(batches=list), editor)
    with rc_context():
        editor.show_preferences = True
        viewer.snapshot()  # the generated widgets for every key draw without errors
        editor.set_preference("ui.scale", 1.25)
        imgui.set_current_context(viewer.imgui.imgui_context)
        assert rc.ui.scale == 1.25 and imgui.get_style().font_scale_main == 1.25
        with pytest.raises(ValueError):
            editor.set_preference("camera.fov", 500)
    editor.set_ui_scale(rc.ui.scale)


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


def menu_editor(kind, viewer, tmp_path):

    from valo import Editor, MeshObject, Scene, SceneEditor, Shape, meshes

    if kind == "editor":
        viewer.show(SimpleNamespace(batches=list), Editor())
    elif kind == "scene editor":
        scene = Scene()
        scene.material("M", "lambert")
        scene.add(MeshObject("Cube", Shape(meshes.cube), "M"))
        viewer.show(scene, SceneEditor(scene))


@pytest.mark.parametrize("kind", ["editor", "scene editor"])  # the studio's menus: tests/test_valo_studio.py
def test_every_menu_opens_without_errors(viewer, tmp_path, kind):
    menu_editor(kind, viewer, tmp_path)
    assert open_every_menu(viewer) >= 3  # File, Edit, View
    assert viewer.gui_error is None


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


def test_dock_left_windows_sit_left_of_the_scene(viewer):
    from imgui_bundle import imgui
    from imgui_bundle.imgui import internal

    from valo import Editor

    class Wide(Editor):
        dock_left = ("Code",)

        def windows(self):
            return {"Tree": lambda: None, "Code": lambda: None}

    editor = Wide()
    viewer.show(SimpleNamespace(batches=list), editor)
    editor.reset_layout()  # the shared viewer kept an earlier editor's layout
    viewer.snapshot()
    imgui.set_current_context(viewer.imgui.imgui_context)
    x, _, w, _ = editor.scene_rect
    code, tree = (internal.find_window_by_name(n) for n in ("Code", "Tree"))
    assert code.pos.x + code.size.x <= x + 1 and tree.pos.x >= x + w - 1  # left and right of the 3D view


@pytest.mark.parametrize("change", [lambda ed: ed.set_projection("orthographic"), lambda ed: ed.set_fov(60.0),
                                    lambda ed: ed.set_view("+X"), lambda ed: ed.fit_view()])
def test_view_changes_move_the_camera_but_are_not_undo_steps(viewer, change):
    import copy

    from valo import Camera, Editor

    editor = Editor()
    viewer.show(SimpleNamespace(batches=list, bounding_radius=lambda: 0.9), editor)
    viewer.camera = Camera(distance=6.0)
    before = copy.copy(viewer.camera)
    change(editor)
    assert viewer.camera != before and editor.undo_stack.done == []  # like Blender: navigation is not an edit
