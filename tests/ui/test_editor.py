"""valo.ui.editor: every shared action undoes exactly (EDITOR_ACTIONS); view changes are not undo steps; layout,
preferences and menus."""

from types import SimpleNamespace

import numpy as np
import pytest

from valo.testing import open_every_menu

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
