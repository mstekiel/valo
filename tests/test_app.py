"""GPU tests: the engine's Viewer rendered offscreen (software Vulkan is fine). Skipped without an adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def viewer():
    pytest.importorskip("wgpu")
    from rendercanvas.offscreen import RenderCanvas

    from valo import Viewer

    try:
        return Viewer(RenderCanvas(size=(640, 400), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")


def test_empty_viewer_draws_the_background(viewer):
    viewer.show(SimpleNamespace(batches=list))
    img = viewer.snapshot()
    assert (img == img[0, 0]).all()


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


def test_any_mesh_draws_instanced_with_its_own_transforms(viewer):
    from valo import instance_matrices, meshes

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.8, 0.8, 0.8))
    cubes = viewer.renderer.instances(meshes.cube(), m)
    arrows = viewer.renderer.instances(meshes.arrow(), m)
    assert viewer.renderer.instances(meshes.cube(), m).mesh is not cubes.mesh  # cache is per MeshData object
    viewer.show(SimpleNamespace(batches=lambda: [cubes.batch(), arrows.batch()]))
    viewer.camera.fit(3.0)
    empty = viewer.snapshot()

    cubes.set(instance_matrices([[-1, 0, 0], [1, 0, 0]], [[0.5], [1.0]]))
    with_cubes = viewer.snapshot()
    assert (with_cubes != empty).any(-1).sum() > 500
    arrows.set(instance_matrices([[0, 0, 0]], (0.3, 0.3, 2.0), z_axis=(0, 1, 1)))
    assert (viewer.snapshot() != with_cubes).any(-1).sum() > 200


def test_vector_fonts_with_icons_are_loaded(viewer):
    from imgui_bundle import imgui

    from valo.ui_style import ICON, mono_font

    imgui.set_current_context(viewer.imgui.imgui_context)
    io = imgui.get_io()
    assert len(io.fonts.fonts) == 2 and io.font_default is io.fonts.fonts[0]
    assert io.font_default.is_glyph_in_font(ord(ICON.FLOPPY_DISK))  # icons merged into the UI font
    assert mono_font() is io.fonts.fonts[1]


def test_style_scale_is_absolute_not_compounding(viewer):
    from imgui_bundle import imgui

    from valo.ui_style import apply_style

    imgui.set_current_context(viewer.imgui.imgui_context)
    style = imgui.get_style()
    apply_style(1.5)
    apply_style(1.5)
    assert (style.frame_padding.x, style.font_scale_main) == (12.0, 1.5)
    apply_style()
    assert style.frame_padding.x == 8.0


def test_scene_renders_into_an_offset_viewport_rect(viewer):
    import wgpu

    from valo import OrbitCamera

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.9, 0.9, 0.9), ambient=1.0)
    ball = viewer.renderer.spheres(m)
    ball.set([[0, 0, 0]], 1.6)  # seen from distance 3, it fills the whole viewport
    scene = SimpleNamespace(batches=lambda: [ball.batch()])
    fmt = viewer.context._config["format"]
    tex = viewer.device.create_texture(size=(400, 300, 1), format=fmt,
                                       usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    viewer.renderer.render(tex, (100, 50, 200, 150), scene, OrbitCamera(distance=3.0))
    data = viewer.device.queue.read_texture({"texture": tex}, {"bytes_per_row": 400 * 4}, (400, 300, 1))
    img = np.frombuffer(data, np.uint8).reshape(300, 400, 4)[:, :, :3]
    inside = np.zeros((300, 400), bool)
    inside[50:200, 100:300] = True
    assert img[inside].min() > 150  # the bright sphere covers the rect
    assert img[~inside].max() < 60  # the window background everywhere else


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


def test_orthographic_shading_is_the_same_for_identical_spheres(viewer):
    """In orthographic mode the view vector is the same everywhere, so two identical spheres shade identically."""
    from valo import OrbitCamera

    m = viewer.renderer.material("Shiny", "pbr", base_color=(0.9, 0.9, 0.9), metallic=1.0, roughness=0.2)
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.2, 0, 0], [1.2, 0, 0]], 0.5)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()]))
    viewer.camera = cam = OrbitCamera(azimuth=90.0, elevation=0.0, distance=3.0)

    centres = np.array([[-1.2, 0, 0, 1], [1.2, 0, 0, 1]])

    def patches_differ(projection):  # mean difference of the two spheres' pixels, each centred on its projection
        cam.set_projection(projection)
        img = viewer.snapshot().astype(int)
        h, w = img.shape[:2]
        clip = centres @ cam.matrices(w / h)[0].T
        px = ((clip[:, 0] / clip[:, 3] + 1) / 2 * w).round().astype(int)
        py = ((1 - clip[:, 1] / clip[:, 3]) / 2 * h).round().astype(int)
        a, b = (img[y - 25 : y + 25, x - 25 : x + 25] for x, y in zip(px, py, strict=True))
        return np.abs(a - b).mean()

    persp, ortho = patches_differ("perspective"), patches_differ("orthographic")
    assert ortho < 0.5 and persp > 3 * ortho


def test_global_ambient_lights_the_unlit_side(viewer):
    from valo import OrbitCamera
    from valo.lighting import Lighting

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.8, 0.8, 0.8))
    ball = viewer.renderer.spheres(m)
    ball.set([[0, 0, 0]], 0.8)
    viewer.show(SimpleNamespace(batches=lambda: [ball.batch()]))
    viewer.camera = OrbitCamera(azimuth=0.0, elevation=0.0, distance=4.0)
    viewer.lighting = Lighting("headlight", azimuth=180.0, elevation=0.0)  # from behind: the visible side is unlit
    lit = viewer.snapshot()
    viewer.lighting.ambient = 0.0
    dark = viewer.snapshot()
    background = lit[0, 0]
    assert lit.max() > 60 and (dark <= background).all()  # only ambient lit the front: without it, black on bg
    viewer.lighting = Lighting()



@pytest.mark.parametrize(("mode", "changes"), [("headlight", False), ("fixed", True)])
def test_orbiting_changes_the_shading_only_with_a_fixed_light(viewer, mode, changes):
    from valo import OrbitCamera
    from valo.lighting import Lighting

    m = viewer.renderer.material("Matte", "lambert", base_color=(0.8, 0.8, 0.8))
    ball = viewer.renderer.spheres(m)
    ball.set([[0, 0, 0]], 0.8)
    viewer.show(SimpleNamespace(batches=lambda: [ball.batch()]))
    viewer.camera, viewer.lighting = OrbitCamera(elevation=0.0, distance=4.0), Lighting(mode, -40.0, 20.0)
    first = viewer.snapshot()
    viewer.camera.azimuth += 180  # a sphere looks the same from behind, unless the light stayed put
    differs = np.abs(viewer.snapshot().astype(int) - first).max() > 30
    assert differs == changes
    viewer.lighting = Lighting()


# -- every shared Editor action: changes something, and undo restores state and image exactly
EDITOR_ACTIONS = {  # name → (action(editor, material), changes the image)
    "projection": (lambda ed, m: ed.set_projection("orthographic"), True),
    "field of view": (lambda ed, m: ed.set_fov(60.0), True),
    "view preset": (lambda ed, m: ed.set_view("+X"), True),
    "fit view": (lambda ed, m: ed.fit_view(), True),
    "light mode": (lambda ed, m: ed.set_light_mode("fixed"), False),  # switching keeps the light where it is
    "light direction": (lambda ed, m: ed.set_light("azimuth", 90.0), True),
    "light colour": (lambda ed, m: ed.set_light("color", (1.0, 0.3, 0.1)), True),
    "light intensity": (lambda ed, m: ed.set_light("intensity", 2.0), True),
    "ambient intensity": (lambda ed, m: ed.set_light("ambient", 1.2), True),
    "sky colour": (lambda ed, m: ed.set_light("sky", (1.0, 0.0, 0.0)), True),
    "flat ambient": (lambda ed, m: ed.set_flat_ambient((0.0, 0.0, 1.0)), True),
    "material parameter": (lambda ed, m: ed.set_param(m, "base_color", (0.0, 1.0, 0.0)), True),
    "material reset": (lambda ed, m: ed.reset(m), True),
}


@pytest.mark.parametrize("name", list(EDITOR_ACTIONS))
def test_editor_action_undoes_exactly(viewer, name):
    import copy

    from valo import Editor, Lighting, OrbitCamera, meshes

    action, visible = EDITOR_ACTIONS[name]
    m = viewer.renderer.material("Shiny", "blinn_phong", base_color=(0.7, 0.3, 0.2), shininess=200.0)
    cube = viewer.renderer.instances(meshes.cube(), m)
    cube.set(np.eye(4)[None])
    viewer.camera, viewer.lighting = OrbitCamera(), Lighting()
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
    from valo import Editor, OrbitCamera

    viewer.camera = cam = OrbitCamera()
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


def test_material_parameters_clamp_round_and_rebuild_from_dict(viewer):
    m = viewer.renderer.material("M", "toon", base_color=(2.0, -1.0, 0.5), bands=3.6, ambient=7.0)
    assert m.get("base_color") == (1.0, 0.0, 0.5)  # clamped to the schema range
    assert m.get("bands") == 4.0 and m.get("ambient") == 1.0  # ints round, floats clamp
    seen = []
    m.listeners.append(lambda *event: seen.append(event[1:]))
    m.set("bands", 4.2)  # rounds to the same value: no event
    m.set("bands", 6)
    assert seen == [("bands", 4.0, 6.0)]
    with pytest.raises(ValueError):
        m.set("shading", "chrome")
    d = m.to_dict()
    copy = viewer.renderer.material("copy", d.pop("shading"), **d)
    assert np.allclose(copy.data, m.data, atol=1e-4)


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
    from valo import Editor, OrbitCamera, rc_context

    viewer.camera = cam = OrbitCamera()
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


def test_batches_carry_their_instances_and_matrices(viewer):
    from valo import instance_matrices, meshes

    m = viewer.renderer.material("M", "lambert")
    cubes = viewer.renderer.instances(meshes.cube(), m, label="cubes", info=lambda i: {"index": i})
    matrices = instance_matrices([[0, 0, 0], [2, 0, 0]])
    cubes.set(matrices)
    b = cubes.batch()
    assert b.source is cubes and b[:4] == (m, cubes.mesh, cubes.bind_group, 2)
    assert np.array_equal(cubes.matrices, matrices) and cubes.label == "cubes" and cubes.info(1) == {"index": 1}


def test_pick_finds_the_batch_and_instance_under_a_pixel(viewer):
    from valo import OrbitCamera, instance_matrices, meshes

    m = viewer.renderer.material("M", "lambert")
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]], 0.5)
    cubes = viewer.renderer.instances(meshes.cube(), m)
    cubes.set(instance_matrices([[0, 0, 1.5]], 0.8))
    scene = SimpleNamespace(batches=lambda: [balls.batch(), cubes.batch()])
    viewer.show(scene)
    viewer.camera = cam = OrbitCamera(azimuth=-90.0, elevation=0.0, distance=8.0)  # looking along +Y
    viewer.snapshot()
    x0, y0, w, h = viewer.viewport

    def pixel(p):
        clip = cam.matrices(w / h)[0] @ np.r_[p, 1.0]
        return x0 + (clip[0] / clip[3] + 1) / 2 * w, y0 + (1 - clip[1] / clip[3]) / 2 * h

    for point, batch, instance in (([1.5, 0, 0], 0, 2), ([-1.5, 0, 0], 0, 0), ([0, 0, 1.5], 1, 0)):
        hit = viewer.renderer.pick(viewer.viewport, scene, cam, *pixel(point))
        assert (hit.batch_index, hit.instance_index) == (batch, instance)
        assert hit.source is (balls if batch == 0 else cubes)
    hit = viewer.renderer.pick(viewer.viewport, scene, cam, *pixel([1.5, 0, 0]))
    assert np.linalg.norm(hit.world_point - [1.5, 0, 0]) == pytest.approx(0.5, abs=0.02)  # on the sphere's surface
    assert hit.world_point[1] < 0  # the near side, facing the camera at -Y
    assert viewer.renderer.pick(viewer.viewport, scene, cam, *pixel([0, 0, -2.5])) is None  # background
    assert viewer.renderer.pick(viewer.viewport, scene, cam, -5, -5) is None  # outside the viewport


def test_hovered_object_is_highlighted_and_nothing_else(viewer):
    from valo import Editor, OrbitCamera, rc_context

    m = viewer.renderer.material("M", "lambert", base_color=(0.5, 0.5, 0.5))
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.2, 0, 0], [1.2, 0, 0]], 0.6)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()]), Editor())
    viewer.camera = cam = OrbitCamera(azimuth=-90.0, elevation=0.0, distance=6.0)
    viewer.snapshot()
    x0, y0, w, h = viewer.viewport
    clip = cam.matrices(w / h)[0] @ np.array([-1.2, 0, 0, 1.0])
    r = viewer.canvas.get_pixel_ratio()
    sx, sy = x0 + (clip[0] / clip[3] + 1) / 2 * w, y0 + (1 - clip[1] / clip[3]) / 2 * h
    viewer.pointer = (sx / r, sy / r)
    with rc_context({"highlight.strength": 0.0}):
        plain = viewer.snapshot().astype(int)
    lit = viewer.snapshot().astype(int)
    assert viewer.hover and viewer.hover.instance_index == 0
    changed = np.abs(lit - plain).max(-1) > 3
    assert changed.sum() > 200
    assert not changed[:, w // 2 :].any()  # the other sphere (right half) and the background stay as they were
    viewer.pointer = None


def test_info_rows_show_how_the_instance_was_placed(viewer):
    from valo import instance_matrices, meshes
    from valo.render import Pick
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
    from valo import Editor, OrbitCamera, rc_context

    m = viewer.renderer.material("M", "lambert")
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.5, 0, 0], [1.5, 0, 0]], 0.5)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()], bounding_radius=lambda: 2.0), Editor())
    viewer.camera = cam = OrbitCamera(azimuth=-90.0, elevation=0.0, distance=8.0)
    viewer.snapshot()
    x0, y0, w, h = viewer.viewport
    r = viewer.canvas.get_pixel_ratio()
    clip = cam.matrices(w / h)[0] @ np.array([1.5, 0, 0, 1.0])
    at = ((x0 + (clip[0] / clip[3] + 1) / 2 * w) / r, (y0 + (1 - clip[1] / clip[3]) / 2 * h) / r)

    def key(name, *mods):
        viewer.on_key({"event_type": "key_down", "key": name, "modifiers": mods})

    def click():
        viewer.on_pointer({"event_type": "pointer_move", "x": at[0], "y": at[1]})
        viewer.snapshot()  # picks what is under the pointer
        viewer.on_pointer({"event_type": "pointer_down", "button": 1, "x": at[0], "y": at[1], "modifiers": ()})
        viewer.on_pointer({"event_type": "pointer_up", "button": 1, "x": at[0], "y": at[1], "modifiers": ()})

    with rc_context({"camera.center_duration": 0.0}):
        click()
        assert viewer.selection.instance_index == 1
        key("f")  # frame the selection: centre on that sphere, zoomed to fit it
        assert cam.target == pytest.approx((1.5, 0.0, 0.0)) and cam.distance < 8.0
        key("Home")  # fit everything again
        assert cam.target == pytest.approx((0.0, 0.0, 0.0))
        key("Escape")
        assert viewer.selection is None
        key("7")
        assert cam.elevation == pytest.approx(89.99)  # top view (undoable through the editor)
        key("5")
        assert cam.projection == "orthographic"
        az = cam.azimuth
        key("ArrowLeft")
        assert cam.azimuth == pytest.approx((az - 15) % 360)
        target = cam.target
        key("ArrowUp", "Shift")
        assert cam.target != target  # Shift+arrow pans
        assert len(viewer.editor.undo_stack.done) == 2  # view preset and projection; navigation is not recorded

    viewer.camera = cam = OrbitCamera(azimuth=-90.0, elevation=0.0, distance=8.0)
    with rc_context({"camera.center_duration": 0.0}):
        click()
        click()  # a double-click centres on the point under the pointer
        assert cam.target[0] == pytest.approx(1.5, abs=0.1) and cam.target[1] < 0  # front of the sphere
    viewer.selection = None
