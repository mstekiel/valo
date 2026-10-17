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
    assert set(editor.visible) == {"A", "B", "Camera", "Lighting", "History"} and all(editor.visible.values())
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


def test_camera_and_light_settings_undo_exactly(viewer):
    import copy

    from valo import Editor, OrbitCamera
    from valo.lighting import Lighting

    m = viewer.renderer.material("Shiny", "blinn_phong", base_color=(0.7, 0.3, 0.2))
    cubes = viewer.renderer.instances(__import__("valo").meshes.cube(), m)
    cubes.set(np.eye(4)[None])
    viewer.camera, viewer.lighting = OrbitCamera(distance=4.0), Lighting()
    editor = Editor()
    viewer.show(SimpleNamespace(batches=lambda: [cubes.batch()], bounding_radius=lambda: 0.9), editor)
    before, cam0 = viewer.snapshot(), copy.copy(viewer.camera)
    editor.set_projection("orthographic")
    editor.set_view("+X")
    editor.set_light_mode("fixed")
    editor.set_light("azimuth", 120.0)
    editor.set_light("color", (1.0, 0.5, 0.2))
    editor.set_flat_ambient((0.2, 0.2, 0.9))
    editor.fit_view()
    assert viewer.camera.projection == "orthographic" and viewer.lighting.sky == viewer.lighting.ground
    assert np.abs(viewer.snapshot().astype(int) - before).max() > 30
    while editor.undo_stack.done:
        editor.undo_stack.undo()
    assert viewer.camera == cam0 and viewer.lighting == Lighting()
    assert np.array_equal(viewer.snapshot(), before)


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
