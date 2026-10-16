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


def snap(viewer) -> np.ndarray:
    for _ in range(2):  # ImGui lays out on the first frame, settles on the second
        img = np.asarray(viewer.canvas.draw())[:, :, :3].copy()
    return img[:, : viewer.viewport_width]  # scene region only


def test_empty_viewer_draws_the_background(viewer):
    viewer.show(SimpleNamespace(batches=list))
    img = snap(viewer)
    assert (img == img[0, 0]).all()


def test_spheres_render_and_material_edits_undo_exactly(viewer):
    from valo import Editor

    m = viewer.renderer.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]], [0.3, 0.5, 0.7])
    editor = Editor()
    editor.watch(m)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()], bounding_radius=lambda: 2.5), editor)
    before = snap(viewer)
    assert before.std() > 5, "viewport is blank"

    editor.set_param(m, "shading", "toon")
    assert np.abs(snap(viewer).astype(int) - before).max() > 0
    editor.undo_stack.undo()
    assert np.array_equal(snap(viewer), before)
