"""Mesh gallery: rendered offscreen, every shape visible, material edits undo exactly. Skipped without an adapter."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gallery():
    pytest.importorskip("wgpu")
    from gallery import build
    from rendercanvas.offscreen import RenderCanvas

    from valo import Viewer

    try:
        viewer = Viewer(RenderCanvas(size=(1000, 560), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")
    return viewer, build(viewer)


def snap(viewer) -> np.ndarray:
    for _ in range(2):  # ImGui lays out on the first frame, settles on the second
        img = np.asarray(viewer.canvas.draw())[:, :, :3].copy()
    return viewer.crop(img)


def test_every_shape_changes_the_image_and_undo_restores_it(gallery):
    viewer, ed = gallery
    before = snap(viewer)
    assert before.std() > 5, "viewport is blank"
    for name, inst in ed.gallery.items:  # emissive red on one shape at a time must show up
        ed.set_param(inst.material, "emissive", (1.0, 0.0, 0.0))
        assert np.abs(snap(viewer).astype(int) - before).max() > 50, f"{name} not visible"
        ed.undo_stack.undo()
    assert np.array_equal(snap(viewer), before)
