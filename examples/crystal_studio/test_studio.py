"""Crystal studio tests: the rock-salt lattice (pure), and the full app rendered offscreen (GPU)."""

import numpy as np
import pytest
from crystal import rock_salt


@pytest.mark.parametrize(("n", "counts"), [((6, 6, 6), (108, 108)), ((9, 5, 3), (68, 67)), ((1, 1, 1), (1, 0))])
def test_rock_salt_counts_and_centering(n, counts):
    centers, kind = rock_salt(n, 1.0)
    assert (np.sum(kind == 0), np.sum(kind == 1)) == counts
    assert np.allclose(centers.mean(0), 0.0)


def test_rock_salt_neighbours_alternate_species():
    centers, kind = rock_salt((4, 1, 1), 1.0)
    assert list(kind) == [0, 1, 0, 1]
    assert np.allclose(np.diff(centers[:, 0]), 1.0)


@pytest.fixture(scope="module")
def studio():
    pytest.importorskip("wgpu")
    from rendercanvas.offscreen import RenderCanvas
    from studio import build

    from valo import Viewer

    try:
        viewer = Viewer(RenderCanvas(size=(900, 520), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")
    return viewer, build(viewer)


def snap(viewer) -> np.ndarray:
    for _ in range(2):  # ImGui lays out on the first frame, settles on the second
        img = np.asarray(viewer.canvas.draw())[:, :, :3].copy()
    return img[:, : viewer.viewport_width]  # scene region only


@pytest.mark.gpu
def test_edits_change_the_image_and_undo_restores_it_exactly(studio):
    viewer, ed = studio
    crystal = ed.crystal
    before = snap(viewer)
    assert before.std() > 5, "viewport is blank"

    ed.set_count(0, 9)
    ed.set_count(1, 5)
    ed.set_count(2, 3)
    assert [s.count for s in crystal.species] == [68, 67]
    ed.selected = 0
    ed.set_radius(0.22)
    ed.set_param(ed.species.material, "metallic", 1.0)
    ed.selected = 1
    ed.assign(next(m for m in crystal.materials if m.name == "Toon blue"))
    ed.set_param(ed.species.material, "bands", 4)
    ed.duplicate()
    ed.set_param(ed.species.material, "base_color", (1.0, 0.0, 0.0))
    edited = snap(viewer)
    assert np.abs(edited.astype(int) - before).max() > 0

    while ed.undo_stack.done:
        ed.undo_stack.undo()
    ed.selected = 0
    assert [s.count for s in crystal.species] == [108, 108]
    assert np.array_equal(snap(viewer), before)
