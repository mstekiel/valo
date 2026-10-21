"""valo.ui.overlays: the compass and the hover / selection bounding boxes drawn over the 3D view, and box_layout."""

from types import SimpleNamespace

import numpy as np
import pytest

from valo.testing import changed, hover, image_point, release_pointer

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


def two_balls(viewer):
    from valo import Camera, Editor

    m = viewer.renderer.material("M", "lambert", base_color=(0.5, 0.5, 0.5))
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.2, 0, 0], [1.2, 0, 0]], 0.6)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()]), Editor())
    viewer.camera = Camera(azimuth=-90.0, elevation=0.0, distance=6.0)
    viewer.selection, viewer.outlined = None, []
    release_pointer(viewer)
    return balls, viewer.snapshot()


def test_the_hovered_object_gets_a_box_not_a_tint(viewer):
    from valo import rc_context

    balls, plain = two_balls(viewer)
    w = plain.shape[1]
    assert hover(viewer, [-1.2, 0, 0]).instance_index == 0
    diff = changed(plain, viewer.snapshot())
    cx, cy = image_point(viewer, [-1.2, 0, 0])
    assert diff.sum() > 50 and not diff[:, w // 2 :].any()  # lines around the left ball only
    assert not diff[cy - 5 : cy + 5, cx - 5 : cx + 5].any()  # its own pixels are not tinted
    with rc_context({"highlight.enabled": False}):
        assert not changed(plain, viewer.snapshot()).any()
    release_pointer(viewer)


def test_the_selection_gets_a_box_in_its_own_colour(viewer):
    from valo.gpu import Pick
    from valo.math import box_corners

    balls, plain = two_balls(viewer)
    w = plain.shape[1]
    viewer.selection = Pick(0, 0, balls, np.zeros(3))
    diff = changed(plain, viewer.snapshot())
    assert diff[:, : w // 2].sum() > 50 and not diff[:, w // 2 :].any()  # no owner: the clicked instance's box
    img = viewer.snapshot().astype(int)
    lines = img[diff]
    assert (lines[:, 0] - lines[:, 2]).mean() > 30  # orange (selected_color), not the grey hover colour
    both = np.eye(4)
    both[:3, 3], both[0, 0], both[1, 1], both[2, 2] = (0, 0, 0), 3.6, 1.2, 1.2
    balls.owner = SimpleNamespace(bounding_box=lambda: both)  # an object owning both balls
    diff = changed(plain, viewer.snapshot())
    assert diff[:, : w // 2].any() and diff[:, w // 2 :].any()  # one box around the whole object
    viewer.selection, viewer.outlined = None, [balls.owner]
    assert changed(plain, viewer.snapshot()).any()
    viewer.outlined = []
    assert not changed(plain, viewer.snapshot()).any()
    del balls.owner
    assert box_corners([0, 0, 0], [1, 2, 3]).shape == (8, 3)


def test_compass_turns_with_the_camera_and_draws_in_its_corner(viewer):
    from valo import Camera, Editor, rc_context
    from valo.ui import compass_layout

    cam = Camera(azimuth=-90.0, elevation=0.0)  # looking along +Y: right = +X, up = +Z
    axes = Editor().compass_axes()
    layout = {label: (end, near) for label, _, end, _, near in compass_layout(cam, axes, (100, 100), 40)}
    assert layout["x"][0] == pytest.approx((140, 100)) and layout["z"][0] == pytest.approx((100, 60))
    assert compass_layout(cam, axes, (100, 100), 40)[0][0] == "y"  # pointing away: drawn first (behind)

    viewer.show(SimpleNamespace(batches=list), Editor())
    with rc_context({"ui.compass": False}):
        plain = viewer.snapshot().astype(int)
    with rc_context({"ui.compass_corner": "bottom-left"}):
        drawn = viewer.snapshot().astype(int)
    changed = np.abs(drawn - plain).max(-1) > 20
    h, w = changed.shape
    assert changed[h // 2 :, : w // 3].sum() > 30 and not changed[: h // 2].any()  # only bottom-left


def test_box_layout_projects_the_12_edges_and_marks_the_back_ones():
    from valo import Camera
    from valo.ui import box_layout

    cam = Camera(azimuth=-60.0, elevation=25.0, distance=6.0)  # looking at the origin from front-right, above
    edges = box_layout(cam, np.eye(4), (0, 0, 800, 600))
    assert len(edges) == 12 and sum(back for *_, back in edges) == 3  # the 3 edges at the hidden corner
    xs = [p[0] for s, e, _ in edges for p in (s, e)]
    assert 0 < min(xs) < 400 < max(xs) < 800  # around the middle of the view
    cam.projection = "orthographic"
    assert sum(back for *_, back in box_layout(cam, np.eye(4), (0, 0, 800, 600))) == 3
    cam.projection = "perspective"

    behind = np.eye(4)
    behind[:3, 3] = cam.eye() - 3 * cam.forward()  # a unit box behind the camera: nothing to draw
    assert box_layout(cam, behind, (0, 0, 800, 600)) == []
    around = np.diag([20.0, 20.0, 20.0, 1.0])  # the camera inside a big box: edges crossing the near side are cut
    edges = box_layout(cam, around, (0, 0, 800, 600))
    assert 0 < len(edges) <= 12 and all(np.isfinite(p).all() for s, e, _ in edges for p in (s, e))
