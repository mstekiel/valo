"""valo.ui.navigation: eased transitions, navigation keys, presets and fitting (a stand-in viewer, no GPU)."""

import numpy as np
import pytest


def navigator(**scene):
    from types import SimpleNamespace

    from valo import Camera
    from valo.ui.navigation import Navigator

    viewer = SimpleNamespace(camera=Camera(distance=10.0), scene=SimpleNamespace(**scene), selection=None,
                             viewport=(0, 0, 800, 600))
    return Navigator(viewer), viewer.camera


def test_center_on_eases_to_the_goal_and_ends_exactly():
    nav, cam = navigator()
    nav.center_on((1.0, 2.0, 3.0), radius=1.0, duration=0.5)
    assert nav.animating and cam.target == (0.0, 0.0, 0.0)
    nav.step(0.25)
    assert cam.target[0] == pytest.approx(0.5) and cam.fit_distance(1.0) < cam.distance < 10.0  # half way
    nav.step(0.3)
    assert not nav.animating
    assert cam.target == pytest.approx((1, 2, 3)) and cam.distance == pytest.approx(cam.fit_distance(1.0))
    nav.center_on((0.0, 0.0, 0.0), duration=0)  # 0 jumps
    assert cam.target == (0.0, 0.0, 0.0) and not nav.animating


def test_navigation_keys_presets_and_fitting_move_the_camera():
    nav, cam = navigator(center=lambda: np.array([4.0, 0, 0]), bounding_radius=lambda: 1.0)
    nav.fit_all()  # eased onto the scene's centre, sized to its radius
    nav.step(10.0)
    assert cam.target == pytest.approx((4.0, 0.0, 0.0)) and cam.distance < 10.0
    nav.set_view("+X")
    nav.toggle_projection()
    assert (cam.azimuth, cam.projection) == (0.0, "orthographic")
    assert nav.key("5") and cam.projection == "perspective"
    assert nav.key("ArrowLeft") and cam.azimuth == pytest.approx(345.0)
    target = cam.target
    assert nav.key("ArrowUp", ("Shift",)) and cam.target != target  # Shift+arrow pans
    assert nav.key("7") and cam.elevation == pytest.approx(89.99)
    assert not nav.key("q")  # not a navigation key
