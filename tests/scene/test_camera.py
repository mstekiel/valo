"""valo.scene.camera: projections keep the scene's size, panning moves the target in the view plane."""

import numpy as np
import pytest


def test_switching_projection_keeps_the_target_plane_the_same_size():
    from valo.scene.camera import Camera

    cam = Camera(distance=10.0)
    point = np.asarray(cam.target) + np.cross(cam.forward(), (0, 0, 1)) * 1.5  # beside the target, on its plane

    def ndc():
        vp, _ = cam.matrices(1.5)
        p = vp @ np.r_[point, 1.0]
        return p[:2] / p[3]

    persp = ndc()
    cam.set_projection("orthographic")
    assert ndc() == pytest.approx(persp, abs=1e-5)
    cam.zoom(2.0)
    cam.set_projection("perspective")
    assert cam.distance == pytest.approx(20.0)


def test_camera_changes_keep_the_scene_size():
    from valo.scene.camera import Camera

    radius = 4.75

    def outline_share(cam):  # visible height share of the bounding sphere's outline
        if cam.projection == "orthographic":
            return 2 * radius / cam.ortho_height
        return np.tan(np.arcsin(radius / cam.distance)) / np.tan(np.radians(cam.fovy / 2))

    cam = Camera(distance=15.0)
    share = outline_share(cam)
    for change in (lambda: cam.set_fov(60, radius), lambda: cam.set_fov(20, radius),
                   lambda: cam.set_projection("orthographic", radius),
                   lambda: cam.set_projection("perspective", radius)):
        change()
        assert outline_share(cam) == pytest.approx(share)
    cam.set_fov(35, radius)
    assert cam.distance == pytest.approx(15.0)  # back to the start: same distance


def test_pan_moves_the_target_in_the_view_plane():
    from valo.scene.camera import Camera

    cam = Camera(azimuth=-90.0, elevation=0.0)  # looking along +Y: right = +X, up = +Z
    cam.pan(1.0, 2.0)
    assert cam.target == pytest.approx((-1.0, 0.0, 2.0))  # the scene follows the cursor
    assert cam.world_per_pixel(500) == pytest.approx(cam._height_at_target() / 500)
