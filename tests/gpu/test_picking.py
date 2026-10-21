"""valo.gpu.renderer.pick: the batch, instance and world point under a pixel."""

from types import SimpleNamespace

import numpy as np
import pytest

from valo.testing import framebuffer_point

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


def test_pick_finds_the_batch_and_instance_under_a_pixel(viewer):
    from valo import Camera, instance_matrices, meshes

    m = viewer.renderer.material("M", "lambert")
    balls = viewer.renderer.spheres(m)
    balls.set([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]], 0.5)
    cubes = viewer.renderer.instances(meshes.cube(), m)
    cubes.set(instance_matrices([[0, 0, 1.5]], 0.8))
    scene = SimpleNamespace(batches=lambda: [balls.batch(), cubes.batch()])
    viewer.show(scene)
    viewer.camera = cam = Camera(azimuth=-90.0, elevation=0.0, distance=8.0)  # looking along +Y
    viewer.snapshot()

    def pixel(p):
        return framebuffer_point(viewer, p)

    for point, batch, instance in (([1.5, 0, 0], 0, 2), ([-1.5, 0, 0], 0, 0), ([0, 0, 1.5], 1, 0)):
        hit = viewer.renderer.pick(viewer.viewport, scene, cam, *pixel(point))
        assert (hit.batch_index, hit.instance_index) == (batch, instance)
        assert hit.source is (balls if batch == 0 else cubes)
    hit = viewer.renderer.pick(viewer.viewport, scene, cam, *pixel([1.5, 0, 0]))
    assert np.linalg.norm(hit.world_point - [1.5, 0, 0]) == pytest.approx(0.5, abs=0.02)  # on the sphere's surface
    assert hit.world_point[1] < 0  # the near side, facing the camera at -Y
    assert viewer.renderer.pick(viewer.viewport, scene, cam, *pixel([0, 0, -2.5])) is None  # background
    assert viewer.renderer.pick(viewer.viewport, scene, cam, -5, -5) is None  # outside the viewport
