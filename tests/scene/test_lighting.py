"""valo.scene.lighting: headlight vs fixed light, mode switches that keep the direction, defaults from rc."""

import pytest


def test_headlight_follows_the_camera_and_fixed_light_does_not():
    from valo.scene.camera import Camera
    from valo.scene.lighting import Lighting

    cam = Camera()
    head, fixed = Lighting("headlight", 0.0, 0.0), Lighting("fixed", 30.0, 40.0)
    assert head.direction(cam) == pytest.approx(-cam.forward())  # azimuth 0, elevation 0: from the viewer
    before = fixed.direction(cam)
    cam.azimuth += 120
    assert head.direction(cam) == pytest.approx(-cam.forward())
    assert fixed.direction(cam) == pytest.approx(before)


@pytest.mark.parametrize("start", ["headlight", "fixed"])
def test_switching_light_mode_keeps_the_direction(start):
    from valo.scene.camera import Camera
    from valo.scene.lighting import MODES, Lighting

    cam = Camera(azimuth=37.0, elevation=-12.0)
    light = Lighting(start, -50.0, 25.0)
    d = light.direction(cam)
    light.set_mode(next(m for m in MODES if m != start), cam)
    assert light.direction(cam) == pytest.approx(d, abs=1e-9)


def test_lighting_defaults_come_from_rc():
    from valo import Lighting, rc_context

    with rc_context({"lighting.mode": "fixed", "lighting.ambient": 0.7}):
        light = Lighting()
    assert (light.mode, light.ambient) == ("fixed", 0.7)
    assert Lighting().mode == "headlight"
