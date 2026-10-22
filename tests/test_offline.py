"""valo.render offline: image size, alpha, background, PNG, supersampling, the same picture as the viewer, scripts."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


@pytest.fixture
def ball_scene():
    from valo import renderer

    m = renderer().material("M", "lambert", base_color=(0.8, 0.8, 0.8))
    ball = renderer().spheres(m)
    ball.set([[0, 0, 0]], 1.0)
    return SimpleNamespace(batches=lambda: [ball.batch()], bounding_radius=lambda: 1.0)


def test_offline_render_size_alpha_background_and_png(ball_scene, tmp_path):
    from PIL import Image

    from valo import render

    img = render(ball_scene, size=(320, 200), samples=1)
    assert img.shape == (200, 320, 4) and (img[..., 3] == 255).all()
    red = render(ball_scene, size=(320, 200), samples=1, background=(1.0, 0.0, 0.0))
    assert (red[0, 0, :3] == (255, 0, 0)).all()
    clear = render(ball_scene, {"image": {"size": [320, 200], "transparent": True}}, samples=1, path=tmp_path / "a.png")
    assert clear[0, 0, 3] == 0 and clear[100, 160, 3] == 255  # background transparent, the ball opaque
    assert Image.open(tmp_path / "a.png").mode == "RGBA"
    with pytest.raises(KeyError, match="unknown image settings"):
        render(ball_scene, {"image": {"resolution": [10, 10]}})


def test_supersampling_smooths_the_outline(ball_scene):
    from valo import render

    def partial_alpha(samples):  # pixels on the outline that are neither background nor ball
        a = render(ball_scene, size=(200, 200), samples=samples, transparent=True)[..., 3]
        return int(((a > 0) & (a < 255)).sum())

    assert partial_alpha(1) == 0 and partial_alpha(4) > 50


def test_offline_render_matches_the_viewer_and_settings_round_trip(viewer, ball_scene):
    from valo import Camera, Lighting, render

    viewer.show(ball_scene)
    viewer.camera, viewer.lighting = Camera(azimuth=30.0, distance=5.0), Lighting(mode="fixed")
    shown = viewer.snapshot()
    h, w = shown.shape[:2]
    settings = {"camera": viewer.camera.settings(), "lighting": viewer.lighting.settings()}
    img = render(ball_scene, settings, size=(w, h), samples=1)
    assert np.array_equal(img[..., :3], shown)  # the same picture as on screen
    assert Camera.from_settings(settings["camera"]) == viewer.camera
    with pytest.raises(KeyError, match="did you mean 'azimuth'"):
        Camera.from_settings({"azimut": 3})
    viewer.lighting = Lighting()


def test_py_literal_writes_readable_valid_python():
    import ast

    from valo.offline import py_literal

    data = {"scene": {"n": [6, 6, 6], "name": 'say "hi"', "materials": [{"name": "M", "rgb": [0.1, 0.2, 0.3]}] * 4},
            "image": {"transparent": True, "background": None, "samples": 2}}
    text = py_literal(data)
    assert ast.literal_eval(text) == data
    assert '"n": [6, 6, 6],' in text and '\n    "image": ' in text and max(map(len, text.splitlines())) <= 100


def test_any_app_can_render_an_image_but_export_needs_scene_settings(viewer, ball_scene, tmp_path):
    from valo import Editor

    editor = Editor()
    viewer.show(ball_scene, editor)
    editor.show_render = True
    viewer.snapshot()  # the Render window draws
    editor.render_options.update(size=[64, 48], samples=1)
    assert editor.render_image(tmp_path / "x.png").exists() and editor.messages[-1].text == "rendered x.png"
    assert not editor.can_export()
    with pytest.raises(ValueError):
        editor.export_script(tmp_path / "x.py")
