"""Fast tests: pure Python/numpy, no GPU."""

import numpy as np

# -- config


def test_py_literal_writes_readable_valid_python():
    import ast

    from valo.offline import py_literal

    data = {"scene": {"n": [6, 6, 6], "name": 'say "hi"', "materials": [{"name": "M", "rgb": [0.1, 0.2, 0.3]}] * 4},
            "image": {"transparent": True, "background": None, "samples": 2}}
    text = py_literal(data)
    assert ast.literal_eval(text) == data
    assert '"n": [6, 6, 6],' in text and '\n    "image": ' in text and max(map(len, text.splitlines())) <= 100


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
