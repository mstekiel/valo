"""Mesh gallery: rendered offscreen, every object visible, edits undo exactly. Skipped without an adapter."""

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


def test_every_object_and_material_shows_and_undo_restores_it(gallery):
    viewer, ed = gallery
    before = viewer.snapshot()
    assert before.std() > 5, "viewport is blank"
    names = [o.name for o in ed.scene.objects()]
    assert names == ["Sphere", "Icosphere", "Cube", "Cylinder", "Cone", "Torus", "Capsule", "Arrow", "Arrows", "Floor"]
    for obj in ed.scene.objects():  # emissive red on one material at a time must show up
        ed.set_param(ed.scene.materials[obj.material], "emissive", (1.0, 0.0, 0.0))
        assert np.abs(viewer.snapshot().astype(int) - before).max() > 50, f"{obj.name} not visible"
        ed.undo_stack.undo()
    assert np.array_equal(viewer.snapshot(), before)


def test_the_ring_is_rebuilt_from_its_parameters(gallery):
    viewer, ed = gallery
    ring = ed.scene.find("Ring")
    before = viewer.snapshot()
    assert ring.children[0].instances.count == 16
    ed.set_procedural_param(ring, "n", 6)
    assert ring.children[0].instances.count == 6
    assert np.abs(viewer.snapshot().astype(int) - before).max() > 50
    ed.undo_stack.undo()
    assert ring.children[0].instances.count == 16 and np.array_equal(viewer.snapshot(), before)


def test_the_gallery_saves_and_rebuilds_from_its_settings(gallery):
    from valo import Scene, render

    viewer, ed = gallery
    d = ed.scene.settings()
    assert d["nodes"][1] == {"kind": "procedural", "name": "Ring", "visible": True,
                             "transform": d["nodes"][1]["transform"], "builder": "gallery:arrow_ring",
                             "params": {"n": 16, "radius": 3.6}}
    cam = viewer.camera
    assert np.array_equal(render(Scene.from_settings(d), size=(200, 120), camera=cam),
                          render(ed.scene, size=(200, 120), camera=cam))
