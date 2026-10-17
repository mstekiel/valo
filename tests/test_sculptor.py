"""MeshSculptor rendered offscreen: edits, undo, hot reload, errors, save to the library. Needs a GPU adapter."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
VASE = Path(__file__).parents[1] / "examples" / "sculpts" / "vase.py"


@pytest.fixture(scope="module")
def viewer():
    pytest.importorskip("wgpu")
    from rendercanvas.offscreen import RenderCanvas

    from valo import Viewer

    try:
        return Viewer(RenderCanvas(size=(900, 600), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")


@pytest.fixture
def sculptor(viewer, tmp_path, monkeypatch):
    from valo import library
    from valo.tools.sculptor import build

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "library")  # never write into the real library
    path = tmp_path / "vase.py"
    shutil.copy(VASE, path)
    return build(viewer, path)


def snap(viewer) -> np.ndarray:
    for _ in range(2):
        img = np.asarray(viewer.canvas.draw())[:, :, :3].copy()
    return viewer.crop(img)


def touch_later(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    t = path.stat().st_mtime + 5  # mtime resolution can be coarse; make the change unmistakable
    os.utime(path, (t, t))


def test_shape_transform_and_material_edits_undo_exactly(sculptor, viewer):
    ed = sculptor
    before = snap(viewer)
    assert before.std() > 5 and ed.error is None and ed.report.ok
    ed.set_shape_param("belly", 0.5)
    ed.set_transform("rotate_deg", [45.0, 0.0, 0.0])
    ed.set_param(ed.material, "base_color", (0.1, 0.8, 0.1))
    assert np.abs(snap(viewer).astype(int) - before).max() > 50
    while ed.undo_stack.done:
        ed.undo_stack.undo()
    assert np.array_equal(snap(viewer), before)


def test_hot_reload_keeps_values_and_survives_errors(sculptor):
    ed = sculptor
    ed.set_shape_param("height", 2.0)
    first = ed.scene.data

    touch_later(ed.path, ed.path.read_text(encoding="utf-8").replace("np.sin(np.pi * t) ** 2", "np.sin(np.pi * t)"))
    ed._polled = 0.0
    ed.poll()
    assert ed.error is None and ed.values["height"] == 2.0
    assert ed.scene.data is not first and not np.array_equal(ed.scene.data.vertices, first.vertices)

    good = ed.scene.data
    touch_later(ed.path, "def mesh(:\n")
    ed._polled = 0.0
    ed.poll()
    assert "SyntaxError" in ed.error
    assert ed.scene.data is good  # last good mesh stays


def test_save_writes_a_library_entry_that_rebuilds_what_is_on_screen(sculptor):
    from valo import library

    ed = sculptor
    ed.set_shape_param("segments", 20)
    ed.set_transform("scale", [2.0, 2.0, 2.0])
    ed.set_param(ed.material, "roughness", 0.9)
    assert ed.save("my_vase") is not None
    assert library.names() == ["my_vase"]
    assert np.allclose(library.mesh("my_vase").vertices, ed.scene.data.vertices, atol=1e-6)
    assert library.load("my_vase").material["roughness"] == pytest.approx(0.9)
    assert meshes_ok(library.mesh("my_vase"))

    assert ed.save("my_vase") is None and "exists" in ed.status  # no silent overwrite
    assert ed.save("my_vase", overwrite=True) is not None
    assert ed.save("not a name") is None


def meshes_ok(data) -> bool:
    from valo import meshes

    return meshes.check(data).ok
