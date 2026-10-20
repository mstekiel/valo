"""valo studio rendered offscreen: shape files in the Code window, re-read on save, New / Open shape, Save to library.
Needs a GPU adapter."""

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
        return Viewer(RenderCanvas(size=(1280, 720), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")


@pytest.fixture
def studio(viewer, tmp_path, monkeypatch):
    from valo import rc
    from valo.studio import build

    monkeypatch.setattr(rc.studio, "procedural_meshes", str(tmp_path / "library"))  # never the real library
    shutil.copy(VASE, tmp_path / "vase.py")
    editor = build(viewer)
    viewer.gui_error = None
    return editor


def files(ed):
    from valo import library

    return {p.name: p.read_text(encoding="utf-8") for p in sorted(library.directory().glob("*.py"))}


def blob(ed):
    ed.new_shape("blob")
    ed.undo_stack.done.clear()


def edited_code(ed):
    blob(ed)
    ed.set_code(ed.code.replace("height: float = 1.5", "height: float = 2.5"))


def moved_object(ed):
    blob(ed)
    obj = ed.scene.find("Blob")
    ed.set_transform(obj, "translate", (0.5, 0, 0))
    ed.set_shape_param(obj, "radius", 0.8)
    ed.undo_stack.done.clear()


STUDIO_ACTIONS = {  # name → (setup, action, changes the image)
    "new shape": (lambda ed: None, lambda ed, tmp: ed.new_shape("cup"), True),
    "open shape": (lambda ed: None, lambda ed, tmp: ed.open_shape(tmp / "vase.py"), True),
    "save code": (edited_code, lambda ed, tmp: ed.save_code(), True),
    "save to library": (moved_object, lambda ed, tmp: ed.save_to_library(ed.scene.find("Blob"), "kept"), False),
}


@pytest.mark.parametrize("name", list(STUDIO_ACTIONS))
def test_studio_actions_undo_exactly(viewer, studio, tmp_path, name):
    setup, action, visible = STUDIO_ACTIONS[name]
    setup(studio)
    file_set = name != "new shape"  # new shape leaves its file behind on undo, by design

    def state():
        return studio.scene.settings(), files(studio) if file_set else None

    before, image = state(), viewer.snapshot()
    action(studio, tmp_path)
    after = state()
    assert len(studio.undo_stack.done) == 1 and after != before
    assert (np.abs(viewer.snapshot().astype(int) - image).max() > 10) == visible
    studio.undo_stack.undo()
    assert state() == before and np.array_equal(viewer.snapshot(), image)
    studio.undo_stack.redo()
    assert state() == after
    assert viewer.gui_error is None


def test_saving_the_code_rereads_the_parameters(viewer, studio):
    blob(studio)
    obj = studio.scene.find("Blob")
    studio.set_shape_param(obj, "radius", 0.8)
    assert studio.code_path.name == "blob.py" and not studio.dirty
    studio.set_code(studio.code.replace("segments: int = 48)", "segments: int = 48, twist: float = 0.25)"))
    assert studio.dirty
    studio.save_code()
    assert not studio.dirty and [p.name for p in obj.shape.specs][-1] == "twist"  # the new parameter
    assert obj.shape.params["radius"] == 0.8 and obj.shape.params["twist"] == 0.25  # kept / new default
    mesh = obj.instances.mesh
    studio.set_code(studio.code.replace("def mesh(", "def mesh(:"))
    studio.save_code()
    assert "SyntaxError" in studio.code_error and obj.instances.mesh is mesh  # the last good mesh stays
    assert studio.code_error.startswith('File "') and "blob.py" in studio.code_error.splitlines()[0]  # your file
    viewer.snapshot()  # the error shows under the code
    studio.set_code(studio.code.replace("def mesh(:", "def mesh("))
    studio.save_code()
    assert studio.code_error is None and viewer.gui_error is None


def test_save_to_library_bakes_the_object_state_into_the_file(viewer, studio):
    from valo import library

    moved_object(studio)
    obj = studio.scene.find("Blob")
    studio.set_param(studio.scene.materials[obj.material], "roughness", 0.9)
    studio.set_transform(obj, "scale", (1, 1, 2))
    image = viewer.snapshot()
    path = studio.save_to_library(obj, "tall_blob")
    assert path == library.path("tall_blob") and library.names() == ["blob", "tall_blob"]
    assert obj.transform.settings() == {"translate": [0.0] * 3, "rotate_deg": [0.0] * 3, "scale": [1.0] * 3}
    assert np.abs(viewer.snapshot().astype(int) - image).max() <= 2  # baked: the same picture, up to rounding
    saved = library.load("tall_blob")
    assert saved.defaults()["radius"] == pytest.approx(0.8) and saved.material["roughness"] == pytest.approx(0.9)
    assert saved.transform["translate"] == pytest.approx([0.5, 0, 0]) and saved.transform["scale"][2] == 2.0
    assert studio.code_path == path  # the Code window follows the object to its new file
    with pytest.raises(FileExistsError):
        studio.save_to_library(studio.open_shape(library.path("blob")), "tall_blob")


def test_shape_files_changed_elsewhere_reload_and_names_are_checked(studio, tmp_path):
    from valo import rc_context

    obj = studio.open_shape(tmp_path / "vase.py")
    assert obj.material == "vase"  # its MATERIAL preset became a scene material
    text = (tmp_path / "vase.py").read_text(encoding="utf-8").replace("height: float = 1.2", "height: float = 2.0")
    (tmp_path / "vase.py").write_text(text, encoding="utf-8")
    import os

    os.utime(tmp_path / "vase.py", (0, (tmp_path / "vase.py").stat().st_mtime + 10))
    studio._polled = 0.0
    with rc_context({"studio.poll_s": 0.05, "sculptor.poll_s": 0.05}):
        studio.poll()
    assert "height: float = 2.0" in studio.code and not studio.dirty
    for bad in ("1abc", "_hidden", "has space"):
        with pytest.raises(ValueError):
            studio.new_shape(bad)
    studio.new_shape("dup")
    with pytest.raises(FileExistsError):
        studio.new_shape("dup")


def test_every_menu_and_window_draws_without_errors(viewer, studio):
    from imgui_bundle import imgui

    blob(studio)
    studio.reset_layout()
    viewer.snapshot()
    x, y, w, h = studio.scene_rect
    opened = 0
    for mx in range(4, 240, 6):
        viewer.ui_click(mx, 10)
        imgui.set_current_context(viewer.imgui.imgui_context)
        opened += imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
        viewer.ui_click(x + w / 2, y + h / 2)
    assert opened >= 3 and viewer.gui_error is None
    for kind in ("new", "open", "save"):  # the File dialogs
        studio._dialog = (kind, "")
        viewer.snapshot()
        studio._dialog = None
        viewer.snapshot()
    assert viewer.gui_error is None


def test_the_normals_overlay_follows_the_selection_and_the_mesh(viewer, studio):
    blob(studio)
    obj = studio.scene.find("Blob")
    studio.set_transform(obj, "translate", (1, 0, 0))
    image = viewer.snapshot()
    studio.show_normals = True
    viewer.snapshot()
    arrows = studio.normals
    assert 0 < arrows.count <= 2000 and arrows.batch() in studio.scene.batches()
    assert np.abs(viewer.snapshot().astype(int) - image).max() > 10
    v = int(studio._normal_rows[0, 1])
    start = arrows.matrices[0][:3, 3]
    assert np.allclose(start, obj.shape.build().positions[v] + [1, 0, 0], atol=1e-5)  # at the vertex, in world space
    assert set(studio.normal_info(0)) == {"Instance", "Vertex", "Normal"}
    count = arrows.count
    studio.set_shape_param(obj, "segments", 8)
    viewer.snapshot()
    assert arrows.count == len(obj.shape.build().positions) < count  # fewer vertices, fewer arrows
    studio.show_normals = False
    viewer.snapshot()
    off = viewer.snapshot()
    assert studio.normals.count == 0  # switched off
    studio.show_normals = True
    studio.select(None)
    viewer.snapshot()
    assert studio.normals.count == 0  # nothing selected
    studio.select(obj)
    studio.show_normals = False
    assert np.array_equal(viewer.snapshot(), off)  # the overlay leaves nothing behind
