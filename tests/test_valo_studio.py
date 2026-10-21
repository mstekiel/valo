"""valo studio rendered offscreen: shape files in the Code window, re-read on save, New / Open shape, Save to library.
Needs a GPU adapter."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from valo.testing import open_every_menu

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (1280, 720)  # room for the Code column
VASE = Path(__file__).parents[1] / "examples" / "sculpts" / "vase.py"


@pytest.fixture
def studio(viewer, tmp_path, tmp_library):
    from valo.studio import build

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
    with rc_context({"studio.poll_s": 0.05}):
        studio.poll()
    assert "height: float = 2.0" in studio.code and not studio.dirty
    for bad in ("1abc", "_hidden", "has space"):
        with pytest.raises(ValueError):
            studio.new_shape(bad)
    studio.new_shape("dup")
    with pytest.raises(FileExistsError):
        studio.new_shape("dup")


def test_every_menu_and_window_draws_without_errors(viewer, studio):

    blob(studio)
    studio.reset_layout()
    assert open_every_menu(viewer) >= 3 and viewer.gui_error is None
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


class FakeFileDialog:
    """Stands in for the system file dialog: not ready on the first poll, then returns ``picked``."""

    def __init__(self, picked, calls):
        self.picked, self.polls = picked, 0
        calls.append(self)

    def ready(self, timeout=20):  # noqa: D102 - the dialog interface
        self.polls += 1
        return self.polls > 1

    def result(self):  # noqa: D102
        return self.picked


@pytest.mark.parametrize("picked", ["vase", None, "missing"])
def test_open_shape_file_uses_the_system_file_dialog(viewer, studio, tmp_path, monkeypatch, picked):
    calls = []
    path = {"vase": tmp_path / "vase.py", "missing": tmp_path / "missing.py", None: None}[picked]
    monkeypatch.setattr(studio, "file_dialog", lambda title, folder, filters: FakeFileDialog(
        [str(path)] if path else [], calls))
    studio.browse_shape()
    studio.browse_shape()  # a second click while it's open doesn't open another
    studio._poll_browsing()  # still open: nothing happens yet
    assert len(calls) == 1 and studio.scene.objects() == [] and studio._browsing is not None
    viewer.snapshot()  # the next frame sees the choice
    names = [o.name for o in studio.scene.objects()]
    if picked == "vase":
        assert names == ["Vase"] and studio.code_path == path
    else:
        assert names == [] and studio._browsing is None
        assert ("could not open missing.py" in studio.status) == (picked == "missing")


def test_without_a_system_dialog_a_path_field_opens(studio, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("no dialog backend")

    monkeypatch.setattr(studio, "file_dialog", unavailable)
    studio.browse_shape()
    assert studio._dialog == ("open", "")


BROKEN = """
MATERIAL = {"shading": "lambert", "base_color": [0.2, 0.4, 0.9]}


def mesh(size: float = 1.0):
    from valo import meshes

    return meshes.cube(points)
"""


def test_a_broken_shape_file_opens_for_fixing(viewer, studio, tmp_path):
    path = tmp_path / "broken.py"
    path.write_text(BROKEN, encoding="utf-8")
    image = viewer.snapshot()
    assert studio.open_shape(path) is None
    assert studio.scene.objects() == [] and studio.scene.materials == {} and not studio.undo_stack.done
    assert studio.code_path == path and "NameError" in studio.code_error and "broken.py" in studio.code_error
    assert np.array_equal(viewer.snapshot(), image) and viewer.gui_error is None  # the studio keeps drawing
    studio.set_code(studio.code.replace("cube(points)", "cube(size)"))
    studio.save_code()
    assert [o.name for o in studio.scene.objects()] == ["Broken"] and studio.code_error is None
    assert len(studio.undo_stack.done) == 1  # the save and the object it made: one step
    studio.undo_stack.undo()
    assert studio.scene.objects() == [] and "cube(points)" in path.read_text(encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        studio.open_shape(tmp_path / "nothing.py")


def test_a_scene_built_by_actions_saves_and_rebuilds_the_same_picture(viewer, studio, tmp_path):
    """End to end, through the editor only: what the refactor must keep working."""
    from valo import Group, Scene, render

    blob(studio)  # its file and object exist before the history starts
    start = studio.scene.settings()
    obj = studio.scene.find("Blob")
    studio.set_shape_param(obj, "waist", 0.2)
    studio.set_transform(obj, "rotate_deg", (20, 0, 30))
    pair = studio.add(Group("Pair"))
    studio.add_shape("valo.meshes.primitives:torus", pair, material=studio.new_material("Gold", "pbr").name)
    studio.add_shape("valo.meshes.primitives:arrow", pair)
    studio.set_transform(pair, "translate", (1.5, 0, 0))
    studio.set_object_material(obj, "Gold")
    studio.set_param(studio.scene.materials["Gold"], "roughness", 0.2)
    studio.set_visible(studio.scene.find("Arrow"), False)
    settings = studio.scene.settings()
    again = Scene.from_settings(settings)
    assert again.settings() == settings
    cam = viewer.camera
    assert np.array_equal(render(again, size=(160, 120), camera=cam), render(studio.scene, size=(160, 120), camera=cam))
    while studio.undo_stack.done:  # and every step undoes, back to where the history started
        studio.undo_stack.undo()
    assert studio.scene.settings() == start
