"""Crystal studio tests: the rock-salt lattice (pure), and the full app rendered offscreen (GPU)."""

from pathlib import Path

import numpy as np
import pytest
from crystal import rock_salt


@pytest.mark.parametrize(("n", "counts"), [((6, 6, 6), (108, 108)), ((9, 5, 3), (68, 67)), ((1, 1, 1), (1, 0))])
def test_rock_salt_counts_and_centering(n, counts):
    centers, kind = rock_salt(n, 1.0)
    assert (np.sum(kind == 0), np.sum(kind == 1)) == counts
    assert np.allclose(centers.mean(0), 0.0)


def test_rock_salt_neighbours_alternate_species():
    centers, kind = rock_salt((4, 1, 1), 1.0)
    assert list(kind) == [0, 1, 0, 1]
    assert np.allclose(np.diff(centers[:, 0]), 1.0)


VIEWER_SIZE = (900, 520)


@pytest.fixture(scope="module")
def studio(viewer):
    from studio import build

    return viewer, build(viewer)


@pytest.mark.gpu
def test_edits_change_the_image_and_undo_restores_it_exactly(studio):
    viewer, ed = studio
    crystal = ed.crystal
    before = viewer.snapshot()
    assert before.std() > 5, "viewport is blank"

    ed.set_count(0, 9)
    ed.set_count(1, 5)
    ed.set_count(2, 3)
    ed.set_spacing(1.4)
    assert [s.count for s in crystal.species] == [68, 67]
    ed.selected = 0
    ed.set_radius(0.22)
    ed.set_param(ed.species.material, "metallic", 1.0)
    ed.selected = 1
    ed.assign(next(m for m in crystal.materials if m.name == "Toon blue"))
    ed.set_param(ed.species.material, "bands", 4)
    ed.duplicate()
    ed.set_param(ed.species.material, "base_color", (1.0, 0.0, 0.0))
    edited = viewer.snapshot()
    assert np.abs(edited.astype(int) - before).max() > 0

    while ed.undo_stack.done:
        ed.undo_stack.undo()
    ed.selected = 0
    assert [s.count for s in crystal.species] == [108, 108]
    assert np.array_equal(viewer.snapshot(), before)


@pytest.mark.gpu
def test_hovered_atom_shows_its_species_and_lattice_site(studio):
    _, ed = studio
    sp = ed.crystal.species[1]
    info = sp.spheres.info(0)
    assert sp.spheres.label == sp.name and info["Species"] == sp.name
    i, j, k = info["Lattice site"]
    assert all(0 <= c < n for c, n in zip((i, j, k), ed.crystal.n, strict=True)) and (i + j + k) % 2 == 1  # B sites


@pytest.mark.gpu
def test_crystal_settings_rebuild_the_same_picture(studio):
    from crystal import Crystal

    import valo

    viewer, ed = studio
    ed.set_count(0, 4)
    ed.set_spacing(1.2)
    ed.selected = 1
    ed.duplicate()
    ed.set_param(ed.species.material, "base_color", (0.9, 0.1, 0.1))
    s = ed.crystal.settings()
    copy = Crystal.from_settings(s)
    assert copy.settings() == s and copy.species[1].material is not ed.crystal.species[1].material
    options = {"camera": viewer.camera.settings(), "image": {"size": [200, 150], "samples": 1}}
    assert np.array_equal(valo.render(copy, options), valo.render(ed.crystal, options))
    while ed.undo_stack.done:
        ed.undo_stack.undo()
    with pytest.raises(KeyError):
        Crystal.from_settings({"size": 3})


@pytest.mark.gpu
def test_scripting_example_writes_a_transparent_png(studio, tmp_path, monkeypatch):
    import runpy

    from PIL import Image

    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(__file__).with_name("crystal.py")), run_name="__main__")
    png = Image.open(tmp_path / "crystal.png")
    assert png.size == (2400, 1600) and png.mode == "RGBA"


@pytest.mark.gpu
def test_exported_script_renders_the_same_picture_as_the_studio(studio, tmp_path):
    import runpy

    from PIL import Image

    viewer, ed = studio
    ed.set_count(2, 3)
    ed.set_projection("orthographic")
    ed.set_light_mode("fixed")
    ed.render_options.update(size=[320, 240], samples=2, transparent=True)
    expected = np.asarray(Image.open(ed.render_image(tmp_path / "studio.png")))
    script = ed.export_script(tmp_path / "again.py")
    namespace = runpy.run_path(str(script))  # importable: SETTINGS without rendering
    assert namespace["SETTINGS"]["scene"]["n"] == [6, 6, 3] and namespace["SETTINGS"]["image"]["size"] == [320, 240]
    runpy.run_path(str(script), run_name="__main__")
    assert np.array_equal(np.asarray(Image.open(tmp_path / "again.png")), expected)
    while ed.undo_stack.done:
        ed.undo_stack.undo()
    viewer.lighting = type(viewer.lighting)()


def test_every_menu_opens_without_errors(studio):
    from valo.testing import open_every_menu

    viewer, _ = studio
    assert open_every_menu(viewer) >= 3 and viewer.gui_error is None
