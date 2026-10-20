"""spinw_studio tests: the data contract (pure) and, below, the studio rendered offscreen (GPU).

Everything runs on a hand-made SystemView, so spinwaves is not needed.
"""

import numpy as np
import pytest
from spinw_studio import adapter
from spinw_studio.model import (
    AtomSite,
    SystemView,
    cell_edges,
    default_radius,
    expand,
    parse_color,
    to_xyz,
    validate_options,
)

A_LFO = np.diag([5.3, 5.6, 7.5])
FE_UVW = [(0, 0.5, 0), (0.5, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]  # Pbnm 4b, like LFO
FE_M = [(-1, 0, 0.1), (1, 0, -0.1), (1, 0, 0.1), (-1, 0, -0.1)]  # G-type, canted


def lfo_like(**kw) -> SystemView:
    atoms = tuple(AtomSite("Fe", i, u, m, spin=2.5) for i, (u, m) in enumerate(zip(FE_UVW, FE_M, strict=True)))
    return SystemView(kw.get("A", A_LFO), kw.get("atoms", atoms), "LFO-like")


def test_uvw_to_xyz_uses_the_columns_of_A():
    A = np.array([[5.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]])  # b leans toward +x
    assert np.allclose(to_xyz(A, [[0, 1, 0]]), [[1.0, 4.0, 0.0]])
    assert np.allclose(to_xyz(A, [[0.5, 0.5, 1]]), [[3.0, 2.0, 3.0]])


def test_expand_finds_boundary_atoms_and_their_translations():
    out = expand(lfo_like(), [[-0.1, 1.1]] * 3)["Fe"]
    assert len(out.xyz) == 4 + 4 + 2 + 2  # coordinates 0 also appear at 1
    assert ((out.uvw >= -0.1) & (out.uvw <= 1.1)).all()
    assert np.allclose(out.xyz, to_xyz(A_LFO, out.uvw))
    i = np.flatnonzero((out.indices == 0) & (out.translations == [1, 0, 1]).all(1))
    assert len(i) == 1 and np.allclose(out.uvw[i[0]], [1, 0.5, 1]) and np.allclose(out.moments[i[0]], FE_M[0])
    assert len(expand(lfo_like(), [[0, 2], [0, 1], [0, 1]])["Fe"].xyz) == 6 + 8 + 3 + 4  # a 2x1x1 box


def test_cell_edges_are_the_12_edges_of_the_cell():
    edges = cell_edges(A_LFO)
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    assert edges.shape == (12, 2, 3) and sorted(lengths) == pytest.approx([5.3] * 4 + [5.6] * 4 + [7.5] * 4)


def test_plot_options_are_validated_and_merged_with_defaults():
    opts = validate_options({"boundaries": ([-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.02]), "atom_colors": {"Fe": "Orange"}})
    assert opts["boundaries"][2] == [-0.1, 1.02] and opts["show_moments"] is True and opts["cell_radius"] == 0.03
    with pytest.raises(KeyError, match="did you mean 'boundaries'"):
        validate_options({"boundary": [[0, 1]] * 3})
    for bad in ({"boundaries": [[1, 0], [0, 1], [0, 1]]}, {"atom_colors": {"Fe": "Orangey"}},
                {"atom_radii": {"Fe": -1}}, {"show_cell": "yes"}):
        with pytest.raises(ValueError):
            validate_options(bad)


def test_colours_and_radii():
    assert parse_color("Orange") == pytest.approx((1.0, 165 / 255, 0.0))
    assert parse_color("#ff0000") == (1.0, 0.0, 0.0) and parse_color((255, 0, 0)) == (1.0, 0.0, 0.0)
    assert default_radius("Fe3+") == default_radius("Fe") == 0.6 and default_radius("Xx") == 0.5


def test_system_view_rejects_a_bad_lattice_matrix():
    with pytest.raises(ValueError, match="invertible 3x3"):
        lfo_like(A=np.zeros((3, 3)))


def test_adapter_stubs_say_what_to_fill_and_the_contract_is_checked(monkeypatch):
    with pytest.raises(NotImplementedError, match="fill in from spinwaves"):
        adapter.system_view(object())
    monkeypatch.setattr(adapter, "lattice_matrix", lambda sw: A_LFO)
    monkeypatch.setattr(adapter, "atom_labels", lambda sw: ["Fe"] * 4)
    monkeypatch.setattr(adapter, "atom_positions_uvw", lambda sw: np.array(FE_UVW))
    monkeypatch.setattr(adapter, "atom_moments_xyz", lambda sw: np.array(FE_M))
    view = adapter.system_view(object())
    assert view.labels == ["Fe"] and view.atoms[2].uvw == (0, 0.5, 0.5) and view.atoms[0].spin == 0.0
    monkeypatch.setattr(adapter, "atom_moments_xyz", lambda sw: np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"atom_moments_xyz must return .* shape \(4, 3\)"):
        adapter.system_view(object())


# -- the studio, offscreen (GPU) ------------------------------------------------------------------


@pytest.fixture(scope="module")
def viewer():
    pytest.importorskip("wgpu")
    from rendercanvas.offscreen import RenderCanvas

    from valo import Viewer

    try:
        return Viewer(RenderCanvas(size=(900, 560), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")


@pytest.fixture
def studio(viewer):
    from spinw_studio.editor import SpinwEditor
    from spinw_studio.scene import SpinwScene

    scene = SpinwScene(lfo_like(), {"atom_colors": {"Fe": "Orange"}})
    editor = SpinwEditor(scene)
    viewer.show(scene, editor)
    viewer.camera.target = tuple(scene.center())
    return viewer, scene, editor


def test_scene_draws_atoms_moments_and_cell(studio):
    viewer, scene, _ = studio
    assert scene.atom_count == 12 and scene.spheres["Fe"].count == 12
    assert scene.arrows["Fe"].count == 12 and scene.cell.count == 12
    assert len(scene.batches()) == 3 and viewer.snapshot().std() > 5
    m = scene.arrows["Fe"].matrices[0]
    assert np.linalg.norm(m[:3, 2]) == pytest.approx(scene.moment_length)  # arrow length along its z axis


SPINW_ACTIONS = {  # name → (action, changes the image)
    "extent": (lambda ed: ed.set_bounds(0, -0.1, 2.1), True),
    "radius": (lambda ed: ed.set_radius("Fe", 1.0), True),
    "duplicate material": (lambda ed: ed.duplicate("Fe"), False),  # an identical copy is assigned
    "material parameter": (lambda ed: ed.set_param(ed.scene.spheres["Fe"].material, "base_color", (0.1, 0.2, 0.9)),
                           True),
    "moment length": (lambda ed: ed.set_moments(length=3.0), True),
    "hide moments": (lambda ed: ed.set_moments(show=False), True),
    "cell": (lambda ed: ed.set_cell(radius=0.15), True),
    "hide cell": (lambda ed: ed.set_cell(show=False), True),
}


@pytest.mark.parametrize("name", list(SPINW_ACTIONS))
def test_studio_actions_undo_exactly(studio, name):
    viewer, scene, editor = studio
    action, visible = SPINW_ACTIONS[name]

    def state():
        return scene.options(), scene.spheres["Fe"].material

    before, image = state(), viewer.snapshot()
    action(editor)
    assert len(editor.undo_stack.done) == 1 and state() != before
    assert (np.abs(viewer.snapshot().astype(int) - image).max() > 10) == visible
    editor.undo_stack.undo()
    assert state() == before and np.array_equal(viewer.snapshot(), image)


def test_info_rows_and_compass_axes(studio):
    _, scene, editor = studio
    k = int(np.flatnonzero((scene.expanded["Fe"].indices == 1) & (scene.expanded["Fe"].translations == 0).all(1))[0])
    rows = scene.spheres["Fe"].info(k)
    assert rows["Cell index"] == "1" and rows["uvw"] == "(0.500, 0.000, 0.000)"
    assert rows["xyz (Å)"] == "(2.650, 0.000, 0.000)"
    assert rows["m (xyz)"] == "(1.000, 0.000, -0.100)" and rows["S"] == "2.5"
    assert scene.arrows["Fe"].info(k)["Cell index"] == "1"
    axes = {label: d for label, d, *_ in editor.compass_axes()}
    assert set(axes) == {"x", "y", "z", "a", "b", "c"} and axes["b"] == pytest.approx((0.0, 1.0, 0.0))


def test_plot_options_round_trip_and_export(studio, tmp_path):
    import runpy

    import spinw_studio

    viewer, scene, editor = studio
    editor.set_bounds(2, -0.1, 1.6)
    editor.set_radius("Fe", 0.45)
    editor.set_param(scene.spheres["Fe"].material, "roughness", 0.8)
    editor.render_options.update(size=[240, 160], samples=1)
    opts = editor.plot_options()
    assert opts["boundaries"][2] == [-0.1, 1.6] and opts["atom_radii"] == {"Fe": 0.45}
    assert opts["camera"]["azimuth"] == viewer.camera.azimuth and opts["image"]["size"] == [240, 160]
    shown = valo_render_of(editor)
    assert np.array_equal(spinw_studio.render(lfo_like(), opts), shown)  # the options rebuild the same picture
    namespace = runpy.run_path(str(editor.export_plot_options(tmp_path / "opts.py")))
    assert namespace["PLOT_OPTIONS"] == opts
    while editor.undo_stack.done:
        editor.undo_stack.undo()


def valo_render_of(editor):
    import valo

    return valo.render(editor.scene, editor.render_settings())


def test_plot_structure_blocks_then_returns_the_final_options(viewer, monkeypatch):
    from spinw_studio import api

    calls = []
    monkeypatch.setattr(api, "_window", lambda title: calls.append(title) or viewer)
    monkeypatch.setattr(viewer, "run", lambda: viewer.editor.set_radius("Fe", 0.9))  # "the user" edits, then closes
    opts = api.plot_structure(lfo_like(), {"atom_colors": {"Fe": "Orange"}})
    assert calls == ["SpinW studio: LFO-like"] and opts["atom_radii"] == {"Fe": 0.9}
    monkeypatch.undo()


if __name__ == '__main__':
    viewer()