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
