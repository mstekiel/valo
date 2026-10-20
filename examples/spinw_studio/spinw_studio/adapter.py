"""The interface between spinwaves and spinw_studio: STUBS for you to fill in by hand.

This is the only module that touches spinwaves objects. Each stub takes your ``SpinW`` object (``sw``) and returns
plain numpy data in the documented coordinate system, units, shape and dtype; ``system_view`` checks every result
and turns it into a ``model.SystemView``. Nothing else in the studio knows about spinwaves.

Conventions (see model.py): crystal coordinates uvw (fractional), Cartesian xyz in Å, r_xyz = A @ r_uvw with the
lattice vectors a, b, c as the COLUMNS of A. Moments are returned in Cartesian xyz: convert them here, from
whatever frame spinwaves stores them in.

Required stubs raise NotImplementedError until filled; optional ones return neutral defaults.
"""

from __future__ import annotations

import numpy as np

from .model import AtomSite, SystemView

# -- required stubs: fill these in ----------------------------------------------------------------


def lattice_matrix(sw) -> np.ndarray:
    """The lattice matrix A: shape (3, 3), float64, Å.

    r_xyz = A @ r_uvw. The COLUMNS of A are the lattice vectors a, b, c expressed in Cartesian xyz (Å), e.g. for
    the orthorhombic LFO cell (a, b, c = 5.3, 5.6, 7.5 Å, all angles 90°): ``np.diag([5.3, 5.6, 7.5])``.
    Use the same Cartesian convention spinwaves uses for moments and couplings.
    """
    return sw.crystal.A
    # raise NotImplementedError("fill in from spinwaves: lattice_matrix(sw) -> (3, 3) A, columns a, b, c in xyz Å")


def atom_positions_uvw(sw) -> np.ndarray:
    """Positions of ALL atoms of one unit cell (after applying the symmetry): shape (N, 3), float64, uvw.

    Crystal (fractional) coordinates, dimensionless, each in [0, 1). Row i is atom i, in the SAME ORDER as the
    indices used by ``Coupling(id1, id2)``. Example for the four Fe of LFO (Pbnm, 4b):
    ``np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])`` (order as spinwaves generates them).
    """
    return [atom.r for atom in sw.crystal.atoms_all]
    # raise NotImplementedError("fill in from spinwaves: atom_positions_uvw(sw) -> (N, 3) uvw in [0, 1)")


def atom_moments_xyz(sw) -> np.ndarray:
    """Magnetic moment of each atom: shape (N, 3), float64, Cartesian xyz, same row order as the positions.

    Convert from spinwaves' frame to Cartesian xyz here. Any length: the direction is drawn, |m| is shown in the
    Info window; a zero row means "no arrow" (non-magnetic atom). Example (G-type along x): ``[[-1, 0, 0],
    [1, 0, 0], [1, 0, 0], [-1, 0, 0]]``.
    """
    return [atom.m for atom in sw.crystal.atoms_all]
    # raise NotImplementedError("fill in from spinwaves: atom_moments_xyz(sw) -> (N, 3) moments in xyz")


def atom_labels(sw) -> list[str]:
    """Label of each atom: list of N str, same order ("Fe", "O1", ...). Selects the material and the radius."""
    return [atom.label for atom in sw.crystal.atoms_all]
    # raise NotImplementedError("fill in from spinwaves: atom_labels(sw) -> list of N labels")


# -- optional stubs: replace the defaults if spinwaves has the data ---------------------------------


def atom_spins(sw) -> np.ndarray:
    """Spin length S of each atom: shape (N,), float64. Optional: zeros by default."""
    return [atom.s for atom in sw.crystal.atoms_all]
    # return np.zeros(len(atom_labels(sw)))


def atom_extra(sw, i: int) -> dict[str, str]:
    """Extra Info rows for atom i (e.g. Wyckoff site, generating symmetry operation). Optional: none by default."""
    return {}


def system_name(sw) -> str:
    """A short name for the window title and exported files. Optional: empty by default."""
    return ""


# -- the contract check ----------------------------------------------------------------------------


def _array(stub: str, value, shape: tuple) -> np.ndarray:
    a = np.asarray(value, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"adapter.{stub} must return a finite array of shape {shape}, got shape {a.shape}")
    return a


def system_view(sw) -> SystemView:
    """Your SpinW object as a SystemView, built from the stubs above; names the stub that breaks the contract."""
    A = _array("lattice_matrix", lattice_matrix(sw), (3, 3))
    labels = [str(s) for s in atom_labels(sw)]
    n = len(labels)
    uvw = _array("atom_positions_uvw", atom_positions_uvw(sw), (n, 3))
    if ((uvw < -1e-9) | (uvw >= 1 + 1e-9)).any():
        raise ValueError("adapter.atom_positions_uvw must return fractional coordinates in [0, 1)")
    moments = _array("atom_moments_xyz", atom_moments_xyz(sw), (n, 3))
    spins = _array("atom_spins", atom_spins(sw), (n,))
    atoms = tuple(AtomSite(labels[i], i, tuple(uvw[i] % 1.0), tuple(moments[i]), float(spins[i]),
                           {str(k): str(v) for k, v in atom_extra(sw, i).items()}) for i in range(n))
    return SystemView(A, atoms, str(system_name(sw)))
