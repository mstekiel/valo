"""The interface between spinwaves and spinw_studio: STUBS for you to fill in by hand.

This is the only module that touches spinwaves objects. Each stub takes your ``SpinW`` object (``sw``) and returns
plain numpy data in the documented coordinate system, units, shape and dtype; ``system_view`` checks every result
and turns it into a ``model.SystemView``. Nothing else in the studio knows about spinwaves.

How the cell is built: the UNIQUE atoms (the sites you define in your script) are copied by the SYMMETRY OPERATIONS
of the magnetic space group. Each operation {R|t}·θ acts differently on
  - positions (polar vectors):  r' = R r + t                  → ``atom_operation`` (4 x 4, uvw)
  - moments (axial vectors):    m' = θ · det(R) · R m         → ``moment_operation`` (3 x 3, lattice basis)
with θ = -1 for operations with time reversal. ``atom_generators`` says which site and which operation made each
atom of the cell, in spinwaves' order (the numbering of ``Coupling(id1, id2)``). The studio then draws every atom's
mesh with its operation, so a mirror or an inversion mirrors the mesh too.

Conventions (see model.py): crystal coordinates uvw (fractional), Cartesian xyz in Å, r_xyz = A @ r_uvw with the
lattice vectors a, b, c as the COLUMNS of A. Operations are in crystal coordinates (the studio converts them with
A, so lattice edits stay consistent). Site moments are Cartesian xyz: convert them here from spinwaves' frame.

Required stubs raise NotImplementedError until filled; optional ones return neutral defaults. ``set_lattice`` is
the one stub that changes ``sw``: the studio calls it when you edit the lattice parameters.
"""

from __future__ import annotations

import numpy as np

from .model import AtomSite, SymOp, SystemView, apply_op, moment_xyz

# -- required stubs: fill these in ----------------------------------------------------------------


def lattice_matrix(sw) -> np.ndarray:
    """The lattice matrix A: shape (3, 3), float64, Å.

    r_xyz = A @ r_uvw. The COLUMNS of A are the lattice vectors a, b, c expressed in Cartesian xyz (Å), e.g. for
    the orthorhombic LFO cell (a, b, c = 5.3, 5.6, 7.5 Å, all angles 90°): ``np.diag([5.3, 5.6, 7.5])``.
    Use the same Cartesian convention spinwaves uses for moments and couplings.
    """
    return sw.crystal.A


def site_positions_uvw(sw) -> np.ndarray:
    """The UNIQUE atoms (the sites of your script, before symmetry): shape (S, 3), float64, uvw.

    Crystal (fractional) coordinates, dimensionless. Example for LFO's four sites (Lu, Fe, O1, O2):
    ``[[0.0221, 0.9284, 0.75], [0, 0.5, 0], [0.1296, 0.4475, 0.25], [0.188, 0.1915, 0.5659]]``.
    """
    raise NotImplementedError("fill in from spinwaves: site_positions_uvw(sw) -> (S, 3) uvw of the unique atoms")


def site_moments_xyz(sw) -> np.ndarray:
    """The magnetic moment of each unique atom: shape (S, 3), float64, Cartesian xyz, same order as the sites.

    Convert from spinwaves' frame to Cartesian xyz here. Any length; a zero row means a non-magnetic site. Example
    for LFO: Fe ``[-1, 0, Fz]``, the others ``[0, 0, 0]``.
    """
    raise NotImplementedError("fill in from spinwaves: site_moments_xyz(sw) -> (S, 3) site moments in xyz")


def site_labels(sw) -> list[str]:
    """Label of each unique atom: list of S str, same order ("Lu", "Fe", "O1", "O2"). Selects material and radius."""
    raise NotImplementedError("fill in from spinwaves: site_labels(sw) -> list of S labels")


def symmetry_operations(sw) -> list:
    """The symmetry operations of the magnetic space group, as spinwaves stores them: a list of G objects (any
    type). The studio passes each one back to ``atom_operation``, ``moment_operation`` and ``operation_label``, and
    numbers them in this order (the op index of ``atom_generators``)."""
    raise NotImplementedError("fill in from spinwaves: symmetry_operations(sw) -> list of the group's operations")


def atom_operation(sw, op) -> np.ndarray:
    """How operation ``op`` moves a POSITION: the affine matrix (4, 4), float64, in crystal coordinates uvw.

    ``[[R, t], [0, 0, 0, 1]]`` with r' = R r + t, R the (3, 3) rotation / mirror part (integers for a lattice
    operation, det ±1) and t the translation (uvw). Time reversal doesn't act on positions. Example for
    "x+1/2,-y+1/2,-z, -1": R = diag(1, -1, -1), t = (1/2, 1/2, 0).
    """
    raise NotImplementedError("fill in from spinwaves: atom_operation(sw, op) -> (4, 4) [[R, t], [0, 0, 0, 1]] in uvw")


def moment_operation(sw, op) -> np.ndarray:
    """How operation ``op`` changes a MOMENT: matrix (3, 3), float64, acting on moment components along the lattice
    vectors a, b, c (the same basis as R).

    Moments are axial vectors: M = θ · det(R) · R, with θ = -1 if the operation includes time reversal, else +1.
    So an inversion (R = -1) leaves a moment unchanged, and time reversal flips it. Example for
    "x+1/2,-y+1/2,-z, -1": θ = -1, det(R) = 1, M = -diag(1, -1, -1) = diag(-1, 1, 1).
    """
    raise NotImplementedError("fill in from spinwaves: moment_operation(sw, op) -> (3, 3) θ·det(R)·R")


def atom_generators(sw) -> np.ndarray:
    """Where each atom of the unit cell comes from: shape (N, 2), int, row i = [site index, operation index].

    One row per atom of ONE unit cell, in spinwaves' order: row i is the atom that ``Coupling(id1, id2)`` calls i.
    The site index points into ``site_positions_uvw``, the operation index into ``symmetry_operations``; the atom
    sits at ``(R r_site + t) mod 1``. Example for LFO's 4 Fe (site 1): ``[[1, 0], [1, 1], [1, 2], [1, 3]]``.
    """
    raise NotImplementedError("fill in from spinwaves: atom_generators(sw) -> (N, 2) [site, operation] per cell atom")


# -- optional stubs: replace the defaults if spinwaves has the data ---------------------------------


def site_spins(sw) -> np.ndarray:
    """Spin length S of each unique atom: shape (S,), float64. Optional: zeros by default."""
    return np.zeros(len(site_labels(sw)))


def operation_label(sw, op) -> str:
    """A short name for ``op`` in the Info window, e.g. "x+1/2,-y+1/2,-z, -1". Optional: empty by default."""
    return ""


def atom_positions_uvw(sw) -> np.ndarray | None:
    """OPTIONAL CHECK: spinwaves' own positions of the cell atoms, (N, 3) uvw in the same order as
    ``atom_generators``. When given, ``system_view`` compares the cell it builds from sites and operations with
    these, and names the stub to fix on a mismatch. Return None to skip the check."""
    return [atom.r for atom in sw.crystal.atoms_all]


def atom_extra(sw, i: int) -> dict[str, str]:
    """Extra Info rows for cell atom i (e.g. Wyckoff site). Optional: none by default."""
    return {}


def system_name(sw) -> str:
    """A short name for the window title and exported files. Optional: empty by default."""
    return ""


# -- changing sw: the lattice parameters edited in the studio --------------------------------------------


def set_lattice(sw, a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> None:
    """Give ``sw`` new lattice parameters: a, b, c in Å, alpha, beta, gamma in degrees (alpha between b and c).

    Change ``sw`` IN PLACE so that afterwards ``lattice_matrix(sw)`` returns the new A, with every atom keeping its
    crystal coordinates uvw. Called by the studio's Structure parameters; until filled, editing them shows this
    error and changes nothing.
    """
    sw.crystal.a = a
    sw.crystal.b = b
    sw.crystal.c = c

    sw.crystal.alpha = alpha
    sw.crystal.beta = beta
    sw.crystal.gamma = gamma


# -- the contract check ----------------------------------------------------------------------------


def _array(stub: str, value, shape: tuple) -> np.ndarray:
    a = np.asarray(value, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"adapter.{stub} must return a finite array of shape {shape}, got shape {a.shape}")
    return a


def _operation(sw, i: int, op) -> SymOp:
    """Operation ``op`` (number ``i``) as a SymOp, checked; errors name the stub."""
    affine = _array("atom_operation", atom_operation(sw, op), (4, 4))
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError(f"adapter.atom_operation must return [[R, t], [0, 0, 0, 1]]; operation {i}'s last row is "
                         f"{affine[3].tolist()}")
    M = _array("moment_operation", moment_operation(sw, op), (3, 3))
    try:
        return SymOp(affine[:3, :3], affine[:3, 3], M, str(operation_label(sw, op)))
    except ValueError as exc:
        raise ValueError(f"adapter.atom_operation, operation {i}: {exc}") from None


def system_view(sw) -> SystemView:
    """Your SpinW object as a SystemView: the cell built from the sites and operations above, checked; errors name
    the stub that breaks the contract."""
    A = _array("lattice_matrix", lattice_matrix(sw), (3, 3))
    labels = [str(s) for s in site_labels(sw)]
    n_sites = len(labels)
    sites = _array("site_positions_uvw", site_positions_uvw(sw), (n_sites, 3))
    moments = _array("site_moments_xyz", site_moments_xyz(sw), (n_sites, 3))
    spins = _array("site_spins", site_spins(sw), (n_sites,))
    ops = tuple(_operation(sw, i, op) for i, op in enumerate(symmetry_operations(sw)))
    if not ops:
        raise ValueError("adapter.symmetry_operations returned no operations (the identity at least)")
    gen = np.asarray(atom_generators(sw))
    if gen.ndim != 2 or gen.shape[1] != 2 or not np.issubdtype(gen.dtype, np.integer):
        raise ValueError(f"adapter.atom_generators must return (N, 2) integers [site, operation], got {gen!r}")
    if ((gen[:, 0] < 0) | (gen[:, 0] >= n_sites) | (gen[:, 1] < 0) | (gen[:, 1] >= len(ops))).any():
        raise ValueError(f"adapter.atom_generators has a site index outside 0..{n_sites - 1} or an operation index "
                         f"outside 0..{len(ops) - 1}")
    uvw = np.vstack([apply_op(ops[g], sites[s]) for s, g in gen]) if len(gen) else np.zeros((0, 3))
    for i in range(len(uvw)):  # two atoms on one spot: the generators repeat a (site, equivalent operation)
        d = (uvw[:i] - uvw[i] + 0.5) % 1.0 - 0.5
        same = np.flatnonzero(np.abs(d).max(axis=1) < 1e-4) if i else []
        if len(same):
            raise ValueError(f"adapter.atom_generators: cell atoms {int(same[0])} and {i} land on the same position "
                             f"{np.round(uvw[i], 4).tolist()}")
    expected = atom_positions_uvw(sw)
    if expected is not None:
        expected = _array("atom_positions_uvw", expected, (len(gen), 3))
        d = (uvw - expected + 0.5) % 1.0 - 0.5
        wrong = np.flatnonzero(np.abs(d).max(axis=1) > 1e-4)
        if len(wrong):
            i = int(wrong[0])
            s, g = (int(x) for x in gen[i])
            raise ValueError(f"cell atom {i} (site {s}, operation {g}) is built at {np.round(uvw[i], 4).tolist()}, "
                             f"but spinwaves has it at {np.round(expected[i], 4).tolist()}: check "
                             "adapter.atom_operation and adapter.atom_generators")
    atoms = tuple(AtomSite(labels[s], i, tuple(uvw[i]), tuple(moment_xyz(A, ops[g], moments[s])), float(spins[s]),
                           {str(k): str(v) for k, v in atom_extra(sw, i).items()}, op=int(g), site=int(s))
                  for i, (s, g) in enumerate(gen))
    return SystemView(A, atoms, str(system_name(sw)), ops)
