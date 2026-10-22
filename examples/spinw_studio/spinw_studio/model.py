"""The data contract between spinwaves and the studio, and the pure geometry on it (numpy only; no valo, no spinwaves).

Coordinates: crystal (fractional) **uvw** and Cartesian **xyz** in Å, with ``r_xyz = A @ r_uvw``; the columns of
``A`` are the lattice vectors a, b, c in xyz. Magnetic moments are always Cartesian xyz here.

Symmetry operations (``SymOp``) are in crystal coordinates: positions r' = R r + t (polar), moments m' = M m in the
lattice basis with M = θ·det(R)·R (axial vectors; θ = -1 with time reversal). Each atom of the cell records the
operation that placed it, and ``orientation(A, op)`` turns that into the Cartesian matrix its mesh is drawn with.

    SystemView(A, atoms=(AtomSite(...), ...), ops=(SymOp(...), ...))   ← adapter.system_view(sw)
    apply_op / moment_xyz / orientation         → a site moved by an operation, its moment, its mesh's matrix
    expand(system, bounds)                      → every atom (with lattice translations) inside a uvw box
    cell_edges(A)                               → the 12 edges of the unit cell [0, 1]³ in xyz
    lattice_params(A) / lattice_matrix(...)     ↔ (a, b, c in Å, alpha, beta, gamma in degrees)
    validate_options(plot_options)              → plot_options merged with DEFAULTS, checked
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

import numpy as np

# -- the contract ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomSite:
    """One atom of the unit cell (after symmetry), as the adapter delivers it."""

    label: str  # "Fe": selects the material and the radius
    index: int  # row in the unit cell, the same numbering as Coupling(id1, id2)
    uvw: tuple  # crystal coordinates, fractional, in [0, 1)
    moment: tuple  # magnetic moment, Cartesian xyz, any length (zero = no arrow)
    spin: float = 0.0  # spin length S
    extra: dict = field(default_factory=dict)  # extra Info rows, str → str
    op: int = 0  # the symmetry operation (index into SystemView.ops) that put the atom here from its site
    site: int | None = None  # the unique site (asymmetric unit) it comes from, if known


@dataclass(frozen=True)
class SymOp:
    """A symmetry operation in crystal coordinates: r' = R r + t for positions (uvw), m' = M m for moments in the
    lattice basis (M = θ·det(R)·R: moments are axial vectors, θ = -1 with time reversal). ``label`` names it in the
    Info window, e.g. "x+1/2,-y+1/2,-z, -1"."""

    R: np.ndarray  # (3, 3), integer-valued for a lattice operation, det ±1
    t: np.ndarray  # (3,) translation, uvw
    M: np.ndarray  # (3, 3) acting on moments in the lattice basis
    label: str = ""

    def __post_init__(self):
        R, t, M = (np.asarray(x, float) for x in (self.R, self.t, self.M))
        shapes_ok = R.shape == (3, 3) and t.shape == (3,) and M.shape == (3, 3)
        if not shapes_ok or not np.isfinite([*R.ravel(), *t, *M.ravel()]).all():
            raise ValueError(f"a SymOp needs R (3, 3), t (3,), M (3, 3), got {R.shape}, {t.shape}, {M.shape}")
        if not np.isclose(abs(np.linalg.det(R)), 1.0, atol=1e-6):
            raise ValueError(f"a symmetry operation's R must have det ±1, got det {np.linalg.det(R):.4g}")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "M", M)


IDENTITY = SymOp(np.eye(3), np.zeros(3), np.eye(3), "x,y,z")


def apply_op(op: SymOp, uvw) -> np.ndarray:
    """Crystal coordinates ``uvw`` (N, 3) moved by ``op`` and brought back into the cell: (R uvw + t) mod 1."""
    out = np.mod(np.asarray(uvw, float).reshape(-1, 3) @ op.R.T + op.t, 1.0)
    out[np.isclose(out, 1.0, rtol=0, atol=1e-9)] = 0.0
    return out


def moment_xyz(A, op: SymOp, moment) -> np.ndarray:
    """A Cartesian moment (3,) moved by ``op``: A M A⁻¹ m (the operation's moment matrix in xyz)."""
    A = np.asarray(A, float)
    return A @ op.M @ np.linalg.solve(A, np.asarray(moment, float))


def orientation(A, op: SymOp) -> np.ndarray:
    """The Cartesian linear part of ``op``, A R A⁻¹ (3, 3): how a site's mesh turns, or mirrors (det < 0), when the
    operation places a copy of it."""
    A = np.asarray(A, float)
    return A @ op.R @ np.linalg.inv(A)


@dataclass(frozen=True)
class SystemView:
    """A crystal system as plain data: the lattice matrix A, the atoms of one unit cell, and the symmetry
    operations that placed them (``AtomSite.op`` indexes ``ops``; without operations every atom has the identity)."""

    A: np.ndarray  # (3, 3) float, r_xyz = A @ r_uvw, columns a, b, c in Å
    atoms: tuple  # tuple[AtomSite, ...]
    name: str = ""
    ops: tuple = (IDENTITY,)  # tuple[SymOp, ...]

    def __post_init__(self):
        a = np.asarray(self.A, float)
        if a.shape != (3, 3) or not np.isfinite(a).all() or abs(np.linalg.det(a)) < 1e-9:
            raise ValueError(f"A must be a finite, invertible 3x3 matrix (columns a, b, c in Å), got {self.A!r}")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "ops", tuple(self.ops))
        bad = [atom.index for atom in self.atoms if not 0 <= atom.op < len(self.ops)]
        if bad:
            raise ValueError(f"atoms {bad} name a symmetry operation outside ops (there are {len(self.ops)})")

    @property
    def labels(self) -> list[str]:
        """Distinct atom labels in order of first appearance."""
        return list(dict.fromkeys(a.label for a in self.atoms))


def to_xyz(A, uvw) -> np.ndarray:
    """Cartesian positions (N, 3) in Å of crystal coordinates ``uvw`` (N, 3): r_xyz = A @ r_uvw."""
    return np.asarray(uvw, float).reshape(-1, 3) @ np.asarray(A, float).T


# -- lattice parameters ----------------------------------------------------------------------------


def lattice_params(A) -> tuple[float, float, float, float, float, float]:
    """(a, b, c, alpha, beta, gamma) of the lattice matrix A: lengths of its columns in Å, and the angles between
    b and c, a and c, a and b in degrees."""
    A = np.asarray(A, float)
    lengths = np.linalg.norm(A, axis=0)

    def angle(i, j):
        return math.degrees(math.acos(np.clip(A[:, i] @ A[:, j] / (lengths[i] * lengths[j]), -1.0, 1.0)))

    return (*map(float, lengths), angle(1, 2), angle(0, 2), angle(0, 1))


def lattice_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """The lattice matrix (columns a, b, c in xyz, Å) in the standard orientation: a along x, b in the xy plane.
    Angles in degrees; raises ValueError if they don't make a cell."""
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    cy = (ca - cb * cg) / sg
    cz2 = 1.0 - cb * cb - cy * cy
    if not (a > 0 and b > 0 and c > 0) or sg <= 0 or cz2 <= 0:
        raise ValueError(f"no unit cell has a, b, c = {a}, {b}, {c} Å and angles {alpha}, {beta}, {gamma} deg")
    return np.array([[a, b * cg, c * cb], [0.0, b * sg, c * cy], [0.0, 0.0, c * math.sqrt(cz2)]])


def with_lattice(system: SystemView, a, b, c, alpha, beta, gamma) -> SystemView:
    """``system`` with other lattice parameters, keeping each atom's uvw and the cell's orientation in space (the
    display-only path: for a SystemView there is no SpinW object to change)."""
    rotation = system.A @ np.linalg.inv(lattice_matrix(*lattice_params(system.A)))  # standard → this orientation
    return SystemView(rotation @ lattice_matrix(a, b, c, alpha, beta, gamma), system.atoms, system.name, system.ops)


# -- extent ----------------------------------------------------------------------------------------


@dataclass
class Expanded:
    """The atoms of one label inside a uvw box: one row per drawn atom."""

    xyz: np.ndarray  # (M, 3) positions, Å
    uvw: np.ndarray  # (M, 3) crystal coordinates including the lattice translation
    moments: np.ndarray  # (M, 3) Cartesian moments
    indices: np.ndarray  # (M,) unit-cell index of each atom
    translations: np.ndarray  # (M, 3) int lattice translation n (uvw = site uvw + n)
    orientations: np.ndarray  # (M, 3, 3) Cartesian linear part of each atom's symmetry operation (det -1: mirrored)


def expand(system: SystemView, bounds, tol: float = 1e-6) -> dict[str, Expanded]:
    """Every atom of ``system``, repeated by lattice translations n ∈ ℤ³, whose uvw falls inside ``bounds``
    = ((umin, umax), (vmin, vmax), (wmin, wmax)) (inclusive, with a small tolerance). Grouped by label; each row
    also carries the orientation of the operation that placed the atom (``orientation``)."""
    lo, hi = np.asarray(bounds, float).T
    turns = [orientation(system.A, op) for op in system.ops]
    rows: dict[str, list] = {label: [] for label in system.labels}
    for atom in system.atoms:
        u = np.asarray(atom.uvw, float)
        ranges = [range(math.ceil(lo[k] - u[k] - tol), math.floor(hi[k] - u[k] + tol) + 1) for k in range(3)]
        for n in np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T:
            rows[atom.label].append((u + n, atom.moment, atom.index, n, turns[atom.op]))
    out = {}
    for label, items in rows.items():
        uvw = np.array([r[0] for r in items], float).reshape(-1, 3)
        out[label] = Expanded(to_xyz(system.A, uvw), uvw, np.array([r[1] for r in items], float).reshape(-1, 3),
                              np.array([r[2] for r in items], int), np.array([r[3] for r in items], int).reshape(-1, 3),
                              np.array([r[4] for r in items], float).reshape(-1, 3, 3))
    return out


def cell_edges(A) -> np.ndarray:
    """The 12 edges of the unit cell [0, 1]³ as (12, 2, 3) start/end points in xyz (Å)."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(corners[a] - corners[b]).sum() == 1]
    return np.stack([to_xyz(A, corners[[a, b]]) for a, b in pairs])


# -- appearance defaults ---------------------------------------------------------------------------

# Display radii in Å (smaller than ionic radii, so neighbours stay apart); the element is the label's first letters.
RADII = {
    "H": 0.30, "Li": 0.45, "C": 0.45, "N": 0.45, "O": 0.50, "F": 0.45, "Na": 0.60, "Cl": 0.60, "K": 0.75,
    "Ti": 0.60, "V": 0.60, "Cr": 0.60, "Mn": 0.60, "Fe": 0.60, "Co": 0.58, "Ni": 0.56, "Cu": 0.56, "Sr": 0.75,
    "Ba": 0.80, "La": 0.80, "Ce": 0.80, "Nd": 0.78, "Gd": 0.75, "Tb": 0.75, "Dy": 0.75, "Ho": 0.74, "Er": 0.73,
    "Yb": 0.70, "Lu": 0.70, "Bi": 0.75,
}
DEFAULT_RADIUS = 0.5
ELEMENT_COLORS = {  # CSS names, chemistry-like
    "O": "Red", "F": "YellowGreen", "Cl": "LimeGreen", "N": "RoyalBlue", "C": "DimGray", "H": "White",
    "Fe": "DarkOrange", "Mn": "MediumPurple", "Co": "SteelBlue", "Ni": "MediumSeaGreen", "Cu": "Peru",
    "Cr": "LightSlateGray", "V": "Silver", "Ti": "LightGray",
}
PALETTE = ["DarkOrange", "SteelBlue", "MediumSeaGreen", "Orchid", "Goldenrod", "Tomato", "Teal", "SlateBlue"]


def element(label: str) -> str:
    """The element symbol a label starts with ("Fe2+" → "Fe", "O1" → "O"), or the label itself."""
    m = re.match(r"[A-Z][a-z]?", label)
    return m.group(0) if m else label


def default_radius(label: str) -> float:
    """Display radius in Å for an atom label (from RADII, else DEFAULT_RADIUS)."""
    return RADII.get(element(label), DEFAULT_RADIUS)


def default_color(label: str, i: int) -> str:
    """Default colour name for the i-th label: the element's colour, else a palette entry."""
    return ELEMENT_COLORS.get(element(label), PALETTE[i % len(PALETTE)])


def parse_color(color) -> tuple[float, float, float]:
    """An sRGB colour (0-1 floats) from a CSS name ("Orange"), hex ("#ff8800") or an (r, g, b) tuple (0-1 or 0-255)."""
    if isinstance(color, str):
        from PIL import ImageColor  # pillow is a valo dependency; imported here only

        try:
            r, g, b = ImageColor.getrgb(color)[:3]
        except ValueError:
            raise ValueError(f"unknown colour {color!r}: use a CSS name, '#rrggbb' or (r, g, b)") from None
        return r / 255, g / 255, b / 255
    c = tuple(float(v) for v in color)
    if len(c) != 3 or min(c) < 0:
        raise ValueError(f"a colour tuple needs 3 non-negative numbers, got {color!r}")
    return tuple(v / 255 for v in c) if max(c) > 1 else c


# -- plot_options ----------------------------------------------------------------------------------

DEFAULTS = {
    "boundaries": [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]],  # uvw extent
    "atom_colors": {},  # {label: colour}; missing labels get default_color
    "atom_radii": {},  # {label: Å}; missing labels get default_radius
    "materials": {},  # {label: Material.to_dict()}: full material, overrides atom_colors
    "show_moments": True,
    "moment_length": 2.0,  # Å, arrow length, centred on the atom (must exceed the atom diameter to show)
    "moment_color": "Crimson",
    "moment_material": {},  # full Material.to_dict() for the arrows (overrides moment_color); written on export
    "show_cell": True,
    "cell_radius": 0.03,  # Å, edge cylinder radius
    "cell_color": "Gray",
    "cell_material": {},  # full Material.to_dict() for the edges (overrides cell_color); written on export
    "shapes": {},  # {structure object: {"func": "module:function" or sculpt file, "params": {...}}}: swapped meshes
    "objects": [],  # valo node settings of the objects added in the studio
    "object_materials": {},  # {name: Material.to_dict()}: the other materials of the scene's library
    "coupling_colors": {},  # {coupling label: colour}; kept for when couplings are drawn
    "camera": {},  # valo Camera.settings() keys
    "lighting": {},  # valo Lighting.settings() keys
    "image": {},  # valo.render image keys (size, samples, transparent, background)
}


def unknown_key(name: str, known, where: str) -> str:
    """Error text for an unknown key, with a "did you mean" hint."""
    hint = difflib.get_close_matches(name, list(known), n=1, cutoff=0.5)
    return f"unknown key {name!r} in {where}" + (f" (did you mean {hint[0]!r}?)" if hint else "")


def validate_options(options: dict | None) -> dict:
    """``options`` checked and merged over DEFAULTS (a new dict; the input is not changed)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for key, value in (options or {}).items():
        if key not in DEFAULTS:
            raise KeyError(unknown_key(key, DEFAULTS, "plot_options"))
        out[key] = dict(value) if isinstance(value, dict) else value
    b = np.asarray(out["boundaries"], float)
    if b.shape != (3, 2) or not (b[:, 0] < b[:, 1]).all():
        raise ValueError(f"boundaries must be three [min, max] uvw ranges with min < max, got {out['boundaries']!r}")
    out["boundaries"] = b.tolist()
    for key in ("atom_colors", "coupling_colors"):
        for color in out[key].values():
            parse_color(color)  # raises with a clear message
    for label, r in out["atom_radii"].items():
        if not float(r) > 0:
            raise ValueError(f"atom_radii[{label!r}] must be > 0 Å, got {r!r}")
    for key in ("moment_length", "cell_radius"):
        if not float(out[key]) > 0:
            raise ValueError(f"{key} must be > 0, got {out[key]!r}")
    for key in ("moment_color", "cell_color"):
        parse_color(out[key])
    for key in ("show_moments", "show_cell"):
        if not isinstance(out[key], bool):
            raise ValueError(f"{key} must be True or False, got {out[key]!r}")
    return out
