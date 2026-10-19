"""The 3D scene of a SystemView in valo: atoms as instanced spheres, moments as arrows, the unit-cell edges.

Reads only the model (``SystemView`` + plot_options); never spinwaves. Positions come from ``model.expand`` in
Cartesian xyz (Å); one material per atom label, plus one each for the moment arrows and the cell edges.
"""

from __future__ import annotations

import numpy as np

from valo import Material, Renderer, instance_matrices, linear_to_srgb, meshes, renderer, srgb_to_linear

from .model import (
    SystemView,
    cell_edges,
    default_color,
    default_radius,
    expand,
    parse_color,
    to_xyz,
    validate_options,
)


def _linear(color) -> tuple:
    return tuple(srgb_to_linear(c) for c in parse_color(color))


def _srgb(m: Material) -> list:
    return [round(linear_to_srgb(c), 4) for c in m.get("base_color")]


def _vec(v, digits: int = 3) -> str:
    return "(" + ", ".join(f"{float(c) + 0.0:.{digits}f}" for c in v) + ")"


class SpinwScene:
    """Everything drawn for one system; ``options()`` returns the current state as plot_options."""

    def __init__(self, system: SystemView, options: dict | None = None, gpu: Renderer | None = None):
        self.system, self.renderer = system, gpu or renderer()
        opts = validate_options(options)
        r = self.renderer
        self.bounds = [list(b) for b in opts["boundaries"]]
        self.library: list[Material] = []  # materials the atom labels can choose from
        self.radii: dict[str, float] = {}
        self.spheres, self.arrows = {}, {}
        self.moment_material = self._material("Moments", opts["moment_material"], "pbr",
                                              base_color=_linear(opts["moment_color"]), roughness=0.5)
        self.cell_material = self._material("Unit cell", opts["cell_material"], "lambert",
                                            base_color=_linear(opts["cell_color"]))
        for i, label in enumerate(system.labels):
            if label in opts["materials"]:
                params = dict(opts["materials"][label])
                m = r.material(label, params.pop("shading"), **params)
            else:
                color = opts["atom_colors"].get(label, default_color(label, i))
                m = r.material(label, "pbr", base_color=_linear(color), roughness=0.4)
            self.library.append(m)
            self.radii[label] = float(opts["atom_radii"].get(label, default_radius(label)))
            self.spheres[label] = r.spheres(m, label=label, info=lambda k, lb=label: self.atom_info(lb, k))
            self.arrows[label] = r.instances(meshes.arrow(), self.moment_material, label=f"{label} moment",
                                             info=lambda k, lb=label: self.moment_info(lb, k))
        self.cell = r.instances(meshes.cylinder(), self.cell_material, label="unit cell")
        self.show_moments, self.moment_length = opts["show_moments"], float(opts["moment_length"])
        self.show_cell, self.cell_radius = opts["show_cell"], float(opts["cell_radius"])
        self.passthrough = {k: opts[k] for k in ("coupling_colors", "camera", "lighting", "image")}
        self._arrow_rows: dict[str, np.ndarray] = {}  # per label: arrow k → expanded row (zero moments skipped)
        self.rebuild()

    def _material(self, name: str, full: dict, shading: str, **params) -> Material:
        """``full`` (Material.to_dict()) when given, else a new material from ``shading`` and ``params``."""
        if full:
            params = dict(full)
            shading = params.pop("shading")
        return self.renderer.material(name, shading, **params)

    # -- geometry
    def rebuild(self) -> None:
        """Recompute the atoms inside the current uvw bounds and upload everything."""
        self.expanded = expand(self.system, self.bounds)
        for label in self.system.labels:
            self.upload_atoms(label)
        self.upload_moments()
        self.upload_cell()

    def upload_atoms(self, label: str) -> None:
        """Place the spheres of ``label`` at its expanded positions with its current radius."""
        self.spheres[label].set(self.expanded[label].xyz, self.radii[label])

    def upload_moments(self) -> None:
        """Arrows of length ``moment_length`` (Å) along each non-zero moment, centred on its atom."""
        L = self.moment_length
        for label, ex in self.expanded.items():
            norms = np.linalg.norm(ex.moments, axis=1)
            keep = norms > 1e-12
            self._arrow_rows[label] = np.flatnonzero(keep)  # arrow k → expanded row
            d = ex.moments[keep] / norms[keep, None] if keep.any() else np.zeros((0, 3))
            w = np.full(len(d), 0.3 * L)
            self.arrows[label].set(instance_matrices(ex.xyz[keep] - d * L / 2, np.c_[w, w, np.full(len(d), L)],
                                                     z_axis=d) if len(d) else np.zeros((0, 4, 4)))

    def upload_cell(self) -> None:
        """The 12 cell edges as cylinders of radius ``cell_radius`` (Å)."""
        e = cell_edges(self.system.A)
        mid, vec = e.mean(1), e[:, 1] - e[:, 0]
        length = np.linalg.norm(vec, axis=1)
        d = 2 * self.cell_radius
        self.cell.set(instance_matrices(mid, np.c_[np.full(12, d), np.full(12, d), length], z_axis=vec))

    @property
    def atom_count(self) -> int:
        """Number of atoms drawn."""
        return sum(len(ex.xyz) for ex in self.expanded.values())

    def batches(self):
        """Atoms, then moments and the cell edges when shown."""
        out = [s.batch() for s in self.spheres.values()]
        if self.show_moments:
            out += [a.batch() for a in self.arrows.values()]
        if self.show_cell:
            out.append(self.cell.batch())
        return out

    def center(self) -> np.ndarray:
        """Centre of the uvw bounds, in xyz (Å): where the camera aims."""
        return to_xyz(self.system.A, [np.mean(self.bounds, axis=1)])[0]

    def bounding_radius(self) -> float:
        """Radius around ``center()`` containing every atom and the cell."""
        pts = [ex.xyz for ex in self.expanded.values()] + [cell_edges(self.system.A).reshape(-1, 3)]
        pts = np.concatenate(pts)
        return float(np.linalg.norm(pts - self.center(), axis=1).max()) + max(self.radii.values(), default=0.5)

    # -- Info rows
    def atom_info(self, label: str, k: int) -> dict:
        """Info rows for drawn atom ``k`` of ``label``: cell index, uvw with translation, xyz, moment, spin."""
        ex = self.expanded[label]
        site = self.system.atoms[int(ex.indices[k])]
        m = np.asarray(site.moment, float)
        rows = {"Label": label, "Cell index": str(site.index), "uvw": _vec(ex.uvw[k]),
                "Translation": _vec(ex.translations[k], 0), "xyz (Å)": _vec(ex.xyz[k]),
                "|m|": f"{np.linalg.norm(m):.3f}", "m (xyz)": _vec(m), "S": f"{site.spin:g}"}
        rows.update(site.extra)
        return rows

    def moment_info(self, label: str, k: int) -> dict:
        """Info rows for moment arrow ``k`` of ``label``: the atom it sits on."""
        return self.atom_info(label, int(self._arrow_rows[label][k]))

    # -- as plot_options
    def options(self) -> dict:
        """The current state as plot_options (materials in full; camera / lighting / image as given)."""
        material_of = {label: s.material for label, s in self.spheres.items()}
        return {
            "boundaries": [list(map(float, b)) for b in self.bounds],
            "atom_colors": {label: _srgb(m) for label, m in material_of.items()},
            "atom_radii": dict(self.radii),
            "materials": {label: m.to_dict(digits=8) for label, m in material_of.items()},
            "show_moments": self.show_moments, "moment_length": self.moment_length,
            "moment_color": _srgb(self.moment_material), "moment_material": self.moment_material.to_dict(digits=8),
            "show_cell": self.show_cell, "cell_radius": self.cell_radius, "cell_color": _srgb(self.cell_material),
            "cell_material": self.cell_material.to_dict(digits=8),
            **{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.passthrough.items()},
        }
