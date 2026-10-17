"""The crystal model: a rock-salt lattice of atom sites, the species on them, and a material library.

GPU work is valo's: each species owns a ``valo.Spheres``; ``batches()`` hands those to the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valo import Material, Renderer, Spheres

MAX_ATOMS_PER_AXIS = 24


def rock_salt(n: tuple[int, int, int], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Centres of an nx × ny × nz grid (centred on the origin) and the species index of each site.

    Species alternate like a 3D chessboard: (i + j + k) % 2 → 0 for A, 1 for B.
    """
    ijk = np.stack(np.meshgrid(*(np.arange(k) for k in n), indexing="ij"), -1).reshape(-1, 3)
    centers = (ijk - (np.asarray(n) - 1) / 2) * spacing
    return centers.astype(np.float32), ijk.sum(1) % 2


@dataclass
class Species:
    """One atom type: a name, a radius, and the instanced spheres of its atoms (which carry the material)."""

    name: str
    radius: float
    spheres: Spheres

    @property
    def material(self) -> Material:
        """Material of this species' atoms."""
        return self.spheres.material

    @material.setter
    def material(self, m: Material) -> None:
        """Change the material of this species' atoms."""
        self.spheres.material = m

    @property
    def count(self) -> int:
        """Number of atoms of this species."""
        return self.spheres.count


class Crystal:
    """A rock-salt crystal: lattice size and spacing, its species, and a shared material library."""
    def __init__(self, renderer: Renderer):
        # Material library. Species point at entries; entries can be shared or swapped.
        new = renderer.material
        sodium = new("Sodium", "pbr", base_color=(0.85, 0.62, 0.12), metallic=0.0, roughness=0.35)
        chlorine = new("Chlorine", "blinn_phong", base_color=(0.12, 0.55, 0.12), shininess=64.0)
        self.materials: list[Material] = [
            sodium, chlorine,
            new("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0, roughness=0.3),
            new("Toon blue", "toon", base_color=(0.08, 0.25, 0.75), bands=3),
            new("Matte white", "lambert", base_color=(0.8, 0.8, 0.8)),
        ]
        self.species = [Species("A (Na)", 0.30, renderer.spheres(sodium)),
                        Species("B (Cl)", 0.42, renderer.spheres(chlorine))]
        self.n = [6, 6, 6]
        self.spacing = 1.0
        self.rebuild()

    # -- materials
    def duplicate(self, m: Material) -> Material:
        """Copy ``m`` under a new unique name and add it to the library."""
        names, i = {x.name for x in self.materials}, 2
        while f"{m.name} {i}" in names:
            i += 1
        copy = m.copy(f"{m.name} {i}")
        self.materials.append(copy)
        return copy

    def users(self, m: Material) -> list[str]:
        """Names of the species that use material ``m``."""
        return [s.name for s in self.species if s.material is m]

    # -- structure
    @property
    def total_atoms(self) -> int:
        """Number of atoms of all species."""
        return sum(s.count for s in self.species)

    def bounding_radius(self) -> float:
        """Radius of a sphere around the origin containing every atom."""
        corner = float(np.linalg.norm(np.asarray(self.n) - 1)) * self.spacing / 2
        return corner + max(s.radius for s in self.species)

    def rebuild(self) -> None:
        """Recompute atom centres for the current boundaries and upload them per species."""
        self._centers, self._kind = rock_salt(tuple(self.n), self.spacing)
        for sp in self.species:
            self._upload(sp)

    def set_radius(self, sp: Species, r: float) -> None:
        """Set the atom radius of species ``sp`` and re-upload it."""
        sp.radius = float(r)
        self._upload(sp)

    def _upload(self, sp: Species) -> None:
        sp.spheres.set(self._centers[self._kind == self.species.index(sp)], sp.radius)

    # -- what the renderer draws
    def batches(self):
        """One instanced batch per species."""
        return [sp.spheres.batch() for sp in self.species]
