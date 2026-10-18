"""The crystal model: a rock-salt lattice of atom sites, the species on them, and a material library.

GPU work is valo's: each species owns a ``valo.Spheres``; ``batches()`` hands those to the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valo import Material, Renderer, Spheres, renderer

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
    """A rock-salt crystal: lattice size and spacing, its species, and a shared material library.

    ``settings()`` / ``from_settings()`` turn it into plain data and back (for scripts and exported renders).
    """

    def __init__(self, gpu: Renderer | None = None):
        """``gpu``: the renderer that owns the GPU objects (default: the shared ``valo.renderer()``)."""
        gpu = gpu or renderer()
        self.renderer = gpu
        # Material library. Species point at entries; entries can be shared or swapped.
        new = gpu.material
        sodium = new("Sodium", "pbr", base_color=(0.85, 0.62, 0.12), metallic=0.0, roughness=0.35)
        chlorine = new("Chlorine", "blinn_phong", base_color=(0.12, 0.55, 0.12), shininess=64.0)
        self.materials: list[Material] = [
            sodium, chlorine,
            new("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0, roughness=0.3),
            new("Toon blue", "toon", base_color=(0.08, 0.25, 0.75), bands=3),
            new("Matte white", "lambert", base_color=(0.8, 0.8, 0.8)),
        ]
        self.species = [Species("A (Na)", 0.30, gpu.spheres(sodium)),
                        Species("B (Cl)", 0.42, gpu.spheres(chlorine))]
        for sp in self.species:  # what the Info window shows for a hovered atom
            sp.spheres.label = sp.name
            sp.spheres.info = lambda i, sp=sp: self.site_info(sp, i)
        self.n = [6, 6, 6]
        self.spacing = 1.0
        self.rebuild()

    def site_info(self, sp: Species, i: int) -> dict:
        """Info rows for atom ``i`` of species ``sp``: species, lattice site (i, j, k) and radius."""
        centre = sp.spheres.matrices[i][:3, 3]
        ijk = np.rint(centre / self.spacing + (np.asarray(self.n) - 1) / 2).astype(int)
        return {"Species": sp.name, "Lattice site": tuple(int(c) for c in ijk), "Radius": f"{sp.radius:.2f}"}

    # -- as plain data
    def settings(self) -> dict:
        """Everything needed to rebuild this crystal: lattice size, spacing, species and the material library."""
        return {
            "n": list(self.n), "spacing": self.spacing,
            "species": [{"name": s.name, "radius": s.radius, "material": s.material.name} for s in self.species],
            "materials": [{"name": m.name, **m.to_dict(digits=6)} for m in self.materials],
        }

    @classmethod
    def from_settings(cls, d: dict, gpu: Renderer | None = None) -> Crystal:
        """A crystal rebuilt from ``settings()`` data (keys left out keep the defaults)."""
        unknown = set(d) - {"n", "spacing", "species", "materials"}
        if unknown:
            raise KeyError(f"unknown crystal settings {sorted(unknown)}")
        c = cls(gpu)
        if "materials" in d:
            library = []
            for entry in d["materials"]:
                params = dict(entry)
                name, shading = params.pop("name"), params.pop("shading")
                library.append(c.renderer.material(name, shading, **params))
            c.materials = library
        by_name = {m.name: m for m in c.materials}
        for sp, entry in zip(c.species, d.get("species", []), strict=False):
            sp.name, sp.radius = entry.get("name", sp.name), float(entry.get("radius", sp.radius))
            sp.spheres.label = sp.name
            if "material" in entry:
                sp.material = by_name[entry["material"]]
        if c.species[0].material not in c.materials or c.species[1].material not in c.materials:
            raise KeyError("every species' material must be in the materials list")
        c.n, c.spacing = [int(v) for v in d.get("n", c.n)], float(d.get("spacing", c.spacing))
        c.rebuild()
        return c

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



if __name__ == "__main__":
    # Scripting example: build a crystal, set every option in Python, render a PNG. No window opens.
    #     uv run python examples/crystal_studio/crystal.py          (writes crystal.png in the current folder)
    # The studio's File > Export render script writes the same kind of script from what you set up there.
    import valo

    crystal = Crystal()  # on valo.renderer(), the shared GPU renderer
    crystal.n = [8, 8, 4]
    crystal.spacing = 1.1
    crystal.rebuild()
    crystal.set_radius(crystal.species[0], 0.25)
    crystal.species[1].material = next(m for m in crystal.materials if m.name == "Matte white")

    options = {
        "camera": {"azimuth": -60.0, "elevation": 30.0, "projection": "orthographic", "ortho_height": 13.0},
        "lighting": {"mode": "fixed", "azimuth": -120.0, "elevation": 45.0, "ambient": 0.35},
        "image": {"size": [2400, 1600], "samples": 2, "transparent": True},
    }
    img = valo.render(crystal, options, path="crystal.png")
    print(f"wrote crystal.png ({img.shape[1]} x {img.shape[0]})")
