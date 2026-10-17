"""Crystal studio: a rock-salt crystal with editable boundaries, radii and materials, built on valo.

Run: ``uv run python examples/crystal_studio/studio.py``
"""

from __future__ import annotations

import numpy as np
from crystal import MAX_ATOMS_PER_AXIS, Crystal, Species
from imgui_bundle import imgui

from valo import ICON, Command, Editor, Material, OrbitCamera, Viewer, mono_font


class CrystalEditor(Editor):
    """Crystal actions with undo; the shared Editor adds material widgets, History and View."""

    title = "Crystal studio"

    def __init__(self, crystal: Crystal, camera: OrbitCamera, triangles_per_atom: int = 0):
        super().__init__()
        self.crystal = crystal
        self.camera = camera
        self.auto_fit = True
        self.triangles_per_atom = triangles_per_atom
        self.selected = 0  # species index
        for m in crystal.materials:
            self.watch(m)

    @property
    def species(self) -> Species:
        return self.crystal.species[self.selected]

    def _rebuild(self) -> None:
        self.crystal.rebuild()
        if self.auto_fit:
            self.camera.fit(self.crystal.bounding_radius())

    # -- crystal actions
    def set_count(self, axis: int, value: int) -> None:
        sc, old = self.crystal, self.crystal.n[axis]
        value = int(np.clip(value, 1, MAX_ATOMS_PER_AXIS))
        if value == old:
            return

        def apply(v):
            def f():
                sc.n[axis] = v
                self._rebuild()
            return f

        apply(value)()
        self.undo_stack.push(Command(f"atoms along {'xyz'[axis]}", apply(old), apply(value), merge_key=("n", axis)))

    def set_spacing(self, value: float) -> None:
        sc, old = self.crystal, self.crystal.spacing

        def apply(v):
            def f():
                sc.spacing = v
                self._rebuild()
            return f

        apply(value)()
        self.undo_stack.push(Command("spacing", apply(old), apply(value), merge_key=("spacing",)))

    def set_radius(self, value: float) -> None:
        sp, old = self.species, self.species.radius
        self.crystal.set_radius(sp, value)
        self.undo_stack.push(Command(f"{sp.name}: radius", lambda: self.crystal.set_radius(sp, old),
                                     lambda: self.crystal.set_radius(sp, value), merge_key=("radius", id(sp))))

    # -- material actions on the selected species
    def assign(self, material: Material) -> None:
        sp, old = self.species, self.species.material
        if material is old:
            return
        sp.material = material
        self.undo_stack.push(Command(f"{sp.name}: material",
                                     lambda: setattr(sp, "material", old), lambda: setattr(sp, "material", material)))

    def duplicate(self) -> None:
        copy = self.crystal.duplicate(self.species.material)
        self.watch(copy)
        self.assign(copy)

    # -- panel
    def windows(self):
        return {"Crystal": self.crystal_window, "Species": self.species_window, "Material as data": self.data_window}

    def crystal_window(self) -> None:
        sc = self.crystal
        imgui.separator_text("Boundaries (rock-salt)")
        for axis, label in enumerate(("Atoms along x", "Atoms along y", "Atoms along z")):
            changed, v = imgui.slider_int(label, sc.n[axis], 1, MAX_ATOMS_PER_AXIS)
            if changed:
                self.set_count(axis, v)
        changed, v = imgui.slider_float("Spacing", sc.spacing, 0.5, 2.0, "%.2f")
        if changed:
            self.set_spacing(v)
        _, self.auto_fit = imgui.checkbox("Auto-fit camera to crystal", self.auto_fit)
        counts = ", ".join(f"{s.name.split()[0]}: {s.count}" for s in sc.species)
        imgui.text(f"{sc.total_atoms} atoms ({counts})")
        imgui.text_disabled(f"{len(sc.species)} instanced draw calls, {self.triangles_per_atom} triangles per atom")

    def species_window(self) -> None:
        sc = self.crystal
        for i, sp in enumerate(sc.species):
            if i:
                imgui.same_line()
            if imgui.radio_button(sp.name, self.selected == i):
                self.selected = i
        changed, r = imgui.slider_float("Radius", self.species.radius, 0.05, 1.0, "%.2f")
        if changed:
            self.set_radius(r)

        mats = sc.materials
        changed, mi = imgui.combo("Material", mats.index(self.species.material), [x.name for x in mats])
        if changed:
            self.assign(mats[mi])
        if imgui.button(f"{ICON.CLONE} Duplicate material"):
            self.duplicate()
        m = self.species.material
        users = sc.users(m)
        imgui.text_disabled("Used by: " + ", ".join(users) + ("  (shared)" if len(users) > 1 else ""))
        self.material_section(m)

    def data_window(self) -> None:
        imgui.text_disabled(f"Material of {self.species.name}")
        imgui.push_font(mono_font(), 0.0)
        imgui.text_unformatted(self.species.material.to_json())
        imgui.pop_font()


def build(viewer: Viewer) -> CrystalEditor:
    crystal = Crystal(viewer.renderer)
    editor = CrystalEditor(crystal, viewer.camera, viewer.renderer.sphere.count // 3)
    viewer.show(crystal, editor)
    return editor


def main() -> None:
    viewer = Viewer(title="Crystal studio")
    build(viewer)
    viewer.run()


if __name__ == "__main__":
    main()
