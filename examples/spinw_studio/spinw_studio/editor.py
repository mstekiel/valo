"""SpinW studio's editor: the extent, atom radii and materials, moments and cell, each as an undoable action,
plus the Structure / Atoms / Moments windows that call them. The compass adds the crystal axes a, b, c."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from imgui_bundle import imgui

from valo import ICON, Command, Editor, Material

from .scene import SpinwScene


class SpinwEditor(Editor):
    """Undoable actions on a SpinwScene and the windows for them."""

    title = "SpinW studio"
    dock_bottom = ()

    def __init__(self, scene: SpinwScene):
        super().__init__()
        self.scene = scene
        self.selected = scene.system.labels[0] if scene.system.labels else ""
        for m in [*scene.library, scene.moment_material, scene.cell_material]:
            self.watch(m)

    def _change(self, label: str, get: Callable, put: Callable, value, merge_key=None) -> None:
        """Apply ``put(value)`` and record an undo step that restores ``get()``'s old value."""
        old = get()
        if old == value:
            return
        put(value)
        self.undo_stack.push(Command(label, lambda: put(old), lambda: put(value), merge_key=merge_key))

    # -- actions
    def set_bounds(self, axis: int, lo: float, hi: float) -> None:
        """Set the uvw extent along ``axis`` (0 = u, 1 = v, 2 = w) to [lo, hi] (undoable)."""
        sc = self.scene
        lo, hi = float(lo), float(hi)
        if not lo < hi:
            return

        def put(b):
            sc.bounds[axis] = list(b)
            sc.rebuild()

        self._change(f"extent {'uvw'[axis]}", lambda: list(sc.bounds[axis]), put, [lo, hi], merge_key=("bounds", axis))

    def set_radius(self, label: str, radius: float) -> None:
        """Set the display radius (Å) of every atom with ``label`` (undoable)."""
        sc = self.scene

        def put(r):
            sc.radii[label] = r
            sc.upload_atoms(label)

        self._change(f"{label}: radius", lambda: sc.radii[label], put, float(radius), merge_key=("radius", label))

    def assign(self, label: str, material: Material) -> None:
        """Give the atoms of ``label`` another material from the library (undoable)."""
        spheres = self.scene.spheres[label]
        self._change(f"{label}: material", lambda: spheres.material, lambda m: setattr(spheres, "material", m),
                     material)

    def duplicate(self, label: str) -> None:
        """Copy the material of ``label`` and assign the copy (undoable)."""
        sc = self.scene
        source = sc.spheres[label].material
        names, i = {m.name for m in sc.library}, 2
        while f"{source.name} {i}" in names:
            i += 1
        copy = source.copy(f"{source.name} {i}")
        sc.library.append(copy)
        self.watch(copy)
        self.assign(label, copy)

    def set_moments(self, show: bool | None = None, length: float | None = None) -> None:
        """Show or hide the moment arrows and/or set their length in Å (undoable)."""
        sc = self.scene
        new = (sc.show_moments if show is None else bool(show), sc.moment_length if length is None else float(length))

        def put(v):
            sc.show_moments, sc.moment_length = v
            sc.upload_moments()

        self._change("moments", lambda: (sc.show_moments, sc.moment_length), put, new, merge_key=("moments",))

    def set_cell(self, show: bool | None = None, radius: float | None = None) -> None:
        """Show or hide the unit-cell edges and/or set their radius in Å (undoable)."""
        sc = self.scene
        new = (sc.show_cell if show is None else bool(show), sc.cell_radius if radius is None else float(radius))

        def put(v):
            sc.show_cell, sc.cell_radius = v
            sc.upload_cell()

        self._change("unit cell", lambda: (sc.show_cell, sc.cell_radius), put, new, merge_key=("cell",))

    # -- compass: the crystal axes next to x, y, z
    def compass_axes(self):
        """x, y, z plus the crystal axes a, b, c (the columns of A, normalised; drawn at 70% length)."""
        A = self.scene.system.A
        abc = [(name, tuple(A[:, k] / np.linalg.norm(A[:, k])), col, 0.7) for k, (name, col) in enumerate(
            (("a", (1.0, 0.62, 0.6)), ("b", (0.7, 0.92, 0.62)), ("c", (0.62, 0.75, 1.0))))]
        return super().compass_axes() + abc

    # -- windows
    def windows(self):
        """Structure, Atoms and Moments windows."""
        return {"Structure": self.structure_window, "Atoms": self.atoms_window, "Moments": self.moments_window}

    def structure_window(self) -> None:
        """uvw extent, atom count and the unit-cell edges."""
        sc = self.scene
        imgui.separator_text("Extent (crystal coordinates)")
        for axis, name in enumerate("uvw"):
            changed, (lo, hi) = imgui.drag_float2(f"{name} range", list(sc.bounds[axis]), 0.01, -10.0, 10.0, "%.2f")
            if changed:
                self.set_bounds(axis, lo, hi)
        imgui.text(f"{sc.atom_count} atoms drawn, {len(sc.system.atoms)} per unit cell")
        imgui.separator_text("Unit cell")
        changed, show = imgui.checkbox("Show edges", sc.show_cell)
        if changed:
            self.set_cell(show=show)
        changed, r = imgui.slider_float("Edge radius (Å)", sc.cell_radius, 0.005, 0.2, "%.3f")
        if changed:
            self.set_cell(radius=r)
        a, b, c = (np.linalg.norm(self.scene.system.A[:, k]) for k in range(3))
        imgui.text_disabled(f"a = {a:.3f}  b = {b:.3f}  c = {c:.3f} Å")

    def atoms_window(self) -> None:
        """Label picker, radius, material choice and the material editor."""
        sc = self.scene
        for label in sc.system.labels:
            if imgui.radio_button(label, label == self.selected):
                self.selected = label
            imgui.same_line()
        imgui.new_line()
        if not self.selected:
            return
        label = self.selected
        changed, r = imgui.slider_float("Radius (Å)", sc.radii[label], 0.05, 2.0, "%.2f")
        if changed:
            self.set_radius(label, r)
        names = [m.name for m in sc.library]
        current = sc.spheres[label].material
        changed, i = imgui.combo("Material", sc.library.index(current), names)
        if changed:
            self.assign(label, sc.library[i])
        if imgui.button(f"{ICON.CLONE} Duplicate material"):
            self.duplicate(label)
        self.material_section(sc.spheres[label].material)

    def moments_window(self) -> None:
        """Moment arrows: show, length and their material."""
        sc = self.scene
        changed, show = imgui.checkbox("Show moments", sc.show_moments)
        if changed:
            self.set_moments(show=show)
        changed, length = imgui.slider_float("Length (Å)", sc.moment_length, 0.1, 5.0, "%.2f")
        if changed:
            self.set_moments(length=length)
        self.material_section(sc.moment_material)
