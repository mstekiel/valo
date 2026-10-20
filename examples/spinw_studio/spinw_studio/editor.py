"""SpinW studio's editor: the valo studio (Scene / Object / Materials / Code windows, undo for everything) plus
Export plot options and a compass with the crystal axes a, b, c. The Structure group's Object window is the
parameters tab: lattice, extent and moment length. Shapes of your own (File > New shape) can sit next to it."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from valo import ICON, Scene, StudioEditor, plain, py_literal

from .scene import STRUCTURE, options, source_of


class SpinwEditor(StudioEditor):
    """The studio for a spinw_studio scene; starts with the Structure (its parameters) selected."""

    title = "SpinW studio"

    def __init__(self, scene: Scene):
        super().__init__(scene)
        self.selected = scene.find(STRUCTURE)
        self._focus.append("Object")

    @property
    def system(self):
        """The SystemView shown now (follows lattice edits)."""
        return source_of(self.scene).system

    def can_export(self) -> bool:
        """No render script: the structure needs your SpinW object. Use File > Export plot options instead."""
        return False

    # -- plot_options out (round trip to the script)
    def plot_options(self) -> dict:
        """Everything shown, as plot_options: the scene's state plus the current camera, lighting and image options."""
        opts = options(self.scene)
        if self.viewer is not None:
            opts.update(camera=self.viewer.camera.settings(), lighting=self.viewer.lighting.settings(),
                        image=dict(self.render_options))
        return plain(opts)

    def export_plot_options(self, path=None) -> Path:
        """Write ``PLOT_OPTIONS = {...}`` (and how to use it) to a .py file; returns the path."""
        name = self.system.name or "system"
        slug = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower() or "system"
        path = Path(path) if path else Path.cwd() / f"plot_options_{slug}_{time.strftime('%Y%m%d-%H%M%S')}.py"
        path.write_text(
            f'''"""plot_options exported from SpinW studio ({name}) on {time.strftime("%Y-%m-%d %H:%M")}.

Use them in your script:
    plot_options = plot_structure(sw, plot_options=PLOT_OPTIONS)      # the studio, as you left it
    spinw_studio.render(sw, PLOT_OPTIONS, path="structure.png")        # the same picture, no window
Keys: see spinw_studio.model.DEFAULTS (colours are sRGB, lengths in Å, boundaries in crystal coordinates).
The lattice is not here: it lives in your SpinW object (the studio's lattice edits change it).
"""

PLOT_OPTIONS = {py_literal(self.plot_options())}
''', encoding="utf-8")
        self.render_status = f"exported {path}"
        return path

    def file_menu(self) -> None:
        """The studio's File items plus Export plot options (valo adds Render below them)."""
        if imgui.menu_item_simple(f"{ICON.FILE_EXPORT} Export plot options..."):
            self.export_plot_options()
        imgui.separator()
        super().file_menu()

    # -- compass: the crystal axes next to x, y, z
    def compass_axes(self):
        """x, y, z plus the crystal axes a, b, c (the columns of the current A, normalised; drawn at 70% length)."""
        A = self.system.A
        abc = [(name, tuple(A[:, k] / np.linalg.norm(A[:, k])), col, 0.7) for k, (name, col) in enumerate(
            (("a", (1.0, 0.62, 0.6)), ("b", (0.7, 0.92, 0.62)), ("c", (0.62, 0.75, 1.0))))]
        return super().compass_axes() + abc
