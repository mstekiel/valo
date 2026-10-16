"""The editor base class: undo history, the docked Dear ImGui panel, and schema-driven material widgets.

An app subclasses ``Editor``: every user action is a method that pushes a ``Command`` for undo, and
``panel()`` draws the app's widgets, which only call those methods. Scripts and tests call the same
methods, so the panel is never the only way in.
"""

from __future__ import annotations

from imgui_bundle import imgui

from .materials import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .undo import Command, UndoStack

PANEL_W = 400  # logical pixels


class Editor:
    title = "valo"

    def __init__(self):
        self.undo_stack = UndoStack()
        self.docked = True
        self.auto_rotate = False

    # -- material actions (material edits reach the undo stack through watch())
    def watch(self, m: Material) -> None:
        """Record every edit of ``m`` on the undo stack."""
        m.listeners.append(self._on_material_event)

    def _on_material_event(self, m, key, old, new) -> None:
        self.undo_stack.push(Command(f"{m.name}: {key}", lambda: m.set(key, old), lambda: m.set(key, new),
                                     merge_key=(id(m), key)))

    def set_param(self, m: Material, key, value) -> None:
        m.set(key, value)

    def reset(self, m: Material) -> None:
        for s in SCHEMA[m.shading]:
            self.set_param(m, s.key, s.default)

    # -- panel
    def panel(self) -> None:
        """The app's own widgets, between the dock checkbox and the shared View/History sections."""

    def material_section(self, m: Material) -> None:
        changed, si = imgui.combo("Shading", SHADINGS.index(m.shading), SHADINGS)
        if changed:
            self.set_param(m, "shading", SHADINGS[si])

        imgui.separator_text(f"Material parameters ({m.shading})")
        for s in SCHEMA[m.shading]:  # the schema decides the widgets
            if s.kind == "color":
                srgb = [linear_to_srgb(c) for c in m.get(s.key)]
                changed, col = imgui.color_edit3(s.label, srgb, imgui.ColorEditFlags_.picker_hue_wheel)
                if changed:
                    self.set_param(m, s.key, [srgb_to_linear(c) for c in col])
            elif s.kind == "float":
                fl = imgui.SliderFlags_.logarithmic if s.log else 0
                changed, v = imgui.slider_float(s.label, m.get(s.key), s.lo, s.hi, "%.0f" if s.log else "%.3f", fl)
                if changed:
                    self.set_param(m, s.key, v)
            else:
                changed, v = imgui.slider_int(s.label, int(m.get(s.key)), int(s.lo), int(s.hi))
                if changed:
                    self.set_param(m, s.key, v)
        if imgui.button("Reset material parameters"):
            self.reset(m)

    def gui(self) -> None:
        """Called by the Viewer's ImguiRenderer every frame."""
        io = imgui.get_io()
        if io.key_ctrl and not io.want_text_input:
            if imgui.is_key_pressed(imgui.Key.z):
                self.undo_stack.undo()
            if imgui.is_key_pressed(imgui.Key.y):
                self.undo_stack.redo()

        flags = 0
        if self.docked:
            imgui.set_next_window_pos(imgui.ImVec2(io.display_size.x - PANEL_W, 0), imgui.Cond_.always)
            imgui.set_next_window_size(imgui.ImVec2(PANEL_W, io.display_size.y), imgui.Cond_.always)
            flags = imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
        else:
            imgui.set_next_window_size(imgui.ImVec2(PANEL_W, 700), imgui.Cond_.first_use_ever)
        imgui.begin(self.title, None, flags)
        _, self.docked = imgui.checkbox("Dock panel to the right", self.docked)

        self.panel()

        imgui.separator_text("History")
        imgui.begin_disabled(not self.undo_stack.done)
        if imgui.button("Undo"):
            self.undo_stack.undo()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not self.undo_stack.undone)
        if imgui.button("Redo"):
            self.undo_stack.redo()
        imgui.end_disabled()
        imgui.same_line()
        last = self.undo_stack.done[-1].label if self.undo_stack.done else "-"
        imgui.text_disabled(f"last: {last}")

        imgui.separator_text("View")
        _, self.auto_rotate = imgui.checkbox("Auto-rotate camera", self.auto_rotate)
        imgui.text_disabled("Left-drag: orbit   Wheel: zoom   Ctrl+Z/Y")
        imgui.end()
