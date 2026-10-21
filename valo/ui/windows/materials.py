"""The Materials window: the scene's material library, with New / Duplicate / Delete and the material editor."""

from __future__ import annotations

from imgui_bundle import imgui

from ..style import ICON


class MaterialsWindow:
    """SceneEditor part: the Materials window."""

    def materials_window(self) -> None:
        """The material library: select one to edit it; New, Duplicate, and Delete for unused materials."""
        mats = self.scene.materials
        if self.material_selected not in mats:
            self.material_selected = next(iter(mats), None)
        if imgui.button(f"{ICON.PLUS} New"):
            self.material_selected = self.new_material().name
        name = self.material_selected
        imgui.same_line()
        imgui.begin_disabled(name is None)
        if imgui.button(f"{ICON.COPY} Duplicate"):
            self.material_selected = self.duplicate_material(name).name
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(name is None or bool(self.scene.users(name)))
        if imgui.button(f"{ICON.TRASH} Delete"):
            self.delete_material(name)
        imgui.end_disabled()
        if imgui.begin_list_box("##materials", imgui.ImVec2(-1, 5 * imgui.get_text_line_height_with_spacing())):
            for n in mats:
                if imgui.selectable(f"{n}  ({len(self.scene.users(n))} users)", n == self.material_selected)[0]:
                    self.material_selected = n
            imgui.end_list_box()
        if self.material_selected in mats:
            self.material_section(mats[self.material_selected])
