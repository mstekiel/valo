"""The Object window (Blender's Object properties): the selected node's shape or builder parameters, transform,
material and mesh diagnostics."""

from __future__ import annotations

from imgui_bundle import imgui

from ... import meshes
from ...meshes import library
from ...meshes.check import check
from ...meshes.params import ShapeParam
from ...scene import (
    InstancedObject,
    MeshObject,
    Node,
    ProceduralGroup,
    Transform,
)
from ..style import ICON
from .outliner import KIND_ICONS


class ObjectWindow:
    """SceneEditor part: the Object window."""

    @staticmethod
    def _param_widget(p: ShapeParam, value) -> tuple[bool, object]:
        if p.kind == "float":
            return imgui.slider_float(p.name, float(value), p.lo, p.hi, "%.4g")
        if p.kind == "int":
            return imgui.slider_int(p.name, int(value), int(p.lo), int(p.hi))
        return imgui.checkbox(p.name, bool(value))

    def object_window(self) -> None:
        """The selected node: its shape or builder parameters, transform, material and mesh diagnostics."""
        node = self.selected
        if node is None:
            imgui.text_disabled("Select an object in the Scene window or the 3D view")
            return
        imgui.text(f"{KIND_ICONS.get(node.kind, '')} {node.name}")
        imgui.same_line()
        imgui.text_disabled(node.kind)
        if node.name in self.errors:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.45, 0.4, 1.0))
            imgui.text_wrapped(self.errors[node.name])
            imgui.pop_style_color()
            imgui.text_disabled("showing the last good result")
        if isinstance(node, MeshObject):
            imgui.separator_text("Shape")
            self._shape_combo(node)
            for p in node.shape.specs:
                changed, v = self._param_widget(p, node.shape.params[p.name])
                if changed:
                    self.set_shape_param(node, p.name, v)
        elif isinstance(node, ProceduralGroup):
            imgui.separator_text(f"Builder: {node.func.__name__}")
            for p in node.specs:
                changed, v = self._param_widget(p, node.params[p.name])
                if changed:
                    self.set_procedural_param(node, p.name, v)
        self._transform_section(node)
        if isinstance(node, MeshObject):
            imgui.separator_text("Material")
            if imgui.begin_combo("Material", node.material):
                for name in self.scene.materials:
                    if imgui.selectable(name, name == node.material)[0]:
                        self.set_object_material(node, name)
                imgui.end_combo()
            if imgui.small_button(f"{ICON.PALETTE} Edit in Materials"):
                self.material_selected = node.material
                self._focus.append("Materials")
            imgui.separator_text("Mesh")
            for line in self.diagnostics(node):
                imgui.text_disabled(line)

    def _shape_combo(self, obj: MeshObject) -> None:
        if imgui.begin_combo("Shape", obj.shape.name):
            for name, func in meshes.STANDARD.items():
                if imgui.selectable(name, func is obj.shape.func)[0]:
                    self.set_shape(obj, func)
            for name in library.names():
                if imgui.selectable(f"{ICON.BOOK} {name}", obj.shape.name == name and obj.shape.sculpt)[0]:
                    self.set_shape(obj, library.path(name))
            imgui.end_combo()

    def _transform_section(self, node: Node) -> None:
        imgui.separator_text("Transform")
        t = node.transform
        changed, v = imgui.drag_float3("Translate", list(t.translate), 0.01)
        if changed:
            self.set_transform(node, "translate", v)
        changed, v = imgui.drag_float3("Rotate xyz (deg)", list(t.rotate_deg), 0.5, -360.0, 360.0, "%.1f")
        if changed:
            self.set_transform(node, "rotate_deg", v)
        _, self.uniform_scale = imgui.checkbox("Uniform scale", self.uniform_scale)
        if self.uniform_scale:
            changed, k = imgui.drag_float("Scale", t.scale[0], 0.01, 0.01, 100.0)
            if changed:
                self.set_transform(node, "scale", (k, k, k))
        else:
            changed, v = imgui.drag_float3("Scale xyz", list(t.scale), 0.01, 0.01, 100.0)
            if changed:
                self.set_transform(node, "scale", v)
        if imgui.small_button(f"{ICON.ARROW_ROTATE_LEFT} Reset transform"):
            for field, value in Transform().settings().items():
                self.set_transform(node, field, value)

    def diagnostics(self, obj: MeshObject) -> list[str]:
        """``meshes.check`` lines for ``obj``'s current mesh (computed again only after the mesh changed)."""
        gpu, report = self._reports.get(id(obj), (None, None))
        if gpu is not obj.instances.mesh:
            report = check(obj.shape.build())
            self._reports[id(obj)] = (obj.instances.mesh, report)
        lines = report.lines()
        if isinstance(obj, InstancedObject):
            lines.insert(0, f"{obj.instances.count} instances")
        return lines
