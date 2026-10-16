"""MeshSculptor: render a mesh defined by a Python sculpt file, tune its parameters, transform and
material, and save it into valo's mesh library.

    uv run valo-sculpt examples/sculpts/vase.py

Edit the file in your editor: it hot-reloads on save (errors show in the panel, the last good mesh stays).
"Save to library" writes ``valo/meshes/library/<name>.py`` with your values as the new defaults.
See ``valo.meshes.sculpt`` for the file contract.
"""

from __future__ import annotations

import argparse
import copy
import time
import traceback
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from .. import meshes
from ..geometry import instance_matrices
from ..meshes import library
from ..meshes.sculpt import IDENTITY, Sculpt
from ..render import Renderer
from ..ui import Editor
from ..undo import Command

POLL_S = 0.3  # how often the sculpt file's mtime is checked
MAX_NORMAL_ARROWS = 2000
ERROR_COLOR = imgui.ImVec4(1.0, 0.45, 0.4, 1.0)


class SculptScene:
    """The sculpted mesh (one identity instance; the transform is baked into the mesh) + a normals overlay."""

    def __init__(self, renderer: Renderer):
        self.renderer = renderer
        self.body = renderer.instances(meshes.cube(), renderer.material("Sculpt", "pbr"))
        self.body.set(np.eye(4)[None])
        self.arrows = renderer.instances(meshes.arrow(), renderer.material("Normals", "unlit",
                                                                          base_color=(1.0, 0.75, 0.1)))
        self.data: meshes.MeshData | None = None
        self.show_normals = False
        self.normal_length = 0.08

    def set_mesh(self, data: meshes.MeshData) -> None:
        self.data = data
        self.body.mesh = self.renderer.mesh(data)  # the replaced GPU mesh is freed with its MeshData
        self.update_normals()

    def update_normals(self) -> None:
        if self.data is None:
            return
        p, n = self.data.positions, self.data.normals
        pick = np.linspace(0, len(p) - 1, min(len(p), MAX_NORMAL_ARROWS)).astype(int)
        length = self.normal_length
        w = np.full(len(pick), 0.12 * length)
        self.arrows.set(instance_matrices(p[pick], np.c_[w, w, np.full(len(pick), length)], z_axis=n[pick]))

    def center(self) -> np.ndarray:
        p = self.data.positions
        return (p.min(0) + p.max(0)) / 2

    def bounding_radius(self) -> float:
        if self.data is None or not len(self.data.positions):
            return 1.0
        return float(np.linalg.norm(self.data.positions - self.center(), axis=1).max())

    def batches(self):
        return [self.body.batch()] + ([self.arrows.batch()] if self.show_normals else [])


class SculptorEditor(Editor):
    title = "MeshSculptor"

    def __init__(self, scene: SculptScene, path):
        super().__init__()
        self.scene = scene
        self.path = Path(path).resolve()
        self.sculpt: Sculpt | None = None
        self.values: dict = {}
        self.transform = copy.deepcopy(IDENTITY)
        self.uniform_scale = True
        self.error: str | None = None
        self.report: meshes.MeshReport | None = None
        self.status = ""
        self.save_name = self.path.stem
        self.confirm_overwrite = False
        self._mtime = 0.0
        self._polled = 0.0
        self.watch(scene.body.material)
        self.reload(first=True)

    @property
    def material(self):
        return self.scene.body.material

    # -- file
    def reload(self, first: bool = False) -> None:
        """Re-read the sculpt file. Values of parameters that still exist (same type) are kept."""
        try:
            self._mtime = self.path.stat().st_mtime
            sculpt = Sculpt.load(self.path)
        except Exception:
            self.error = traceback.format_exc(limit=-3)
            return
        old = {p.name: p.kind for p in self.sculpt.params} if self.sculpt else {}
        self.sculpt = sculpt
        self.values = {p.name: self.values[p.name] if old.get(p.name) == p.kind else p.default
                       for p in sculpt.params}
        if first:
            self.transform = copy.deepcopy(sculpt.transform)
            for key, value in (sculpt.material or {}).items():
                self.material.set(key, value, notify=False)
        self.status = f"loaded {time.strftime('%H:%M:%S')}"
        self.rebuild()

    def poll(self) -> None:
        """Hot reload: re-read the file when its modification time changes."""
        now = time.monotonic()
        if now - self._polled < POLL_S:
            return
        self._polled = now
        try:
            if self.path.stat().st_mtime != self._mtime:
                self.reload()
        except OSError:
            pass

    def rebuild(self) -> None:
        if self.sculpt is None:
            return
        try:
            data = self.sculpt.build(self.values, self.transform)
        except Exception:
            self.error = traceback.format_exc(limit=-3)
            return
        self.error = None
        self.scene.set_mesh(data)
        self.report = meshes.check(data)

    def save(self, name: str | None = None, overwrite: bool = False) -> Path | None:
        """Write the sculpt with the current values, material and transform to ``valo/meshes/library/<name>.py``."""
        name = (name or self.save_name).strip()
        if not name.isidentifier() or name.startswith("_"):
            self.status = f"'{name}' is not a valid name (letters, digits, _; not starting with _)"
            return None
        target = library.path(name)
        if self.sculpt is None or (target.exists() and not overwrite and target != self.path):
            self.status = f"{target.name} exists in the library" if self.sculpt else "nothing to save"
            return None
        self.sculpt.save(target, self.values, self.material.to_dict(), self.transform)
        self.status = f"saved to valo/meshes/library/{target.name}"
        return target

    # -- actions with undo
    def set_shape_param(self, name: str, value) -> None:
        old = self.values.get(name)
        if name not in self.values or value == old:
            return

        def apply(v):
            def f():
                if name in self.values:  # the parameter may have gone in a reload
                    self.values[name] = v
                    self.rebuild()
            return f

        apply(value)()
        self.undo_stack.push(Command(f"shape: {name}", apply(old), apply(value), merge_key=("shape", name)))

    def set_transform(self, kind: str, value) -> None:
        """``kind`` is "translate", "rotate_deg" or "scale"; ``value`` three numbers."""
        old, value = list(self.transform[kind]), [float(x) for x in value]
        if value == old:
            return

        def apply(v):
            def f():
                self.transform[kind] = list(v)
                self.rebuild()
            return f

        apply(value)()
        self.undo_stack.push(Command(f"transform: {kind}", apply(old), apply(value), merge_key=("transform", kind)))

    # -- panel
    def panel(self) -> None:
        self.poll()

        imgui.separator_text("Sculpt file")
        imgui.text_wrapped(str(self.path))
        imgui.text_disabled(f"{self.status}   (hot-reloads on save)")
        if imgui.button("Reload now"):
            self.reload()
        if self.error:
            imgui.push_style_color(imgui.Col_.text, ERROR_COLOR)
            imgui.text_wrapped(self.error)
            imgui.pop_style_color()
            imgui.text_disabled("showing the last good mesh")
        _, self.save_name = imgui.input_text("Name", self.save_name)
        if imgui.button("Save to library"):
            self.confirm_overwrite = self.save() is None and library.path(self.save_name.strip()).exists()
        if self.confirm_overwrite:
            imgui.same_line()
            if imgui.button("Overwrite"):
                self.save(overwrite=True)
                self.confirm_overwrite = False

        if self.sculpt:
            imgui.separator_text("Shape parameters")
            for p in self.sculpt.params:
                v = self.values[p.name]
                if p.kind == "float":
                    changed, v = imgui.slider_float(p.name, v, p.lo, p.hi, "%.4g")
                elif p.kind == "int":
                    changed, v = imgui.slider_int(p.name, int(v), int(p.lo), int(p.hi))
                else:
                    changed, v = imgui.checkbox(p.name, bool(v))
                if changed:
                    self.set_shape_param(p.name, v)
            if imgui.button("Reset parameters"):
                for p in self.sculpt.params:
                    self.set_shape_param(p.name, p.default)

        imgui.separator_text("Transform (baked into the saved mesh)")
        changed, v = imgui.drag_float3("Translate", self.transform["translate"], 0.01)
        if changed:
            self.set_transform("translate", v)
        changed, v = imgui.slider_float3("Rotate xyz (deg)", self.transform["rotate_deg"], -180.0, 180.0, "%.1f")
        if changed:
            self.set_transform("rotate_deg", v)
        _, self.uniform_scale = imgui.checkbox("Uniform scale", self.uniform_scale)
        if self.uniform_scale:
            changed, s = imgui.drag_float("Scale", self.transform["scale"][0], 0.01, 0.01, 100.0)
            if changed:
                self.set_transform("scale", [s, s, s])
        else:
            changed, v = imgui.drag_float3("Scale xyz", self.transform["scale"], 0.01, 0.01, 100.0)
            if changed:
                self.set_transform("scale", v)
        if imgui.button("Reset transform"):
            for kind, value in IDENTITY.items():
                self.set_transform(kind, value)

        imgui.separator_text("Material")
        self.material_section(self.material)

        imgui.separator_text("Diagnostics")
        if self.report:
            for line in self.report.lines():
                if line.startswith(("normals: ", "winding: ")) and not line.endswith("ok"):
                    imgui.text_colored(ERROR_COLOR, line)
                else:
                    imgui.text(line)
        changed, self.scene.show_normals = imgui.checkbox("Show normals", self.scene.show_normals)
        if self.scene.show_normals:
            changed, length = imgui.slider_float("Normal length", self.scene.normal_length, 0.01, 0.5)
            if changed:
                self.scene.normal_length = length
                self.scene.update_normals()


def build(viewer, path) -> SculptorEditor:
    scene = SculptScene(viewer.renderer)
    editor = SculptorEditor(scene, path)
    viewer.show(scene, editor)  # fits the camera distance to the mesh
    if scene.data is not None:  # ...and aim at the mesh's centre
        viewer.camera.target = tuple(float(c) for c in scene.center())
    return editor


def main(argv=None) -> None:
    from ..app import Viewer

    ap = argparse.ArgumentParser(prog="valo-sculpt", description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path, help="sculpt file: a .py file with a mesh(...) function")
    ap.add_argument("--size", default="1360x800", help="window size, e.g. 1600x900")
    args = ap.parse_args(argv)
    if not args.file.exists():
        ap.error(f"{args.file} does not exist")
    w, h = (int(x) for x in args.size.lower().split("x"))
    viewer = Viewer(title=f"MeshSculptor: {args.file.name}", size=(w, h))
    build(viewer, args.file)
    viewer.run()


if __name__ == "__main__":
    main()
