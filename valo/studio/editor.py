"""valo studio: the base studio app. valo's Scene / Object / Materials windows plus a Code window, where you write a
procedural mesh in Python and tune its parameters in the Object tab.

    uv run valo-studio                 # an empty scene; File > New shape starts a procedural mesh
    uv run valo-studio my_shape.py     # open a shape file

A **shape file** is plain Python with a ``mesh(...)`` function whose parameters all have defaults (``float``,
``int``, ``bool``), optionally ``RANGES``, ``MATERIAL`` and ``TRANSFORM`` (contract: ``valo.meshes.shapefile``).
Every save in the Code window (Ctrl+S) re-reads the file: the Object tab's parameters are refreshed (values kept
where name and type still match) and the mesh is rebuilt; on an error the last good mesh stays and the error shows
under the code. Files changed by another editor reload too.

**Save to library** writes the file with the object's parameter values as new defaults, its material as
``MATERIAL`` and its transform baked into ``TRANSFORM`` to the procedural mesh library
(``rc.studio.procedural_meshes``, default ``data/procedural_meshes/``).

Domain studios (spinw_studio, ...) subclass ``StudioEditor``.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs
from imgui_bundle import imgui_color_text_edit as text_edit

from .. import meshes
from ..config import rc
from ..geometry import decompose, euler_xyz, instance_matrices
from ..meshes import library
from ..meshes.shapefile import IDENTITY, transform_matrix
from ..scene import MeshObject, Scene, Shape, Transform
from ..ui.scene_editor import SceneEditor, shape_error
from ..ui.style import ICON, mono_font

ERROR_COLOR = imgui.ImVec4(1.0, 0.45, 0.4, 1.0)

TEMPLATE = '''"""{name}: a procedural mesh. Edit, then save (Ctrl+S): the Object tab re-reads the parameters.

mesh(...) returns a valo MeshData. Every parameter needs a default and a float / int / bool type; RANGES sets the
slider ranges. Build with valo.meshes: revolve ((rho, z) profile around Z), parametric (P(u, v)), heightfield,
tube (a circle swept along a 3D path), the primitives, transform and merge.
"""

import numpy as np

from valo import meshes

RANGES = {{"radius": (0.1, 2.0), "height": (0.1, 4.0), "waist": (0.0, 0.9), "segments": (3, 128)}}


def mesh(radius: float = 0.5, height: float = 1.5, waist: float = 0.4, segments: int = 48):
    """A closed body of revolution with a narrower waist."""
    z = np.linspace(0, height, 24)
    rho = radius * (1 - waist * np.sin(np.pi * z / height))
    side = list(zip(rho, z))
    # bottom centre -> bottom rim (repeated: a hard edge) -> side -> top rim (repeated) -> top centre
    return meshes.revolve([(0, 0), side[0], *side, side[-1], (0, height)], segments)
'''


def _same(a, b) -> bool:
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


class StudioEditor(SceneEditor):
    """The studio: SceneEditor plus the Code window for shape files, New / Open shape and Save to library."""

    title = "valo studio"
    dock_left = ("Code",)
    file_dialog = staticmethod(portable_file_dialogs.open_file)  # the OS's file dialog (swapped out in tests)

    def __init__(self, scene: Scene):
        super().__init__(scene)
        self.code_path: Path | None = None  # the shape file in the Code window
        self.code = ""  # the Code window's text (what you typed)
        self.saved_code = ""  # the file's text when last opened, saved or reloaded
        self.status = ""
        self._widget = None  # the TextEditor, made on first draw
        self._in_widget = None  # the text last put into the widget
        self._code_mtime = 0.0
        self._dialog: tuple[str, str] | None = None  # ("new" | "open" | "save", text being typed)
        self._browsing = None  # the open file dialog, polled each frame until you pick a file or cancel
        self._pending: tuple[Path, str] | None = None  # (file, error): opened but its object couldn't be made yet
        self._confirm_overwrite = False
        self.show_normals, self.normal_length = False, 0.08  # the normals overlay (view state, not undoable)
        self.normals = scene.renderer.instances(meshes.arrow(), scene.renderer.material(
            "Normals", "unlit", base_color=(1.0, 0.75, 0.1)), label="normal", info=self.normal_info)
        self._normal_rows = np.zeros((0, 2), int)  # arrow → (instance, vertex)
        self._normal_key = None
        self._normal_data: tuple | None = None  # (gpu mesh, MeshData) of the object the overlay shows
        scene.overlays.append(self.normals)

    # -- the Code window's file (viewing a file is not an edit)
    @property
    def dirty(self) -> bool:
        """True when the Code window has changes that aren't saved."""
        return self.code != self.saved_code

    def open_code(self, path) -> None:
        """Show shape file ``path`` in the Code window (unsaved changes to the previous file are dropped)."""
        self.code_path = Path(path)
        self.code = self.saved_code = self.code_path.read_text(encoding="utf-8")
        self._code_mtime = self.code_path.stat().st_mtime
        self.status = f"opened {self.code_path.name}"

    def set_code(self, text: str) -> None:
        """Replace the Code window's text (what typing does; typing has the widget's own undo, so not recorded)."""
        self.code = text

    def users(self, path) -> list[MeshObject]:
        """The objects built from shape file ``path``."""
        return [o for o in self.scene.objects() if o.shape.sculpt is not None and _same(o.shape.source, path)]

    @property
    def code_error(self) -> str | None:
        """The error of the last reload of the Code window's file, if any."""
        if self._pending is not None and _same(self._pending[0], self.code_path):
            return self._pending[1]
        return next((self.errors[o.name] for o in self.users(self.code_path) if o.name in self.errors), None)

    # -- actions with undo
    def _write_and_reload(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8", newline="")
        if _same(path, self.code_path):
            self.code = self.saved_code = text
            self._code_mtime = path.stat().st_mtime
        for obj in self.users(path):
            self.reload_shape(obj)

    def save_code(self) -> None:
        """Write the Code window's text to its file and re-read it for every object built from it (undoable:
        undo writes the previous text back)."""
        if self.code_path is None or not self.dirty:
            return
        path, old, new = self.code_path, self.saved_code, self.code
        with self.one_step(f"Save {path.name}"):
            self._write_and_reload(path, new)
            self._push(f"Save {path.name}", lambda: self._write_and_reload(path, old),
                       lambda: self._write_and_reload(path, new))
            if self._pending is not None and _same(self._pending[0], path) and not self.users(path):
                self._try_add(path)  # opened while broken: the object appears once the file builds
        self.status = f"saved {path.name}" + (" (with errors)" if self.code_error else "")

    def add_shape(self, source, parent=None, material: str | None = None) -> MeshObject:
        """Add an object made by ``source``; a shape file with a ``MATERIAL`` preset gets it as a new scene
        material named after the file (unless ``material`` is given). See ``SceneEditor.add_shape``."""
        shape = Shape(source)
        with self.one_step(f"Add {shape.name.capitalize()}"):
            if material is None and shape.sculpt is not None and shape.sculpt.material:
                preset = dict(shape.sculpt.material)
                material = self.new_material(shape.name, preset.pop("shading", "pbr"), **preset).name
            return super().add_shape(source, parent, material)

    def _try_add(self, path: Path) -> MeshObject | None:
        try:
            obj = self.add_shape(path)
        except Exception:
            self._pending = (path, shape_error(traceback.format_exc(), path))
            return None
        self._pending = None
        return obj

    def open_shape(self, path) -> MeshObject | None:
        """Add an object built from shape file ``path`` and show the file in the Code window; returns the object.
        A file that fails to load or build still opens, with its error under the code and no object (returns None):
        fix it there, and the object appears on the first save that builds."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a file")
        obj = self._try_add(path)
        self.open_code(path)
        if obj is None:
            self.status = f"! {path.name} doesn't build yet: fix the code and save"
        return obj

    def new_shape(self, name: str) -> MeshObject:
        """Write a new shape file ``<name>.py`` from a template into the procedural mesh library, add an object built
        from it and open it in the Code window; returns the object. (Undo removes the object; the file stays.)"""
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"{name!r} is not a valid shape name (letters, digits, _; not starting with _)")
        path = library.path(name)
        if path.exists():
            raise FileExistsError(f"{path} exists; open it, or choose another name")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE.format(name=name), encoding="utf-8")
        return self.open_shape(path)

    def save_to_library(self, obj: MeshObject, name: str, overwrite: bool = False) -> Path:
        """Write ``obj``'s shape file to the library as ``<name>.py`` with its parameter values as defaults, its
        material as ``MATERIAL`` and its transform baked into ``TRANSFORM``; the object then uses that file with an
        identity transform, so it looks the same. Undoable (undo restores the file and the object). Returns the path."""
        if obj.shape.sculpt is None:
            raise ValueError(f"{obj.name} isn't built from a shape file")
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"{name!r} is not a valid library name (letters, digits, _; not starting with _)")
        target = library.path(name)
        if target.exists() and not overwrite and not _same(target, obj.shape.source):
            raise FileExistsError(f"{target.name} exists in the library")
        sculpt = obj.shape.sculpt
        t, r, s = decompose(obj.transform.matrix() @ transform_matrix(sculpt.transform))
        baked = {"translate": t.tolist(), "rotate_deg": [round(a, 6) for a in euler_xyz(r)], "scale": s.tolist()}
        baked = {k: v if not np.allclose(v, IDENTITY[k]) else IDENTITY[k] for k, v in baked.items()}
        old = (target.read_text(encoding="utf-8") if target.exists() else None, obj.shape, obj.transform)
        text = sculpt.rewrite(obj.shape.params, self.scene.materials[obj.material].to_dict(), baked)
        new_shape = []  # made once, then reused by redo

        def put(file_text, shape, transform):
            def f():
                if file_text is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(file_text, encoding="utf-8", newline="")
                if shape is None and not new_shape:
                    new_shape.append(Shape(target, dict(obj.shape.params)))
                obj.shape, obj.transform = shape or new_shape[0], transform
                obj.rebuild_mesh()
                self.scene.refresh(obj)
                if _same(self.code_path, target) or _same(self.code_path, old[1].source):
                    self.open_code(obj.shape.source)
            return f

        put(text, None, Transform())()
        self._push(f"Save {obj.name} to library as {name}", put(*old), put(text, None, Transform()))
        self.status = f"saved to {target}"
        return target

    # -- hot reload and following the selection
    def poll(self) -> None:
        """SceneEditor's hot reload, plus: the Code window's file changed elsewhere and has no unsaved changes →
        show the new text."""
        super().poll()
        path = self.code_path
        if path is not None and not self.dirty and path.exists() and path.stat().st_mtime != self._code_mtime:
            self.open_code(path)
            self.status = f"reloaded {path.name} (changed on disk)"

    # -- normals overlay
    def update_normals(self) -> None:
        """Arrows along the vertex normals of the selected object (every instance, at most
        ``rc.studio.max_normal_arrows``), ``normal_length`` long; nothing when ``show_normals`` is off."""
        obj = self.selected
        if not (self.show_normals and isinstance(obj, MeshObject) and obj.shown() and obj.instances.count):
            if self.normals.count:
                self.normals.set(np.zeros((0, 4, 4)))
            self._normal_key = None
            return
        inst = obj.instances
        key = (id(obj), id(inst.mesh), inst.matrices.tobytes(), self.normal_length, rc.studio.max_normal_arrows)
        if key == self._normal_key:
            return
        self._normal_key = key
        if self._normal_data is None or self._normal_data[0] is not inst.mesh:
            self._normal_data = (inst.mesh, obj.shape.build())
        data = self._normal_data[1]
        n_inst, n_vert = len(inst.matrices), len(data.positions)
        pick = np.linspace(0, n_inst * n_vert - 1, min(n_inst * n_vert, rc.studio.max_normal_arrows)).astype(int)
        rows = np.c_[pick // n_vert, pick % n_vert]
        m = inst.matrices[rows[:, 0]]
        p = np.einsum("nij,nj->ni", m[:, :3, :3], data.positions[rows[:, 1]]) + m[:, :3, 3]
        normal_m = np.linalg.inv(m[:, :3, :3]).transpose(0, 2, 1)  # normals transform with the inverse transpose
        n = np.einsum("nij,nj->ni", normal_m, data.normals[rows[:, 1]])
        w, L = np.full(len(p), 0.12 * self.normal_length), np.full(len(p), self.normal_length)
        self.normals.set(instance_matrices(p, np.c_[w, w, L], z_axis=n))
        self._normal_rows = rows

    def normal_info(self, i: int) -> dict:
        """Info rows for normal arrow ``i``: the instance and vertex it starts at, and that vertex's normal."""
        k, v = (int(c) for c in self._normal_rows[i])
        n = self._normal_data[1].normals[v]
        return {"Instance": str(k), "Vertex": str(v), "Normal": "(" + ", ".join(f"{c + 0.0:.3f}" for c in n) + ")"}

    def object_window(self) -> None:
        """SceneEditor's Object window plus the normals overlay switch for mesh objects."""
        super().object_window()
        if isinstance(self.selected, MeshObject):
            _, self.show_normals = imgui.checkbox("Show normals", self.show_normals)
            if self.show_normals:
                _, self.normal_length = imgui.slider_float("Normal length", self.normal_length, 0.01, 0.5)

    def browse_shape(self) -> None:
        """Open the system's file dialog (in the library folder) to pick a shape file; the studio keeps running and
        ``open_shape`` follows when you choose one. Without a system dialog, a path field opens instead."""
        if self._browsing is not None:
            return
        try:
            self._browsing = self.file_dialog("Open shape file", str(library.directory()),
                                              ["Python shape files (*.py)", "*.py", "All files", "*"])
        except Exception:
            self._dialog = ("open", "")

    def _poll_browsing(self) -> None:
        if self._browsing is None or not self._browsing.ready(0):
            return
        paths, self._browsing = self._browsing.result(), None
        if paths:
            try:
                self.open_shape(Path(paths[0]))
            except Exception as exc:
                self.status = f"! could not open {Path(paths[0]).name}: {exc}"

    def gui(self) -> None:
        """Show the selected object's shape file in the Code window (when nothing is unsaved), then draw."""
        sel = self.selected
        if (isinstance(sel, MeshObject) and sel.shape.sculpt is not None and not self.dirty
                and not _same(sel.shape.source, self.code_path)):
            self.open_code(sel.shape.source)
        self._poll_browsing()
        self.update_normals()
        super().gui()
        self._dialogs()

    # -- windows and menus
    def windows(self):
        """Scene, Object, Materials and Code."""
        return {**super().windows(), "Code": self.code_window}

    def menus(self):
        """The File menu: New shape, Open shape file, Save to library."""
        return {"File": self.file_menu}

    def file_menu(self) -> None:
        """File menu items (valo adds Render below them)."""
        if imgui.menu_item_simple(f"{ICON.FILE} New shape..."):
            self._dialog = ("new", "")
        if imgui.menu_item_simple(f"{ICON.FOLDER_OPEN} Open shape file..."):
            self.browse_shape()
        obj = self.selected
        can_save = isinstance(obj, MeshObject) and obj.shape.sculpt is not None
        if imgui.menu_item(f"{ICON.BOOK} Save to library as...", "", False, can_save)[0]:
            self._dialog, self._confirm_overwrite = ("save", Path(str(obj.shape.source)).stem), False
        imgui.separator()

    def _dialogs(self) -> None:
        if self._dialog is None:
            return
        kind, text = self._dialog
        title = {"new": "New shape", "open": "Open shape file", "save": "Save to library"}[kind]
        if not imgui.is_popup_open(title):
            imgui.open_popup(title)
        if not imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        hint = {"new": f"name: a file in {library.directory()}", "open": "path of a .py shape file",
                "save": f"library name: {library.directory()}/<name>.py"}[kind]
        imgui.text_disabled(hint)
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        done, text = imgui.input_text("##text", text, imgui.InputTextFlags_.enter_returns_true)
        self._dialog = (kind, text)
        if self.status.startswith("! "):
            imgui.text_colored(ERROR_COLOR, self.status[2:])
        ok = imgui.button("OK") or done
        if kind == "save" and self._confirm_overwrite:
            imgui.same_line()
            ok = imgui.button(f"{ICON.TRIANGLE_EXCLAMATION} Overwrite") or ok
        if ok:
            try:
                if kind == "new":
                    self.new_shape(text.strip())
                elif kind == "open":
                    self.open_shape(Path(text.strip().strip('"')))
                else:
                    self.save_to_library(self.selected, text.strip(), overwrite=self._confirm_overwrite)
                self._dialog = None
                imgui.close_current_popup()
            except FileExistsError as exc:
                self._confirm_overwrite = kind == "save"
                self.status = f"! {exc}"
            except Exception as exc:
                self.status = f"! {exc}"
        imgui.same_line()
        if imgui.button("Cancel"):
            self._dialog = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _sync_widget(self):
        if self._widget is None:
            self._widget = text_edit.TextEditor()
            self._widget.set_language(text_edit.TextEditor.Language.python())
            self._widget.set_palette(text_edit.TextEditor.get_dark_palette())
            self._widget.set_tab_size(4)
            self._widget.set_insert_spaces_on_tabs(True)
        if self.code != self._in_widget:  # changed outside the widget: opened, saved, undone, set_code
            self._widget.set_text(self.code)
            self._in_widget = self.code = self._widget.get_text()
            if self.saved_code.rstrip("\n") == self.code.rstrip("\n"):
                self.saved_code = self.code  # the widget normalises the final newline: not a change
        return self._widget

    def code_window(self) -> None:
        """The shape file being edited: Save (Ctrl+S), the text, and the last reload error under it."""
        if self.code_path is None:
            imgui.text_disabled("File > New shape..., or File > Open shape file...,")
            imgui.text_disabled("or select an object built from a shape file.")
            return
        widget = self._sync_widget()
        imgui.begin_disabled(not self.dirty)
        if imgui.button(f"{ICON.FLOPPY_DISK} Save"):
            self.save_code()
        imgui.end_disabled()
        imgui.same_line()
        imgui.text(f"{self.code_path.name}{'  ●' if self.dirty else ''}")
        imgui.same_line()
        imgui.text_disabled(self.status if not self.status.startswith("! ") else "")
        error = self.code_error
        footer = 0 if not error else 6 * imgui.get_text_line_height_with_spacing()
        imgui.push_font(mono_font(), 0.0)
        widget.render("##code", imgui.ImVec2(0, -footer))
        imgui.pop_font()
        if imgui.is_item_focused() or imgui.is_window_focused(imgui.FocusedFlags_.root_and_child_windows):
            io = imgui.get_io()
            if io.key_ctrl and imgui.is_key_pressed(imgui.Key.s):
                self.save_code()
        text = widget.get_text()
        if text != self._in_widget:
            self.code = self._in_widget = text
        if error:
            imgui.push_style_color(imgui.Col_.text, ERROR_COLOR)
            imgui.push_font(mono_font(), 0.0)
            imgui.text_wrapped(error)
            imgui.pop_font()
            imgui.pop_style_color()


def build(viewer, path=None) -> StudioEditor:
    """An empty studio scene in ``viewer`` (plus an object for shape file ``path``, if given); returns the editor."""
    editor = StudioEditor(Scene(viewer.renderer))
    viewer.show(editor.scene, editor)
    if path is not None:
        editor.open_shape(path)
        viewer.fit_all()
    return editor


def main(argv=None) -> None:
    """Command-line entry point: ``valo-studio [shape.py] [--size WxH]``."""
    from ..ui.viewer import Viewer

    ap = argparse.ArgumentParser(prog="valo-studio", description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path, nargs="?", help="a shape file to open: a .py file with a mesh(...) function")
    ap.add_argument("--size", default="1600x900", help="window size, e.g. 1920x1080")
    args = ap.parse_args(argv)
    if args.file is not None and not args.file.exists():
        ap.error(f"{args.file} does not exist")
    w, h = (int(x) for x in args.size.lower().split("x"))
    viewer = Viewer(title="valo studio", size=(w, h))
    build(viewer, args.file)
    viewer.run()


if __name__ == "__main__":
    main()
