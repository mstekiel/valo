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

import traceback
from pathlib import Path

import numpy as np
from imgui_bundle import portable_file_dialogs

from .. import meshes
from ..geometry import decompose, euler_xyz
from ..meshes import library
from ..meshes.shapefile import IDENTITY, transform_matrix
from ..scene import MeshObject, Scene, Shape, Transform
from ..ui.scene_editor import SceneEditor, shape_error
from .code import CodeWindow
from .normals import NormalsOverlay

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


class StudioEditor(CodeWindow, NormalsOverlay, SceneEditor):
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
