"""Shapes: a mesh made by a function from parameters. The function is a callable, a ``"module:function"`` path or the
path of a shape file (``valo.meshes.shapefile``); ``meshes.params.shape_params`` reads its editable parameters."""

from __future__ import annotations

import importlib
from pathlib import Path

from ..meshes.data import MeshData
from ..meshes.params import ShapeParam, shape_params


def resolve(ref):
    """A callable from a callable or a ``"package.module:function"`` path."""
    if callable(ref):
        return ref
    module, _, name = str(ref).partition(":")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def reference(func) -> str:
    """The ``"module:qualname"`` path of an importable function (what ``resolve`` turns back into it)."""
    if isinstance(func, str):
        return func
    path = f"{func.__module__}:{func.__qualname__}"
    if "<" in path or func.__module__ == "__main__":
        raise ValueError(f"{path} is not importable by name; use a module-level function to save the scene")
    return path


class Shape:
    """A mesh made by a function from parameters: ``build()`` returns its MeshData.

    ``source`` is a callable, a ``"module:function"`` path or the path of a sculpt file (``.py`` with ``mesh(...)``,
    whose own TRANSFORM is baked in, as in the library). ``specs`` lists the editable parameters.
    """

    def __init__(self, source, params: dict | None = None):
        self.source = source
        self.sculpt = None
        self._load()
        known = {p.name for p in self.specs}
        unknown = set(params or {}) - known
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)} for {self.name}; it takes {sorted(known)}")
        self.params = {**{p.name: p.default for p in self.specs}, **(params or {})}

    def _load(self) -> None:
        if isinstance(self.source, (str, Path)) and str(self.source).endswith(".py"):
            from ..meshes.shapefile import ShapeFile

            mtime = Path(self.source).stat().st_mtime  # before reading: a write during the read shows next time
            self.sculpt = ShapeFile.load(self.source)
            self.func, self.specs, self.mtime = self.sculpt.build, self.sculpt.params, mtime
        else:
            self.func = resolve(self.source)
            self.specs: list[ShapeParam] = shape_params(self.func)

    @property
    def name(self) -> str:
        """A short label: the function name or the sculpt file's stem."""
        return Path(str(self.source)).stem if self.sculpt else getattr(self.func, "__name__", str(self.source))

    def build(self) -> MeshData:
        """The mesh for the current parameters."""
        data = self.func(self.params) if self.sculpt else self.func(**self.params)
        if not isinstance(data, MeshData):
            raise TypeError(f"{self.name} returned {type(data).__name__}, expected MeshData")
        return data

    def stale(self) -> bool:
        """True when the shape's sculpt file changed on disk since it was read (always False for functions)."""
        try:
            return self.sculpt is not None and Path(self.source).stat().st_mtime != self.mtime
        except OSError:
            return False

    def reload(self) -> None:
        """Re-read a sculpt file. A parameter you changed from its default keeps your value (if it still exists with
        the same kind); the others take the file's new defaults, so editing a default in the code shows at once."""
        old = {p.name: p for p in self.specs}
        values = self.params
        self._load()

        def value(p):
            before = old.get(p.name)
            edited = before is not None and before.kind == p.kind and values[p.name] != before.default
            return values[p.name] if edited else p.default

        self.params = {p.name: value(p) for p in self.specs}

    def settings(self) -> dict:
        """As plain data: the function reference (or file) and the parameters."""
        return {"func": str(self.source) if self.sculpt else reference(self.source), "params": dict(self.params)}
