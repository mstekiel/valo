"""Sculpt files: a Python file whose ``mesh(...)`` function builds a MeshData. Load, build, save.

The contract (only ``mesh`` is required; every parameter needs a default)::

    RANGES = {"height": (0.2, 3.0)}                       # slider ranges (optional)
    MATERIAL = {"shading": "pbr", "base_color": [...]}    # material preset (optional)
    TRANSFORM = {"translate": [0, 0, 0], "rotate_deg": [0, 0, 0], "scale": [1, 1, 1]}  # baked (optional)

    def mesh(height: float = 1.0, segments: int = 48, closed: bool = True):
        return revolve(...)

``Sculpt.build`` applies TRANSFORM, so the MeshSculptor preview and ``valo.library.mesh`` agree.
``Sculpt.save`` rewrites only the defaults and the MATERIAL / TRANSFORM values; comments and the rest
of the file are kept byte for byte.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from .. import geometry
from .data import MeshData
from .ops import transform
from .params import ShapeParam, shape_params

IDENTITY = {"translate": [0.0, 0.0, 0.0], "rotate_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}
_counter = itertools.count()


__all__ = ["IDENTITY", "Sculpt", "ShapeParam", "transform_matrix"]


def transform_matrix(t: dict) -> np.ndarray:
    """T · Rz · Ry · Rx · S from a TRANSFORM dict (rotations in degrees, applied x, then y, then z)."""
    t = {**IDENTITY, **(t or {})}
    return geometry.transform_matrix(t["translate"], t["rotate_deg"], t["scale"])


def _literal(v) -> str:
    """A tidy Python literal: floats to 6 significant digits, containers recursively."""
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(f"{float(v):.6g}"))
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, dict):
        items = [f"{json.dumps(k)}: {_literal(x)}" for k, x in v.items()]
        one_line = "{" + ", ".join(items) + "}"
        return one_line if len(one_line) <= 96 else "{\n" + "".join(f"    {i},\n" for i in items) + "}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_literal(x) for x in v) + "]"
    return repr(v)


class Sculpt:
    """A loaded sculpt file: its ``mesh()`` function, parameters, MATERIAL and TRANSFORM."""
    def __init__(self, path: Path, source: str, module):
        self.path, self.source, self.module = path, source, module
        self.func = getattr(module, "mesh", None)
        if not callable(self.func):
            raise AttributeError(f"{path.name} has no mesh(...) function")
        self.params: list[ShapeParam] = shape_params(self.func, getattr(module, "RANGES", {}))
        self.material: dict | None = getattr(module, "MATERIAL", None)
        self.transform: dict = {**IDENTITY, **getattr(module, "TRANSFORM", {})}

    @classmethod
    def load(cls, path) -> Sculpt:
        """Execute ``path`` as a fresh module (never cached, so a reload sees the new file)."""
        path = Path(path).resolve()
        source = path.read_text(encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"_valo_sculpt_{path.stem}_{next(_counter)}", path)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)
        return cls(path, source, module)

    def defaults(self) -> dict:
        """The parameter defaults of ``mesh()`` as a dict."""
        return {p.name: p.default for p in self.params}

    def build(self, params: dict | None = None, transform_: dict | None = None) -> MeshData:
        """``mesh(**params)`` (unknown names ignored, missing ones default) with the transform baked in."""
        known = {p.name for p in self.params}
        kwargs = {**self.defaults(), **{k: v for k, v in (params or {}).items() if k in known}}
        data = self.func(**kwargs)
        if not isinstance(data, MeshData):
            raise TypeError(f"mesh() returned {type(data).__name__}, expected MeshData")
        m = transform_matrix(self.transform if transform_ is None else transform_)
        return data if np.allclose(m, np.eye(4)) else transform(data, m)

    # -- saving
    def rewrite(self, params: dict, material: dict | None, transform_: dict) -> str:
        """The source with new defaults and MATERIAL / TRANSFORM values; everything else untouched."""
        src = self.source.encode("utf-8")  # ast offsets are UTF-8 byte offsets
        line_start = [0, *itertools.accumulate(len(line) for line in src.splitlines(keepends=True))]

        def offset(line, col):
            return line_start[line - 1] + col

        def span(node):
            return offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)

        tree = ast.parse(self.source)
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mesh")
        args = func.args
        positional = args.posonlyargs + args.args
        defaults = dict(zip((a.arg for a in positional[len(positional) - len(args.defaults):]), args.defaults,
                            strict=True))
        defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d is not None})

        edits = []  # (start, end, text)
        kinds = {p.name: p.kind for p in self.params}
        for name, value in params.items():
            if name in defaults:
                cast = {"int": int, "bool": bool, "float": float}[kinds[name]]
                edits.append((*span(defaults[name]), _literal(cast(value))))

        values = {"MATERIAL": material, "TRANSFORM": transform_}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in values):
                value = values.pop(node.targets[0].id)
                if value is not None:
                    edits.append((*span(node.value), _literal(value)))
        top = func.decorator_list[0].lineno if func.decorator_list else func.lineno
        new = "".join(f"{k} = {_literal(v)}\n" for k, v in values.items() if v is not None)
        if new:
            edits.append((offset(top, 0), offset(top, 0), new + "\n\n"))

        for start, end, text in sorted(edits, reverse=True):
            src = src[:start] + text.encode("utf-8") + src[end:]
        return src.decode("utf-8")

    def save(self, target, params: dict, material: dict | None, transform_: dict) -> Path:
        """Write the sculpt with new defaults, material and transform to ``target``; returns the path."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.rewrite(params, material, transform_), encoding="utf-8", newline="")
        return target
