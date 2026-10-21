"""Shape parameters: the editable arguments of a shape function, read from its signature.

A shape function (``valo.meshes.primitives.sphere``, a shape file's ``mesh``, an app's builder) declares its
parameters with defaults and ``float`` / ``int`` / ``bool`` annotations; ``shape_params`` turns them into
``ShapeParam`` records that panels show as sliders and checkboxes. Ranges come from a ``ranges`` dict (a shape
file's ``RANGES``, or a ``func.ranges`` attribute) or a default around the default value.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass
class ShapeParam:
    """One parameter of a shape function: name, widget kind, default and slider range."""

    name: str
    kind: str  # "float" | "int" | "bool" → decides the widget
    default: float | int | bool
    lo: float = 0.0
    hi: float = 1.0


def param_kind(annotation, default) -> str:
    """"float", "int" or "bool" from a parameter's annotation (or its default's type when unannotated)."""
    for kind, typ in (("bool", bool), ("int", int), ("float", float)):  # bool first: it's an int too
        if annotation is typ or annotation == typ.__name__ or (annotation is inspect.Parameter.empty
                                                                and type(default) is typ):
            return kind
    raise TypeError(f"unsupported parameter type {annotation!r} (use float, int or bool)")


def default_range(kind: str, default) -> tuple[float, float]:
    """A slider range for a parameter without an explicit one: 0 .. 2 × default (ints: 1 .. 4 × default)."""
    if kind == "int":
        return min(1, default), max(4, 4 * default)
    if kind == "float":
        return (0.0, 2 * default) if default > 0 else ((2 * default, 0.0) if default < 0 else (-1.0, 1.0))
    return 0, 1


def shape_params(func, ranges: dict | None = None, skip: int = 0) -> list[ShapeParam]:
    """The editable parameters of ``func`` (after the first ``skip`` ones, e.g. a builder's ``group`` argument):
    every one needs a default and a float / int / bool type."""
    ranges = {**getattr(func, "ranges", {}), **(ranges or {})}
    out = []
    for p in list(inspect.signature(func).parameters.values())[skip:]:
        if p.default is inspect.Parameter.empty:
            raise TypeError(f"parameter {p.name!r} of {getattr(func, '__name__', func)}() needs a default value")
        kind = param_kind(p.annotation, p.default)
        lo, hi = ranges.get(p.name, default_range(kind, p.default))
        out.append(ShapeParam(p.name, kind, p.default, lo, hi))
    return out
