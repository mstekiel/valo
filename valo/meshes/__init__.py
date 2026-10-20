"""Meshes (pure numpy, no GPU code): import everything from here, e.g. ``from valo.meshes import revolve``.

    data.py        MeshData (position + unit normal per vertex, uint32 triangles)
    ops.py         transform, merge
    builders.py    revolve, parametric, heightfield, tube    (what shape files use)
    primitives.py  sphere, icosphere, cube, plane, cylinder, cone, torus, capsule, arrow, STANDARD
    check.py       check → MeshReport                  (winding, normals, watertight, volume)
    sculpt.py      Sculpt: load / build / save a sculpt file
    library/       saved sculpts: library.names(), library.mesh(name, **params)

Conventions: triangles counter-clockwise seen from outside; Z-up; unit-sized and centred at the origin
unless a function says otherwise.
"""

from . import library
from .builders import heightfield, parametric, revolve, tube
from .check import MeshReport, check
from .data import MeshData
from .ops import merge, transform
from .primitives import STANDARD, arrow, capsule, cone, cube, cylinder, icosphere, plane, sphere, torus
from .sculpt import Sculpt

__all__ = ["STANDARD", "MeshData", "MeshReport", "Sculpt", "arrow", "capsule", "check", "cone", "cube", "cylinder",
           "heightfield", "icosphere", "library", "merge", "parametric", "plane", "revolve", "sphere", "torus",
           "transform", "tube"]
