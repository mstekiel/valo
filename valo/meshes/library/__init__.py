"""valo's mesh library: sculpt files saved by MeshSculptor (``uv run valo-sculpt <file.py>``; this folder is valo/meshes/library).

Every ``*.py`` file in this folder is an entry, named after the file::

    from valo import library
    library.names()                          # ["vase", ...]
    data = library.mesh("vase", height=2.0)  # MeshData, saved transform baked in
    gold_vase = library.material(renderer, "vase")
"""

from __future__ import annotations

from pathlib import Path

from ...materials import Material
from ..data import MeshData
from ..sculpt import Sculpt

LIBRARY_DIR = Path(__file__).parent


def names() -> list[str]:
    """Names of all saved library meshes, sorted."""
    return sorted(p.stem for p in LIBRARY_DIR.glob("*.py") if not p.name.startswith("_"))


def path(name: str) -> Path:
    """File path of library entry ``name`` (it may not exist yet)."""
    return LIBRARY_DIR / f"{name}.py"


def load(name: str) -> Sculpt:
    """The library entry ``name`` as a Sculpt (raises KeyError if there is none)."""
    if not path(name).exists():
        raise KeyError(f"no library mesh {name!r}; have {names()}")
    return Sculpt.load(path(name))


def mesh(name: str, **params) -> MeshData:
    """Build library mesh ``name`` with its saved transform; ``params`` override saved defaults."""
    return load(name).build(params)


def material(renderer, name: str) -> Material:
    """The entry's saved MATERIAL preset as a new material (a plain grey PBR one if it has none)."""
    preset = dict(load(name).material or {"shading": "pbr"})
    return renderer.material(name, preset.pop("shading"), **preset)
