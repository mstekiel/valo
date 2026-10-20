"""valo's library of procedural meshes: shape files saved by the studio, in ``rc.studio.procedural_meshes``
(default ``data/procedural_meshes/`` in the project folder; a relative path is resolved against that folder).

Every ``*.py`` file there is an entry, named after the file::

    from valo import library
    library.names()                          # ["vase", ...]
    data = library.mesh("vase", height=2.0)  # MeshData, saved transform baked in
    gold_vase = library.material(renderer, "vase")
"""

from __future__ import annotations

from pathlib import Path

from ...config import rc
from ...materials import Material
from ..data import MeshData
from ..sculpt import Sculpt

PROJECT_DIR = Path(__file__).resolve().parents[3]  # the folder holding the valo package


def directory() -> Path:
    """The library folder: ``rc.studio.procedural_meshes``, relative paths resolved against the project folder."""
    return PROJECT_DIR / Path(rc.studio.procedural_meshes).expanduser()


def names() -> list[str]:
    """Names of all saved library meshes, sorted."""
    return sorted(p.stem for p in directory().glob("*.py") if not p.name.startswith("_"))


def path(name: str) -> Path:
    """File path of library entry ``name`` (it may not exist yet)."""
    return directory() / f"{name}.py"


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
