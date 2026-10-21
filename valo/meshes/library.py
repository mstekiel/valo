"""valo's library of procedural meshes: shape files in two folders, read together.

- **Built-in** (``BUILTIN_DIR``, ``valo/data/procedural_meshes/``): starter meshes shipped with valo; only read.
- **User** (``directory()``, from ``rc.studio.procedural_meshes``, default ``data/procedural_meshes``; a relative
  path is relative to the current working directory): the studio's New shape and Save to library write here. A user
  file hides a built-in one of the same name.

Every ``*.py`` file there is an entry, named after the file::

    from valo import library
    library.names()                          # ["vase", ...]
    data = library.mesh("vase", height=2.0)  # MeshData, saved transform baked in
    gold_vase = library.material(renderer, "vase")
"""

from __future__ import annotations

from pathlib import Path

from ..config import rc
from .data import MeshData
from .shapefile import ShapeFile

BUILTIN_DIR = Path(__file__).resolve().parents[1] / "data" / "procedural_meshes"  # shipped inside the package


def directory() -> Path:
    """The user library folder: ``rc.studio.procedural_meshes`` (``~`` expanded; relative to the working directory)."""
    return Path(rc.studio.procedural_meshes).expanduser().resolve()


def _entries(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.glob("*.py") if not p.name.startswith("_")}


def names() -> list[str]:
    """Names of all library meshes, built-in and user, sorted."""
    return sorted({**_entries(BUILTIN_DIR), **_entries(directory())})


def path(name: str) -> Path:
    """File of library entry ``name``: the user's, else the built-in one, else where a new one would be saved."""
    for folder in (directory(), BUILTIN_DIR):
        if (folder / f"{name}.py").exists():
            return folder / f"{name}.py"
    return save_path(name)


def save_path(name: str) -> Path:
    """Where entry ``name`` is saved: always the user folder, so saving never writes into the installed package."""
    return directory() / f"{name}.py"


def load(name: str) -> ShapeFile:
    """The library entry ``name`` as a ShapeFile (raises KeyError if there is none)."""
    if not path(name).exists():
        raise KeyError(f"no library mesh {name!r}; have {names()}")
    return ShapeFile.load(path(name))


def mesh(name: str, **params) -> MeshData:
    """Build library mesh ``name`` with its saved transform; ``params`` override saved defaults."""
    return load(name).build(params)


def material(renderer, name: str):
    """The entry's saved MATERIAL preset as a new material (a plain grey PBR one if it has none)."""
    preset = dict(load(name).material or {"shading": "pbr"})
    return renderer.material(name, preset.pop("shading"), **preset)
