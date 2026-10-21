"""valo.studio: the base studio app (``valo-studio``): valo's Scene / Object / Materials windows plus a Code window
for procedural meshes. Domain studios (spinw_studio, ...) subclass ``StudioEditor``."""

from .cli import build, main
from .editor import TEMPLATE, StudioEditor

__all__ = ["TEMPLATE", "StudioEditor", "build", "main"]
