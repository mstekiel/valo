"""valo.studio: the base studio app (``valo-studio``): valo's Scene / Object / Materials windows plus a Code window
for procedural meshes. Domain studios (spinw_studio, ...) subclass ``StudioEditor``."""

from .editor import TEMPLATE, StudioEditor, build, main

__all__ = ["TEMPLATE", "StudioEditor", "build", "main"]
