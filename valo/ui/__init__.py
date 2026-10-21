"""valo.ui: the editors (Blender's editors and window manager): the Editor base with its dock layout and menus,
the shared windows, the overlays drawn over the 3D view, and the Viewer that runs it all."""

from .editor import SHARED_WINDOWS, UI_SCALES, Editor
from .overlays import BOX_EDGES, box_layout, compass_layout
from .windows.info import pick_info

__all__ = ["BOX_EDGES", "SHARED_WINDOWS", "UI_SCALES", "Editor", "box_layout", "compass_layout", "pick_info"]
