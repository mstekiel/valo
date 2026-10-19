"""spinw_studio: show spinwaves systems (atoms, moments, unit cell) in valo.

Layers: spinwaves objects → ``adapter.system_view`` (stubs you fill) → ``model.SystemView`` (plain data) →
scene / editor → valo. See README.md for the rules and the data contract.
"""

from .api import plot_structure, render, show, system_of
from .model import DEFAULTS, AtomSite, SystemView, expand, validate_options

__all__ = ["DEFAULTS", "AtomSite", "SystemView", "expand", "plot_structure", "render", "show", "system_of",
           "validate_options"]
