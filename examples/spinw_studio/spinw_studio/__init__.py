"""spinw_studio: show spinwaves systems (atoms, moments, unit cell) in valo.

Layers: spinwaves objects → ``adapter.system_view`` (stubs you fill) → ``model.SystemView`` (plain data) →
scene / editor → valo. See README.md for the rules and the data contract.
"""

from .model import AtomSite, SystemView, expand, validate_options

__all__ = ["AtomSite", "SystemView", "expand", "validate_options"]
