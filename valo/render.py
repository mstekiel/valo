"""Deprecated location: the GPU side is ``valo.gpu`` and the camera is ``valo.scene.camera.Camera`` (this module is
removed at the end of the refactor)."""

from .gpu.instances import *  # noqa: F403
from .gpu.mesh import *  # noqa: F403
from .gpu.renderer import *  # noqa: F403
from .scene.camera import PROJECTIONS, VIEWS  # noqa: F401
from .scene.camera import Camera as OrbitCamera  # noqa: F401
