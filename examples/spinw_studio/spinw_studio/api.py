"""The two calls a script uses: ``plot_structure`` (the studio window, blocking) and ``render`` (an image, no window).

Both take your ``SpinW`` object (read through ``adapter.system_view``) or a ``SystemView``, and ``plot_options``
(see ``model.DEFAULTS``). ``plot_structure`` returns the plot_options as they are when you close the window, so
your adjustments can go straight back into the script (or use File > Export plot options).
"""

from __future__ import annotations

import numpy as np

import valo

from . import adapter
from .editor import SpinwEditor
from .model import SystemView, validate_options
from .scene import SpinwScene


def system_of(sw_or_view) -> SystemView:
    """A SystemView: passed through, or read from a SpinW object with the adapter stubs."""
    return sw_or_view if isinstance(sw_or_view, SystemView) else adapter.system_view(sw_or_view)


def show(viewer: valo.Viewer, sw_or_view, plot_options: dict | None = None) -> SpinwEditor:
    """Build the scene and editor for ``sw_or_view`` in ``viewer`` (camera and lighting from plot_options)."""
    system = system_of(sw_or_view)
    opts = validate_options(plot_options)
    scene = SpinwScene(system, opts, viewer.renderer)
    editor = SpinwEditor(scene)
    viewer.show(scene, editor)
    if opts["camera"]:
        viewer.camera = valo.OrbitCamera.from_settings(opts["camera"])
    else:
        viewer.camera.target = tuple(float(c) for c in scene.center())
    if opts["lighting"]:
        viewer.lighting = valo.Lighting.from_settings(opts["lighting"])
    if opts["image"]:
        editor.render_options.update(opts["image"])
    return editor


def _window(title: str) -> valo.Viewer:
    return valo.Viewer(title=title)


def plot_structure(sw_or_view, plot_options: dict | None = None, *, title: str | None = None) -> dict:
    """Open SpinW studio on ``sw_or_view``; block until the window is closed; return the final plot_options."""
    system = system_of(sw_or_view)
    viewer = _window(title or ("SpinW studio" + (f": {system.name}" if system.name else "")))
    editor = show(viewer, system, plot_options)
    viewer.run()
    return editor.plot_options()


def render(sw_or_view, plot_options: dict | None = None, *, path=None, **image) -> np.ndarray:
    """The picture ``plot_structure`` would show, rendered offline: (h, w, 4) uint8 sRGB, saved as PNG if ``path``.

    ``image`` keywords (size, samples, transparent, background) override ``plot_options["image"]``.
    """
    system = system_of(sw_or_view)
    opts = validate_options(plot_options)
    scene = SpinwScene(system, opts)
    camera = valo.OrbitCamera.from_settings(opts["camera"]) if opts["camera"] else None
    if camera is None:  # the studio's starting view: fitted, aimed at the extent's centre
        camera = valo.OrbitCamera(fovy=valo.rc.camera.fov)
        camera.target = tuple(float(c) for c in scene.center())
        camera.fit(scene.bounding_radius())
    settings = {"lighting": opts["lighting"], "image": {**opts["image"], **image}}
    return valo.render(scene, settings, camera=camera, path=path)
