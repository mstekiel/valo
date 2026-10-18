"""Offline rendering: an image of a scene without a window, for scripts and the studio's "Render image".

    import valo
    crystal = Crystal()                                   # built on valo.renderer(), no window needed
    img = valo.render(crystal, size=(2400, 1600), samples=2, transparent=True, path="nacl.png")

``settings`` is the dict an exported render script holds: ``{"camera": {...}, "lighting": {...},
"image": {"size": [w, h], "samples": k, "transparent": bool, "background": [r, g, b]}}`` (any part optional);
keyword arguments override it, and anything still missing comes from ``rc.render`` / ``rc.lighting``, with the
camera fitted to the scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wgpu

from .config import rc
from .lighting import Lighting
from .render import OrbitCamera, renderer

IMAGE_KEYS = ("size", "samples", "transparent", "background")


def default_camera(scene) -> OrbitCamera:
    """The starting camera for ``scene``: rc field of view, aimed at ``scene.center()`` and fitted to
    ``scene.bounding_radius()`` when the scene has them."""
    cam = OrbitCamera(fovy=rc.camera.fov)
    if hasattr(scene, "center"):
        cam.target = tuple(float(c) for c in scene.center())
    if hasattr(scene, "bounding_radius"):
        cam.fit(scene.bounding_radius())
    return cam


def image_settings(settings: dict | None = None, **overrides) -> dict:
    """The image options (size, samples, transparent, background): ``overrides`` (not None) over
    ``settings["image"]`` over ``rc.render``."""
    image = {k: getattr(rc.render, k) for k in IMAGE_KEYS}
    given = dict((settings or {}).get("image", {}))
    unknown = set(given) - set(IMAGE_KEYS)
    if unknown:
        raise KeyError(f"unknown image settings {sorted(unknown)}; use {list(IMAGE_KEYS)}")
    image.update(given)
    image.update({k: v for k, v in overrides.items() if v is not None})
    return image


def render(scene, settings: dict | None = None, *, size=None, samples=None, transparent=None, background=None,
           camera: OrbitCamera | None = None, lighting: Lighting | None = None, path=None) -> np.ndarray:
    """Render ``scene`` offline and return the image, (height, width, 4) uint8 sRGB with alpha; save it as a
    PNG if ``path`` is given. See the module docstring for ``settings`` and the defaults."""
    settings = settings or {}
    image = image_settings(settings, size=size, samples=samples, transparent=transparent, background=background)
    if camera is None:
        camera = OrbitCamera.from_settings(settings["camera"]) if "camera" in settings else default_camera(scene)
    if lighting is None:
        lighting = Lighting.from_settings(settings.get("lighting", {}))
    r = renderer()
    w, h = (int(c) for c in image["size"])
    limit = r.device.limits["max-texture-dimension-2d"]
    if max(w, h) > limit:
        raise ValueError(f"image size {w}x{h} exceeds this GPU's limit of {limit} px per side")
    k = max(1, min(int(image["samples"]), limit // max(w, h)))  # supersampled scene must fit the GPU too
    target = r.device.create_texture(size=(w, h, 1), format="rgba8unorm",
                                     usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    background = None if image["background"] is None else tuple(image["background"])
    r.render(target, (0, 0, w, h), scene, camera, lighting, samples=k, transparent=bool(image["transparent"]),
             background=background)
    data = r.device.queue.read_texture({"texture": target}, {"bytes_per_row": w * 4}, (w, h, 1))
    img = np.frombuffer(data, np.uint8).reshape(h, w, 4).copy()
    if path is not None:
        save_png(img, path)
    return img


def save_png(img: np.ndarray, path) -> Path:
    """Write an (h, w, 4) uint8 image as PNG (RGB when it is fully opaque); returns the path."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opaque = bool((img[..., 3] == 255).all())
    Image.fromarray(img[..., :3] if opaque else img, "RGB" if opaque else "RGBA").save(path)
    return path
