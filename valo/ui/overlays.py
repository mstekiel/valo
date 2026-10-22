"""What the editor draws over the 3D view: the corner compass (ImGui, on top) and the bounding boxes of the selected
and hovered objects (overlay lines in the scene pass, so objects in front hide them), plus the pure layout functions
behind them (``compass_layout``, ``box_edges``, ``box_layout``)."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ..config import rc
from ..gpu import Lines, srgb_to_linear

BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]  # corner index 4x+2y+z


def compass_layout(camera, axes, center, size: float) -> list[tuple]:
    """Where to draw each compass axis: (label, start, end, rgb, toward_viewer), back-to-front.

    ``axes`` are (label, direction xyz, rgb[, length share]) as from ``Editor.compass_axes``; the optional share
    (default 1) draws an axis shorter, e.g. so crystal axes parallel to x, y, z stay readable. ``center`` and
    ``size`` are in logical pixels. A direction is projected onto the camera's right / up vectors (screen y points
    down), so the gizmo turns with the view; axes pointing away from the viewer come first, so nearer ones are drawn
    on top.
    """
    right, up = camera.right_up()
    forward = camera.forward()
    out = []
    for label, direction, rgb, *share in axes:
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) or 1.0) * (share[0] if share else 1.0)
        end = (center[0] + size * float(d @ right), center[1] - size * float(d @ up))
        out.append((label, tuple(center), end, tuple(rgb), float(d @ forward) < 0))
    return sorted(out, key=lambda a: a[4])


def box_edges(camera, box) -> list[tuple]:
    """The 12 edges of box matrix ``box`` (unit cube [-0.5, 0.5]^3 → world) as world segments (start, end, back),
    with ``back`` True for the edges behind the box as ``camera`` sees it (both faces meeting there turn away)."""
    from ..math import box_corners

    box = np.asarray(box, float)
    corners = (box @ np.c_[box_corners([-0.5] * 3, [0.5] * 3), np.ones(8)].T).T[:, :3]
    eye = np.asarray(camera.eye(), float)
    axes = box[:3, :3].T  # rows: the box's x, y, z edge vectors in world space
    centre = box[:3, 3]
    normals = [np.cross(axes[(a + 1) % 3], axes[(a + 2) % 3]) for a in range(3)]  # outward of the +a face
    perspective = camera.projection == "perspective"

    def away(axis, side):  # does the face on ``side`` (±1) of ``axis`` turn away from the camera?
        n = side * normals[axis]
        towards = (centre + side * axes[axis] / 2 - eye) if perspective else camera.forward()
        return float(n @ towards) > 1e-12

    out = []
    for a, b in BOX_EDGES:
        axis = 2 - (a ^ b).bit_length() + 1  # the bit that differs: 4 → x, 2 → y, 1 → z
        others = [k for k in range(3) if k != axis]
        sides = [1 if (a >> (2 - k)) & 1 else -1 for k in others]
        out.append((corners[a], corners[b], all(away(k, s) for k, s in zip(others, sides, strict=True))))
    return out


def box_layout(camera, box, rect) -> list[tuple]:
    """The 12 edges of box matrix ``box`` (unit cube [-0.5, 0.5]^3 → world) as drawn by ``camera`` into ``rect`` =
    (x, y, w, h) logical pixels: (start, end, back) with ``back`` as in ``box_edges``. Edges are clipped at the
    camera's near side (perspective)."""
    x, y, w, h = rect
    view_proj, _ = camera.matrices(w / h)
    perspective = camera.projection == "perspective"

    def screen(c):
        return (x + (c[0] / c[3] + 1) / 2 * w, y + (1 - c[1] / c[3]) / 2 * h)

    out = []
    near = 1e-4
    for start, end, back in box_edges(camera, box):
        p, q = view_proj @ np.r_[start, 1.0], view_proj @ np.r_[end, 1.0]
        if perspective:
            if p[3] < near and q[3] < near:
                continue  # behind the camera
            if p[3] < near or q[3] < near:  # clip at the near side
                t = (near - p[3]) / (q[3] - p[3])
                cut = p + t * (q - p)
                p, q = (cut, q) if p[3] < near else (p, cut)
        out.append((screen(p), screen(q), back))
    return out


class Overlays:
    """Editor part: the compass and the selection / hover boxes, drawn over the 3D view."""

    def draw_compass(self) -> None:
        """The axis gizmo in a corner of the 3D view (``rc.ui.compass``), drawn over the scene by ImGui."""
        if not (rc.ui.compass and self.viewer and self.scene_rect):
            return
        x, y, w, h = self.scene_rect
        size, margin = rc.ui.compass_size, rc.ui.compass_size + 18.0
        cx = x + margin if "left" in rc.ui.compass_corner else x + w - margin
        cy = y + h - margin if "bottom" in rc.ui.compass_corner else y + margin
        draw = imgui.get_foreground_draw_list()
        for label, start, end, rgb, near in compass_layout(self.viewer.camera, self.compass_axes(), (cx, cy), size):
            col = imgui.get_color_u32(imgui.ImVec4(*rgb, 1.0 if near else 0.45))
            draw.add_line(imgui.ImVec2(*start), imgui.ImVec2(*end), col, 2.5 if near else 1.5)
            dx, dy = end[0] - start[0], end[1] - start[1]
            n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            draw.add_text(imgui.ImVec2(end[0] + 7 * dx / n - 4, end[1] + 7 * dy / n - 8), col, label)

    def box_lines(self) -> Lines | None:
        """The bounding boxes of the selected and hovered objects (``viewer.boxes()``) as overlay lines for the
        renderer, which draws them against the scene's depth (objects in front hide them): ``rc.highlight`` colours
        (sRGB, converted to linear) and width (logical px), back edges fainter. None when there is nothing to box."""
        if not self.viewer:
            return None
        h = rc.highlight
        segments, colors = [], []
        for box, kind in self.viewer.boxes():
            rgb = [srgb_to_linear(c) for c in (h.selected_color if kind == "selected" else h.hover_color)]
            for start, end, back in box_edges(self.viewer.camera, box):
                segments.append((start, end))
                colors.append((*rgb, h.back_alpha if back else 1.0))
        if not segments:
            return None
        return Lines(np.array(segments), np.array(colors), h.box_width * self.viewer.canvas.get_pixel_ratio())
