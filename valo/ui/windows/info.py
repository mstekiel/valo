"""The shared Info window: the properties of the object under the pointer (or the selected one), from
``pick_info``."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ...math import axis_angle, decompose, euler_xyz


def _vec(v, digits: int = 3) -> str:
    return "(" + ", ".join(f"{float(c) + 0.0:.{digits}f}" for c in v) + ")"  # + 0.0: no "-0.000"


def pick_info(pick, scene) -> dict[str, str]:
    """Human-readable properties of a picked object (the Info window's rows).

    Object, material, instance number, mesh size, the instance's translation / rotation / scale decomposed
    from its matrix, the point under the mouse, then scene-specific rows from ``scene.describe(pick)`` or,
    failing that, the Instances' ``info(i)`` callback.
    """
    src, i = pick.source, pick.instance_index
    rows: dict[str, str] = {}
    if src is not None:
        rows["Object"] = src.label or src.material.name
        rows["Material"] = f"{src.material.name} ({src.material.shading})"
        rows["Instance"] = f"{i + 1} of {src.count}"
        rows["Mesh"] = f"{src.mesh.count // 3} triangles"
        if i < len(src.matrices):
            t, rotation, s = decompose(src.matrices[i])
            axis, angle = axis_angle(rotation)
            rows["Translation"] = _vec(t)
            rows["Rotation"] = f"{angle:.1f} deg about {_vec(axis, 2)}" if angle > 1e-3 else "none"
            rows["Euler XYZ"] = _vec(euler_xyz(rotation), 1) + " deg"
            rows["Scale"] = f"{s[0]:.3f}" if np.allclose(s, s[0], rtol=1e-4) else _vec(s)
    else:
        rows["Object"] = f"batch {pick.batch_index}, instance {i}"
    rows["Point"] = _vec(pick.world_point)
    extra = scene.describe(pick) if hasattr(scene, "describe") else (src.info(i) if src and src.info else {})
    rows.update({str(k): str(v) for k, v in (extra or {}).items()})
    return rows


class InfoWindow:
    """Editor part: the Info window."""

    def info_window(self) -> None:
        """Properties of the hovered object, or of the selected one when nothing is hovered."""
        viewer = self.viewer
        pick, what = viewer.hover, "Hovered"
        if pick is None and getattr(viewer, "selection", None) is not None:
            pick, what = viewer.selection, "Selected"
        if pick is None:
            imgui.text_disabled("Point at an object in the 3D view")
            return
        imgui.text_disabled(what)
        if imgui.begin_table("info", 2):
            imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            for key, value in pick_info(pick, viewer.scene).items():
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(key)
                imgui.table_next_column()
                imgui.text_wrapped(value)
            imgui.end_table()
