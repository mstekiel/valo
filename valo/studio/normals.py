"""The normals overlay: arrows along the vertex normals of the selected object, switched on in the Object window."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ..config import rc
from ..math import instance_matrices
from ..scene import MeshObject
from ..ui.widgets import SliderSpec


class NormalsOverlay:
    """StudioEditor part: the normals overlay and its switch in the Object window."""

    def update_normals(self) -> None:
        """Arrows along the vertex normals of the selected object (every instance, at most
        ``rc.studio.max_normal_arrows``), ``normal_length`` long; nothing when ``show_normals`` is off."""
        obj = self.selected
        if not (self.show_normals and isinstance(obj, MeshObject) and obj.shown() and obj.instances.count):
            if self.normals.count:
                self.normals.set(np.zeros((0, 4, 4)))
            self._normal_key = None
            return
        inst = obj.instances
        key = (id(obj), id(inst.mesh), inst.matrices.tobytes(), self.normal_length, rc.studio.max_normal_arrows)
        if key == self._normal_key:
            return
        self._normal_key = key
        if self._normal_data is None or self._normal_data[0] is not inst.mesh:
            self._normal_data = (inst.mesh, obj.shape.build())
        data = self._normal_data[1]
        n_inst, n_vert = len(inst.matrices), len(data.positions)
        pick = np.linspace(0, n_inst * n_vert - 1, min(n_inst * n_vert, rc.studio.max_normal_arrows)).astype(int)
        rows = np.c_[pick // n_vert, pick % n_vert]
        m = inst.matrices[rows[:, 0]]
        p = np.einsum("nij,nj->ni", m[:, :3, :3], data.positions[rows[:, 1]]) + m[:, :3, 3]
        normal_m = np.linalg.inv(m[:, :3, :3]).transpose(0, 2, 1)  # normals transform with the inverse transpose
        n = np.einsum("nij,nj->ni", normal_m, data.normals[rows[:, 1]])
        w, L = np.full(len(p), 0.12 * self.normal_length), np.full(len(p), self.normal_length)
        self.normals.set(instance_matrices(p, np.c_[w, w, L], z_axis=n))
        self._normal_rows = rows

    def normal_info(self, i: int) -> dict:
        """Info rows for normal arrow ``i``: the instance and vertex it starts at, and that vertex's normal."""
        k, v = (int(c) for c in self._normal_rows[i])
        n = self._normal_data[1].normals[v]
        return {"Instance": str(k), "Vertex": str(v), "Normal": "(" + ", ".join(f"{c + 0.0:.3f}" for c in n) + ")"}

    def object_window(self) -> None:
        """SceneEditor's Object window plus the normals overlay switch for mesh objects."""
        super().object_window()
        if isinstance(self.selected, MeshObject):
            _, self.show_normals = imgui.checkbox("Show normals", self.show_normals)
            if self.show_normals:
                spec = SliderSpec(0.01, 0.5, 0.08, hard_lo=1e-4)
                _, self.normal_length = self.slider("normals.length", "Normal length", self.normal_length, spec)
