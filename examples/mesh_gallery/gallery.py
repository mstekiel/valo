"""Mesh gallery: every standard mesh in valo.meshes on a floor plane, with a ring of arrows, each shape
with its own editable material.

Run: ``uv run python examples/mesh_gallery/gallery.py``
"""

from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from valo import Editor, Instances, Renderer, Viewer, instance_matrices, meshes

# name, mesh, shading, parameters
SHAPES = [
    ("sphere", meshes.sphere(), "pbr", {"base_color": (0.85, 0.62, 0.12), "roughness": 0.35}),
    ("icosphere", meshes.icosphere(), "blinn_phong", {"base_color": (0.12, 0.55, 0.12), "shininess": 64.0}),
    ("cube", meshes.cube(), "pbr", {"base_color": (1.0, 0.71, 0.29), "metallic": 1.0, "roughness": 0.3}),
    ("cylinder", meshes.cylinder(), "lambert", {"base_color": (0.7, 0.2, 0.2)}),
    ("cone", meshes.cone(), "toon", {"base_color": (0.08, 0.25, 0.75), "bands": 3}),
    ("torus", meshes.torus(), "pbr", {"base_color": (0.9, 0.9, 0.92), "metallic": 1.0, "roughness": 0.15}),
    ("capsule", meshes.capsule(), "blinn_phong", {"base_color": (0.6, 0.3, 0.7), "shininess": 32.0}),
    ("arrow", meshes.arrow(), "pbr", {"base_color": (0.1, 0.6, 0.65), "roughness": 0.5}),
]


class Gallery:
    """The scene: one Instances per shape (one instance each), a floor, and a ring of arrows."""

    def __init__(self, renderer: Renderer):
        self.items: list[tuple[str, Instances]] = []
        for k, (name, mesh, shading, params) in enumerate(SHAPES):
            inst = renderer.instances(mesh, renderer.material(name.capitalize(), shading, **params))
            x, y = -2.4 + 1.6 * (k % 4), 0.9 - 1.8 * (k // 4)
            if name == "arrow":  # tail at the origin: stands on the floor
                inst.set(instance_matrices([(x, y, 0.01)], (0.5, 0.5, 1.0)))
            elif name == "torus":  # tilted so its hole shows
                inst.set(instance_matrices([(x, y, 0.5)], 1.6, z_axis=(0, -1, 1)))
            else:
                inst.set(instance_matrices([(x, y, 0.51)]))
            self.items.append((name, inst))

        ring = renderer.instances(meshes.arrow(), renderer.material("Ring arrows", "pbr",
                                                                    base_color=(0.85, 0.3, 0.1), roughness=0.4))
        a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        origins = np.c_[3.6 * np.cos(a), 3.6 * np.sin(a), np.full_like(a, 0.4)]
        vectors = np.c_[-np.sin(a), np.cos(a), 0.3 * np.sin(2 * a)]  # tangential, tilting up and down (above the floor)
        w = np.full_like(a, 0.25)
        ring.set(instance_matrices(origins, np.c_[w, w, np.linalg.norm(vectors, axis=1)], z_axis=vectors))
        self.items.append(("ring of arrows", ring))

        floor = renderer.instances(meshes.plane(), renderer.material("Floor", "lambert", base_color=(0.3, 0.3, 0.32)))
        floor.set(instance_matrices([(0, 0, 0)], (9.5, 9.5, 1)))
        self.items.append(("floor", floor))

    def batches(self):
        return [inst.batch() for _, inst in self.items]

    def bounding_radius(self) -> float:
        return 4.8


class GalleryEditor(Editor):
    title = "Mesh gallery"
    dock_bottom = ("Material",)  # under the shape list, with Camera, Lighting and History
    bottom_share = 0.6

    def __init__(self, gallery: Gallery):
        super().__init__()
        self.gallery = gallery
        self.selected = 0
        for _, inst in gallery.items:
            self.watch(inst.material)

    def windows(self):
        return {"Shapes": self.shapes_window, "Material": self.material_window}

    def shapes_window(self) -> None:
        for i, (name, inst) in enumerate(self.gallery.items):
            if imgui.selectable(name, i == self.selected)[0]:
                self.selected = i
            imgui.same_line(imgui.get_content_region_avail().x * 0.6)
            imgui.text_disabled(f"{inst.count} x {inst.mesh.count // 3} tris")

    def material_window(self) -> None:
        name, inst = self.gallery.items[self.selected]
        imgui.text(name)
        self.material_section(inst.material)

def build(viewer: Viewer) -> GalleryEditor:
    gallery = Gallery(viewer.renderer)
    editor = GalleryEditor(gallery)
    viewer.show(gallery, editor)
    viewer.camera.elevation = 35.0
    return editor


def main() -> None:
    viewer = Viewer(title="Mesh gallery")
    build(viewer)
    viewer.run()


if __name__ == "__main__":
    main()
