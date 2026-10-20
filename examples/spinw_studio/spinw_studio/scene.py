"""The 3D scene of a spinwaves system as a ``valo.Scene``: the structure is one locked procedural group.

    Structure        ProceduralGroup(structure): lattice a, b, c (Å), alpha, beta, gamma (deg), uvw extent,
    ├─ Atoms             moment_length (Å); its children are rebuilt when one changes, keeping your edits
    │   └─ Fe, O1, …     one InstancedObject per label: Shape(sphere, r = radius), material named after the label
    ├─ Moments
    │   └─ Fe moments    arrows centred on the atoms per magnetic label, material "Moments"
    └─ Cell              the 12 unit-cell edges: Shape(cylinder, r = edge radius), material "Unit cell"

All of it is locked (hide it, edit it, but it can't be deleted); objects you add in the studio sit next to it.
A lattice change goes through ``adapter.set_lattice`` for a SpinW object (changing it in place) or
``model.with_lattice`` for a plain SystemView. Reads only the model and the adapter; never spinwaves directly.
"""

from __future__ import annotations

import numpy as np

from valo import (
    Group,
    InstancedObject,
    Material,
    ProceduralGroup,
    Renderer,
    Scene,
    Shape,
    instance_matrices,
    linear_to_srgb,
    meshes,
    node_from_settings,
    srgb_to_linear,
)

from . import adapter
from .model import (
    SystemView,
    cell_edges,
    default_color,
    default_radius,
    expand,
    lattice_params,
    parse_color,
    validate_options,
    with_lattice,
)

STRUCTURE, LATTICE = "Structure", ("a", "b", "c", "alpha", "beta", "gamma")
EXTENT = ("umin", "umax", "vmin", "vmax", "wmin", "wmax")
OWN_MATERIALS = ("Moments", "Unit cell")  # besides one per label
BASE_SHAPES = {"Atoms": meshes.sphere, "Moments": meshes.arrow, STRUCTURE: meshes.cylinder}  # by parent node name


def _linear(color) -> tuple:
    return tuple(srgb_to_linear(c) for c in parse_color(color))


def _srgb(m: Material) -> list:
    return [round(linear_to_srgb(c), 4) for c in m.get("base_color")]


def _vec(v, digits: int = 3) -> str:
    return "(" + ", ".join(f"{float(c) + 0.0:.{digits}f}" for c in v) + ")"


class StructureSource:
    """Where the structure comes from (the procedural group's context): a SpinW object or a SystemView, the current
    SystemView, and the starting looks the builder gives new children (radii, edge radius, what is shown)."""

    def __init__(self, sw_or_view, options: dict):
        self.sw = None if isinstance(sw_or_view, SystemView) else sw_or_view
        self.original = sw_or_view if self.sw is None else adapter.system_view(sw_or_view)
        self.system = self.original
        self.radii = {label: float(options["atom_radii"].get(label, default_radius(label)))
                      for label in self.system.labels}
        self.cell_radius = float(options["cell_radius"])
        self.show_moments, self.show_cell = options["show_moments"], options["show_cell"]

    def set_lattice(self, params) -> SystemView:
        """The system with lattice ``params`` (a, b, c, alpha, beta, gamma): through ``adapter.set_lattice`` for a
        SpinW object, ``with_lattice`` for a SystemView; nothing is changed when they are the current ones."""
        if np.allclose(params, lattice_params(self.system.A), rtol=0, atol=1e-9):
            return self.system
        if self.sw is None:
            same = np.allclose(params, lattice_params(self.original.A), rtol=0, atol=1e-9)
            self.system = self.original if same else with_lattice(self.original, *params)
        else:
            adapter.set_lattice(self.sw, *params)
            system = adapter.system_view(self.sw)
            if not np.allclose(lattice_params(system.A), params, rtol=1e-6, atol=1e-6):
                raise ValueError("adapter.set_lattice ran, but adapter.lattice_matrix(sw) still gives the old cell: "
                                 "set_lattice must change what lattice_matrix reads")
            self.system = system
        return self.system


def structure(group, a: float = 5.0, b: float = 5.0, c: float = 5.0, alpha: float = 90.0, beta: float = 90.0,
              gamma: float = 90.0, umin: float = -0.1, umax: float = 1.1, vmin: float = -0.1, vmax: float = 1.1,
              wmin: float = -0.1, wmax: float = 1.1, moment_length: float = 2.0):
    """The Structure builder: atoms, moment arrows and cell edges for the lattice (Å, degrees) and uvw extent."""
    source: StructureSource = group.context
    bounds = [[umin, umax], [vmin, vmax], [wmin, wmax]]
    if not all(lo < hi for lo, hi in bounds):
        raise ValueError(f"each extent needs min < max, got u {bounds[0]}, v {bounds[1]}, w {bounds[2]}")
    system = source.set_lattice((a, b, c, alpha, beta, gamma))
    expanded = expand(system, bounds)

    def atom_info(label, k):
        ex = expanded[label]
        site = system.atoms[int(ex.indices[k])]
        m = np.asarray(site.moment, float)
        rows = {"Label": label, "Cell index": str(site.index), "uvw": _vec(ex.uvw[k]),
                "Translation": _vec(ex.translations[k], 0), "xyz (Å)": _vec(ex.xyz[k]),
                "|m|": f"{np.linalg.norm(m):.3f}", "m (xyz)": _vec(m), "S": f"{site.spin:g}"}
        return {**rows, **site.extra}

    atoms = group.add(Group("Atoms", locked=True))
    moments = group.add(Group("Moments", visible=source.show_moments, locked=True))
    for label in system.labels:
        ex = expanded[label]
        atoms.add(InstancedObject(label, Shape(meshes.sphere, {"n_lat": 24, "n_lon": 48, "r": source.radii[label]}),
                                  label, instance_matrices(ex.xyz) if len(ex.xyz) else [],
                                  info=lambda k, lb=label: atom_info(lb, k), locked=True))
        norms = np.linalg.norm(ex.moments, axis=1)
        if not any(np.any(site.moment) for site in system.atoms if site.label == label):
            continue  # a non-magnetic label gets no moments object
        rows = np.flatnonzero(norms > 1e-12)  # arrow k → expanded row
        d = ex.moments[rows] / norms[rows, None]
        w, L = np.full(len(rows), 0.3 * moment_length), np.full(len(rows), moment_length)
        moments.add(InstancedObject(f"{label} moments", Shape(meshes.arrow), "Moments",
                                    instance_matrices(ex.xyz[rows] - d * moment_length / 2, np.c_[w, w, L], z_axis=d)
                                    if len(rows) else [],
                                    info=lambda k, lb=label, rows=rows: atom_info(lb, int(rows[k])), locked=True))
    e = cell_edges(system.A)
    vec = e[:, 1] - e[:, 0]
    group.add(InstancedObject("Cell", Shape(meshes.cylinder, {"r": source.cell_radius}), "Unit cell",
                              instance_matrices(e.mean(1), np.c_[np.ones(12), np.ones(12), np.linalg.norm(vec, axis=1)],
                                                z_axis=vec), visible=source.show_cell, locked=True))


def build_scene(sw_or_view, plot_options: dict | None = None, renderer: Renderer | None = None) -> Scene:
    """The valo.Scene for a SpinW object (read through the adapter) or a SystemView, set up from plot_options."""
    opts = validate_options(plot_options)
    source = StructureSource(sw_or_view, opts)
    scene = Scene(renderer)

    def material(name, full, shading, **params):
        if full:
            params = dict(full)
            shading = params.pop("shading")
        scene.material(name, shading, **params)

    for i, label in enumerate(source.system.labels):
        color = opts["atom_colors"].get(label, default_color(label, i))
        material(label, opts["materials"].get(label), "pbr", base_color=_linear(color), roughness=0.4)
    material("Moments", opts["moment_material"], "pbr", base_color=_linear(opts["moment_color"]), roughness=0.5)
    material("Unit cell", opts["cell_material"], "lambert", base_color=_linear(opts["cell_color"]))
    for name, full in opts["object_materials"].items():
        material(name, full, None)

    lattice = lattice_params(source.system.A)
    params = {**dict(zip(LATTICE, lattice, strict=True)),
              **dict(zip(EXTENT, np.ravel(opts["boundaries"]).tolist(), strict=True)),
              "moment_length": float(opts["moment_length"])}
    ranges = {**{k: (0.5 * v, 1.5 * v) for k, v in zip(LATTICE[:3], lattice[:3], strict=True)},
              **{k: (30.0, 150.0) for k in LATTICE[3:]}, **{k: (-3.0, 4.0) for k in EXTENT},
              "moment_length": (0.1, 5.0)}
    scene.add(ProceduralGroup(STRUCTURE, structure, params, ranges=ranges, context=source, locked=True))
    for name, shape in opts["shapes"].items():  # base meshes changed in the studio: kept like any edit
        node = scene.find(name)
        node.shape = Shape(shape["func"], shape.get("params"))
        node.rebuild_mesh()
    for node in opts["objects"]:
        scene.add(node_from_settings(node))
    return scene


def source_of(scene: Scene) -> StructureSource:
    """The StructureSource behind a scene made by ``build_scene`` (holds the current SystemView)."""
    return scene.find(STRUCTURE).context


def options(scene: Scene) -> dict:
    """The scene's state as plot_options: extent, looks per label, moments, cell, materials in full, and the objects
    added in the studio (camera / lighting / image are the editor's)."""
    group, source = scene.find(STRUCTURE), source_of(scene)
    p = group.params
    labels = {label: scene.find(label) for label in source.system.labels}
    arrows = [n for n in scene.find("Moments").children]
    cell = scene.find("Cell")
    moment_material = scene.materials[arrows[0].material if arrows else "Moments"]
    extra = [n for n in scene.root.children if n is not group]
    swapped = [n for n in group.walk() if hasattr(n, "shape") and n.shape.func is not BASE_SHAPES[n.parent.name]]
    return {
        "boundaries": [[float(p[lo]), float(p[hi])] for lo, hi in zip(EXTENT[::2], EXTENT[1::2], strict=True)],
        "atom_colors": {label: _srgb(scene.materials[n.material]) for label, n in labels.items()},
        "atom_radii": {label: float(n.shape.params.get("r", default_radius(label))) for label, n in labels.items()},
        "materials": {label: scene.materials[n.material].to_dict(digits=8) for label, n in labels.items()},
        "show_moments": scene.find("Moments").visible, "moment_length": float(p["moment_length"]),
        "moment_color": _srgb(moment_material), "moment_material": moment_material.to_dict(digits=8),
        "show_cell": cell.visible, "cell_radius": float(cell.shape.params.get("r", 0.03)),
        "cell_color": _srgb(scene.materials[cell.material]),
        "cell_material": scene.materials[cell.material].to_dict(digits=8),
        "shapes": {n.name: n.shape.settings() for n in swapped},
        "objects": [n.settings() for n in extra],
        "object_materials": {name: m.to_dict(digits=8) for name, m in scene.materials.items()
                             if name not in OWN_MATERIALS and name not in labels},
    }
