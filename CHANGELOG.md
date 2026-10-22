# Changelog

All notable changes to valo. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/) (while at 0.x, a minor version may break the API).

## [Unreleased]

### Added
- Messages at the mouse (`editor.notify`: info / warning / error) and a read-only **Console** window with every
  message and its detail; what shape files print shows there.
- Blender-like sliders everywhere: filled bar, drag in steps, soft / hard limits, Esc to cancel, right-click menu
  (reset, copy / paste, range and step). Save to library keeps changed ranges in `RANGES`, which may give a step.
- Window icon (`valo/assets/icons/`, `Viewer(icon=...)`) and a dark title bar on Windows 11 (`rc.ui.title_bar`).
- Preferences: Expand all / Collapse all.
- Mirrored instances: instance matrices with det < 0 (mirrors, inversion) draw right side out (`Batch.mirrored`);
  picks and highlights keep your instance order.
- spinw_studio: the cell is built from unique sites and symmetry operations (adapter stubs `atom_operation`,
  `moment_operation`, `atom_generators`); atoms are drawn with their operation.

### Changed
- An `InstancedObject`'s Transform now acts on each instance's mesh (parent world · M_i · T, like Blender's
  instancing), not on the set as a whole.
- Instanced objects are boxed instance by instance: the selection shows a box per instance, hover the one under the
  pointer.
- Font size and UI scale apply live (between frames; in Preferences when the slider is released); font size in
  0.5 px steps.

### Fixed
- Bounding boxes no longer show through objects in front of them (drawn in the scene pass, depth-tested).
- GPU tests no longer skip silently when the renderer fails to build (only without a GPU adapter).
- Preferences → Scene background raised "Missing End()" (the image size had no widget).
- The test suite no longer reads the user's `valorc.toml`.

## [0.1.0] - 2026-10-21

First public release.

### Added
- **Engine** (`valo/`, layered packages `math` ← `meshes` ← `gpu` ← `scene` ← (`offline`, `ui`) ← `studio`):
  - wgpu renderer with reversed-Z depth, linear colour, instanced drawing, GPU picking, PBR and other shading models;
  - `valo.Scene`: an object tree (groups, procedural groups, mesh and instanced objects) with a material library,
    settings round trip and hot reload of shape files;
  - `valo.meshes`: primitives, `revolve`, `parametric`, `heightfield`, `tube`, mesh checks, shape files and the
    procedural mesh library (built-in meshes shipped with valo, plus your own folder for saves);
  - `valo.render`: offline images of any size, with supersampling, transparency and background, plus exported
    render scripts.
- **Editors**: Dear ImGui docking UI with undo/redo for every action; Scene, Object, Materials, Camera, Lighting,
  Info, History, Render and Preferences windows; hover and selection bounding boxes; corner compass; Blender-like
  navigation (orbit, pan, zoom, view presets, framing); view changes are not undoable.
- **valo-studio**: procedural meshes in code, with a Code window, the system file dialog, a normals overlay and
  Save to library.
- **Configuration**: `valo.rc` with typed sections, TOML files, a Preferences window and `valo-config`.
- **Example**: `examples/spinw_studio`, spinwaves magnetic structures in valo.
- **Testing**: `valo.testing` helpers, an offscreen `viewer` fixture, and structure tests for the public API,
  the layering and undo coverage.

[Unreleased]: https://github.com/mstekiel/valo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mstekiel/valo/releases/tag/v0.1.0
