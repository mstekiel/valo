# Changelog

All notable changes to valo. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/) (while at 0.x, a minor version may break the API).

## [Unreleased]

## [0.1.0] - Unreleased

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

[Unreleased]: https://github.com/OWNER/valo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/valo/releases/tag/v0.1.0
