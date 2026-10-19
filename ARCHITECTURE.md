# valo architecture (skeleton)

Grown from the wgpu-py exploration examples (example 06, "crystal studio"). `valo/` is a standalone
engine; apps such as `examples/crystal_studio/` are built on its public API (`valo/__init__.py`).
Deliberately small: structure is added when a feature needs it, not before.

## Writing an app

```python
from types import SimpleNamespace
from valo import Editor, Viewer, instance_matrices, meshes

viewer = Viewer(title="my app")                       # window, device, renderer, camera, ImGui
gold = viewer.renderer.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
balls = viewer.renderer.spheres(gold)                 # instanced spheres, one draw call
balls.set(centers, radii)                             # (N, 3) and a scalar or N values
arrows = viewer.renderer.instances(meshes.arrow(), gold)   # any MeshData, one draw call
arrows.set(instance_matrices(origins, np.c_[w, w, lengths], z_axis=vectors))
editor = Editor(); editor.watch(gold)                 # material edits now undoable
viewer.show(SimpleNamespace(batches=lambda: [balls.batch(), arrows.batch()]), editor)
viewer.run()
```

Meshes live in the `valo/meshes/` package; import everything from `valo.meshes`.
- **`primitives.py`**: the standard meshes, all unit-sized, Z-up and CCW outward. They are `sphere`, `icosphere`, `cube`,
  `cylinder`, `cone`, `torus`, `capsule`, `arrow` (tail at origin, tip at z=1) and `plane` (double-sided).
  `examples/mesh_gallery/` shows them all.
- **`ops.py`**: `transform` and `merge`.
- **`builders.py`**: `revolve`, `parametric` and `heightfield`.
- **`check.py`**: diagnostics.
- **`sculpt.py`** and **`library/`**: the sculpting workflow below.

## Sculpting meshes into the library

```
your_shape.py ──valo-sculpt──▶ MeshSculptor ──Save──▶ valo/meshes/library/<name>.py ──▶ library.mesh("<name>")
 mesh(a=1.0, n=32)              sliders from the       same file, tuned values     Sculpt.build(): mesh(**params)
 RANGES / MATERIAL / TRANSFORM  signature, transform,  as defaults + MATERIAL +    with TRANSFORM baked in
                                material, check()      TRANSFORM (AST rewrite)
```
A sculpt file is plain Python (contract in `valo/meshes/sculpt.py`): a `mesh(...)` function whose parameters
all have defaults (`float`, `int`, `bool`), optionally `RANGES`, `MATERIAL`, `TRANSFORM`. Build shapes with
`meshes.revolve` ((ρ, z) profile; repeat a point for a hard edge), `meshes.parametric` (P(u, v), outward is
∂u × ∂v) or `meshes.heightfield` (z = f(x, y)). `meshes.check` (winding, unit normals, watertight + volume)
drives the Diagnostics panel and the tests, which run it on every library entry. Preview and library share
`Sculpt.build`, so what you see is what's saved. Starter sculpts: `examples/sculpts/`.

A real app subclasses `Editor`: its actions are methods that push `Command`s, and `windows()` returns its
dockable windows as `{title: draw_function}`; `material_section(m)` gives the schema-driven material editor.
Optional: `menus()` adds menus (e.g. File) before Edit and View; `dock_bottom` / `bottom_share` / `right_column`
shape the default layout. The base class draws the menu bar (Edit: undo/redo, View: window toggles, auto-rotate,
UI scale, Reset layout), the History window and the dockspace; the scene shows through the empty **central dock
node**. The layout is saved to `~/.valo/<title>.ini` (real windows only, also on close). A scene with
`bounding_radius()` gets the camera fitted on `show`.

## Camera and lighting

- **`OrbitCamera`** (`render.py`): azimuth/elevation/distance around a target; `projection` is "perspective" or
  "orthographic" (`set_projection` keeps the target the same size on screen; `zoom` scales distance or
  `ortho_height`); `VIEWS` has ±X/±Y/±Z/Iso presets.
- **`Lighting`** (`lighting.py`, `viewer.lighting`): one directional light, a **headlight** (angles in the camera's
  frame, so it moves with you) or **fixed** (world angles); `set_mode` converts so it never jumps. Plus a **global
  hemisphere ambient**: sky colour, ground colour and an intensity shared by every object. The same sky/ground
  is what PBR materials reflect. A material's `ambient` only scales that global ambient (0–1, default 1).
- The Editor's shared **Camera** and **Lighting** windows change these through undoable actions; mouse orbit/zoom
  is navigation and is not recorded.

Frame uniform (group 0, 160 bytes): `view_proj` mat4 · `eye` · `view_dir` (w = 1 perspective, 0 orthographic:
shading uses a constant view vector) · `light_dir` · `light_color` (× intensity) · `sky` (w = ambient intensity)
· `ground`.

Look: `valo/ui_style.py` loads Roboto with Font Awesome icons merged in (`ICON.FLOPPY_DISK` etc.) and
Inconsolata (`mono_font()`), and applies the theme (`apply_style(scale)`). ImGui 1.92 rasterises them at the
real pixel density, so text is crisp at any DPI.

## Configuration: `valo.rc`

`valo/config.py` holds every tunable as typed sections with defaults in code: camera, lighting, highlight,
picking, ui, keys, sculptor. Read them as `rc.camera.orbit_speed` or `rc["camera.orbit_speed"]`; every set is
validated (type, range, choices; typos get a "did you mean"). Files override the defaults, in this order:
`~/.valo/valorc.toml` → `./valorc.toml` → `$VALO_RC`. `valo-config --write-defaults` writes a commented file of
every key. View → Preferences edits `rc` live (generated from the schema) and saves the changed keys.
`rc_context({...})` changes keys temporarily (tests). `rc` holds **defaults and preferences**; live state stays on
the objects, so `Lighting()` starts from `rc.lighting`, but the Lighting window edits `viewer.lighting`.

## Picking, Info and navigation

- **Picking:** `Renderer.pick(viewport, scene, camera, x, y)` draws (batch + 1, instance, depth) into an
  `rgba32uint` target inside a 1-pixel scissor and reads that pixel back. It returns a `Pick` (batch_index,
  instance_index, source, world_point), or None for the background. The Viewer picks under the pointer every
  frame (`viewer.hover`); a click sets `viewer.selection`.
- **Batches:** `Instances.batch()` returns a `Batch` NamedTuple `(material, mesh, bind_group, count, source)`.
  Plain 4-tuples still draw, but can't be described. `Instances` keep `matrices` (CPU copy), `label` and
  `info(i) -> dict`.
- **Info window:** `ui.pick_info(pick, scene)` gives:
  - object, material, instance, mesh size;
  - translation / rotation / scale from `geometry.decompose`;
  - the point hit;
  - then `scene.describe(pick)` or `source.info(i)`.
- **Highlight:** the hovered and selected instances are drawn again over the scene, with an additive tint + rim
  pipeline at the same depth (`@invariant` position, `greater-equal`). Colour and strength come from `rc.highlight`.
- **Navigation** (`Viewer.on_pointer` / `on_key`, keys in `rc.keys`):
  - mouse: left-drag orbit, middle- or Shift-drag pan, wheel zoom, click select, double-click centre on the point;
  - keys: F frame selection, Home fit all, 1/3/7 (+Ctrl) views, 5 projection, arrows orbit (Shift pan), Esc deselect.

  `OrbitCamera.center_on` animates over `rc.camera.center_duration`. Selection and navigation are view state
  (not undoable); view presets and projection go through the Editor (undoable).

## Scripting and offline rendering

- **One GPU:** there is one GPU device and one `Renderer` per process (`valo.renderer()`); every `Viewer` uses it,
  so a scene built in a script can be shown in a window and the reverse.
- **`valo.render(scene, settings=None, *, size, samples, transparent, background, camera, lighting, path)`**
  (`offline.py`) renders without a window into an `rgba8unorm` texture and returns (h, w, 4) uint8 sRGB.
  Given a `path`, it saves a PNG.
  - **Supersampling:** the scene target is `samples`× larger and the post pass averages each block, premultiplied
    by alpha.
  - **Transparency:** the scene clears to alpha 0, and the post pass passes alpha through.
  - **Defaults:** missing options come from `rc.render` / `rc.lighting`, and the camera is fitted to the scene.
- **`settings`** is plain data: `{"scene": ..., "camera": OrbitCamera.settings(), "lighting": Lighting.settings(),
  "image": {"size", "samples", "transparent", "background"}}`.
- **The scene settings protocol:** a scene that implements `settings() -> dict` and the classmethod
  `from_settings(d, gpu=None)` can be rebuilt from data. The Editor's File → Render window then exports a
  **runnable script**: `SETTINGS` written by `offline.py_literal`, plus a main that calls `from_settings` and
  `valo.render`. Its picture is pixel-identical to the studio's Render PNG. Any app can Render PNG; Export needs
  the protocol. `examples/crystal_studio/crystal.py`'s `__main__` shows scripting from scratch.

## One frame

```
Viewer.draw()
 ├─ Renderer.render(surface_texture, viewport=(x, y, w, h), scene, camera)   rect = editor's central dock node
 │    ├─ write frame uniform (camera, light, ambient)              group 0
 │    ├─ flush dirty materials (64-byte uniform each)              group 1
 │    ├─ scene pass → rgba16float texture, one instanced draw per batch
 │    │     batch = Batch(material, gpu mesh, instance bind group, count, source)   group 2 = mat4x4f per instance
 │    │     (vs_main: world = M·p, normal = cofactor(M)·n, so non-uniform scale shades correctly)
 │    ├─ highlight: hovered / selected instance drawn again, additive tint + rim, same depth
 │    └─ post pass → encode linear → sRGB into the viewport region of the window
 ├─ Renderer.pick(...) under the pointer → viewer.hover (1-pixel id pass, read back)
 └─ ImguiRenderer.render()  → editor.gui(): menu bar, dockspace, windows on top (skipped without an editor)
    (if that frame moved the central node, scene + UI are drawn once more, so the scene never lags the layout)
```

## Dependencies

```
engine:  geometry, materials, undo              (leaves: numpy only, materials holds a GPU buffer)
         meshes/  data ← ops ← builders ← primitives;  data ← check;  ops ← sculpt ← library
                  (sculpt → geometry, library → materials)
         render  → geometry, materials, meshes.data, meshes.primitives
         ui      → materials, undo
         app     → render, ui
tools:   tools/sculptor → app, meshes  (the `valo-sculpt` command)
apps:    examples/crystal_studio: studio → crystal → valo
         examples/mesh_gallery:   gallery → valo
```
`render` draws anything with a `batches()` method; nothing in `valo/` imports from `examples/`.

## Extension points (where a typical feature request goes)

| Request | Change |
|---|---|
| New shading model | `valo/materials.py`: WGSL in `FRAGMENT_WGSL` + parameter list in `SCHEMA`. Panels and pipelines pick it up. |
| New mesh shape | sculpt it: `uv run valo-sculpt my_shape.py`, then Save to library (→ `valo/meshes/library/`, checked by the tests). A primitive every app needs goes in `valo/meshes/primitives.py` + `STANDARD`. |
| New mesh builder | `valo/meshes/builders.py`, next to `revolve` / `parametric` / `heightfield`; test it with `check()`. |
| Bonds, vector fields | app: `renderer.instances(meshes.cylinder() / arrow(), m)` + `instance_matrices(..., z_axis=...)`. |
| Polyhedra, isosurfaces (per-scene meshes) | app: build a `MeshData` and draw it with one identity instance. |
| Animation (vibrations, precession) | engine: time in the frame uniform; per-instance amplitudes; displacement in `vs_main`. |
| Object properties in Info | app: `renderer.instances(..., label=, info=lambda i: {...})`, or `scene.describe(pick)`. |
| A new tunable | add a field to a section in `valo/config.py` (default, label, range, unit, help); read `rc.<section>.<key>`. |
| Image export | done: `valo.render` / File → Render. Video: render frames with `valo.render` in a loop. |
| Scripting / export for an app's scene | app: `settings()` + `from_settings(d, gpu=None)` on the scene class (in an importable module). |
| New panel or window | app: an action method on its `Editor` subclass (with undo) + an entry in `windows()`. |
| New crystal structure (CIF, POSCAR, other lattice), more species | `examples/crystal_studio/crystal.py`: centres + species per site; keep `batches()`. |
| A new app | `examples/<name>/` with a scene, an `Editor` subclass, a `main()` and a test. |
| A real application (domain library → valo) | follow `examples/spinw_studio/`: an adapter turning the library's objects into plain data, a pure model, then scene / editor on valo's public API (its README states the layering rules). |

## Parked ideas (pull in only when a feature needs them)

From the earlier v2 plan, kept here so they aren't lost: ray-cast impostor spheres for >10k atoms,
order-independent transparency, keyframe tracks and a clock, golden-image tests, a Rust/wgpu core
behind the same Python API, multi-agent workflow with worktrees. Each one is a feature request
when its time comes.
