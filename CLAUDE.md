# valo — instructions for Claude Code

Interactive scientific 3D rendering engine: wgpu (rendering) + glfw (window) + Dear ImGui (panel), Python only.
Apps built on it live in `examples/`. Design and extension points: `ARCHITECTURE.md`. Keep both files short and current.

## Commands (uv only — never pip, venv, conda, poetry)

| | |
|---|---|
| Install | `uv sync` |
| Run an example app | `uv run python examples/crystal_studio/studio.py` (or `mesh_gallery/gallery.py`) |
| Sculpt a mesh | `uv run valo-sculpt examples/sculpts/vase.py` (MeshSculptor; saves to `valo/meshes/library/`) |
| Test | `uv run pytest` (engine + examples; GPU tests render offscreen and skip without an adapter) |
| Lint / format | `uv run ruff check --fix .` |
| Add a dependency | `uv add <pkg>` (ask the user first) |

## Engine modules (`valo/`)

| Module | Owns | GPU? |
|---|---|---|
| `geometry.py` | matrices, `instance_matrices` (pure numpy) | no |
| `meshes/` | package, pure numpy; import everything from `valo.meshes` (its `__init__` re-exports) | no |
| `meshes/data.py` | `MeshData` (+ `_mesh`, `_grid_indices` helpers) | no |
| `meshes/ops.py` | `transform`, `merge` | no |
| `meshes/builders.py` | `revolve`, `parametric`, `heightfield` | no |
| `meshes/primitives.py` | `sphere`, `icosphere`, `cube`, `plane`, `cylinder`, `cone`, `torus`, `capsule`, `arrow`, `STANDARD` | no |
| `meshes/check.py` | `check` → `MeshReport` (winding, unit normals, watertight + volume) | no |
| `meshes/sculpt.py` | sculpt files: load (params from `mesh()`'s signature), build (transform baked), save (AST rewrite of defaults, `MATERIAL`, `TRANSFORM`) | no |
| `meshes/library/` | saved sculpts, one `<name>.py` each; `names()`, `mesh(name, **params)`, `material(renderer, name)` | no |
| `tools/sculptor.py` | MeshSculptor app (`valo-sculpt`): `SculptScene`, `SculptorEditor` with hot reload, diagnostics, save | yes |
| `materials.py` | shading models (WGSL), parameter schema, observable `Material`, colour conversion | buffer only |
| `render.py` | shared WGSL, pipelines, render targets, `OrbitCamera`, `Instances` (mat4 per instance), `Spheres`, scene pass → post pass | yes |
| `undo.py` | undo/redo stack | no |
| `ui.py` | `Editor` base: undo, docked panel, material widgets, View/History sections | no |
| `app.py` | `Viewer`: canvas, device, renderer, camera, ImGui, mouse; `show(scene, editor)`, `run()` | yes |

## Example apps (`examples/<name>/`)

| App | Files |
|---|---|
| `crystal_studio` | `crystal.py` (rock-salt model, species, material library), `studio.py` (`CrystalEditor` + `main`), `test_studio.py` |
| `mesh_gallery` | `gallery.py` (every standard mesh + a ring of arrows, `GalleryEditor` + `main`), `test_gallery.py` |

## Rules that keep it working

1. Matrices are math convention (`p' = M @ p`), Z-up. `geometry.pack_mat4` is the only place that transposes for WGSL.
2. Depth is reversed-Z (clear 0, compare `greater`). Colour is linear everywhere; sRGB encoding happens only in the post pass.
3. Every user-facing change is a method on the app's `Editor` subclass that pushes a `Command` for undo. The panel only
   calls these methods, so tests and scripts can do everything the UI can.
4. The renderer draws whatever `scene.batches()` returns: `(material, gpu_mesh, instance_bind_group, count)`
   tuples, made by `Instances.batch()`. Meshes are CCW-outward (back faces are culled). `valo/` never imports from `examples/` and knows no domain
   (crystals, molecules, …); apps use only the public API in `valo/__init__.py`.
5. Prefer the smallest change that works. Don't add layers, plugins or abstractions before a feature needs them;
   when it does, update `ARCHITECTURE.md` in the same change.

## Workflow for a feature request

1. For anything touching more than one module, use plan mode first and show the plan.
2. Write or extend a test: engine → `tests/test_core.py` (pure) or `tests/test_app.py` (offscreen);
   app → `examples/<name>/test_*.py`.
3. Implement. Run `uv run ruff check --fix . && uv run pytest` until green.
4. If the change is visual, render offscreen and look at the image before saying it's done.
5. Commit with a short conventional message (`feat:`, `fix:`, `refactor:`). Never push.
