# valo — instructions for Claude Code

Interactive scientific 3D rendering engine: wgpu (rendering) + glfw (window) + Dear ImGui (docking UI), Python only.
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
| `render.py` | shared WGSL, pipelines, render targets, `OrbitCamera` (projections, `VIEWS`, pan, `center_on`), `Instances` / `Batch` / `Spheres`, scene pass → highlight → post pass, `pick` | yes |
| `config.py` | `rc`: typed config sections (camera, lighting, highlight, picking, ui, keys, sculptor), TOML files, `rc_context`, `valo-config` | no |
| `offline.py` | `valo.render` (offline images: size, supersampling, transparency, background), `export_script`, `py_literal` | yes |
| `lighting.py` | `Lighting`: headlight or fixed directional light, global hemisphere ambient (sky, ground, intensity) | no |
| `undo.py` | undo/redo stack | no |
| `ui.py` | `Editor` base: undo, `windows()` / `menus()`, dockspace + default layout, menu bar, History, material widgets | no |
| `ui_style.py` | fonts (Roboto + Font Awesome `ICON`, `mono_font()`), theme `apply_style(scale)` | no |
| `app.py` | `Viewer`: canvas, shared device/renderer, camera, ImGui, mouse; `show(scene, editor)`, `run()` | yes |

## Example apps (`examples/<name>/`)

| App | Files |
|---|---|
| `crystal_studio` | `crystal.py` (rock-salt model, species, material library), `studio.py` (`CrystalEditor` + `main`), `test_studio.py` |
| `mesh_gallery` | `gallery.py` (every standard mesh + a ring of arrows, `GalleryEditor` + `main`), `test_gallery.py` |

## Rules that keep it working

1. Matrices are math convention (`p' = M @ p`), Z-up. `geometry.pack_mat4` is the only place that transposes for WGSL.
2. Depth is reversed-Z (clear 0, compare `greater`). Colour is linear everywhere; sRGB encoding happens only in the post pass.
3. Every user-facing change is a method on the app's `Editor` subclass that pushes a `Command` for undo. The windows only
   call these methods, so tests and scripts can do everything the UI can.
4. The renderer draws whatever `scene.batches()` returns: `(material, gpu_mesh, instance_bind_group, count)`
   tuples, made by `Instances.batch()`. Meshes are CCW-outward (back faces are culled). `valo/` never imports from `examples/` and knows no domain
   (crystals, molecules, …); apps use only the public API in `valo/__init__.py`.
5. Prefer the smallest change that works. Don't add layers, plugins or abstractions before a feature needs them;
   when it does, update `ARCHITECTURE.md` in the same change.
6. Every public module, class, function and method has a docstring: at least one sentence saying what it does or
   returns, with units and conventions where they matter (degrees, linear vs sRGB, logical vs framebuffer pixels).
   Ruff's `D1` rules enforce it; tests and saved library entries are exempt. Private helpers (`_name`) get one
   when the code isn't obvious.
7. Tunables (speeds, sizes, colours, default settings, key bindings) live in `valo.rc` (`config.py`), never as
   module constants or literals in code. A new one gets a field with label, range, unit and help, which also
   gives it a Preferences widget. `rc` holds defaults; live state stays on objects (camera, lighting).

## Testing (enough to trust a change, no more)

1. Test behaviour through the public entry point: the function, or the `Editor` action the UI calls. Assert a
   result, not just that the code ran.
2. A bug fix comes with a test that fails without the fix. Check it once by reverting the fix.
3. Every new `Editor` action gets a row in an undo table (`EDITOR_ACTIONS` in `tests/test_app.py` for shared
   actions, the app's own test otherwise). It must change state, undo exactly (state and image) and redo.
4. Where tests go:
   - pure engine code → `tests/test_core.py`, meshes and sculpts → `tests/test_meshes.py`;
   - offscreen GPU → `tests/test_app.py` and `tests/test_sculptor.py` (share the module-scoped `viewer`, use
     `viewer.snapshot()`);
   - apps → `examples/<name>/test_*.py`.
5. Prefer `parametrize` or a table over copied tests. No tests of ImGui widget layout, and no golden images
   (GPU drivers differ): compare images only with each other (changed / restored exactly).
6. Keep the whole suite under ~30 s. Every public function should be called by some test; `main()` / `run()`
   (they need a real window) are the exceptions.

## Workflow for a feature request

1. For anything touching more than one module, use plan mode first and show the plan.
2. Write or extend a test (see Testing).
3. Implement. Run `uv run ruff check --fix . && uv run pytest` until green.
4. If the change is visual, render offscreen and look at the image before saying it's done.
5. Commit with a short conventional message (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`). Never push.
   Keep commits small: at most 3–5 files each. Split a feature into a series of commits, each one coherent and
   green on its own (e.g. engine change + its test → app using it → docs). Stage with explicit paths, not `git add -A`.
