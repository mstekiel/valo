# valo — instructions for Claude Code

Interactive scientific 3D rendering engine: wgpu (rendering) + glfw (window) + Dear ImGui (docking UI), Python only.
Apps built on it live in `examples/`. Design and extension points: `ARCHITECTURE.md`. Keep both files short and current.

## Commands (uv only — never pip, venv, conda, poetry)

| | |
|---|---|
| Install | `uv sync` |
| Run the studio | `uv run valo-studio [examples/shapes/vase.py]` (procedural meshes in code; Save to library → `data/procedural_meshes/`) |
| Run the example app | `uv run --group spinw python examples/spinw_studio/lfo_spinw.py` (needs the local spinwaves) |
| Test | `uv run pytest` (engine + examples; GPU tests render offscreen and skip without an adapter); `-m "not gpu"` for the fast part, `--cov` for coverage |
| Lint / format | `uv run ruff check --fix .` |
| Add a dependency | `uv add <pkg>` (ask the user first) |

## Engine packages (`valo/`): math ← meshes ← gpu ← scene ← (offline, ui) ← studio

Lower layers never import higher ones (`tests/test_structure.py` checks it); `config.py` is used by all. Apps import
only from `valo` (and `valo.meshes`). Blender's split: data (`scene`, `meshes`), draw engine (`gpu`), editors (`ui`).

| Module | Owns | GPU? |
|---|---|---|
| `math/` | `transforms.py`: matrices (`pack_mat4`, projections, `instance_matrices`, `decompose`); `bounds.py`: `box_corners`, `bounding_box` | no |
| `meshes/` | pure numpy, import from `valo.meshes`: `data` (`MeshData`), `ops` (`transform`, `merge`), `builders` (`revolve`, `parametric`, `heightfield`, `tube`), `primitives` (`STANDARD`), `params` (`shape_params`), `check` (`MeshReport`), `shapefile` (`ShapeFile`: load / build / save a shape file), `library` (`data/procedural_meshes/`) | no |
| `gpu/` | the draw engine: `shaders` (all WGSL), `material` (`Material`, `SCHEMA`, colour conversion), `mesh` (`GpuMesh`), `instances` (`Instances`, `Spheres`, `Batch`), `renderer` (`Renderer`: passes, `pick`, `renderer()`) | yes |
| `scene/` | the data model: `camera` (`Camera`: view state, projections, `VIEWS`), `lighting` (`Lighting`), `shape` (`Shape`), `nodes` (`Transform`, `Group`, `ProceduralGroup`, `MeshObject`, `InstancedObject`, bounding boxes), `scene` (`Scene`: tree, material library, `batches()`, `settings()`) | via a Renderer |
| `offline.py` | `valo.render` (images without a window), `export_script`, `py_literal` | yes |
| `ui/` | editors: `editor` (`Editor`: dockspace, layout, menus), `windows/` (camera, lighting, info, history, render, preferences, outliner, object, materials: mixins with their actions), `overlays` (compass, hover / selection boxes, `box_layout`), `widgets`, `navigation` (`Navigator`: orbit / pan / zoom, keys, presets, transitions; not undoable), `viewer` (`Viewer`), `scene_editor` (`SceneEditor` actions, `one_step`), `undo`, `style` (`ICON`, fonts) | yes |
| `studio/` | `StudioEditor` (`valo-studio`): `editor` (actions), `code` (Code window, File menu, dialogs), `normals` (overlay), `cli` (`build`, `main`); base of domain studios | yes |
| `config.py` | `rc`: typed sections (camera, lighting, highlight, picking, ui, keys, studio), TOML files, `rc_context`, `valo-config` | no |
| `testing.py` | test helpers for valo and apps: `offscreen_viewer`, `screen_point` / `hover` / `click` at a world point, `press`, `open_every_menu`, `changed` | — |

## Example apps (`examples/<name>/`)

| App | Files |
|---|---|
| `spinw_studio` | spinwaves systems in valo: `spinw_studio/` package (`adapter`, `model`, `scene` = a locked procedural `valo.Scene`, `editor` = a `StudioEditor`, `api`), `lfo_spinw.py`, `test_spinw_studio.py`. **Read its `README.md` (the defining document) before changing it**; it needs `uv sync --group spinw` only to run the real script |

## Rules that keep it working

1. Matrices are math convention (`p' = M @ p`), Z-up. `valo.math.pack_mat4` is the only place that transposes for WGSL.
2. Depth is reversed-Z (clear 0, compare `greater`). Colour is linear everywhere; sRGB encoding happens only in the post pass.
3. Every user-facing change is a method on the app's `Editor` subclass that pushes a `Command` for undo. The windows only
   call these methods, so tests and scripts can do everything the UI can.
4. The renderer draws whatever `scene.batches()` returns: `(material, gpu_mesh, instance_bind_group, count)`
   tuples, made by `Instances.batch()`. Build scenes as a `valo.Scene` (objects with shapes, transforms and
   library materials, edited by `SceneEditor`); raw `Instances` are the low level. Meshes are CCW-outward (back faces are culled). `valo/` never imports from `examples/` and knows no domain
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
3. Every new `Editor` action gets a row in an undo table: `EDITOR_ACTIONS` (`tests/ui/test_editor.py`),
   `SCENE_ACTIONS` (`tests/scene/test_scene.py`), `STUDIO_ACTIONS` (`tests/studio/test_valo_studio.py`), or the app's
   own test. It must change state, undo exactly (state and image) and redo. `tests/test_structure.py` fails for an
   action without a row (view changes, queries and windows are on its allow-list).
4. Where tests go: `tests/` mirrors the packages (`tests/math/`, `meshes/`, `gpu/`, `scene/`, `ui/`, `studio/`, plus
   `test_config.py`, `test_offline.py`); apps → `examples/<name>/test_*.py`. Test file names are unique across the
   repo (no `__init__.py`). GPU tests take the `viewer` fixture (root `conftest.py`: one offscreen Viewer per module,
   size from `VIEWER_SIZE`), mark the module `pytestmark = pytest.mark.gpu`, and use `valo.testing` for pointer,
   keys, menus and image comparison. `tests/test_structure.py` guards the public API, the layering and the undo
   tables: change it on purpose when a rename or a new dependency is intended.
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
