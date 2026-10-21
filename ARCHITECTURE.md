# valo architecture (skeleton)

Grown from the wgpu-py exploration examples. `valo/` is a standalone engine; apps such as
`examples/spinw_studio/` are built on its public API (`valo/__init__.py`).
Deliberately small: structure is added when a feature needs it, not before.

## Package layout

```
valo/
  math/        transforms (matrices, projections, decomposition), bounds (box matrices)     pure numpy
  meshes/      MeshData, builders, primitives, check, shape files (ShapeFile), the library  pure numpy
  gpu/         the draw engine: shaders (WGSL), Material, GpuMesh, Instances, Renderer      wgpu
  scene/       the data model: Camera, Lighting, Shape, nodes, Scene                        GPU via a Renderer
  offline.py   valo.render (images without a window), render scripts
  ui/          editors: Editor + windows/ (mixins), overlays, navigation, Viewer, SceneEditor
  studio/      the base studio app (valo-studio): StudioEditor, Code window, normals, CLI
  config.py    rc, used by every layer                 testing.py   helpers for tests of valo and its apps
```
- **Layering:** `math` ← `meshes` ← `gpu` ← `scene` ← (`offline`, `ui`) ← `studio`. A layer imports only from
  lower ones; `tests/test_structure.py` checks it, along with the public API names and signatures.
- **Blender's model:**
  - data is `scene` and `meshes`;
  - the draw engine is `gpu`, with the overlays in `ui/overlays`;
  - the editors are `ui` (the windows are Blender's Outliner / Properties / Text editor / Preferences);
  - the operators are the undoable `Editor` actions.
- **Mixins:** each window is a mixin with its actions (`ui/windows/camera.py` → `editor.set_projection`), so the
  `Editor` classes stay small and scripts call the same methods the windows do.

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

That is the low level: `Instances` and a `batches()` function. Most apps build a **`valo.Scene`** instead
(next section) and get the Scene / Object / Materials windows for free.

## The scene model: `valo.Scene`

```python
from valo import Group, InstancedObject, MeshObject, ProceduralGroup, Scene, SceneEditor, Shape, Transform

scene = Scene()                                                  # on the shared valo.renderer()
scene.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)   # the library: shared by name
shapes = scene.add(Group("Shapes", transform=Transform((0, 0, 1))))
scene.add(MeshObject("Ball", Shape("valo.meshes.primitives:sphere", {"r": 0.4}), "Gold"), parent=shapes)
scene.add(InstancedObject("Arrows", Shape(meshes.arrow), "Gold", instances=matrices))   # (N, 4, 4)
scene.add(ProceduralGroup("Ring", "my_app.builders:ring", {"n": 8}))   # ring(group, n=8) adds the children
viewer.show(scene, SceneEditor(scene))
```
- **A tree** (`valo/scene/nodes.py`, `scene.py`): `Group` (transform + visibility for its children), `ProceduralGroup` (a group
  whose children `builder(group, **params)` regenerates whenever a parameter changes), `MeshObject` (one mesh)
  and `InstancedObject` (one mesh at world · M_i per instance matrix). world = parent world · own `Transform`
  (scale, rotate x→y→z in degrees, translate: `valo.math.transform_matrix`). Hidden ancestors hide a node.
- **Shapes**: a `Shape` is a function (callable, `"module:function"`, or a shape file) plus parameters; the
  editable ones (float/int/bool with defaults; ranges from `func.ranges` / `RANGES`) come from its signature
  (`meshes/params.py`, shared by shape files and procedural builders). Shape files hot-reload.
- **Rebuilds keep children:** a procedural rebuild matches the new children to the old ones by name path and
  type, and keeps the old nodes. A property the user changed from what the builder gave (visibility, transform,
  material, shape) stays; the rest (instance matrices, info) comes from the builder. So selection and undo history
  survive a rebuild. A failed build restores the old children, and SceneEditor shows the error.
- **App structures:** `ProceduralGroup(..., ranges=, context=)`: per-instance slider ranges, and a non-saved
  context object for the builder (an app's data source). A scene with one can render but not export.
  `locked=True` nodes can be hidden and edited but not deleted, renamed, duplicated or moved; nothing is added
  inside a procedural group. `MeshObject(info=)` adds per-instance Info rows. `examples/spinw_studio/` is
  built this way.
- **Names** of nodes and materials are unique per scene (" 2" suffix), so they work as references.
- **GPU**: each object owns an `Instances` whose `owner` is the object; `scene.batches()` draws the visible ones.
  A click on any instance selects the whole object. `Node.bounding_box()` is the box around a node's content in
  its own frame (it tilts and scales with the node); the selected and hovered nodes are drawn with it.
- **Serialization**: `scene.settings()` / `Scene.from_settings(d)`: materials plus the tree as plain data,
  functions as `"module:function"` paths (so they must be importable), procedural groups as builder + parameters
  only. So Export render script works for any Scene.
- **`SceneEditor`** (`valo/ui/scene_editor.py`, windows in `ui/windows/{outliner,object,materials}.py`): undoable actions for the tree (add, add_shape, delete, duplicate,
  rename, reparent, visibility), an object (transform, base shape, shape / builder parameters, material) and the library
  (new, duplicate, delete unused); windows **Scene** (tree with type and eye toggle, context menu, drag onto a
  group to reparent, "+ Add"), **Object** (parameters, transform, material, `meshes.check` diagnostics) and
  **Materials**. `SceneEditor.one_step(label)` groups several actions into one undo step.
- **Why only a tree, no scene graph of dependencies:** shared transforms and one-click visibility are what
  grouping needs; "parameter → regenerate a structure" (lattice constants → atoms, cell edges) is a procedural
  group's rebuild, with the physics links kept in the app's builder. A procedural group rebuilds all its children
  (no partial updates) and there are no links across groups; add either when a real case needs it.

Meshes live in the `valo/meshes/` package; import everything from `valo.meshes`.
- **`primitives.py`**: the standard meshes, all unit-sized, Z-up and CCW outward. They are `sphere`, `icosphere`, `cube`,
  `cylinder`, `cone`, `torus`, `capsule`, `arrow` (tail at origin, tip at z=1) and `plane` (double-sided).
  The studio's Scene window (+ Add > Shape) adds any of them.
- **`ops.py`**: `transform` and `merge`.
- **`builders.py`**: `revolve`, `parametric`, `heightfield` and `tube` (a circle swept along a 3D path).
- **`check.py`**: diagnostics.
- **`shapefile.py`** and **`library.py`**: shape files and the procedural mesh library (below).

## valo studio: procedural meshes in code

```
File > New shape ──▶ data/procedural_meshes/<name>.py ──Code window, Ctrl+S──▶ parameters re-read in the Object tab
valo-studio my.py      mesh(a=1.0, n=32)                  (your edited values stay;     ──Save to library──▶
                       RANGES / MATERIAL / TRANSFORM       untouched ones follow the code)    defaults + MATERIAL +
                                                                                              baked TRANSFORM
```
- **`StudioEditor`** (`valo/studio/`, `uv run valo-studio [shape.py]`): `SceneEditor` plus a **Code** window
  (`imgui_color_text_edit`, docked left through `Editor.dock_left`). It's the base of every domain studio
  (spinw_studio derives from it).
- **A shape file** is plain Python (contract in `valo/meshes/shapefile.py`): a `mesh(...)` function whose parameters
  all have defaults (`float`, `int`, `bool`), optionally `RANGES`, `MATERIAL`, `TRANSFORM`. Build with
  `meshes.revolve` ((ρ, z) profile; repeat a point for a hard edge), `meshes.parametric` (P(u, v), outward is
  ∂u × ∂v), `meshes.heightfield` (z = f(x, y)) or `meshes.tube` (around a 3D path: springs, wires, rings). Starter shapes: `examples/shapes/`.
- **Saving the code** re-reads the file for every object built from it (`Shape.reload`). On an error the last good
  mesh stays and the error, cut to your file's frames, shows under the code.
- **Save to library** writes the file with the object's values as defaults, its material as `MATERIAL` and its
  transform baked into `TRANSFORM` (AST rewrite, `ShapeFile.save`) to `rc.studio.procedural_meshes` (default
  `data/procedural_meshes/`); the object's own transform becomes identity. `valo.library.mesh("<name>")` reads
  it back. One undo step.
- **Diagnostics:** the Object window shows `meshes.check` (winding, unit normals, watertight + volume); "Show
  normals" draws the selected object's normals through `Scene.overlays`. The tests check every library entry.

A real app subclasses `Editor`: its actions are methods that push `Command`s, and `windows()` returns its
dockable windows as `{title: draw_function}`; `material_section(m)` gives the schema-driven material editor.
Optional: `menus()` adds menus (e.g. File) before Edit and View; `dock_bottom` / `bottom_share` / `right_column` /
`dock_left` / `left_column` shape the default layout. The base class draws the menu bar (Edit: undo/redo, View: window toggles, auto-rotate,
UI scale, Reset layout), the History window and the dockspace; the scene shows through the empty **central dock
node**. The layout is saved to `~/.valo/<title>.ini` (real windows only, also on close). A scene with
`bounding_radius()` gets the camera fitted on `show`.

## Camera and lighting

- **`Camera`** (`scene/camera.py`, Blender's viewport view): plain state and maths. Azimuth / elevation /
  distance around a target; `projection` is "perspective" or "orthographic" (`set_projection` keeps the target the
  same size on screen; `zoom` scales distance or `ortho_height`); `VIEWS` has ±X/±Y/±Z/Iso presets.
- **`Navigator`** (`ui/navigation.py`, `viewer.navigator`) moves it: mouse orbit / pan / zoom, the navigation keys,
  presets, perspective / orthographic, fit and frame, and eased transitions (`center_on`). Like Blender's view
  navigation, none of it is an undo step.
- **`Lighting`** (`scene/lighting.py`, `viewer.lighting`): one directional light, a **headlight** (angles in the camera's
  frame, so it moves with you) or **fixed** (world angles); `set_mode` converts so it never jumps. Plus a **global
  hemisphere ambient**: sky colour, ground colour and an intensity shared by every object. The same sky/ground
  is what PBR materials reflect. A material's `ambient` only scales that global ambient (0–1, default 1).
- The shared **Camera** window calls the navigator (not recorded); the **Lighting** window's changes are undoable
  actions (the light is part of the picture).

Frame uniform (group 0, 160 bytes): `view_proj` mat4 · `eye` · `view_dir` (w = 1 perspective, 0 orthographic:
shading uses a constant view vector) · `light_dir` · `light_color` (× intensity) · `sky` (w = ambient intensity)
· `ground`.

Look: `valo/ui/style.py` loads Roboto with Font Awesome icons merged in (`ICON.FLOPPY_DISK` etc.) and
Inconsolata (`mono_font()`), and applies the theme (`apply_style(scale)`). ImGui 1.92 rasterises them at the
real pixel density, so text is crisp at any DPI.

## Configuration: `valo.rc`

`valo/config.py` holds every tunable as typed sections with defaults in code: camera, lighting, highlight,
picking, ui, keys, studio. Read them as `rc.camera.orbit_speed` or `rc["camera.orbit_speed"]`; every set is
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
  - translation / rotation / scale from `valo.math.decompose`;
  - the point hit;
  - then `scene.describe(pick)` or `source.info(i)`.
- **Hover and selection boxes:** `Viewer.boxes()` lists the selected node's box (`viewer.outlined`, set by
  `SceneEditor.select`, else the clicked pick's owner, else the clicked instance) and the hovered one, as box
  matrices (unit cube → world). `Editor.draw_boxes` projects them with `ui.box_layout` and draws them with ImGui over
  the 3D view: `rc.highlight` colours (sRGB), line width, back edges fainter; not in `valo.render` images.
  `Renderer.render(highlights=...)` (an additive tint + rim pass at the same depth) remains as a low-level option.
- **Navigation** (`Viewer.on_pointer` / `on_key` hand the events to the `Navigator`; keys in `rc.keys`):
  - mouse: left-drag orbit, middle- or Shift-drag pan, wheel zoom, click select, double-click centre on the point;
  - keys: F frame selection, Home fit all, 1/3/7 (+Ctrl) views, 5 projection, arrows orbit (Shift pan), Esc deselect.

  `Navigator.center_on` eases over `rc.camera.center_duration`. Selection and navigation are view state: never undo
  steps.

## Scripting and offline rendering

- **One GPU:** there is one GPU device and one `Renderer` per process (`valo.renderer()`); every `Viewer` uses it,
  so a scene built in a script can be shown in a window and the reverse.
- **`valo.render(scene, settings=None, *, size, samples, transparent, background, camera, lighting, path)`**
  (`valo/offline.py`) renders without a window into an `rgba8unorm` texture and returns (h, w, 4) uint8 sRGB.
  Given a `path`, it saves a PNG.
  - **Supersampling:** the scene target is `samples`× larger and the post pass averages each block, premultiplied
    by alpha.
  - **Transparency:** the scene clears to alpha 0, and the post pass passes alpha through.
  - **Defaults:** missing options come from `rc.render` / `rc.lighting`, and the camera is fitted to the scene.
- **`settings`** is plain data: `{"scene": ..., "camera": Camera.settings(), "lighting": Lighting.settings(),
  "image": {"size", "samples", "transparent", "background"}}`.
- **The scene settings protocol:** a scene that implements `settings() -> dict` and the classmethod
  `from_settings(d, gpu=None)` can be rebuilt from data. The Editor's File → Render window then exports a
  **runnable script**: `SETTINGS` written by `offline.py_literal`, plus a main that calls `from_settings` and
  `valo.render`. Its picture is pixel-identical to the studio's Render PNG. Any app can Render PNG; Export needs
  the protocol. `valo.Scene` implements it, so every studio scene exports.

## One frame

```
Viewer.draw()
 ├─ Renderer.render(surface_texture, viewport=(x, y, w, h), scene, camera)   rect = editor's central dock node
 │    ├─ write frame uniform (camera, light, ambient)              group 0
 │    ├─ flush dirty materials (64-byte uniform each)              group 1
 │    ├─ scene pass → rgba16float texture, one instanced draw per batch
 │    │     batch = Batch(material, gpu mesh, instance bind group, count, source)   group 2 = mat4x4f per instance
 │    │     (vs_main: world = M·p, normal = cofactor(M)·n, so non-uniform scale shades correctly)
 │    ├─ (highlights=..., if an app passes them: instances drawn again with an additive tint)
 │    └─ post pass → encode linear → sRGB into the viewport region of the window
 ├─ Renderer.pick(...) under the pointer → viewer.hover (1-pixel id pass, read back)
 └─ ImguiRenderer.render()  → editor.gui(): menu bar, dockspace, windows on top (skipped without an editor)
    (if that frame moved the central node, scene + UI are drawn once more, so the scene never lags the layout)
```

## Dependencies

```
math     ← (nothing)
meshes   → math, config                        (shapefile → math; library → shapefile)
gpu      → math, meshes, config                (renderer → shaders, material, mesh, instances)
scene    → math, meshes, gpu, config           (nodes own gpu.Instances; Scene uses a gpu.Renderer)
offline  → scene, gpu, config
ui       → scene, gpu, meshes, math, offline, config
studio   → ui, scene, meshes, math, config
apps:    examples/spinw_studio:   api → editor, scene → model, adapter; scene / editor → valo (Scene, StudioEditor)
```
The renderer draws anything with a `batches()` method; nothing in `valo/` imports from `examples/`.

## Extension points (where a typical feature request goes)

| Request | Change |
|---|---|
| New shading model | `valo/gpu/shaders.py`: WGSL in `FRAGMENT_WGSL` + parameter list in `valo/gpu/material.py` `SCHEMA`. Panels and pipelines pick it up. |
| New mesh shape | write it in the studio: `uv run valo-studio`, File > New shape, then Save to library (→ `data/procedural_meshes/`, checked by the tests). A primitive every app needs goes in `valo/meshes/primitives.py` + `STANDARD`. |
| A new domain studio | subclass `valo.StudioEditor` (scene tree, objects, materials, code window come with it); see `examples/spinw_studio/`. |
| New mesh builder | `valo/meshes/builders.py`, next to `revolve` / `parametric` / `heightfield` / `tube`; test it with `check()`. |
| Bonds, vector fields | app: `renderer.instances(meshes.cylinder() / arrow(), m)` + `instance_matrices(..., z_axis=...)`. |
| Polyhedra, isosurfaces (per-scene meshes) | app: build a `MeshData` and draw it with one identity instance. |
| Animation (vibrations, precession) | engine: time in the frame uniform; per-instance amplitudes; displacement in `vs_main`. |
| Object properties in Info | app: `renderer.instances(..., label=, info=lambda i: {...})`, or `scene.describe(pick)`. |
| A new tunable | add a field to a section in `valo/config.py` (default, label, range, unit, help); read `rc.<section>.<key>`. |
| Image export | done: `valo.render` / File → Render. Video: render frames with `valo.render` in a loop. |
| Scripting / export for an app's scene | app: `settings()` + `from_settings(d, gpu=None)` on the scene class (in an importable module). |
| Objects with parameters in the Scene / Object windows | app: build a `valo.Scene` and show it with `SceneEditor` (or a subclass adding windows). |
| A parameter that regenerates part of the scene | app: a `ProceduralGroup` with a builder `fn(group, **params)` in an importable module (data the builder needs: `context=`; fixed parts: `locked=True`). |
| New panel or window | app: an action method on its `Editor` subclass (with undo) + an entry in `windows()`. |
| A new app | `examples/<name>/` with a scene, an `Editor` subclass, a `main()` and a test. |
| A real application (domain library → valo) | follow `examples/spinw_studio/`: an adapter turning the library's objects into plain data, a pure model, then scene / editor on valo's public API (its README states the layering rules). |

## Parked ideas (pull in only when a feature needs them)

From the earlier v2 plan, kept here so they aren't lost: ray-cast impostor spheres for >10k atoms,
order-independent transparency, keyframe tracks and a clock, golden-image tests, a Rust/wgpu core
behind the same Python API, multi-agent workflow with worktrees. Each one is a feature request
when its time comes.
