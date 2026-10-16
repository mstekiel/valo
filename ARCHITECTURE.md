# valo architecture (skeleton)

Grown from the wgpu-py exploration examples (example 06, "crystal studio"). `valo/` is a standalone
engine; apps such as `examples/crystal_studio/` are built on its public API (`valo/__init__.py`).
Deliberately small: structure is added when a feature needs it, not before.

## Writing an app

```python
from types import SimpleNamespace
from valo import Editor, Viewer

viewer = Viewer(title="my app")                       # window, device, renderer, camera, ImGui
gold = viewer.renderer.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
balls = viewer.renderer.spheres(gold)                 # instanced spheres, one draw call
balls.set(centers, radii)                             # (N, 3) and a scalar or N values
editor = Editor(); editor.watch(gold)                 # material edits now undoable
viewer.show(SimpleNamespace(batches=lambda: [balls.batch()]), editor)
viewer.run()
```

A real app subclasses `Editor`: its actions are methods that push `Command`s, and `panel()` draws its
widgets (`material_section(m)` gives the schema-driven material editor). The base class draws the
docked window, Ctrl+Z/Y, History and View. A scene with `bounding_radius()` gets the camera fitted on `show`.

## One frame

```
Viewer.draw()
 ├─ Renderer.render(surface_texture, viewport_width, scene, camera)
 │    ├─ write frame uniform (camera, light)                       group 0
 │    ├─ flush dirty materials (64-byte uniform each)              group 1
 │    ├─ scene pass → rgba16float texture, one instanced draw per batch
 │    │     batch = (material, instance bind group, count)         group 2 = vec4(centre, radius)
 │    └─ post pass → encode linear → sRGB into the viewport region of the window
 └─ ImguiRenderer.render()  → editor.gui() draws the panel on top (skipped without an editor)
```

## Dependencies

```
engine:  geometry ← materials ← render ← app (Viewer)
                        ↑                  ↓
                        └──── ui (Editor) → undo
app:     examples/crystal_studio: crystal (model) ← studio (CrystalEditor, main) → valo
```
`render` draws anything with a `batches()` method; nothing in `valo/` imports from `examples/`.

## Extension points (where a typical feature request goes)

| Request | Change |
|---|---|
| New shading model | `valo/materials.py`: WGSL in `FRAGMENT_WGSL` + parameter list in `SCHEMA`. Panels and pipelines pick it up. |
| Bonds, polyhedra, isosurfaces (non-sphere geometry) | engine: let a batch carry its mesh, `(material, mesh, instance_bg, count)`, next to `Spheres`; `Renderer` draws `batch.mesh`. |
| Animation (vibrations, precession) | engine: time in the frame uniform; per-instance amplitudes; displacement in `vs_main`. |
| Picking | engine: extra pass writing (batch, instance_index) to an `rg32uint` texture; `Viewer` reads the pixel under the mouse. |
| Image / video export | engine: render to an offscreen canvas at the requested size (the tests already do this). |
| New panel section | app: an action method on its `Editor` subclass (with undo) + widgets in `panel()`. |
| New crystal structure (CIF, POSCAR, other lattice), more species | `examples/crystal_studio/crystal.py`: centres + species per site; keep `batches()`. |
| A new app | `examples/<name>/` with a scene, an `Editor` subclass, a `main()` and a test. |

## Parked ideas (pull in only when a feature needs them)

From the earlier v2 plan, kept here so they aren't lost: ray-cast impostor spheres for >10k atoms,
order-independent transparency, keyframe tracks and a clock, golden-image tests, a Rust/wgpu core
behind the same Python API, multi-agent workflow with worktrees. Each one is a feature request
when its time comes.
