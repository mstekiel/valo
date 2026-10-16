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

Standard meshes (`valo/meshes.py`, all unit-sized, Z-up, CCW outward): `sphere`, `icosphere`, `cube`,
`cylinder`, `cone`, `torus`, `capsule`, `arrow` (tail at origin, tip at z=1), `plane` (double-sided);
`transform(mesh, M)` and `merge(*meshes)` build new ones. `examples/mesh_gallery/` shows them all.

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
 │    │     batch = (material, gpu mesh, instance bind group, count)   group 2 = mat4x4f model per instance
 │    │     (vs_main: world = M·p, normal = cofactor(M)·n, so non-uniform scale shades correctly)
 │    └─ post pass → encode linear → sRGB into the viewport region of the window
 └─ ImguiRenderer.render()  → editor.gui() draws the panel on top (skipped without an editor)
```

## Dependencies

```
engine:  geometry, meshes, materials, undo     (leaves: numpy only, materials holds a GPU buffer)
         render  → geometry, meshes, materials
         ui      → materials, undo
         app     → render, ui
apps:    examples/crystal_studio: studio → crystal → valo
         examples/mesh_gallery:   gallery → valo
```
`render` draws anything with a `batches()` method; nothing in `valo/` imports from `examples/`.

## Extension points (where a typical feature request goes)

| Request | Change |
|---|---|
| New shading model | `valo/materials.py`: WGSL in `FRAGMENT_WGSL` + parameter list in `SCHEMA`. Panels and pipelines pick it up. |
| New standard mesh | `valo/meshes.py`: a function returning `MeshData` (a `_revolve` profile for round shapes), add it to `STANDARD`; the mesh tests check winding and volume. |
| Bonds, vector fields | app: `renderer.instances(meshes.cylinder() / arrow(), m)` + `instance_matrices(..., z_axis=...)`. |
| Polyhedra, isosurfaces (per-scene meshes) | app: build a `MeshData` and draw it with one identity instance. |
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
