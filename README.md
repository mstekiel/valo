# valo

An interactive scientific 3D rendering engine in Python: wgpu for rendering, glfw for the window,
Dear ImGui for a docking UI. Apps hand the `Viewer` a scene of instanced, shaded geometry and an
`Editor` whose windows dock around the 3D view, with undo/redo.

```powershell
uv sync
uv run valo-studio                                # the studio: scene, objects, materials, procedural meshes in code
uv run valo-studio valo/data/procedural_meshes/vase.py   # ...opening a shape file
uv run pytest                                     # engine + example tests (GPU tests render offscreen)
```

Apps build a `valo.Scene`: a tree of objects, each a shape function with parameters, a transform and a material
from the scene's library. `SceneEditor` gives it Scene (tree, visibility, drag to reparent), Object (parameters,
transform, material) and Materials windows, with undo for everything.

In the studio, File > New shape starts a procedural mesh in the Code window. Every save (Ctrl+S) re-reads its
parameters into the Object tab; files changed in another editor reload too. "Save to library" writes
`data/procedural_meshes/<name>.py` in the working directory (your library; set `studio.procedural_meshes` in
`valorc.toml` to move it), and apps use it as `valo.library.mesh("<name>")`. Built-in meshes (vase, helix, shell,
wave) ship with valo and show in the same list.

| Mouse / key | Action |
|---|---|
| left-drag / middle- or Shift-drag / wheel | orbit / pan / zoom |
| hover / click / double-click | Info + grey box / select (orange box) / centre on that point |
| F / Home | frame the selection / fit everything |
| 1, 3, 7 (Ctrl: opposite side) / 5 | front, right, top views / toggle orthographic |
| arrows (Shift: pan) / Esc | orbit in steps / deselect |
| Ctrl+Z / Ctrl+Y | undo / redo |

Drag window tabs to re-dock them; View → Reset layout restores the default (layouts are saved in `~/.valo/`).
Every speed, colour, default and key is in `valo.rc`: View → Preferences, or a TOML file.
`examples/valorc.toml` is an annotated example to copy to `~/.valo/valorc.toml`;
`uv run valo-config --write-defaults` writes every key with its range.

## Scripting

Render pictures without opening a window; every option is plain Python:

```python
import valo
from valo import MeshObject, Scene, Shape, meshes

scene = Scene()                                   # built on valo.renderer(), no window needed
scene.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0, roughness=0.3)
scene.add(MeshObject("Torus", Shape(meshes.torus, {"major": 0.6}), "Gold"))
valo.render(scene, {"camera": {"azimuth": -60.0, "elevation": 30.0, "projection": "orthographic"},
                    "lighting": {"mode": "fixed"},
                    "image": {"size": [2400, 1600], "samples": 2, "transparent": True}},
            path="torus.png")
```

Or set things up in the studio and use File → Render image / export script: Render PNG saves the picture, and
Export writes a runnable `render_*.py` with every setting in an editable `SETTINGS` dict. Running it re-renders
the same picture.

See `ARCHITECTURE.md` for how it fits together, how to write an app, and where new features go.
`DEV.md` covers development: checks, versions, releases and the GitHub housekeeping.

## License

valo is free software under the GNU General Public License v3.0 or later; see `LICENSE`.
Changes are listed in `CHANGELOG.md`.
