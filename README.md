# valo

An interactive scientific 3D rendering engine in Python: wgpu for rendering, glfw for the window,
Dear ImGui for a docking UI. Apps hand the `Viewer` a scene of instanced, shaded geometry and an
`Editor` whose windows dock around the 3D view, with undo/redo.

```powershell
uv sync
uv run python examples/crystal_studio/studio.py   # example app: rock-salt crystal + material editor
uv run python examples/mesh_gallery/gallery.py    # example app: a valo.Scene of every standard mesh
uv run valo-sculpt examples/sculpts/vase.py      # MeshSculptor: sculpt a mesh from a .py file, save to the library
uv run pytest                                     # engine + example tests (GPU tests render offscreen)
```

Apps build a `valo.Scene`: a tree of objects, each a shape function with parameters, a transform and a material
from the scene's library. `SceneEditor` gives it Scene (tree, visibility, drag to reparent), Object (parameters,
transform, material) and Materials windows, with undo for everything.

MeshSculptor hot-reloads the sculpt file whenever you save it in your editor. "Save to library" writes
`valo/meshes/library/<name>.py`, and apps use it as `valo.library.mesh("<name>")`.

| Mouse / key | Action |
|---|---|
| left-drag / middle- or Shift-drag / wheel | orbit / pan / zoom |
| hover / click / double-click | Info + highlight / select / centre on that point |
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
from crystal import Crystal          # examples/crystal_studio

crystal = Crystal()
crystal.n = [8, 8, 4]
crystal.rebuild()
valo.render(crystal, {"camera": {"azimuth": -60.0, "projection": "orthographic"},
                      "lighting": {"mode": "fixed"},
                      "image": {"size": [2400, 1600], "samples": 2, "transparent": True}},
            path="crystal.png")
```

Or set things up in the studio and use File → Render image / export script: Render PNG saves the picture, and
Export writes a runnable `render_*.py` with every setting in an editable `SETTINGS` dict. Running it re-renders
the same picture. A full example is at the bottom of `examples/crystal_studio/crystal.py`
(`uv run python examples/crystal_studio/crystal.py`).

See `ARCHITECTURE.md` for how it fits together, how to write an app, and where new features go.
