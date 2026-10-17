# valo

An interactive scientific 3D rendering engine in Python: wgpu for rendering, glfw for the window,
Dear ImGui for a docking UI. Apps hand the `Viewer` a scene of instanced, shaded geometry and an
`Editor` whose windows dock around the 3D view, with undo/redo.

```powershell
uv sync
uv run python examples/crystal_studio/studio.py   # example app: rock-salt crystal + material editor
uv run python examples/mesh_gallery/gallery.py    # example app: every standard mesh, instanced arrows
uv run valo-sculpt examples/sculpts/vase.py      # MeshSculptor: sculpt a mesh from a .py file, save to the library
uv run pytest                                     # engine + example tests (GPU tests render offscreen)
```

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
Every speed, colour, default and key is in `valo.rc`: View → Preferences, or `uv run valo-config --write-defaults`
for a commented `~/.valo/valorc.toml`.

See `ARCHITECTURE.md` for how it fits together, how to write an app, and where new features go.
