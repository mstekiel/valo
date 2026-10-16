# valo

An interactive scientific 3D rendering engine in Python: wgpu for rendering, glfw for the window,
Dear ImGui for the side panel. Apps hand the `Viewer` a scene of instanced, shaded geometry and an
`Editor` panel with undo/redo.

```powershell
uv sync
uv run python examples/crystal_studio/studio.py   # example app: rock-salt crystal + material editor
uv run pytest                                     # engine + example tests (GPU tests render offscreen)
```

Mouse: left-drag orbits, wheel zooms. Ctrl+Z / Ctrl+Y undo / redo.

See `ARCHITECTURE.md` for how it fits together, how to write an app, and where new features go.
