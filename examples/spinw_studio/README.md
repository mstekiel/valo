# spinw_studio

**What it is.** The middle layer between **spinwaves**, the physics library, and **valo**, the rendering engine.
It takes the well-defined objects of a spinwaves system (atoms, crystal, couplings, wrapped in a `SpinW` object)
and shows them in an interactive valo studio, or renders them to an image.

It's the first real application on valo, and its needs steer the engine's development.

This document is the **defining document** of spinw_studio: the layers, the data contract, the options and the
rules. Change it whenever one of those changes. The original design notes are in [PLAN.md](PLAN.md).

## Status
**Works now:**
- atoms as instanced spheres, one material per label;
- magnetic moments as arrows centred on the atoms;
- the unit-cell edges;
- a continuous uvw extent;
- a compass with x, y, z and a, b, c;
- the Info window for hovered atoms;
- undo for every edit;
- `plot_structure` / `render` / Export plot options.

**To do:**
1. **Fill the adapter stubs** in `spinw_studio/adapter.py` from spinwaves (by hand).
2. **Couplings** (bonds between atoms, coloured by `coupling_colors`).

## Usage
```bash
uv sync --group spinw                                             # spinwaves (editable, local) + lmfit
uv run --group spinw python examples/spinw_studio/lfo_spinw.py    # the LFO example
```
```python
from spinw_studio import plot_structure, render

plot_options = plot_structure(sw, plot_options={"boundaries": ([-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.02])})
#   ↑ blocks until you close the window, then returns plot_options as you left them (camera, lighting included)
render(sw, plot_options, path="lfo.png", size=(2400, 1600), samples=2, transparent=True)   # same picture, no window
```
In the studio, File → Export plot options writes `PLOT_OPTIONS = {...}` to a `.py` file, and File → Render image
saves a PNG.

## The three layers, and what each may import
```
your script (e.g. lfo_spinw.py)      physics: Atom, Crystal, MSG, Coupling, SpinW from spinwaves
      │  sw: SpinW
      ▼
spinw_studio/adapter.py               SpinW → plain numpy data. The ONLY file that knows spinwaves objects.
      │  SystemView(A, atoms)
      ▼
spinw_studio/model.py                 the data contract + pure geometry (numpy; pillow for colour names)
spinw_studio/scene.py, editor.py      the valo scene and editor (valo's public API only)
spinw_studio/api.py                   plot_structure, render
      ▼
valo                                  rendering; knows nothing about crystals or spinwaves
```
**Rules:**
1. **spinwaves stays in the adapter.** spinwaves objects are touched only in `adapter.py`, and duck-typed there:
   no spinwaves imports are needed. spinwaves never imports spinw_studio.
2. **The model stays pure.** `model.py` imports no valo and no spinwaves, and everything after it works on a
   `SystemView`. So a hand-made `SystemView` (the tests) or another library's adapter works the same way.
3. **The rest uses valo's public API.** `scene.py`, `editor.py` and `api.py` import only from `valo` itself,
   and valo never imports spinw_studio. If spinw_studio needs an engine feature, it's added to valo in a
   domain-free form; the corner compass came about this way.
4. **Defaults are app conventions.** spinw_studio's defaults live in `model.DEFAULTS`, not in `valo.rc`.

## Coordinates and units
- **Crystal coordinates uvw:** fractional, dimensionless. **Cartesian xyz:** Å.
- **r_xyz = A @ r_uvw.** The **columns** of `A` are the lattice vectors a, b, c expressed in xyz (Å).
- **Magnetic moments are Cartesian xyz** everywhere in spinw_studio; the adapter converts from spinwaves' frame.
- **Colours** in plot_options are **sRGB** (CSS names, `#rrggbb` or (r, g, b)); valo converts them to linear.

## The data contract (filled by `adapter.py`)
The adapter stubs return plain arrays. `adapter.system_view(sw)` checks their shapes, types and ranges, names any
stub that breaks the contract, and builds the `SystemView`.

| Stub | Returns | Frame and units |
|---|---|---|
| `lattice_matrix(sw)` | `ndarray` (3, 3) float | A; columns a, b, c in xyz, Å |
| `atom_positions_uvw(sw)` | `ndarray` (N, 3) float | every atom of one cell after symmetry; uvw in [0, 1); row i = atom i = `Coupling` id |
| `atom_moments_xyz(sw)` | `ndarray` (N, 3) float | moment per atom, Cartesian xyz, any length; zero row = no arrow |
| `atom_labels(sw)` | `list[str]` (N) | "Fe", "O1", …: selects the material and the radius |
| `atom_spins(sw)` *(optional)* | `ndarray` (N,) float | spin length S (default 0) |
| `atom_extra(sw, i)` *(optional)* | `dict[str, str]` | extra Info rows for atom i |
| `system_name(sw)` *(optional)* | `str` | window title and file names |

```python
AtomSite(label, index, uvw, moment, spin=0.0, extra={})   # one atom of the cell
SystemView(A, atoms, name="")                             # what everything after the adapter uses
```
`model.expand(system, boundaries)` repeats the atoms by lattice translations n ∈ ℤ³, keeping those whose uvw falls
inside the box. The Info window shows each drawn atom's cell index, uvw, translation n, xyz, moment and S.

## plot_options
One dict, validated, going in and out. Unknown keys give an error with "did you mean". The defaults are in
`model.DEFAULTS`.

| Key | Default | Meaning |
|---|---|---|
| `boundaries` | `[[-0.1, 1.1]] * 3` | uvw extent ((umin, umax), (vmin, vmax), (wmin, wmax)) |
| `atom_colors` | element colour or palette | `{label: colour}` |
| `atom_radii` | per-element display radii (`model.RADII`) | `{label: Å}` |
| `materials` | — | `{label: Material.to_dict()}`: the full material, overrides `atom_colors` |
| `show_moments`, `moment_length`, `moment_color`, `moment_material` | True, 2.0 Å, "Crimson", — | arrows centred on the atoms (the length must exceed the atom diameter to show) |
| `show_cell`, `cell_radius`, `cell_color`, `cell_material` | True, 0.03 Å, "Gray", — | unit-cell edges |
| `coupling_colors` | — | `{coupling label: colour}`; kept for the couplings to come |
| `camera`, `lighting`, `image` | — | valo `OrbitCamera.settings()`, `Lighting.settings()`, `valo.render` image options |

What `plot_structure` returns, and what Export writes, contains the materials in full, so feeding it back rebuilds
the same picture pixel for pixel (a test checks this).

## Notes
- **Compass:** it's drawn by ImGui, so it appears in the studio but not in `render()` images.
- **Tests:** `uv run pytest` runs spinw_studio's tests on a hand-made LFO-like `SystemView`, without spinwaves.
  The adapter itself is tested through its contract checks.


# Issues

For future development

### spinw studio
- model.AtomSite reimplements spinwaves.Atom. Looks like some wrong architectural choice overall. I want a simpler, lower level behaviour, where the adapter doesn


### valo

