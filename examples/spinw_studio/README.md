# spinw_studio

**What it is.** The middle layer between **spinwaves**, the physics library, and **valo**, the rendering engine.
It takes the well-defined objects of a spinwaves system (atoms, crystal, couplings, wrapped in a `SpinW` object)
and shows them in an interactive valo studio, or renders them to an image.

It's the first real application on valo, and its needs steer the engine's development.

This document is the **defining document** of spinw_studio: the layers, the data contract, the options and the
rules. Change it whenever one of those changes.

## Status
**Works now:**
- the structure as a `valo.Scene` (see "The scene" below): atoms per label, moment arrows, unit-cell edges;
- a parameters tab: lattice a, b, c, α, β, γ, the uvw extent and the moment length, which rebuild the structure;
- valo's Scene / Object / Materials windows: per-label radius, base mesh, material and visibility;
- objects of your own added next to the structure;
- a compass with x, y, z and a, b, c (follows the lattice);
- the Info window for hovered atoms;
- undo for every edit;
- `plot_structure` / `render` / Export plot options.

**To do:**
1. **Fill the `set_lattice` stub** in `spinw_studio/adapter.py`, so lattice edits reach your `sw`.
2. **Couplings** (bonds between atoms, coloured by `coupling_colors`): a locked "Couplings" group in the
   structure.

## Usage
```bash
uv sync --group spinw                                             # spinwaves (editable, local)
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
spinw_studio/scene.py, editor.py      the valo.Scene and its StudioEditor (valo's public API only)
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

## The scene
`scene.build_scene(sw_or_view, plot_options)` makes a `valo.Scene` with one locked **procedural group**:
```
Structure         ProceduralGroup(scene.structure): a, b, c (Å), alpha, beta, gamma (deg), umin … wmax, moment_length (Å)
├─ Atoms
│   └─ Fe, O1, …   one InstancedObject per label: Shape(sphere, r = radius), material named after the label
├─ Moments         one "<label> moments" InstancedObject per magnetic label: arrows, material "Moments"
└─ Cell            the 12 edges: Shape(cylinder, r = edge radius), material "Unit cell"
+ your own objects (Scene window → Add), deletable as usual
```
- **Rebuild:** changing a Structure parameter calls the builder again. It applies the lattice, re-reads the system,
  runs `model.expand`, and updates the children's instances. Your edits to the children (radius, base mesh,
  material, visibility) are kept: valo keeps rebuilt children by name.
- **Locked:** the structure's nodes can be hidden and edited but not deleted, renamed or moved, and nothing can be
  added inside it.
- **Lattice edits** go through `adapter.set_lattice(sw, …)`, which **changes your `sw` in place**. Then
  `adapter.system_view(sw)` is read again. For a plain `SystemView` (the tests), `model.with_lattice` changes only
  what's shown. Until the stub is filled, a lattice edit shows the stub's error and changes nothing.
- **The connection lives here:** "lattice → atoms, cell edges, compass" is spinw_studio's builder. valo only knows
  "a parameter changed → call the builder".

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
| `set_lattice(sw, a, b, c, alpha, beta, gamma)` | — | **changes sw in place**: new lattice (Å, degrees; alpha between b and c), uvw kept; then `lattice_matrix(sw)` returns the new A |

```python
AtomSite(label, index, uvw, moment, spin=0.0, extra={})   # one atom of the cell
SystemView(A, atoms, name="")                             # what everything after the adapter uses
```
`model.lattice_params(A)` / `model.lattice_matrix(a, b, c, alpha, beta, gamma)` convert between A and the six
parameters. The standard orientation puts a along x and b in the xy plane; `with_lattice` keeps the current
orientation. `model.expand(system, boundaries)` repeats the atoms by lattice translations n ∈ ℤ³, keeping those whose uvw falls
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
| `show_moments`, `moment_length`, `moment_color`, `moment_material` | True, 2.0 Å, "Crimson", — | arrows centred on the atoms (the length must exceed the atom diameter to show); `show_moments` is the Moments eye |
| `show_cell`, `cell_radius`, `cell_color`, `cell_material` | True, 0.03 Å, "Gray", — | unit-cell edges; `show_cell` is the Cell eye, `cell_radius` its cylinder's `r` |
| `shapes` | — | `{structure object: {"func", "params"}}`: base meshes you swapped (e.g. Fe as an icosphere) |
| `objects`, `object_materials` | — | valo node settings of the objects you added, and the library's other materials |
| `coupling_colors` | — | `{coupling label: colour}`; kept for the couplings to come |
| `camera`, `lighting`, `image` | — | valo `Camera.settings()`, `Lighting.settings()`, `valo.render` image options |

What `plot_structure` returns, and what Export writes, contains the materials in full, so feeding it back rebuilds
the same picture pixel for pixel (a test checks this). **The lattice isn't in plot_options:** it lives in your `sw`.

## Notes
- **Compass:** it's drawn by ImGui, so it appears in the studio but not in `render()` images.
- **Tests:** `uv run pytest` runs spinw_studio's tests on a hand-made LFO-like `SystemView`, without spinwaves.
  The adapter is tested through its contract checks on a duck-typed stand-in for `sw`.
- **No render script:** valo's Export render script is off, because the structure needs your `sw`. Use Export
  plot options + `render(sw, PLOT_OPTIONS)`. Render image works.