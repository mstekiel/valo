# spinw_studio — agreed plan (paused)

**Status (2026-10-19):** design agreed, **nothing implemented yet**.
- **Blocker:** installing `spinwaves` (exact, old dependency pins) — see [DEPENDENCIES.md](DEPENDENCIES.md).
- **What can start before it's fixed:** steps 2–5 below need no spinwaves (tests use a hand-made `SystemView`).
- **What waits:** step 1 (the dependency group) and running `lfo_spinw.py`.

**Resume by:** reading this file + `lfo_spinw.py`, then following "Implementation steps". Follow CLAUDE.md:
- plan mode first;
- commits of ≤ 5 files;
- tests per the Testing rules;
- render offscreen and look before calling a visual step done.

## Goal
spinw_studio is the first real application of valo, and it guides valo's development. It is the middle layer
between **spinwaves** (your physics library: `Atom`, `Crystal`, `MSG`, `Coupling`, wrapped in a `SpinW` object)
and **valo** (rendering).

The prototype renders the **atoms** of a spinwaves system:
- instanced spheres placed from spinwaves data;
- magnetic moments as arrows;
- the unit-cell edges;
- a compass showing Cartesian (xyz) and crystal (abc) axes.

The extent (in crystal coordinates) and the materials are adjustable, as in crystal_studio. **Couplings come later.**

**Coordinates:** crystal coordinates **uvw** (fractional) next to Cartesian **xyz** (Å), with
**r_xyz = A · r_uvw**. The columns of A are the lattice vectors a, b, c in xyz.

## Decisions (asked and answered)
| Topic | Decision |
|---|---|
| Dependencies | `spinw` **dependency group**: spinwaves as a local editable path + lmfit; `uv sync --group spinw`. valo itself never depends on spinwaves. |
| Entry point | the **script calls spinw_studio**: `from spinw_studio import plot_structure`. spinwaves never imports spinw_studio. |
| Window | **blocks** until closed, like `plt.show()` |
| Extent | **continuous uvw ranges**: your `boundaries` = ((umin, umax), (vmin, vmax), (wmin, wmax)); every translated atom inside is shown (−0.1…1.1 shows the boundary atoms) |
| Materials | **one per atom label** ("Fe"), editable like crystal_studio (library, duplicate, assign) |
| Moments | **arrows on atoms**: one instanced arrow batch per label, along m, show/hide + length |
| Compass | **corner inset**: rotates with the camera; x, y, z plus a, b, c labelled |
| Cell edges | **one cell [0,1]³, thin cylinders** (12 instanced cylinders, editable thickness and material) |
| Moment frame from the stubs | **xyz**: you convert inside the stub; the studio only sees Cartesian moments |
| Radii | **per-element defaults table** (display radii, Å) overridable by `plot_options["atom_radii"]` + a slider per label |
| Round trip | `plot_structure` **returns** the final `plot_options`; an **"Export plot options"** button; offline **`spinw_studio.render(sw, plot_options, path=...)`** |
| Defining document | `examples/spinw_studio/README.md` (write it in the last step) |

## Three layers, strict imports
```
lfo_spinw.py  (physics: Atom, Crystal, MSG, Coupling, SpinW from spinwaves)
      │  sw: SpinW
      ▼
spinw_studio/adapter.py   STUBS you fill: SpinW → plain numpy data (the only file that knows spinwaves objects)
      │  SystemView(A, atoms=(AtomSite, ...))      pure numpy: no spinwaves, no valo
      ▼
spinw_studio/model.py     uvw → xyz, extent expansion, cell edges, plot_options validation (pure)
spinw_studio/scene.py, editor.py, api.py  → valo (public API only)
```
**What each part may import:**
- `adapter.py` may touch spinwaves objects (duck-typed);
- `model.py` imports only numpy;
- `scene.py`, `editor.py` and `api.py` import `valo` and `model`, never spinwaves;
- `valo/` never imports spinw_studio.

Tests use a hand-made `SystemView`, so they run without spinwaves.

**Package layout:**
```
examples/spinw_studio/
  lfo_spinw.py, README.md, PLAN.md, DEPENDENCIES.md, test_spinw_studio.py
  spinw_studio/{__init__, adapter, model, scene, editor, api}.py
```

## 1. Data contract (`model.py`) + adapter stubs (`adapter.py`)
```python
@dataclass(frozen=True)
class AtomSite:          # one atom of the unit cell (after symmetry)
    label: str           # "Fe": selects the material and radius
    index: int           # row in the unit cell (= id1/id2 of Coupling)
    uvw: tuple           # crystal coordinates, fractional, in [0, 1)
    moment: tuple        # magnetic moment, CARTESIAN xyz, any length
    spin: float = 0.0    # S
    extra: dict = {}     # extra Info rows (str → str)

@dataclass(frozen=True)
class SystemView:
    A: np.ndarray        # (3, 3), r_xyz = A @ r_uvw, columns a, b, c in Å
    atoms: tuple[AtomSite, ...]
    name: str = ""
```
`adapter.system_view(sw) -> SystemView` calls the stubs below.
- **Until filled:** each stub raises `NotImplementedError("fill in from spinwaves: ...")`.
- **Docstrings:** each states the coordinate system, units, array shape and dtype, plus an example value.
- **Checks:** `system_view` validates shapes and types, and names the stub that broke the contract.

| Stub | Returns | Coordinate system and units |
|---|---|---|
| `lattice_matrix(sw)` | `np.ndarray` **(3, 3)** float64 | **A**, r_xyz = A @ r_uvw; **columns** = a, b, c in **Cartesian xyz, Å**. Example: `np.diag([5.3, 5.6, 7.5])` |
| `atom_positions_uvw(sw)` | `np.ndarray` **(N, 3)** float64 | all atoms of one cell after symmetry, **crystal uvw** (fractional, in [0, 1)); row order = `Coupling(id1, id2)` indices |
| `atom_moments_xyz(sw)` | `np.ndarray` **(N, 3)** float64 | moment per atom in **Cartesian xyz** (converted from spinwaves' frame); any length; zero row = no arrow |
| `atom_labels(sw)` | `list[str]` length **N** | label per atom |
| `atom_spins(sw)` | `np.ndarray` **(N,)** float64, optional (default zeros) | spin length S |
| `atom_extra(sw, i)` | `dict[str, str]`, optional (default `{}`) | extra Info rows for atom i |
| `system_name(sw)` | `str`, optional (default `""`) | window title / file names |

**Pure helpers in `model.py`:**
- `to_xyz(A, uvw)`;
- `expand(system, bounds)`: every translation n ∈ ℤ³ with uvw + n inside the bounds (small tolerance). Gives per
  label the xyz positions, uvw (with n), moments and indices;
- `cell_edges(A)`: 12 segments;
- a radius table (about 20 common magnetic ions, fallback 0.5 Å);
- colours: CSS names, hex or sRGB tuples via `PIL.ImageColor`, converted to linear.

## 2. plot_options: one validated dict, in and out
Every key is optional. Unknown keys raise with "did you mean"; this reuses `valo.config._unknown`, made public as
`unknown_key_message`.

| Key | Default | Meaning |
|---|---|---|
| `boundaries` | `([-0.1, 1.1],) * 3` | uvw extent |
| `atom_colors` | palette per label | `{label: "Orange" / "#ff8800" / (r, g, b) sRGB}` |
| `atom_radii` | radius table | `{label: Å}` |
| `materials` | — | `{label: Material.to_dict()}`: full material, overrides the colour; written on export |
| `show_moments`, `moment_length`, `moment_color` | True, 1.0 Å, "Crimson" | arrows centred on atoms, along m |
| `show_cell`, `cell_radius`, `cell_color` | True, 0.03 Å, "Gray" | unit-cell edges |
| `coupling_colors` | — | accepted and kept for later (not drawn yet) |
| `camera`, `lighting`, `image` | — | `OrbitCamera.settings()`, `Lighting.settings()`, `valo.render` image options |

These defaults live in spinw_studio (`DEFAULTS`), not in `valo.rc`: they are app conventions.

## 3. valo engine addition: corner compass (generic)
- **Hook:** `Editor.compass_axes() -> list[(label, xyz_direction, rgb)]`; the default is x/y/z in red, green, blue.
- **Drawing:** in the corner of the scene rect on ImGui's foreground draw list, projected with the camera's
  right/up vectors, axes behind drawn first, labels at the tips.
- **Settings:** `rc.ui.compass`, `rc.ui.compass_size` and the corner, all in Preferences.
- **Use:** every app gets the xyz compass; spinw_studio adds a, b, c (normalised columns of A).
- **Limit:** it's not in offline renders (ImGui only); note this in the README.

## 4. Scene and editor
- **`SpinwScene(system, options)`:**
  - a `Spheres` per label (Info: label, cell index, uvw with translation, xyz, |m|, m xyz, S, extras);
  - an arrow `Instances` per label (`instance_matrices(..., z_axis=m)`);
  - 12 cylinders for the cell edges;
  - `batches()`, `bounding_radius()`, `center()` (the cell centre), and `options()` returning the current plot_options.
- **`SpinwEditor(Editor)`:**
  - **Windows:**
    - Structure: six uvw extent sliders, atom count, cell show/thickness;
    - Atoms: label list, radius, material picker/duplicate, `material_section`;
    - Moments: show, length, colour.
  - **Undo:** every edit is an undoable action (`set_bounds`, `set_radius`, `assign`, `set_moments`, `set_cell`)
    with undo-table tests.
  - **File menu:** "Export plot options…" writes `plot_options_<name>_<time>.py` with `PLOT_OPTIONS = {...}`
    (`valo.offline.py_literal`). valo's Render PNG works too; Export render script stays disabled (can't rebuild `sw`).
  - **Compass:** `compass_axes()` adds a, b, c.

## 5. API (`spinw_studio/__init__.py` → `api.py`)
```python
plot_structure(sw_or_view, plot_options=None, *, title=None) -> dict   # opens the studio, blocks, returns final plot_options
render(sw_or_view, plot_options=None, *, path=None, **image) -> ndarray   # same picture, no window (valo.render)
```
Both accept a `SpinW` (through the adapter) or a `SystemView`.

**`lfo_spinw.py` changes:**
- `from spinw_studio import plot_structure` replaces `spinwaves.plotting`;
- keep `boundaries` and `coupling_colors`, and add `atom_colors={'Fe': 'Orange'}` and `show_moments=True`;
- `plot_options = plot_structure(sw, plot_options=plot_opts)`, dropping `engine='valo'`;
- add a `__main__` with placeholder `lmfit.Parameters` (all 0) calling `load_system(params, show_struct=True)`.

## Implementation steps (each a commit of ≤ 5 files, green)
1. `build: spinw dependency group`: `pyproject.toml`, `uv.lock`. **Blocked: see DEPENDENCIES.md.**
2. `feat: corner compass in every app`: `valo/ui.py`, `valo/config.py`, `examples/valorc.toml`, `tests/test_app.py`.
3. `feat: spinw_studio data contract and adapter stubs`: `spinw_studio/{__init__,model,adapter}.py`,
   `test_spinw_studio.py` (pure tests: expand/bounds/translations, cell edges, options validation, colours).
4. `feat: spinw_studio scene and editor`: `scene.py`, `editor.py`, tests (GPU: picture, undo table, Info rows, compass).
5. `feat: plot_structure / render / export plot options`: `api.py`, `__init__.py`, tests (returned options rebuild
   the same image).
6. `feat: lfo_spinw uses spinw_studio`: `lfo_spinw.py`.
7. `docs: spinw_studio README (defining document)`: `README.md` + pointers in CLAUDE.md and ARCHITECTURE.md.

## Verification
- **Tests:**
  - `uv run pytest`: everything passes without spinwaves, using a fake `SystemView` (4 Fe atoms of a Pbnm-like
    cell, `A = diag(5.3, 5.6, 7.5)`);
  - `uv run --group spinw pytest` also imports spinwaves; the adapter test skips until the stubs are filled.
- **By eye:** offscreen render of the fake LFO-like system (atoms, arrows, cell edges, compass), looked at.
- **Your step:** after filling the stubs, run `uv run --group spinw python examples/spinw_studio/lfo_spinw.py`.
