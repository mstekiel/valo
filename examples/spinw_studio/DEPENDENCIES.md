# Brief: make `spinwaves` installable next to valo (Python 3.12, numpy 2)

Paste this into the agent that works on spinwaves.

---

**Task.** Make the `spinwaves` library installable as a local, editable dependency of the **valo** project, and
make it work there. valo runs on Python 3.12 with numpy 2. spinwaves currently pins exact, old versions of its
dependencies, so it can't be installed in that environment.

**Repositories**
- spinwaves (to change): `C:/Users/stekiel.JCNSOPC-MICHAL.001/Documents/GitHub/spinwaves`, version 0.2.0, has a `pyproject.toml`.
- valo (the consumer; don't change it beyond its dependency settings):
  `C:/Users/stekiel.JCNSOPC-MICHAL.001/Documents/tests/valo-skeleton/valo`. Managed with **uv only**
  (never pip/venv/conda/poetry). Windows 11, uv 0.9.18.

**What valo requires** (fixed, from its `pyproject.toml`):
- `requires-python = ">=3.12"`; the interpreter is CPython 3.12.12;
- `numpy>=2.0`, `wgpu>=0.32,<0.33`, `imgui-bundle>=1.92,<1.92.900`, `glfw>=2.11`, `rendercanvas>=2.7`,
  `pillow>=12.3`;
- the app also uses `lmfit>=1.3.4`, which needs `scipy>=1.10`.

valo adds spinwaves to a dependency group, as the user specified:
```toml
[dependency-groups]
spinw = ["lmfit>=1.3.4", "spinwaves"]

[tool.uv.sources]
spinwaves = { path = "C:/Users/stekiel.JCNSOPC-MICHAL.001/Documents/GitHub/spinwaves", editable = true }
```

**What went wrong** (run in the valo folder: `uv add --group spinw --editable <spinwaves path>`)
1. `spinwaves==0.2.0 depends on numpy==1.23.5` conflicts with valo's `numpy>=2.0`. numpy 1.23.5 also has no
   Python 3.12 wheels.
2. With a uv override for numpy, the next failure: `spinwaves==0.2.0 depends on scipy==1.9.1`, while lmfit 1.3.4
   needs `scipy>=1.10`. scipy 1.9.1 has no 3.12 wheels either.
3. With numpy and scipy overridden, resolution succeeds, but the install fails building **`matplotlib==3.5.2`**
   from source (`error: command 'msbuild' failed`), and **`vispy==0.12.1`** also builds from source. Neither has
   Python 3.12 wheels.

So spinwaves pins **exact old versions** of at least numpy, scipy, matplotlib and vispy. There may be more pins
that weren't reached yet.

**Goal / acceptance criteria**
1. **Loosen the pins in spinwaves' `pyproject.toml`.** Use lower bounds that allow current versions, e.g.
   `numpy>=1.26` (or `>=2.0`), `scipy>=1.10`, and modern matplotlib/vispy, and check every other dependency the
   same way. Keep exact pins only if something truly requires them, and say why.
2. **Make plotting backends optional if they aren't needed for the physics.** The 3D plotting moves to valo
   (`spinw_studio`), so `matplotlib` and `vispy` could become an optional extra such as
   `spinwaves[plot]`. The core (Atom, Crystal, MSG, Coupling, SpinW, DMI) then installs without them. Decide based
   on the code, and report what you chose.
3. **Make the code run on numpy 2 and current scipy.** Watch for numpy 2 removals and renames: `np.float_`,
   `np.complex_`, `np.NaN`/`np.Inf`, `np.in1d` → `np.isin`, `np.product` → `np.prod`, `np.cumproduct`,
   `np.alltrue`/`np.sometrue`, `np.asfarray`, `np.trapz` → `np.trapezoid`, copy semantics of
   `np.array(..., copy=False)`, and dtype promotion (NEP 50). Also scipy API changes since 1.9, and any
   matplotlib/vispy API changes if they stay.
4. **Verify from the valo folder**, after the user (or you, if allowed) removes valo's temporary overrides
   (see "Current state of valo" below):
   ```bash
   uv add --group spinw --editable "C:/Users/stekiel.JCNSOPC-MICHAL.001/Documents/GitHub/spinwaves"
   uv sync --group spinw
   uv run --group spinw python -c "import spinwaves; from spinwaves import Atom, Crystal, SpinW, Coupling; from spinwaves.symmetry.magnetic_symmetry import MSG; from spinwaves.utils.linalg import DMI; print('ok')"
   uv run --group spinw python -c "import sys; sys.path.insert(0, 'examples/spinw_studio'); import lfo_spinw"
   ```
   Also run spinwaves' own test suite, if it has one, on Python 3.12 / numpy 2.
5. **Don't change valo's own requirements** (numpy 2, Python 3.12). valo must keep working:
   `uv run pytest` in the valo folder should still pass (113 tests).

**Context: what valo will need from spinwaves later.** This isn't needed for this task, but helps you know what
must keep working. `examples/spinw_studio/lfo_spinw.py` (in valo) builds a system with:
- `Atom(label, r, m, s)`;
- `MSG.from_xyz_strings(generators=[...])`;
- `Crystal(lattice_parameters=[a, b, c, alpha, beta, gamma], atoms, MSG)`;
- `Coupling(label, n_uvw, id1, id2, J)`;
- `DMI([...])`;
- `SpinW(crystal, couplings, magnetic_modulation={'k': ..., 'n': ...})` and `sw.couplings_all`.

Later, valo's adapter will need from a `SpinW` object:
- the lattice matrix A (r_xyz = A · r_uvw, columns a, b, c in Å);
- all atom positions of one cell after symmetry, in fractional uvw, with an order matching `Coupling(id1, id2)`;
- their magnetic moments, converted to Cartesian xyz;
- labels and spin lengths.

If you notice how to get these cleanly, a short note is welcome (don't implement valo-side code).

**Current state of valo** (uncommitted, left as-is for you or the user to finish):
- `pyproject.toml` has `[dependency-groups] spinw = ["lmfit>=1.3.4"]` (spinwaves not added yet).
- `pyproject.toml` has temporary overrides, which should be **removed** once spinwaves' pins are fixed:
  ```toml
  [tool.uv]
  override-dependencies = ["numpy>=2.0", "scipy>=1.10"]
  ```
- `uv.lock` changed accordingly (lmfit and its dependencies added).
