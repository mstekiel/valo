# Developing valo

How to set up, check, version and release valo, and keep its GitHub pages tidy. For the design and where code goes, see
`ARCHITECTURE.md`; for the rules Claude Code follows, `CLAUDE.md` (they apply to humans too).

## Setup and daily checks
uv only (never pip, venv, conda or poetry).
```bash
uv sync                          # engine + dev tools
uv sync --inexact --group spinw  # also spinwaves (local path) for examples/spinw_studio/lfo_spinw.py
uv run ruff check --fix .
uv run pytest -rs                # -rs lists skipped tests: GPU tests must run on a machine with a GPU
```
`uv sync` without `--inexact` removes groups you installed before (e.g. spinw); add `--inexact` to keep them.

**Commits:**
- small: at most 3–5 files;
- each one green on its own;
- conventional messages: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`;
- stage explicit paths, not `git add -A`;
- add every user-facing change to `CHANGELOG.md` under `## [Unreleased]`, in the same series of commits.

## Versions
[Semantic Versioning](https://semver.org/), `MAJOR.MINOR.PATCH`, in `pyproject.toml` (`version = ...`).
- **While at 0.x:**
  - **minor** (0.2.0 → 0.3.0) for new features, and for anything that changes behaviour or the public API (e.g.
    0.2.0 changed what an instanced object's Transform does);
  - **patch** (0.2.0 → 0.2.1) for fixes only.
- **1.0.0** when the public API in `valo/__init__.py` is meant to stay stable. After that, breaking changes need a
  major bump.
- `tests/test_structure.py` guards the public API: a change there is a hint that the minor version must go up.

## Releasing
**With Claude Code:** run `/release <version>` (`.claude/skills/release/SKILL.md`). It does steps 1–5 below and
stops before anything is pushed. **By hand:** follow the same steps.

1. **Preconditions:**
   - on `main`, with a clean `git status`;
   - the tag `v<version>` doesn't exist (`git tag -l v<version>`);
   - the last CI run on `main` passed.
2. **Checks:** `uv sync --inexact`, `uv run ruff check .`, `uv run pytest -rs`. All green, and the GPU tests ran
   (not skipped).
3. **Version and changelog:**
   - set `version` in `pyproject.toml`, then run `uv lock` (it updates valo's own entry in `uv.lock`);
   - in `CHANGELOG.md`, rename `## [Unreleased]` to `## [<version>] - <YYYY-MM-DD>` and add an empty
     `## [Unreleased]` above it;
   - complete the entries from `git log v<previous>..HEAD --format=%s`, grouped for users under Added / Changed /
     Removed / Fixed;
   - update the links at the bottom: `[Unreleased]` → `v<version>...HEAD`, and a new `[<version>]` →
     `v<previous>...v<version>`.
4. **Build and smoke test:**
   ```bash
   rm -rf dist
   uv build
   ```
   - **The wheel** (`dist/valo-<version>-py3-none-any.whl`) holds only `valo/` and its `dist-info`. It includes
     `licenses/LICENSE`, `valo/data/procedural_meshes/*.py` and `valo/assets/icons/valo.png`.
   - **The sdist** has no `.venv`, `.claude/`, `CLAUDE.md` or `__pycache__`.
   - **From a folder outside the repo:**
     ```bash
     uv run --isolated --no-project --with <path to the wheel> python -c "from valo.meshes import library; print(library.names(), library.mesh('vase').positions.shape)"
     ```
5. **Commit and tag:**
   ```bash
   git add pyproject.toml uv.lock CHANGELOG.md
   git commit -m "chore: release v<version>"
   git tag -a v<version> -m "valo <version>"
   ```
6. **Push**, the branch first:
   ```bash
   git push origin main
   ```
   Wait until the CI run of that push passes (GitHub → Actions). If it fails: delete the local tag
   (`git tag -d v<version>`), fix and commit, and tag again. Then:
   ```bash
   git push origin v<version>
   ```
7. **GitHub release** (github.com/mstekiel/valo → Releases → Draft a new release):
   - choose the tag `v<version>`, titled `valo <version>`;
   - paste that version's CHANGELOG section as the notes;
   - attach the two files in `dist/` (wheel and sdist);
   - tick "Set as the latest release";
   - publish.
8. **Optional, PyPI:** `uv publish` with your own PyPI token (`UV_PUBLISH_TOKEN`). Check the name `valo` is free
   or yours first.

A tag that is already pushed is never moved. A broken release gets a new patch version.

## Housekeeping on GitHub
**After each release:**
- **Releases:** the new one is marked latest, its notes match the CHANGELOG, and both `dist/` files are attached.
- **Actions:** the CI run for `main` is green. Re-run or fix failed runs before starting new work.
- **Issues and pull requests:**
  - close the ones the release fixed, with a comment naming the version (e.g. "fixed in v0.2.0");
  - label open ones (`bug`, `enhancement`, `question`; `upstream` for problems in wgpu-py / imgui-bundle);
  - close stale questions.
- **Upstream workarounds:** valo carries fixes for bugs in other libraries, each a small function with a comment
  naming the library and version. Check whether those bugs are fixed upstream; if so, raise the dependency's
  minimum in `pyproject.toml` and remove the workaround. Known ones:
  - **wgpu-py's ImGui backend uploads a new texture only partly** (`valo/ui/viewer.py`,
    `_upload_new_textures_whole`);
  - **the same backend never frees destroyed textures** (not worked around; harmless).

**Now and then:**
- **Repository "About"** (the gear on the repo page): one-line description, topics (`3d`, `rendering`, `wgpu`,
  `imgui`, `scientific-visualization`, `python`), website link if there is one.
- **Branch protection for `main`** (Settings → Branches): require the CI check to pass before merging pull
  requests; no force pushes.
- **Dependencies:** `uv lock --upgrade`, run the checks, and commit `uv.lock` on its own (`chore: update
  dependencies`). wgpu and imgui-bundle are pinned to a minor range in `pyproject.toml` on purpose: read their
  changelogs before widening it.
- **GitHub Pages:** not set up. When the docs grow beyond README / ARCHITECTURE / DEV, publish them from a `docs/`
  folder (Settings → Pages → deploy from branch `main`, folder `/docs`), and link the site in "About" and
  `pyproject.toml` (`[project.urls]`).

## Files that must stay current
| File | Update when |
|---|---|
| `CHANGELOG.md` | every user-facing change (under Unreleased) |
| `ARCHITECTURE.md` | design changes, new extension points |
| `CLAUDE.md` | modules or rules change |
| `examples/valorc.toml` | a new `rc` key (a test checks that every key is documented there) |
| `examples/spinw_studio/README.md` | spinw_studio's contract or behaviour changes |
| `DEV.md` | this workflow changes (keep `.claude/skills/release/SKILL.md` in step) |
