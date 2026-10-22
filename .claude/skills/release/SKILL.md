---
name: release
description: Prepare a valo release (version bump, CHANGELOG, checks, build, wheel smoke test, release commit and annotated tag), stopping before any push. Use when the user asks to release, cut, tag or publish a version of valo.
---

# Release valo

This skill prepares a release locally. **It never pushes and never publishes.** It ends by printing the commands for
the user to run. Stop and report at the first step that fails; do not work around a failing check.

## 1. Preconditions
- The release version comes from the user, or ask for it. Follow semver: while at 0.x, a minor bump (0.1 → 0.2) may
  break the API; a patch bump (0.1.0 → 0.1.1) is for fixes only.
- `git status --short` must be clean, apart from untracked files the user knows about. Uncommitted work means stop
  and ask.
- You must be on the main branch (`git branch --show-current`). The tag `v<version>` must not exist yet
  (`git tag -l v<version>`).

## 2. Checks (all must pass)
The same checks CI runs (`.github/workflows/ci.yml`), locally:
```bash
uv sync --inexact
uv run ruff check .
uv run pytest -rs
```
`--inexact` keeps extra groups the user installed (e.g. `--group spinw`); a plain `uv sync` would uninstall them.
Check pytest's own exit code, not the exit code of a pipe. Report the test count, the skips (`-rs`: GPU tests must
run here, not skip) and the duration; the suite should stay under ~30 s.

## 3. Version and changelog
- `pyproject.toml`: set `version = "<version>"`, then run `uv lock` (this updates valo's own entry in `uv.lock`;
  `uv lock` asks for permission).
- `CHANGELOG.md`:
  - Move the entries under `## [Unreleased]` into a new `## [<version>] - <YYYY-MM-DD>` section (today's date), and
    leave `## [Unreleased]` empty above it. For the first release, give the existing `[0.1.0] - Unreleased` heading
    a date instead.
  - Sections: Added / Changed / Fixed / Removed.
  - Write the entries from `git log v<previous>..HEAD --format=%s` (all of history for the first release), grouped
    by feature for users, not one line per commit.
  - Update the link references at the bottom: add `[<version>]`, and point `[Unreleased]` at `v<version>...HEAD`.
- If the links still say `OWNER`, ask the user for the GitHub owner and replace it everywhere (`pyproject.toml`
  `[project.urls]`, `CHANGELOG.md`).

## 4. Build and smoke-test the wheel
```bash
rm -rf dist
uv build
uv run --isolated --no-project --with dist/valo-<version>-py3-none-any.whl python -c "import valo, valo.meshes; print(valo.__name__, len(valo.__all__))"
uv run --isolated --no-project --with dist/valo-<version>-py3-none-any.whl valo-config --help
```
- **Check the wheel** (open it with `zipfile`):
  - it contains only `valo/` and `valo-<version>.dist-info/`;
  - `dist-info/licenses/LICENSE` is there;
  - `METADATA` has the new version.
- **Check the sdist** (`tar tzf dist/valo-<version>.tar.gz`): no `.venv`, `out/`, `.coverage`, `__pycache__`, `.claude/` or `CLAUDE.md`.
- **Check the built-in meshes:** the wheel has `valo/data/procedural_meshes/*.py`, and from a folder outside the
  repo, `uv run --isolated --no-project --with dist/valo-<version>-py3-none-any.whl python -c "from valo.meshes import library; print(library.names(), library.mesh('vase').positions.shape)"`
  lists them and builds one.

## 5. Release commit and tag
```bash
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore: release v<version>"
git tag -a v<version> -m "valo <version>"
```
End the commit message with the session's attribution line.

## 6. Hand over to the user (do not run these)
Print the following, with the version filled in, in this order:
```bash
git push origin main
```
Then wait for the CI run of that push to pass (GitHub → Actions). Only then:
```bash
git push origin v<version>
```
Then: on GitHub, create the release from tag `v<version>`, paste the CHANGELOG section as the notes and attach the
files in `dist/`. Publishing to PyPI (`uv publish`) needs the user's own API token: the user runs it.
