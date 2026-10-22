---
name: feature
description: Implement one or more feature requests (or bug reports) for valo to the project's standards — Blender-like behaviour, a plan for multi-module work, a test for every change, offscreen visual checks, small green commits, docs kept current. Use when the user asks for new valo features, changes or fixes, e.g. "/feature add a grid floor" or a numbered list of requests.
---

# Feature requests for valo

The user lists one or more features or bugs. Deliver each one to the standards below. CLAUDE.md rules still apply
in full; this skill adds the habits learned while building valo. **Never push.**

## 1. Understand every item before touching code
- Split the request into numbered items. Restate each in one line, and say what "done" means for it.
- **Bugs:** reproduce first. Usually that is an offscreen script in the scratchpad that renders the state and saves
  a PNG; look at it. Find the cause before fixing. If it reproduces only on the user's machine, say what differs
  (DPI / pixel ratio, their `~/.valo/valorc.toml`, a real window) and find evidence another way, e.g. by logging the
  calls involved.
- **Explore** the modules involved (see CLAUDE.md's package table) and reuse what exists:
  - `editor.notify` / `log` for messages to the user;
  - `editor.slider(key, label, value, SliderSpec)` for numbers;
  - `rc` fields for tunables;
  - `valo.testing` helpers;
  - `SceneEditor` actions and `one_step` for undo.
- **Ask** (AskUserQuestion) only for real decisions: behaviour with several sensible options, names, locations. Put
  the recommended option first. When the user asks "what do you think", give a recommendation with the reason.

## 2. Blender-like by default
valo follows Blender's model:
- data = `scene` / `meshes`;
- draw engine = `gpu` (+ `ui/overlays`);
- editors = `ui` windows (mixins);
- operators = undoable `Editor` actions.

Before designing the UI or interaction, ask "how does Blender do this?", and say so in the plan:
- **View changes are not undoable:** camera, navigation, selection, slider ranges, window layout.
- **Scene edits are undoable:** objects, transforms, materials, lighting, shape parameters.
- **Selection** draws an object-frame bounding box; **hover** draws a fainter one.
- **Numbers** use the Blender-like slider: drag in steps, soft and hard limits, right-click menu.
- **Feedback** is a message box at the mouse (info / warning / error) plus the Console; there are no status-text
  fields.
- **Keys** live in `rc.keys` and are Blender-like where Blender has a convention.

When Blender has no equivalent, borrow from ParaView, VESTA, Mantid or similar, and say which.

## 3. Plan
- Anything touching more than one module → plan mode first (EnterPlanMode).
- **The plan contains:**
  - Context: why, and what exploring found, with file paths;
  - the design per item;
  - the list of commits, each ≤ 5 files and green on its own: engine + test → app → docs;
  - verification.
- **Several items:**
  - order them so shared groundwork lands first;
  - after approval, follow the system prompt's rule on independent steps (separate sessions only for items that
    touch no common files);
  - items touching `config.py`, `ui/editor.py` or `ui/windows/preferences.py` are not independent.
- **The user's open-plan pane may show an older plan.** Name the plan file in your message.

## 4. Test first, then implement
For every item:
- **Test through the public entry point** (the function, or the Editor action the UI calls). Assert a result, not
  just that it ran.
- **Bug fix:** the test must fail without the fix. Check it once by stashing only the source files
  (`git stash push -- <paths>`, never the whole tree: `uv.lock` and the user's files must not move), then
  `git stash pop`.
- **New Editor action:** add a row to the right undo table (`EDITOR_ACTIONS`, `SCENE_ACTIONS`, `STUDIO_ACTIONS` or
  the app's), so it changes state, undoes exactly (state and image) and redoes. UI-only or view-state methods go on
  `NOT_ACTIONS` in `tests/test_structure.py`, with a reason.
- **Public API or signature changes:** update `tests/test_structure.py` on purpose (`API`, `SIGNATURES`). Never put
  machine-specific values in a signature (use `None` defaults, not absolute paths).
- **New `rc` field:** label, range, unit and help. Document it, commented out, in `examples/valorc.toml`; a test
  checks every key is there.
- **GPU tests:**
  - take the module's shared `viewer` and `pytestmark = pytest.mark.gpu`;
  - compare images only with each other;
  - reset state you change (`viewer.gui_error = None`, `rc_context`, `set_ui_scale(1.0)`);
  - remember the viewer is shared with earlier tests in the module.
- **Where tests go:** CLAUDE.md, Testing §4 (`tests/` mirrors the packages; unique file names).

## 5. Implementation habits
- **Layering:** math ← meshes ← gpu ← scene ← (offline, ui) ← studio. `valo/` never imports `examples/` and knows no
  domain. Apps use the public API.
- **Tunables live in `rc`,** never as module constants. Every public name has a docstring.
- **ImGui:**
  - change style, fonts and scale only between frames (`set_ui_scale` → applied at the next `gui()`);
  - never let a widget raise mid-window. A raised error becomes "Missing End()"; the Viewer recovers and records
    it in `viewer.gui_error` / an error message.
- **Workarounds for third-party bugs** (wgpu, imgui-bundle, rendercanvas) go in valo, as a small function with a
  comment naming the library version and the bug. Tell the user it's worth reporting upstream.
- **Editing files:**
  - write helper scripts with the Write tool, not bash heredocs (`\n` and quotes get mangled);
  - give exact-match replacements an `assert count == 1`.
- **New dependency:** ask the user first (`uv add`). uv only.

## 6. Verify
- **Per commit:** `uv run ruff check --fix .` then `uv run pytest -q`. Gate the commit on pytest's own exit code
  (`${PIPESTATUS[0]}`), not on a pipe. The suite stays under ~30 s.
- **Visual changes:**
  - render offscreen (`valo.testing.offscreen_viewer`, or `build(viewer)` for the studio) and look at the PNG,
    zoomed in where details matter;
  - compare with the state before the change when a glitch might be older.
- **Interaction that only a real window shows** (title bar, icon, file dialogs, DPI): check it in a real window
  (`uv run valo-studio`) when you can. Otherwise say plainly what wasn't verified.
- **Packaging** (assets, data files): `uv build`, then check the wheel's file list.

## 7. Commit and document
- **Conventional messages** (`feat:` `fix:` `refactor:` `test:` `docs:`), ≤ 5 files each, explicit paths. End with
  the session's attribution line.
- **In the same series:**
  - `ARCHITECTURE.md` (design, extension points);
  - `CLAUDE.md` (the package table, when modules change);
  - `CHANGELOG.md` under `## [Unreleased]` (Added / Changed / Fixed / Removed).
- **spinw_studio:** read `examples/spinw_studio/README.md` before changing anything there.

## 8. Report
Close with `mcp__ccd_turn__wrap_up`, then a short report per item:
- what changed, for the user (behaviour, where to find it);
- the commits;
- what was verified and how;
- what wasn't verified;
- anything found along the way (older bugs, upstream bugs) as a follow-up card, not as a silent extra fix.
