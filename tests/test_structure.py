"""Guards for the structure of valo: the public API, the layering of its modules, and "every editor action has an
undo test". They keep refactors honest: a rename or a new dependency direction has to be made here, on purpose."""

import ast
import inspect
from pathlib import Path

import pytest

import valo
from valo import meshes

VALO = Path(valo.__file__).parent

# -- the public API ------------------------------------------------------------------------------------------------

API = ["ICON", "SCHEMA", "SHADINGS", "Command", "Editor", "Group", "InstancedObject", "Instances", "Lighting",
       "Material", "MeshData", "MeshObject", "OrbitCamera", "ProceduralGroup", "Renderer", "Scene", "SceneEditor",
       "StudioEditor", "ShapeFile", "Shape", "Spheres", "Transform", "UndoStack", "Viewer", "instance_matrices",
       "library", "linear_to_srgb", "meshes", "mono_font", "node_from_settings", "plain", "py_literal", "rc",
       "rc_context", "render", "renderer", "save_png", "srgb_to_linear"]
MESHES_API = ["STANDARD", "MeshData", "MeshReport", "ShapeFile", "arrow", "capsule", "check", "cone", "cube",
              "cylinder",
              "heightfield", "icosphere", "library", "merge", "parametric", "plane", "revolve", "sphere", "torus",
              "transform", "tube"]

# How the main entry points are called: parameter names, kinds and defaults (annotations left out on purpose).
SIGNATURES = {
    "Viewer": "(canvas=None, title='valo', size=(1360, 800), layout_file='auto')",
    "Viewer.show": "(self, scene, editor=None)",
    "Scene": "(renderer=None)",
    "Scene.add": "(self, node, parent=None, index=None)",
    "Scene.from_settings": "(d, renderer=None)",
    "Transform": "(translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))",
    "Shape": "(source, params=None)",
    "MeshObject": "(name, shape, material, *, info=None, **kw)",
    "InstancedObject": "(name, shape, material, instances=None, **kw)",
    "ProceduralGroup": "(name, builder, params=None, *, ranges=None, context=None, **kw)",
    "Group": "(name, *, visible=True, transform=None, locked=False)",
    "OrbitCamera": "(azimuth=-55.0, elevation=25.0, distance=14.0, target=(0.0, 0.0, 0.0), fovy=35.0, "
                   "projection='perspective', ortho_height=10.0, _anim=None)",
    "render": "(scene, settings=None, *, size=None, samples=None, transparent=None, background=None, camera=None, "
              "lighting=None, path=None)",
    "Renderer.instances": "(self, mesh, material, label=None, info=None)",
    "Instances.set": "(self, matrices)",
    "SceneEditor": "(scene)",
    "StudioEditor": "(scene)",
    "instance_matrices": "(positions, scale=1.0, z_axis=None)",
    "node_from_settings": "(d)",
    "meshes.revolve": "(profile, segments=32)",
    "meshes.parametric": "(f, nu=64, nv=64, u=(0.0, 1.0), v=(0.0, 1.0), flip=False)",
    "meshes.heightfield": "(f, size=1.0, n=64)",
    "meshes.tube": "(path, radius=0.1, segments=16, closed=False, caps=True)",
    "meshes.sphere": "(n_lat=48, n_lon=96, r=0.5)",
    "meshes.cube": "(size=1.0)",
    "meshes.arrow": "(shaft_r=0.15, head_r=0.5, head_len=0.35, segments=24)",
}


def _call_shape(f) -> str:
    sig = inspect.signature(f)
    params = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]
    return str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty))


def _lookup(dotted: str):
    obj = valo
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def test_the_public_api_keeps_its_names():
    assert valo.__all__ == API
    assert meshes.__all__ == MESHES_API
    assert all(hasattr(valo, n) for n in API) and all(hasattr(meshes, n) for n in MESHES_API)


@pytest.mark.parametrize("name", list(SIGNATURES))
def test_entry_points_keep_how_they_are_called(name):
    assert _call_shape(_lookup(name)) == SIGNATURES[name]


# -- layering ------------------------------------------------------------------------------------------------------

# Lower layers never import higher ones: math ← meshes ← gpu ← scene ← (offline, ui) ← studio; config, testing and the
# package __init__ are outside the order. Keys are module paths inside valo (packages as "meshes" prefixes).
LAYERS = {
    "math": 0, "geometry": 0,
    "meshes": 1,
    "render": 2, "materials": 2,
    "scene": 3, "lighting": 3,
    "offline": 4, "ui": 4, "ui_style": 4, "undo": 4, "app": 4, "scene_editor": 4,
    "studio": 5,
}
OUTSIDE = {"config", "testing", "__init__"}
KNOWN_VIOLATIONS = {  # to remove in the package refactor (they are why it exists)
    ("render", "lighting"),  # the renderer will take a Lighting as an argument instead of making one
}


def _module(path: Path) -> str:
    rel = path.relative_to(VALO).with_suffix("")
    return ".".join(p for p in rel.parts if p != "__init__") or "__init__"


def _layer(module: str):
    for key in sorted(LAYERS, key=len, reverse=True):
        if module == key or module.startswith(key + "."):
            return LAYERS[key]
    return None


def _imports(path: Path, module: str) -> set[str]:
    """valo modules imported by ``path`` (relative imports resolved; imports anywhere in the file count)."""
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
                names = [target] if node.module else [f"{target}.{a.name}".strip(".") for a in node.names]
            elif node.module and node.module.split(".")[0] == "valo":
                names = [node.module.removeprefix("valo").strip(".")]
            else:
                continue
            out |= {n for n in names if n}
        elif isinstance(node, ast.Import):
            out |= {a.name.removeprefix("valo").strip(".") for a in node.names if a.name.split(".")[0] == "valo"}
    return {n for n in out if n}


def test_every_module_has_a_layer():
    modules = {_module(p) for p in VALO.rglob("*.py")}
    assert {m for m in modules if _layer(m) is None and m.split(".")[0] not in OUTSIDE} == set()


def test_lower_layers_never_import_higher_ones():
    bad = set()
    for path in VALO.rglob("*.py"):
        module = _module(path)
        mine = _layer(module)
        if mine is None:
            continue
        for target in _imports(path, module):
            theirs = _layer(target)
            if theirs is not None and theirs > mine and (module, target.split(".")[0]) not in KNOWN_VIOLATIONS:
                bad.add((module, target))
    assert bad == set()


def test_known_violations_are_still_real():  # so the list shrinks as they are fixed
    found = {(_module(p), t.split(".")[0]) for p in VALO.rglob("*.py") for t in _imports(p, _module(p))}
    assert KNOWN_VIOLATIONS <= found


def test_valo_never_imports_the_examples():
    for path in VALO.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        absolute = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level]
        absolute += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in absolute if m.split(".")[0] in ("examples", "spinw_studio", "crystal")], path


# -- every editor action has a row in an undo table ----------------------------------------------------------------

NOT_ACTIONS = {
    # windows, menus and drawing: UI only, they call the actions
    "windows", "panel", "menus", "all_windows", "gui", "file_menu", "draw_compass", "draw_boxes", "material_section",
    "compass_axes", "build_layout", "reset_layout", "camera_window", "lighting_window", "info_window",
    "history_window", "render_window", "preferences_window", "scene_window", "object_window", "materials_window",
    "code_window",
    # queries
    "render_settings", "can_export", "diagnostics", "users", "dirty", "code_error", "normal_info",
    # plumbing and view state (not scene edits, so not recorded)
    "watch", "one_step", "poll", "reload_shape", "update_normals", "file_dialog", "select", "open_code",
    "browse_shape",
    # documented as not recorded: typing (the code widget has its own undo), preferences, files written out
    "set_code", "set_preference", "set_ui_scale", "render_image", "export_script",
    # class settings
    "title", "right_column", "dock_bottom", "bottom_share", "dock_left", "left_column",
}


def _called(table) -> set[str]:
    """Names called by each row's action: rows are (action, changes image) or (setup, action, changes image)."""
    out = set()
    for row in table.values():
        action = row[1] if len(row) == 3 else row[0]
        out |= set(action.__code__.co_names)
    return out


def test_every_editor_action_is_in_an_undo_table():
    import test_app
    import test_scene
    import test_valo_studio

    tested = _called(test_app.EDITOR_ACTIONS) | _called(test_scene.SCENE_ACTIONS) | _called(
        test_valo_studio.STUDIO_ACTIONS)
    public = {n for cls in (valo.Editor, valo.SceneEditor, valo.StudioEditor) for n in vars(cls)
              if not n.startswith("_")}
    assert public - NOT_ACTIONS - tested == set()
    assert NOT_ACTIONS & tested == set()  # an allow-listed name that is tested belongs out of the list
