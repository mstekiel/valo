"""The scene object model: tree, world transforms, visibility, names, shapes and procedural groups (needs a GPU)."""

import numpy as np
import pytest

from valo import Transform, meshes
from valo.geometry import transform_matrix

pytestmark = pytest.mark.gpu


@pytest.fixture
def scene():
    pytest.importorskip("wgpu")
    from valo import Scene

    try:
        s = Scene()
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")
    s.material("Matte", "lambert")
    return s


def cube(name="Cube", size=1.0, **kw):
    from valo import MeshObject, Shape

    return MeshObject(name, Shape("valo.meshes.primitives:cube", {"size": size}), "Matte", **kw)


def ring(group, n: int = 3, radius: float = 2.0):
    """A procedural builder: n cubes on a circle."""
    from valo import Transform

    for i in range(n):
        a = 2 * np.pi * i / n
        group.add(cube("Bead", 0.2, transform=Transform((radius * np.cos(a), radius * np.sin(a), 0))))


def test_world_matrix_is_parent_world_times_own_transform(scene):
    from valo import Group, Transform

    g = scene.add(Group("G", transform=Transform((1, 0, 0), (0, 0, 90), (2, 2, 2))))
    c = scene.add(cube(transform=Transform((1, 0, 0))), parent=g)
    expected = transform_matrix((1, 0, 0), (0, 0, 90), (2, 2, 2)) @ transform_matrix((1, 0, 0))
    assert np.allclose(c.world_matrix(), expected)
    assert np.allclose(c.instances.matrices[0] @ [0, 0, 0, 1], [1, 2, 0, 1])  # uploaded: (1,0,0) scaled, turned, moved
    g.transform = Transform((0, 0, 5))
    scene.refresh(g)
    assert np.allclose(c.instances.matrices[0, :3, 3], [1, 0, 5])


def test_instances_are_drawn_at_object_times_instance(scene):
    from valo import InstancedObject, Shape, Transform

    local = np.stack([transform_matrix((x, 0, 0)) for x in (0, 1, 2)])
    obj = scene.add(InstancedObject("Row", Shape(meshes.cube), "Matte", local,
                                    transform=Transform((0, 0, 3), (0, 0, 90))))
    assert np.allclose(obj.instances.matrices[:, :3, 3], [[0, 0, 3], [0, 1, 3], [0, 2, 3]])
    obj.set_instances(local[:2])
    assert obj.instances.count == 2


def test_hidden_ancestors_hide_the_object_and_batches_skip_it(scene):
    from valo import Group

    outer = scene.add(Group("Outer"))
    inner = scene.add(Group("Inner"), parent=outer)
    a, b = scene.add(cube("A"), parent=inner), scene.add(cube("B"))
    assert {x.source.owner for x in scene.batches()} == {a, b}
    outer.visible = False
    assert not a.shown() and [x.source.owner for x in scene.batches()] == [b]
    assert np.allclose(scene.center(), [0, 0, 0])


def test_names_are_unique_and_found(scene):
    from valo import Group, MeshObject, Shape

    names = [scene.add(cube()).name for _ in range(3)]
    assert names == ["Cube", "Cube 2", "Cube 3"]
    assert scene.find("Cube 2").name == "Cube 2" and scene.find("Cube 2").instances.label == "Cube 2"
    assert scene.material("Matte").name == "Matte 2"
    with pytest.raises(KeyError, match="no node"):
        scene.find("nope")
    with pytest.raises(KeyError, match="no material"):
        scene.add(MeshObject("X", Shape(meshes.cube), "Gold"))
    with pytest.raises(TypeError, match="only groups"):
        scene.add(Group("G"), parent=scene.find("Cube"))


def test_changing_a_shape_parameter_rebuilds_the_mesh(scene):
    from valo import Shape

    c = scene.add(cube(size=1.0))
    r0 = c.instances.mesh.radius
    c.shape.params["size"] = 3.0
    c.rebuild_mesh()
    assert np.isclose(c.instances.mesh.radius, 3 * r0)
    assert scene.bounding_radius() == pytest.approx(3 * r0)
    with pytest.raises(KeyError, match="unknown parameters"):
        Shape(meshes.cube, {"sise": 1.0})


def test_material_is_shared_by_name(scene):
    a, b = scene.add(cube("A")), scene.add(cube("B"))
    gold = scene.material("Gold", "pbr", metallic=1.0)
    a.set_material("Gold")
    assert a.instances.material is gold and b.instances.material is scene.materials["Matte"]
    assert scene.users("Gold") == [a]


def test_procedural_group_builds_children_from_its_parameters(scene):
    from valo import ProceduralGroup

    g = scene.add(ProceduralGroup("Ring", ring, {"n": 4}))
    assert [p.name for p in g.specs] == ["n", "radius"]
    assert [c.name for c in g.children] == ["Bead", "Bead 2", "Bead 3", "Bead 4"]
    g.params.update(n=2, radius=5.0)
    g.rebuild()
    assert [c.name for c in g.children] == ["Bead", "Bead 2"] and len(scene.objects()) == 2
    assert np.allclose(g.children[0].instances.matrices[0, :3, 3], [5, 0, 0])
    scene.remove(g)
    assert scene.batches() == []


def labelled_points(group, spread: float = 1.0):
    """A builder using a context: a locked group of one instanced object per label in ``group.context``."""
    from valo import Group, InstancedObject, Shape

    points = group.add(Group("Points", locked=True))
    for label, xs in group.context.items():
        points.add(InstancedObject(label, Shape(meshes.sphere, {"r": 0.1}), "Matte",
                                   np.stack([transform_matrix((spread * x, 0, 0)) for x in xs]), locked=True,
                                   info=lambda i, lb=label: {"Label": lb, "i": str(i)}))


def test_a_rebuild_keeps_its_children_and_the_users_edits(scene):
    from valo import ProceduralGroup, Transform

    scene.material("Gold", "pbr")
    g = scene.add(ProceduralGroup("Ring", ring, {"n": 4}))
    bead, moved, gone = scene.find("Bead 2"), scene.find("Bead 3"), scene.find("Bead 4")
    bead.set_material("Gold")
    bead.visible = False
    bead.shape.params["size"] = 0.5
    moved.transform = Transform((0, 0, 3))
    g.params.update(n=3, radius=5.0)
    g.rebuild()
    assert [c.name for c in g.children] == ["Bead", "Bead 2", "Bead 3"] and gone.scene is None
    assert scene.find("Bead 2") is bead and scene.find("Bead 3") is moved  # the same nodes: undo and selection hold
    assert bead.material == "Gold" and not bead.visible and bead.shape.params["size"] == 0.5  # the user's edits
    assert np.allclose(bead.instances.matrices[0, :3, 3], [5 * np.cos(2 * np.pi / 3), 5 * np.sin(2 * np.pi / 3), 0])
    assert np.allclose(moved.instances.matrices[0, :3, 3], [0, 0, 3])  # an edited transform stays
    assert scene.find("Bead").shape.params["size"] == 0.2


def test_procedural_groups_take_ranges_a_context_and_info(scene):
    from valo import ProceduralGroup

    data = {"Fe": [0, 1], "O": [0.5]}
    g = scene.add(ProceduralGroup("Pts", labelled_points, {"spread": 2.0}, ranges={"spread": (0.5, 8.0)},
                                  context=data, locked=True))
    assert (g.specs[0].lo, g.specs[0].hi) == (0.5, 8.0) and g.locked and scene.find("Fe").locked
    fe = scene.find("Fe")
    assert np.allclose(fe.instances.matrices[:, 0, 3], [0, 2])
    rows = fe.info(1)
    assert rows["Object"] == "Fe" and (rows["Label"], rows["i"]) == ("Fe", "1")  # object rows, then the hook
    data["Fe"].append(2)
    g.rebuild()
    assert scene.find("Fe") is fe and fe.instances.count == 3  # computed data flows in
    with pytest.raises(ValueError, match="context"):
        scene.settings()
    scene.remove(g)
    from valo.scene import node_from_settings

    ranged = scene.add(ProceduralGroup("Ring", ring, ranges={"radius": (1.0, 9.0)}, locked=True))
    again = node_from_settings(ranged.settings())
    assert again.locked and (again.specs[1].lo, again.specs[1].hi) == (1.0, 9.0)


SCULPT = '''
from valo import Transform, meshes

TRANSFORM = {"translate": [0.0, 0.0, 1.0]}


def mesh(size: float = 0.5, tall: bool = False):
    """A box."""
    return meshes.cube(2 * size if tall else size)
'''


def test_settings_round_trip_rebuilds_the_identical_image(scene, tmp_path):
    from valo import Group, InstancedObject, MeshObject, OrbitCamera, ProceduralGroup, Scene, Shape, Transform, render

    path = tmp_path / "box.py"
    path.write_text(SCULPT, encoding="utf-8")
    scene.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0, roughness=0.37)
    g = scene.add(Group("G", transform=Transform((0.5, 0, 0), (0, 0, 30))))
    scene.add(cube("A", 0.7, transform=Transform(scale=(1, 2, 1))), parent=g)
    scene.add(cube("Hidden", visible=False), parent=g)
    scene.add(InstancedObject("Row", Shape(meshes.sphere, {"r": 0.3}), "Gold",
                              np.stack([transform_matrix((x, -1, 0)) for x in (-1, 0, 1)])))
    scene.add(ProceduralGroup("Ring", ring, {"n": 5}))
    box = Shape(path, {"size": 0.4})
    assert [p.name for p in box.specs] == ["size", "tall"]
    assert np.isclose(box.build().positions[:, 2].min(), 0.8)  # the file's TRANSFORM is baked in
    scene.add(MeshObject("Box", box, "Gold"))
    d = scene.settings()
    assert "children" not in d["nodes"][2] and d["nodes"][3]["shape"]["func"] == str(path)

    again = Scene.from_settings(d)
    assert again.settings() == d
    cam = OrbitCamera(distance=8.0, elevation=30.0)
    assert np.array_equal(render(scene, size=(160, 120), camera=cam), render(again, size=(160, 120), camera=cam))


def test_unknown_kinds_and_unimportable_functions_are_reported():
    from valo.scene import node_from_settings, reference, resolve

    assert resolve("valo.meshes.primitives:sphere") is meshes.sphere
    assert reference(meshes.sphere) == "valo.meshes.primitives:sphere"
    with pytest.raises(ValueError, match="not importable"):
        reference(lambda: None)
    with pytest.raises(ValueError, match="unknown node kind"):
        node_from_settings({"kind": "light", "name": "L"})


# -- SceneEditor: every action changes the scene, undoes exactly (state and image) and redoes

@pytest.fixture(scope="module")
def viewer():
    pytest.importorskip("wgpu")
    from rendercanvas.offscreen import RenderCanvas

    from valo import Viewer

    try:
        return Viewer(RenderCanvas(size=(480, 320), pixel_ratio=1))
    except Exception as exc:  # no adapter on this machine
        pytest.skip(f"no GPU adapter: {exc}")


def edited_scene(viewer):
    """A scene with a group, plain and instanced objects, a procedural ring and an unused material, in an editor."""
    from valo import Group, InstancedObject, OrbitCamera, ProceduralGroup, Scene, SceneEditor, Shape, Transform

    scene = Scene()
    scene.material("Matte", "lambert", base_color=(0.6, 0.6, 0.6))
    scene.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
    g = scene.add(Group("G", transform=Transform((0, 0, 1))))
    scene.add(cube("A", 0.6), parent=g)
    scene.add(cube("B", 0.5, transform=Transform((-1.5, 0, 0))))
    scene.add(InstancedObject("Row", Shape(meshes.sphere, {"r": 0.2}), "Matte",
                              np.stack([transform_matrix((x, 1.5, 0)) for x in (-1, 0, 1)])))
    scene.add(ProceduralGroup("Ring", ring, {"n": 3, "radius": 2.5}))
    editor = SceneEditor(scene)
    viewer.show(scene, editor)
    viewer.camera = OrbitCamera(distance=9.0, elevation=30.0)
    return scene, editor


SCENE_ACTIONS = {  # name → (action(editor, scene), changes the image; None: may or may not)
    "add": (lambda ed, s: ed.add(cube("C", transform=Transform((1.5, 0, 0)))), True),
    "add shape": (lambda ed, s: ed.add_shape("valo.meshes.primitives:torus", s.find("G")), True),
    "delete": (lambda ed, s: ed.delete(s.find("A")), True),
    "duplicate": (lambda ed, s: ed.duplicate(s.find("B")), None),  # the copy sits on the original
    "rename": (lambda ed, s: ed.rename(s.find("B"), "Box"), False),
    "reparent": (lambda ed, s: ed.reparent(s.find("B"), s.find("G")), True),  # moves up with G
    "visibility": (lambda ed, s: ed.set_visible(s.find("G"), False), True),
    "transform": (lambda ed, s: ed.set_transform(s.find("B"), "rotate_deg", (0, 0, 45)), True),
    "shape": (lambda ed, s: ed.set_shape(s.find("B"), meshes.icosphere), True),
    "shape parameter": (lambda ed, s: ed.set_shape_param(s.find("A"), "size", 1.2), True),
    "object material": (lambda ed, s: ed.set_object_material(s.find("B"), "Gold"), True),
    "procedural parameter": (lambda ed, s: ed.set_procedural_param(s.find("Ring"), "n", 5), True),
    "new material": (lambda ed, s: ed.new_material("Glass", "blinn_phong"), False),
    "duplicate material": (lambda ed, s: ed.duplicate_material("Gold"), False),
    "delete material": (lambda ed, s: ed.delete_material("Gold"), False),
    "material parameter": (lambda ed, s: ed.set_param(s.materials["Matte"], "base_color", (0, 0, 1)), True),
}


@pytest.mark.parametrize("name", list(SCENE_ACTIONS))
def test_scene_editor_action_undoes_exactly(viewer, name):
    action, visible = SCENE_ACTIONS[name]
    scene, editor = edited_scene(viewer)
    before, image = scene.settings(), viewer.snapshot()
    action(editor, scene)
    after = scene.settings()
    assert after != before
    if visible is not None:
        assert (np.abs(viewer.snapshot().astype(int) - image).max() > 10) == visible
    assert len(editor.undo_stack.done) == 1
    editor.undo_stack.undo()
    assert scene.settings() == before
    assert np.array_equal(viewer.snapshot(), image)
    editor.undo_stack.redo()
    assert scene.settings() == after


def test_scene_editor_refuses_impossible_edits_and_selection_highlights(viewer):
    scene, editor = edited_scene(viewer)
    with pytest.raises(ValueError, match="is used by A, B"):
        editor.delete_material("Matte")
    with pytest.raises(ValueError, match="own subtree"):
        editor.reparent(scene.find("G"), scene.find("G"))
    with pytest.raises(TypeError, match="only groups"):
        editor.reparent(scene.find("G"), scene.find("B"))
    with pytest.raises(KeyError, match="no parameter"):
        editor.set_shape_param(scene.find("A"), "radius", 1.0)
    assert not editor.undo_stack.done
    editor.select(scene.find("Ring"))
    ring = scene.find("Ring")
    assert viewer.outlined == [ring] and [kind for _, kind in viewer.boxes()] == ["selected"]
    beads = np.concatenate([b.instances.matrices[:, :3, 3] for b in ring.children])
    corners = unit_cube_corners(viewer.boxes()[0][0])
    assert (beads >= corners.min(0) - 1e-6).all() and (beads <= corners.max(0) + 1e-6).all()  # one box, all beads
    editor.delete(scene.find("Ring"))
    assert editor.selected is None and viewer.outlined == []




def test_windows_draw_for_every_kind_of_selection(viewer):
    scene, editor = edited_scene(viewer)
    calls = []
    for name, draw in list(editor.windows().items()):
        setattr(editor, draw.__name__, lambda draw=draw, name=name: (calls.append(name), draw()))
    for node in [None, *scene.walk()]:
        editor.select(node)
        for window in ("Object", "Materials"):
            editor._focus = [window]  # bring the tab to the front
            viewer.snapshot()
    assert {"Scene", "Object", "Materials"} <= set(calls)
    assert editor.diagnostics(scene.find("A"))[-1].startswith("watertight")
    assert editor.diagnostics(scene.find("Row"))[0] == "3 instances"
    assert not editor.undo_stack.done  # drawing edits nothing


def test_a_click_in_the_3d_view_selects_the_object(viewer):
    scene, editor = edited_scene(viewer)
    viewer.snapshot()
    x0, y0, w, h = viewer.viewport
    clip = viewer.camera.matrices(w / h)[0] @ np.array([-1.5, 0, 0, 1.0])  # cube B
    r = viewer.canvas.get_pixel_ratio()
    at = ((x0 + (clip[0] / clip[3] + 1) / 2 * w) / r, (y0 + (1 - clip[1] / clip[3]) / 2 * h) / r)
    viewer.on_pointer({"event_type": "pointer_move", "x": at[0], "y": at[1]})
    viewer.snapshot()
    viewer.on_pointer({"event_type": "pointer_down", "button": 1, "x": at[0], "y": at[1], "modifiers": ()})
    viewer.on_pointer({"event_type": "pointer_up", "button": 1, "x": at[0], "y": at[1], "modifiers": ()})
    viewer.snapshot()
    assert editor.selected is scene.find("B") and viewer.selection.source.owner is editor.selected
    assert viewer.outlined == [editor.selected] and [kind for _, kind in viewer.boxes()][0] == "selected"
    viewer.on_key({"event_type": "key_down", "key": "Escape", "modifiers": ()})
    viewer.snapshot()
    assert editor.selected is None and viewer.outlined == []
    viewer.pointer, viewer._last_click = None, None


def test_shapes_from_sculpt_files_reload_when_the_file_changes(viewer, tmp_path):
    import os

    from valo import rc_context

    scene, editor = edited_scene(viewer)
    path = tmp_path / "box.py"
    path.write_text(SCULPT, encoding="utf-8")
    box = editor.add_shape(path)
    editor.set_shape_param(box, "size", 0.8)
    mesh = box.instances.mesh

    def change(text, step):
        path.write_text(text, encoding="utf-8")
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + step))  # filesystems with coarse mtimes
        editor._polled = 0.0
        with rc_context({"studio.poll_s": 0.05}):
            editor.poll()

    change(SCULPT.replace("2 * size if tall else size", "3 * size"), 10)
    assert box.instances.mesh is not mesh and box.shape.params == {"size": 0.8, "tall": False}  # values kept
    width = lambda: np.ptp(box.shape.build().positions[:, 0])  # noqa: E731
    assert np.isclose(width(), 3 * 0.8)
    good = box.instances.mesh
    change("def mesh(:\n", 20)
    assert "SyntaxError" in editor.errors["Box"] and box.instances.mesh is good  # the last good mesh stays
    change(SCULPT, 30)
    assert "Box" not in editor.errors and np.isclose(width(), 0.8) and box.instances.mesh is not good
    change(SCULPT.replace("tall: bool = False", "tall: bool = True"), 40)
    assert box.shape.params == {"size": 0.8, "tall": True}  # your value stays; an untouched one follows the code
    assert len(editor.undo_stack.done) == 2  # add and set_shape_param; reloads are not edits


def test_adding_a_shape_to_an_empty_scene_is_one_undo_step(viewer):
    from valo import Scene, SceneEditor

    scene = Scene()
    editor = SceneEditor(scene)
    viewer.show(scene, editor)
    editor.add_shape(meshes.cube)  # makes a default material too
    assert list(scene.materials) == ["Material"] and len(editor.undo_stack.done) == 1
    editor.undo_stack.undo()
    assert scene.materials == {} and scene.objects() == []
    editor.undo_stack.redo()
    assert list(scene.materials) == ["Material"] and [o.name for o in scene.objects()] == ["Cube"]


def picky_ring(group, n: int = 2):
    """A builder that fails above 3 beads."""
    if n > 3:
        raise RuntimeError("at most 3 beads")
    ring(group, n)


def test_locked_nodes_stay_and_a_failing_builder_changes_nothing(viewer):
    from valo import Group, ProceduralGroup

    scene, editor = edited_scene(viewer)
    pts = scene.add(ProceduralGroup("Pts", labelled_points, context={"Fe": [0, 1]}, locked=True))
    fe = scene.find("Fe")
    for action, match in ((lambda: editor.delete(fe), "not deleted"), (lambda: editor.delete(pts), "not deleted"),
                          (lambda: editor.rename(fe, "X"), "not renamed"),
                          (lambda: editor.duplicate(scene.find("Points")), "not duplicated"),
                          (lambda: editor.reparent(fe, scene.find("G")), "not moved"),
                          (lambda: editor.add(Group("In"), scene.find("Points")), "nothing can be added"),
                          (lambda: editor.reparent(scene.find("B"), pts), "nothing can be added")):
        with pytest.raises(ValueError, match=match):
            action()
    editor.set_visible(fe, False)  # hiding is fine
    editor.set_shape_param(fe, "r", 0.3)  # and so is editing
    assert len(editor.undo_stack.done) == 2

    g = scene.add(ProceduralGroup("Picky", picky_ring))
    beads = list(g.children)
    image = viewer.snapshot()
    editor.set_procedural_param(g, "n", 5)
    assert g.params["n"] == 2 and g.children == beads and "at most 3 beads" in editor.errors["Picky"]
    assert np.array_equal(viewer.snapshot(), image) and len(editor.undo_stack.done) == 2
    editor.set_procedural_param(g, "n", 3)
    assert len(g.children) == 3 and "Picky" not in editor.errors


def test_a_scene_with_a_context_group_renders_but_does_not_export(viewer):
    from valo import ProceduralGroup

    scene, editor = edited_scene(viewer)
    assert editor.can_export()
    scene.add(ProceduralGroup("Pts", labelled_points, context={"Fe": [0, 1]}))
    assert not editor.can_export() and "scene" not in editor.render_settings()


def broken_shape(size: float = 1.0):
    """A shape function with a bug."""
    return points  # noqa: F821 - the bug


def broken_builder(group, n: int = 2):
    """A builder with a bug."""
    raise RuntimeError("builder bug")


def test_a_shape_that_fails_to_build_leaves_the_scene_as_it_was(viewer):
    from valo import MeshObject, ProceduralGroup, Shape

    scene, editor = edited_scene(viewer)
    before, image = scene.settings(), viewer.snapshot()
    with pytest.raises(NameError):
        scene.add(MeshObject("Bad", Shape(broken_shape), "Matte"))
    with pytest.raises(RuntimeError, match="builder bug"):
        scene.add(ProceduralGroup("BadGroup", broken_builder))
    assert scene.settings() == before and np.array_equal(viewer.snapshot(), image)  # still draws
    assert viewer.gui_error is None


def unit_cube_corners(box):
    from valo.geometry import box_corners

    return (box @ np.c_[box_corners([-0.5] * 3, [0.5] * 3), np.ones(8)].T).T[:, :3]


def test_bounding_boxes_live_in_the_nodes_frame(scene):
    from valo import Group, InstancedObject, Shape, Transform

    tilted = Transform((1, 2, 3), (30, 0, 45), (1, 2, 1))
    c = scene.add(cube("C", 0.5, transform=tilted))
    assert np.allclose(c.bounding_box(), tilted.matrix() @ np.diag([0.5, 0.5, 0.5, 1]), atol=1e-6)  # turns with it

    g = scene.add(Group("G", transform=Transform(rotate_deg=(0, 0, 90))))
    scene.add(cube("A", 1.0, transform=Transform((2, 0, 0))), parent=g)
    row = scene.add(InstancedObject("Row", Shape(meshes.cube, {"size": 0.2}), "Matte",
                                    np.stack([transform_matrix((0, y, 0)) for y in (-1, 0, 3)])), parent=g)
    lo, hi = (np.linalg.inv(g.world_matrix()) @ np.c_[unit_cube_corners(g.bounding_box()), np.ones(8)].T)[:3].T[[0, -1]]
    assert np.allclose(lo, [-0.1, -1.1, -0.5]) and np.allclose(hi, [2.5, 3.1, 0.5])  # in the group's own axes
    corners = unit_cube_corners(row.bounding_box())
    assert np.allclose(np.ptp(corners, axis=0), [4.2, 0.2, 0.2])  # the rotated row lies along x in the world
    g.visible = False
    assert g.bounding_box() is None and row.bounding_box() is None
