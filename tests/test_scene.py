"""The scene object model: tree, world transforms, visibility, names, shapes and procedural groups (needs a GPU)."""

import numpy as np
import pytest

from valo import meshes
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
