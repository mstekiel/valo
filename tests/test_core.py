"""Fast tests: pure Python/numpy, no GPU."""

import numpy as np
import pytest

from valo.geometry import instance_matrices, pack_mat4, perspective_reversed_z, translate
from valo.materials import ALL_SPECS, SCHEMA, linear_to_srgb, srgb_to_linear
from valo.undo import Command, UndoStack


def test_instance_matrices_point_z_along_axis_with_given_length():
    origins = np.array([[0, 0, 0], [1, 2, 3], [-1, 0, 0], [0, 0, 5], [2, 2, 2]], float)
    vectors = np.array([[1, 0, 0], [0, 2, 0], [0, 0, -3], [0, 0, 4], [1, 1, 1]], float)
    length = np.linalg.norm(vectors, axis=1)
    m = instance_matrices(origins, np.c_[0.1 * np.ones(5), 0.1 * np.ones(5), length], z_axis=vectors)
    tips = np.einsum("nij,j->ni", m, [0, 0, 1, 1])[:, :3]
    assert np.allclose(tips, origins + vectors, atol=1e-5)
    rot = m[:, :3, :3] / np.array([0.1, 0.1, 1])[None, :] / np.c_[np.ones(5), np.ones(5), length][:, None, :]
    assert np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    assert np.allclose(np.linalg.det(rot), 1.0, atol=1e-5)


def test_pack_mat4_puts_translation_where_wgsl_expects_it():
    """WGSL mat4x4f is column-major: column 3 (translation) is floats 12..14."""
    assert np.allclose(pack_mat4(translate(1, 2, 3)).ravel()[12:15], [1, 2, 3])


def test_reversed_z_depth_is_one_at_near_plane():
    p = perspective_reversed_z(40.0, 1.0, near=0.1) @ np.array([0, 0, -0.1, 1.0])
    assert p[2] / p[3] == pytest.approx(1.0)


def test_schema_slots_fit_the_material_struct_without_overlap():
    for model, specs in SCHEMA.items():
        used = [i for s in specs for i in range(s.offset, s.offset + s.size)]
        assert len(used) == len(set(used)), model
        assert max(used) < 16, model
    assert {s.key for s in ALL_SPECS} >= {s.key for specs in SCHEMA.values() for s in specs}


def test_srgb_round_trip():
    for c in (0.0, 0.002, 0.2, 0.5, 1.0):
        assert srgb_to_linear(linear_to_srgb(c)) == pytest.approx(c, abs=1e-9)


def test_undo_merges_rapid_edits_of_same_key():
    value = {"x": 0}
    stack = UndoStack()

    def edit(new):
        old, value["x"] = value["x"], new
        stack.push(Command("x", lambda: value.update(x=old), lambda: value.update(x=new), merge_key=("x",)))

    for v in (1, 2, 3):  # like a slider drag
        edit(v)
    assert len(stack.done) == 1
    stack.undo()
    assert value["x"] == 0
    stack.redo()
    assert value["x"] == 3


def test_orthographic_reversed_z_maps_near_to_one_and_far_to_zero():
    from valo.geometry import orthographic_reversed_z

    m = orthographic_reversed_z(4.0, 2.0, 1.0, 9.0)
    for z, depth in ((-1.0, 1.0), (-9.0, 0.0)):
        p = m @ np.array([0, 0, z, 1.0])
        assert p[2] / p[3] == pytest.approx(depth)
    assert (m @ np.array([4.0, 2.0, -5, 1]))[:2] == pytest.approx([1, 1])  # half width = height·aspect/2


def test_switching_projection_keeps_the_target_plane_the_same_size():
    from valo.render import OrbitCamera

    cam = OrbitCamera(distance=10.0)
    point = np.asarray(cam.target) + np.cross(cam.forward(), (0, 0, 1)) * 1.5  # beside the target, on its plane

    def ndc():
        vp, _ = cam.matrices(1.5)
        p = vp @ np.r_[point, 1.0]
        return p[:2] / p[3]

    persp = ndc()
    cam.set_projection("orthographic")
    assert ndc() == pytest.approx(persp, abs=1e-5)
    cam.zoom(2.0)
    cam.set_projection("perspective")
    assert cam.distance == pytest.approx(20.0)


def test_headlight_follows_the_camera_and_fixed_light_does_not():
    from valo.lighting import Lighting
    from valo.render import OrbitCamera

    cam = OrbitCamera()
    head, fixed = Lighting("headlight", 0.0, 0.0), Lighting("fixed", 30.0, 40.0)
    assert head.direction(cam) == pytest.approx(-cam.forward())  # azimuth 0, elevation 0: from the viewer
    before = fixed.direction(cam)
    cam.azimuth += 120
    assert head.direction(cam) == pytest.approx(-cam.forward())
    assert fixed.direction(cam) == pytest.approx(before)


@pytest.mark.parametrize("start", ["headlight", "fixed"])
def test_switching_light_mode_keeps_the_direction(start):
    from valo.lighting import MODES, Lighting
    from valo.render import OrbitCamera

    cam = OrbitCamera(azimuth=37.0, elevation=-12.0)
    light = Lighting(start, -50.0, 25.0)
    d = light.direction(cam)
    light.set_mode(next(m for m in MODES if m != start), cam)
    assert light.direction(cam) == pytest.approx(d, abs=1e-9)
