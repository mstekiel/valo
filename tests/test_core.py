"""Fast tests: pure Python/numpy, no GPU."""

import numpy as np
import pytest

from valo.geometry import pack_mat4, perspective_reversed_z, translate
from valo.materials import ALL_SPECS, SCHEMA, linear_to_srgb, srgb_to_linear
from valo.undo import Command, UndoStack


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
