"""valo.gpu.material: the parameter schema fits the uniform struct, colour conversion, parameters clamp and rebuild."""

import numpy as np
import pytest

from valo.gpu.material import ALL_SPECS, SCHEMA, linear_to_srgb, srgb_to_linear


def test_schema_slots_fit_the_material_struct_without_overlap():
    for model, specs in SCHEMA.items():
        used = [i for s in specs for i in range(s.offset, s.offset + s.size)]
        assert len(used) == len(set(used)), model
        assert max(used) < 16, model
    assert {s.key for s in ALL_SPECS} >= {s.key for specs in SCHEMA.values() for s in specs}


def test_srgb_round_trip():
    for c in (0.0, 0.002, 0.2, 0.5, 1.0):
        assert srgb_to_linear(linear_to_srgb(c)) == pytest.approx(c, abs=1e-9)


@pytest.mark.gpu
def test_material_parameters_clamp_round_and_rebuild_from_dict(viewer):
    m = viewer.renderer.material("M", "toon", base_color=(2.0, -1.0, 0.5), bands=3.6, ambient=7.0)
    assert m.get("base_color") == (1.0, 0.0, 0.5)  # clamped to the schema range
    assert m.get("bands") == 4.0 and m.get("ambient") == 1.0  # ints round, floats clamp
    seen = []
    m.listeners.append(lambda *event: seen.append(event[1:]))
    m.set("bands", 4.2)  # rounds to the same value: no event
    m.set("bands", 6)
    assert seen == [("bands", 4.0, 6.0)]
    with pytest.raises(ValueError):
        m.set("shading", "chrome")
    d = m.to_dict()
    copy = viewer.renderer.material("copy", d.pop("shading"), **d)
    assert np.allclose(copy.data, m.data, atol=1e-4)
