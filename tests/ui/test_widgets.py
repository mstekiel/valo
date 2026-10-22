"""valo.ui.widgets: the Blender-like slider (drag in steps, soft and hard limits, Esc, range overrides)."""

from types import SimpleNamespace

import pytest

from valo.ui.widgets import SliderSpec, fill_fraction, limit, parse_number, snap

VIEWER_SIZE = (640, 400)


@pytest.mark.parametrize(("value", "step", "origin", "expected"), [
    (0.37, 0.25, 0.0, 0.25), (0.38, 0.25, 0.0, 0.5), (1.3, 0.5, 0.1, 1.1), (0.123, None, 0.0, 0.123), (7.0, 2, 1, 7)])
def test_snap_rounds_to_the_step_grid(value, step, origin, expected):
    assert snap(value, step, origin) == pytest.approx(expected)


def test_typed_values_may_leave_the_soft_range_but_not_the_hard_limits():
    spec = SliderSpec(0.0, 1.0, hard_lo=0.0)
    assert limit(5.0, spec) == 5.0 and limit(-1.0, spec) == 0.0  # soft hi, hard lo
    assert limit(2.6, SliderSpec(1, 4, hard_hi=4, integer=True)) == 3
    assert limit(9, SliderSpec(1, 4, hard_hi=4, integer=True)) == 4


def test_parse_number_and_fill_fraction():
    assert parse_number(" 2.5 ") == 2.5 and parse_number("abc") is None and parse_number("nan") is None
    assert fill_fraction(0.5, SliderSpec(0, 2)) == 0.25 and fill_fraction(9, SliderSpec(0, 2)) == 1.0
    assert fill_fraction(10.0, SliderSpec(1.0, 100.0, log=True)) == pytest.approx(0.5)


def test_range_overrides_change_the_slider_and_can_be_reset():
    from valo import Editor

    editor, spec = Editor(), SliderSpec(0.0, 1.0, 0.5)
    editor.set_slider_range("shape.vase.height", hi=4.0, step=0.5)
    assert editor.slider_spec("shape.vase.height", spec) == SliderSpec(0.0, 4.0, 0.5, step=0.5)
    editor.set_slider_range("shape.vase.height", step=0)  # smooth again
    assert editor.slider_spec("shape.vase.height", spec).step is None
    with pytest.raises(ValueError):
        editor.set_slider_range("shape.vase.height", lo=5.0)  # lo >= hi
    with pytest.raises(KeyError):
        editor.set_slider_range("shape.vase.height", width=2)
    editor.set_slider_range("shape.vase.height")
    assert editor.slider_spec("shape.vase.height", spec) == spec


@pytest.mark.gpu
def test_dragging_moves_in_steps_and_escape_puts_the_value_back(viewer):
    from imgui_bundle import imgui

    from valo import Editor

    editor, state = Editor(), {"value": 2.0}
    spec = SliderSpec(0.0, 10.0, 2.0, step=1.0)

    def panel():
        state["bar"] = imgui.calc_item_width()  # the bar's width, without the label
        changed, v = editor.slider("test.value", "Value", state["value"], spec)
        if changed:
            state["value"] = v
        state["rect"] = (imgui.get_item_rect_min(), imgui.get_item_rect_max())

    editor.panel = panel
    viewer.show(SimpleNamespace(batches=list), editor)
    viewer.snapshot()
    imgui.set_current_context(viewer.imgui.imgui_context)
    io = imgui.get_io()
    lo, hi = state["rect"]
    x, y, width = lo.x + 4, (lo.y + hi.y) / 2, state["bar"]

    def move(dx, down=None):
        io.add_mouse_pos_event(x + dx, y)
        if down is not None:
            io.add_mouse_button_event(0, down)
        viewer.canvas.draw()

    move(0)
    move(0, True)
    for k in range(1, 6):
        move(width * 0.33 * k / 5)  # a third of the bar ≈ a third of the range: from 2 to about 5.3
    assert state["value"] == pytest.approx(round(state["value"])) and 4 <= state["value"] <= 6  # whole steps
    io.add_key_event(imgui.Key.escape, True)
    move(width * 0.33)
    io.add_key_event(imgui.Key.escape, False)
    move(width * 0.33, False)
    assert state["value"] == 2.0 and viewer.gui_error is None
