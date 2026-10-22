"""valo.ui.messages: notify keeps messages and shows a box at the mouse until the user moves on."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu
VIEWER_SIZE = (640, 400)


@pytest.fixture
def editor(viewer):
    from valo import Editor

    ed = Editor()
    ed.now = 100.0
    ed.clock = lambda: ed.now  # seconds, moved by the test
    viewer.show(SimpleNamespace(batches=list), ed)
    viewer.snapshot()
    viewer.gui_error = None
    return ed


def frame(viewer, mouse=(300, 200)):
    from imgui_bundle import imgui

    imgui.set_current_context(viewer.imgui.imgui_context)
    imgui.get_io().add_mouse_pos_event(*mouse)
    viewer.canvas.draw()


def test_notify_keeps_every_message_with_its_level_and_detail(editor):
    editor.notify("saved vase.py")
    editor.notify("vase.py doesn't build yet", "warning")
    editor.notify("boom", "error", "Traceback ...")
    assert [(m.level, m.text, m.detail) for m in editor.messages] == [
        ("info", "saved vase.py", ""), ("warning", "vase.py doesn't build yet", ""), ("error", "boom", "Traceback ...")]
    with pytest.raises(ValueError):
        editor.notify("?", "debug")


def test_the_box_stays_until_the_mouse_moves_away(viewer, editor):
    from valo import rc

    frame(viewer)
    editor.notify("saved")
    for _ in range(3):
        frame(viewer, (300 + rc.ui.message_move_px / 2, 200))  # small moves near it: it stays
    editor.now += 600  # no time limit by default
    frame(viewer, (300 + rc.ui.message_move_px / 2, 200))
    assert editor.toast is not None and viewer.gui_error is None
    frame(viewer, (300 + rc.ui.message_move_px * 2, 200))
    assert editor.toast is None


def test_the_box_closes_after_message_seconds_or_a_click(viewer, editor):
    from imgui_bundle import imgui

    from valo import rc_context

    with rc_context({"ui.message_seconds": 2.0}):
        editor.notify("rendered")
        frame(viewer)
        editor.now += 1.0
        frame(viewer)
        assert editor.toast is not None
        editor.now += 1.5
        frame(viewer)
        assert editor.toast is None
    editor.notify("again", "warning")
    frame(viewer)
    editor.now += 1.0  # after the grace time: a click is the user moving on
    io = imgui.get_io()
    io.add_mouse_button_event(0, True)
    frame(viewer)
    io.add_mouse_button_event(0, False)
    frame(viewer)
    assert editor.toast is None


def test_an_error_in_a_window_becomes_an_error_message(viewer, editor):
    def broken():
        raise RuntimeError("widget failed")

    editor.panel = broken
    viewer.gui_error = None
    frame(viewer)
    assert editor.messages[-1].level == "error" and editor.messages[-1].text == "RuntimeError: widget failed"
    assert "Traceback" in editor.messages[-1].detail
    editor.panel = lambda: None
    frame(viewer)


def test_the_console_lists_messages_and_copies_them_as_text(viewer, editor):
    editor.notify("saved vase.py")
    editor.log("> hello from mesh()")
    editor.notify("build failed", "error", "Traceback (most recent call last):\n  line 3\nValueError: bad\n")
    frame(viewer)  # the Console is a shared window, open by default
    assert editor.visible["Console"] and viewer.gui_error is None
    lines = editor.console_text().splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == [
        "INFO    saved vase.py", "INFO    > hello from mesh()", "ERROR   build failed",
        "   Traceback (most recent call last):", "     line 3", "   ValueError: bad"]
    editor.messages.clear()
    assert editor.console_text() == ""
