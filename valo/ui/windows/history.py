"""The shared History window: undo / redo and the list of recorded edits."""

from __future__ import annotations

from imgui_bundle import imgui

from ..style import ICON


class HistoryWindow:
    """Editor part: the History window."""

    def history_window(self) -> None:
        """Undo / redo buttons and the list of recorded edits."""
        stack = self.undo_stack
        imgui.begin_disabled(not stack.done)
        if imgui.button(f"{ICON.ROTATE_LEFT} Undo"):
            stack.undo()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not stack.undone)
        if imgui.button(f"{ICON.ROTATE_RIGHT} Redo"):
            stack.redo()
        imgui.end_disabled()
        for cmd in reversed(stack.undone):  # newest redo at the top, greyed out
            imgui.text_disabled(f"  {cmd.label}")
        for i, cmd in enumerate(reversed(stack.done)):
            imgui.text(f"{ICON.CARET_RIGHT if i == 0 else ' '} {cmd.label}")
        if not stack.done and not stack.undone:
            imgui.text_disabled("No edits yet")
