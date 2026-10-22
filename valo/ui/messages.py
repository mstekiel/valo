"""Messages for the user: ``editor.notify(text, level, detail)`` shows a short box at the mouse and keeps the message
(with its detail, e.g. a traceback) for the Console window.

The box looks like ImGui's tooltip with a coloured edge (``rc.ui`` info / warning / error colours). It stays where it
appeared until the mouse moves away from that spot (``rc.ui.message_move_px``), a button or key is pressed, or, if
``rc.ui.message_seconds`` > 0, that many seconds have passed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from imgui_bundle import imgui

from ..config import rc
from .style import ICON

LEVELS = ("info", "warning", "error")
LEVEL_ICONS = {"info": ICON.CIRCLE_CHECK, "warning": ICON.TRIANGLE_EXCLAMATION, "error": ICON.CIRCLE_XMARK}
GRACE_S = 0.3  # clicks and keys within this time of the message don't dismiss it (the click that caused it)


@dataclass
class Message:
    """One message: level ("info" | "warning" | "error"), the short text, optional detail, and when (local time)."""

    text: str
    level: str = "info"
    detail: str = ""
    stamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


def level_color(level: str) -> imgui.ImVec4:
    """The colour of a message level, from ``rc.ui`` (sRGB 0-255) as an ImGui colour."""
    r, g, b = getattr(rc.ui, f"{level}_color")
    return imgui.ImVec4(r / 255, g / 255, b / 255, 1.0)


class Messages:
    """Editor part: ``notify`` and the box it shows at the mouse; ``messages`` keeps them for the Console."""

    clock = staticmethod(time.monotonic)  # seconds; tests replace it

    @property
    def messages(self) -> deque:
        """Every message so far, oldest first (at most ``rc.ui.console_lines``)."""
        if "_messages" not in self.__dict__ or self._messages.maxlen != rc.ui.console_lines:
            self._messages = deque(self.__dict__.get("_messages", ()), maxlen=rc.ui.console_lines)
        return self._messages

    def notify(self, text: str, level: str = "info", detail: str = "") -> Message:
        """Tell the user ``text`` in a box at the mouse (``level``: "info" green, "warning" orange, "error" red);
        ``detail`` (e.g. a traceback) goes only to the Console. Returns the message."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
        message = Message(text, level, detail)
        self.messages.append(message)
        self.toast = (message, self.clock(), None)  # anchor: the mouse position, taken when it is first drawn
        return message

    def draw_toast(self) -> None:
        """Draw the current message box, or drop it once the user has moved on (called every frame)."""
        toast = getattr(self, "toast", None)
        if toast is None:
            return
        message, since, anchor = toast
        io = imgui.get_io()
        mouse = (io.mouse_pos.x, io.mouse_pos.y)
        valid = imgui.is_mouse_pos_valid()
        if anchor is None:
            anchor = mouse if valid else (imgui.get_main_viewport().get_center().x, 80.0)
            self.toast = (message, since, anchor)
        age = self.clock() - since
        moved = valid and ((mouse[0] - anchor[0]) ** 2 + (mouse[1] - anchor[1]) ** 2) ** 0.5 > rc.ui.message_move_px
        pressed = age > GRACE_S and (any(imgui.is_mouse_clicked(b) for b in range(3)) or _any_key_pressed())
        if moved or pressed or (rc.ui.message_seconds > 0 and age > rc.ui.message_seconds):
            self.toast = None
            return
        color = level_color(message.level)
        imgui.set_next_window_pos(imgui.ImVec2(anchor[0] + 16, anchor[1] + 12))
        imgui.push_style_color(imgui.Col_.border, color)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 2.0)
        flags = (imgui.WindowFlags_.tooltip | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize
                 | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_inputs | imgui.WindowFlags_.no_nav
                 | imgui.WindowFlags_.no_focus_on_appearing)
        if imgui.begin("##message", None, flags)[0]:
            imgui.text_colored(color, LEVEL_ICONS[message.level])
            imgui.same_line()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 30)
            imgui.text_unformatted(message.text)
            imgui.pop_text_wrap_pos()
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color()


def _any_key_pressed() -> bool:
    """Whether any keyboard key went down this frame (mouse buttons excluded)."""
    return any(imgui.is_key_pressed(imgui.Key(k), False)
               for k in range(imgui.Key.named_key_begin, imgui.Key.reserved_for_mod_ctrl)
               if not imgui.Key.mouse_left <= k <= imgui.Key.mouse_wheel_y)
