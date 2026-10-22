"""The shared Console window: every message (``editor.notify`` / ``editor.log``) with its time, level and detail,
read-only, newest at the bottom."""

from __future__ import annotations

from imgui_bundle import imgui

from ..messages import LEVEL_ICONS, LEVELS, level_color
from ..style import ICON, mono_font


class ConsoleWindow:
    """Editor part: the Console window (a read-only log of the messages)."""

    def console_window(self) -> None:
        """Filters per level, Clear and Copy all, then the messages in the mono font; follows new ones while
        scrolled to the bottom."""
        shown = self.__dict__.setdefault("console_levels", dict.fromkeys(LEVELS, True))
        for level in LEVELS:
            imgui.push_style_color(imgui.Col_.text, level_color(level))
            _, shown[level] = imgui.checkbox(f"{LEVEL_ICONS[level]}##{level}", shown[level])
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Show {level} messages")
            imgui.same_line()
        if imgui.button(f"{ICON.TRASH_CAN} Clear"):
            self.messages.clear()
        imgui.same_line()
        if imgui.button(f"{ICON.COPY} Copy all"):
            imgui.set_clipboard_text(self.console_text())
        imgui.separator()
        imgui.begin_child("##console", imgui.ImVec2(0, 0), 0, imgui.WindowFlags_.horizontal_scrollbar)
        at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1
        imgui.push_font(mono_font(), 0.0)
        for m in self.messages:
            if not shown[m.level]:
                continue
            imgui.text_disabled(m.stamp)
            imgui.same_line()
            imgui.text_colored(level_color(m.level), LEVEL_ICONS[m.level])
            imgui.same_line()
            imgui.text_unformatted(m.text)
            if m.detail:
                imgui.indent()
                imgui.push_style_color(imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled))
                imgui.text_unformatted(m.detail.rstrip("\n"))
                imgui.pop_style_color()
                imgui.unindent()
        imgui.pop_font()
        if at_bottom:
            imgui.set_scroll_here_y(1.0)
        imgui.end_child()

    def console_text(self) -> str:
        """The messages as plain text, one per line with its time and level, details indented (Copy all)."""
        lines = []
        for m in self.messages:
            lines.append(f"{m.stamp} {m.level.upper():7s} {m.text}")
            lines += [f"    {line}" for line in m.detail.rstrip("\n").splitlines()]
        return "\n".join(lines)
