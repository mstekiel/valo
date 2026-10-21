"""The studio's Code window (an in-app Python editor for the shape file of the selected object, saved with Ctrl+S),
and its File menu: New shape, Open shape file (the system file dialog), Save to library."""

from __future__ import annotations

from pathlib import Path

from imgui_bundle import imgui
from imgui_bundle import imgui_color_text_edit as text_edit

from ..meshes import library
from ..scene import MeshObject
from ..ui.style import ICON, mono_font

ERROR_COLOR = imgui.ImVec4(1.0, 0.45, 0.4, 1.0)


class CodeWindow:
    """StudioEditor part: the Code window, the File menu and its dialogs."""

    def browse_shape(self) -> None:
        """Open the system's file dialog (in the library folder) to pick a shape file; the studio keeps running and
        ``open_shape`` follows when you choose one. Without a system dialog, a path field opens instead."""
        if self._browsing is not None:
            return
        try:
            self._browsing = self.file_dialog("Open shape file", str(library.directory()),
                                              ["Python shape files (*.py)", "*.py", "All files", "*"])
        except Exception:
            self._dialog = ("open", "")

    def _poll_browsing(self) -> None:
        if self._browsing is None or not self._browsing.ready(0):
            return
        paths, self._browsing = self._browsing.result(), None
        if paths:
            try:
                self.open_shape(Path(paths[0]))
            except Exception as exc:
                self.status = f"! could not open {Path(paths[0]).name}: {exc}"

    def file_menu(self) -> None:
        """File menu items (valo adds Render below them)."""
        if imgui.menu_item_simple(f"{ICON.FILE} New shape..."):
            self._dialog = ("new", "")
        if imgui.menu_item_simple(f"{ICON.FOLDER_OPEN} Open shape file..."):
            self.browse_shape()
        obj = self.selected
        can_save = isinstance(obj, MeshObject) and obj.shape.file is not None
        if imgui.menu_item(f"{ICON.BOOK} Save to library as...", "", False, can_save)[0]:
            self._dialog, self._confirm_overwrite = ("save", Path(str(obj.shape.source)).stem), False
        imgui.separator()

    def _dialogs(self) -> None:
        if self._dialog is None:
            return
        kind, text = self._dialog
        title = {"new": "New shape", "open": "Open shape file", "save": "Save to library"}[kind]
        if not imgui.is_popup_open(title):
            imgui.open_popup(title)
        if not imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        hint = {"new": f"name: a file in {library.directory()}", "open": "path of a .py shape file",
                "save": f"library name: {library.directory()}/<name>.py"}[kind]
        imgui.text_disabled(hint)
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        done, text = imgui.input_text("##text", text, imgui.InputTextFlags_.enter_returns_true)
        self._dialog = (kind, text)
        if self.status.startswith("! "):
            imgui.text_colored(ERROR_COLOR, self.status[2:])
        ok = imgui.button("OK") or done
        if kind == "save" and self._confirm_overwrite:
            imgui.same_line()
            ok = imgui.button(f"{ICON.TRIANGLE_EXCLAMATION} Overwrite") or ok
        if ok:
            try:
                if kind == "new":
                    self.new_shape(text.strip())
                elif kind == "open":
                    self.open_shape(Path(text.strip().strip('"')))
                else:
                    self.save_to_library(self.selected, text.strip(), overwrite=self._confirm_overwrite)
                self._dialog = None
                imgui.close_current_popup()
            except FileExistsError as exc:
                self._confirm_overwrite = kind == "save"
                self.status = f"! {exc}"
            except Exception as exc:
                self.status = f"! {exc}"
        imgui.same_line()
        if imgui.button("Cancel"):
            self._dialog = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _sync_widget(self):
        if self._widget is None:
            self._widget = text_edit.TextEditor()
            self._widget.set_language(text_edit.TextEditor.Language.python())
            self._widget.set_palette(text_edit.TextEditor.get_dark_palette())
            self._widget.set_tab_size(4)
            self._widget.set_insert_spaces_on_tabs(True)
        if self.code != self._in_widget:  # changed outside the widget: opened, saved, undone, set_code
            self._widget.set_text(self.code)
            self._in_widget = self.code = self._widget.get_text()
            if self.saved_code.rstrip("\n") == self.code.rstrip("\n"):
                self.saved_code = self.code  # the widget normalises the final newline: not a change
        return self._widget

    def code_window(self) -> None:
        """The shape file being edited: Save (Ctrl+S), the text, and the last reload error under it."""
        if self.code_path is None:
            imgui.text_disabled("File > New shape..., or File > Open shape file...,")
            imgui.text_disabled("or select an object built from a shape file.")
            return
        widget = self._sync_widget()
        imgui.begin_disabled(not self.dirty)
        if imgui.button(f"{ICON.FLOPPY_DISK} Save"):
            self.save_code()
        imgui.end_disabled()
        imgui.same_line()
        imgui.text(f"{self.code_path.name}{'  ●' if self.dirty else ''}")
        imgui.same_line()
        imgui.text_disabled(self.status if not self.status.startswith("! ") else "")
        error = self.code_error
        footer = 0 if not error else 6 * imgui.get_text_line_height_with_spacing()
        imgui.push_font(mono_font(), 0.0)
        widget.render("##code", imgui.ImVec2(0, -footer))
        imgui.pop_font()
        if imgui.is_item_focused() or imgui.is_window_focused(imgui.FocusedFlags_.root_and_child_windows):
            io = imgui.get_io()
            if io.key_ctrl and imgui.is_key_pressed(imgui.Key.s):
                self.save_code()
        text = widget.get_text()
        if text != self._in_widget:
            self.code = self._in_widget = text
        if error:
            imgui.push_style_color(imgui.Col_.text, ERROR_COLOR)
            imgui.push_font(mono_font(), 0.0)
            imgui.text_wrapped(error)
            imgui.pop_font()
            imgui.pop_style_color()
