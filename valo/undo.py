"""Linear undo/redo history. Consecutive edits with the same merge_key within MERGE_S seconds merge
into one step (a slider drag = one undo)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Command:
    label: str
    undo: Callable[[], None]
    redo: Callable[[], None]
    merge_key: tuple | None = None
    stamp: float = field(default_factory=time.monotonic)


class UndoStack:
    MERGE_S = 0.8  # consecutive edits of the same (material, key) within this window merge

    def __init__(self):
        self.done: list[Command] = []
        self.undone: list[Command] = []
        self.applying = False

    def push(self, cmd: Command) -> None:
        if self.applying:
            return
        top = self.done[-1] if self.done else None
        if top and cmd.merge_key and top.merge_key == cmd.merge_key and cmd.stamp - top.stamp < self.MERGE_S:
            top.redo, top.stamp = cmd.redo, cmd.stamp
        else:
            self.done.append(cmd)
        self.undone.clear()

    def _move(self, src, dst, action) -> None:
        if src:
            cmd = src.pop()
            self.applying = True
            try:
                getattr(cmd, action)()
            finally:
                self.applying = False
            dst.append(cmd)

    def undo(self):
        self._move(self.done, self.undone, "undo")

    def redo(self):
        self._move(self.undone, self.done, "redo")
