"""valo studio: the base studio app. valo's Scene / Object / Materials windows plus a Code window for procedural
meshes.

    uv run valo-studio                 # an empty scene; File > New shape starts a procedural mesh
    uv run valo-studio my_shape.py     # open a shape file"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..scene import Scene
from .editor import StudioEditor


def build(viewer, path=None) -> StudioEditor:
    """An empty studio scene in ``viewer`` (plus an object for shape file ``path``, if given); returns the editor."""
    editor = StudioEditor(Scene(viewer.renderer))
    viewer.show(editor.scene, editor)
    if path is not None:
        editor.open_shape(path)
        viewer.fit_all()
    return editor


def main(argv=None) -> None:
    """Command-line entry point: ``valo-studio [shape.py] [--size WxH]``."""
    from ..ui.viewer import Viewer

    ap = argparse.ArgumentParser(prog="valo-studio", description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path, nargs="?", help="a shape file to open: a .py file with a mesh(...) function")
    ap.add_argument("--size", default="1600x900", help="window size, e.g. 1920x1080")
    args = ap.parse_args(argv)
    if args.file is not None and not args.file.exists():
        ap.error(f"{args.file} does not exist")
    w, h = (int(x) for x in args.size.lower().split("x"))
    viewer = Viewer(title="valo studio", size=(w, h))
    build(viewer, args.file)
    viewer.run()


if __name__ == "__main__":
    main()
