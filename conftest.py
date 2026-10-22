"""Shared pytest fixtures for valo's tests and the example apps' tests (built on ``valo.testing``)."""

import pytest

from valo import testing


def pytest_configure(config):
    """Tests run with the default config: forget what ``~/.valo/valorc.toml`` or ``./valorc.toml`` set at import."""
    from valo import rc

    rc.reset()


@pytest.fixture(scope="module")
def viewer(request):
    """An offscreen Viewer shared by the tests of one module (its size: the module's ``VIEWER_SIZE``, default
    900 x 560). Skips the test without a GPU adapter."""
    v = testing.offscreen_viewer(getattr(request.module, "VIEWER_SIZE", (900, 560)))
    if v is None:
        pytest.skip("no GPU adapter")
    return v


@pytest.fixture
def tmp_library(tmp_path, monkeypatch):
    """The procedural mesh library redirected to empty temporary folders (never the real ones): the user folder
    (returned) and the built-in one (``library.BUILTIN_DIR``)."""
    from valo import rc
    from valo.meshes import library

    folder = tmp_path / "library"
    monkeypatch.setattr(library, "BUILTIN_DIR", tmp_path / "builtin")
    monkeypatch.setattr(rc.studio, "procedural_meshes", str(folder))
    return folder
