"""Shared pytest fixtures for valo's tests and the example apps' tests (built on ``valo.testing``)."""

import pytest

from valo import testing


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
    """The procedural mesh library redirected to an empty temporary folder (never the real data/ one); its path."""
    from valo import rc

    folder = tmp_path / "library"
    monkeypatch.setattr(rc.studio, "procedural_meshes", str(folder))
    return folder
