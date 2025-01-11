"""Tests for verifying correct installation of optional dependencies."""

import pytest


def test_base_dependencies():
    """Verify base installation has no optional dependencies."""
    with pytest.raises(ImportError):
        import cv2  # noqa: F401

    with pytest.raises(ImportError):
        import pypdfium2  # noqa: F401


def test_video_dependencies():
    """Verify video dependencies are available."""
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    # Verify we can import and get versions
    assert cv2.__version__, "cv2 version should be available"
    assert np.__version__, "numpy version should be available"


def test_doc_dependencies():
    """Verify doc dependencies are available."""
    import pypdfium2  # noqa: F401

    # Verify we can import and get version
    assert pypdfium2.__version__, "pypdfium2 version should be available"


def test_all_dependencies():
    """Verify all dependencies are available."""
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    import pypdfium2  # noqa: F401

    # Verify we can import and get versions
    assert cv2.__version__, "cv2 version should be available"
    assert np.__version__, "numpy version should be available"
    assert pypdfium2.__version__, "pypdfium2 version should be available"
