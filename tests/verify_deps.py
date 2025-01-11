"""Verify that dependencies are correctly installed based on extras."""
import sys

def check_base():
    """Verify base installation has no optional dependencies."""
    try:
        import cv2
        print("ERROR: cv2 should not be available in base installation")
        sys.exit(1)
    except ImportError:
        print("SUCCESS: cv2 not available in base installation")

    try:
        import pypdfium2
        print("ERROR: pypdfium2 should not be available in base installation")
        sys.exit(1)
    except ImportError:
        print("SUCCESS: pypdfium2 not available in base installation")

def check_video():
    """Verify video dependencies are available."""
    import cv2
    import numpy
    print("SUCCESS: video dependencies available")

def check_doc():
    """Verify doc dependencies are available."""
    import pypdfium2
    print("SUCCESS: doc dependencies available")

def check_all():
    """Verify all dependencies are available."""
    import cv2
    import numpy
    import pypdfium2
    print("SUCCESS: all dependencies available")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_deps.py [base|video|doc|all]")
        sys.exit(1)

    check_type = sys.argv[1]
    if check_type == "base":
        check_base()
    elif check_type == "video":
        check_video()
    elif check_type == "doc":
        check_doc()
    elif check_type == "all":
        check_all()
    else:
        print(f"Unknown check type: {check_type}")
        sys.exit(1)
