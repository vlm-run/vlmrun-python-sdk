"""Script to create a sample test image for unit tests."""

from PIL import Image
import numpy as np
from pathlib import Path

# Ensure fixtures directory exists
fixtures_dir = Path(__file__).parent / "fixtures"
fixtures_dir.mkdir(exist_ok=True)

# Create a small test image (100x100 black square)
img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
img.save(fixtures_dir / "sample.jpg")
