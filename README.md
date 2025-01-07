# vlmrun-python-sdk
Python SDK for VLMRun.

## Installation

```bash
pip install vlmrun
```

## Usage

### Client
```python
from vlmrun.client import Client

client = Client()
```

### Image Utilities
```python
from vlmrun.common import encode_image, download_image
from PIL import Image

# Convert image to base64 or binary
image = Image.open("image.jpg")
base64_str = encode_image(image, format="PNG")  # or format="JPEG"
binary_data = encode_image(image, format="binary")

# Download image from URL
image = download_image("https://example.com/image.jpg")
```
