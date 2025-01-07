# vlmrun-python-sdk
Python SDK for VLMRun.

## Installation

Basic installation:
```bash
pip install vlmrun
```

With optional torch dependencies (required for video utilities):
```bash
pip install vlmrun[torch]
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

### Video Utilities
Requires torch extras: `pip install vlmrun[torch]`
```python
from vlmrun.common import VideoItertools
import cv2

# Initialize video frame sampler
video = VideoItertools()

# Sample unique frames from video
cap = cv2.VideoCapture("video.mp4")
frames = (cap.read()[1] for _ in iter(lambda: cap.isOpened(), False))
unique_frames = video.islice(frames, step=10, similarity_threshold=0.9)

# Process unique frames
for frame in unique_frames:
    # Your frame processing code here
    pass
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

### Video Utilities
Requires torch extras: `pip install vlmrun[torch]`
```python
from vlmrun.common import VideoItertools
import cv2

# Initialize video frame sampler
video = VideoItertools()

# Sample unique frames from video
cap = cv2.VideoCapture("video.mp4")
frames = (cap.read()[1] for _ in iter(lambda: cap.isOpened(), False))
unique_frames = video.islice(frames, step=10, similarity_threshold=0.9)

# Process unique frames
for frame in unique_frames:
    # Your frame processing code here
    pass
```
