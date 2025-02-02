<div align="center">
<p align="center" style="width: 100%;">
    <img src="https://raw.githubusercontent.com/vlm-run/.github/refs/heads/main/profile/assets/vlm-black.svg" alt="VLM Run Logo" width="80" style="margin-bottom: -5px; color: #2e3138; vertical-align: middle; padding-right: 5px;"><br>
</p>
<h2>Python SDK</h2>
<p align="center"><a href="https://docs.vlm.run"><b>Website</b></a> | <a href="https://docs.vlm.run/"><b>Docs</b></a> | <a href="https://docs.vlm.run/blog"><b>Blog</b></a> | <a href="https://discord.gg/AMApC2UzVY"><b>Discord</b></a>
</p>
<p align="center">
<a href="https://pypi.org/project/vlmrun/"><img alt="PyPI Version" src="https://badge.fury.io/py/vlmrun.svg"></a>
<a href="https://pypi.org/project/vlmrun/"><img alt="PyPI Version" src="https://img.shields.io/pypi/pyversions/vlmrun"></a>
<a href="https://www.pepy.tech/projects/vlmrun"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/vlmrun"></a><br>
<a href="https://github.com/vlm-run/vlmrun-python-sdk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/vlm-run/vlmrun-python-sdk.svg"></a>
<a href="https://discord.gg/AMApC2UzVY"><img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord"></a>
<a href="https://twitter.com/vlmrun"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/vlmrun.svg?style=social&logo=twitter"></a>
</p>
</div>

The [VLM Run Python SDK](https://pypi.org/project/vlmrun/) is the official Python SDK for [VLM Run API platform](https://docs.vlm.run), providing a convenient way to interact with our REST APIs.


## 🚀 Getting Started

### Installation

```bash
pip install vlmrun
```

### Installation with Optional Features

The package provides optional features that can be installed based on your needs:

- Video processing features (numpy, opencv-python):
  ```bash
  pip install "vlmrun[video]"
  ```

- Document processing features (pypdfium2):
  ```bash
  pip install "vlmrun[doc]"
  ```

- All optional features:
  ```bash
  pip install "vlmrun[all]"
  ```

### Basic Usage

```python
from vlmrun.client import VLMRun
from vlmrun.hub.schemas.document.invoice import Invoice

# Initialize the client
client = VLMRun(api_key="your-api-key")

# Process an image
response = client.image.generate(
    image="https://example.com/invoice.jpg",
    model="vlm-1",
    domain="document.invoice",
    json_schema=Invoice.model_json_schema(),
)
```

### Image Utilities

```python
from vlmrun.common.image import encode_image
from vlmrun.common.utils import download_image
from PIL import Image

# Convert image to base64 or binary
image = Image.open("image.jpg")
base64_str = encode_image(image, format="PNG")  # or format="JPEG"
binary_data = encode_image(image, format="binary")

# Download image from URL
image = download_image("https://example.com/image.jpg")
```

</details>

## 📂 Directory Structure

```bash
vlmrun/
├── client/               # Client implementation
│   ├── client.py         # Main VLMRun class
│   ├── base_requestor.py # Low-level request logic
│   ├── files.py          # File operations
│   ├── models.py         # Model operations
│   ├── finetune.py       # Fine-tuning operations
│   └── types.py          # Type definitions
├── common/              # Common utilities
│   ├── auth.py          # Authentication utilities
│   └── image.py         # Image processing utilities
└── types/              # Type definitions
    └── abstract.py     # Abstract base classes
```

## 🔗 Quick Links

* 💬 Need help? Email us at [support@vlm.run](mailto:support@vlm.run) or join our [Discord](https://discord.gg/AMApC2UzVY)
* 📚 Check out our [Documentation](https://docs.vlm.run/)
* 📣 Follow us on [Twitter](https://x.com/vlmrun) and [LinkedIn](https://www.linkedin.com/company/vlm-run)
