## VLM Run Python SDK

Welcome to the **VLM Run Python SDK**, a powerful and intuitive Python interface for interacting with VLM Run's Vision Language Model APIs. This SDK provides seamless integration capabilities for processing images, videos, and documents using state-of-the-art vision language models.

<p align="center">
<a href="https://vlm.run"><b>Website</b></a> | <a href="https://docs.vlm.run/"><b>Docs</b></a> | <a href="https://docs.vlm.run/blog"><b>Blog</b></a> | <a href="https://discord.gg/CCY8cYNC"><b>Discord</b></a>
</p>
<p align="center">
<a href="https://pypi.org/project/vlmrun/"><img alt="PyPI Version" src="https://badge.fury.io/py/vlmrun.svg"></a>
<a href="https://pypi.org/project/vlmrun/"><img alt="PyPI Version" src="https://img.shields.io/pypi/pyversions/vlmrun"></a>
<a href="https://www.pepy.tech/projects/vlmrun"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/vlmrun"></a><br>
<a href="https://github.com/vlm-run/vlmrun-python-sdk/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/vlm-run/vlmrun-python-sdk.svg"></a>
<a href="https://discord.gg/4jgyECY4rq"><img alt="Discord" src="https://img.shields.io/badge/discord-chat-purple?color=%235765F2&label=discord&logo=discord"></a>
<a href="https://twitter.com/vlmrun"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/vlmrun.svg?style=social&logo=twitter"></a>
</p>

## 💡 Motivation

The VLM Run Python SDK is designed to make it easy for developers to integrate vision language models into their applications. It provides:

- 🔑 **Simple Authentication**: Easy-to-use client setup with API key management
- 🎯 **Type Safety**: Full type hints and modern Python typing support
- 🔄 **Async Support**: Both synchronous and asynchronous API interfaces
- 🛠️ **Utility Functions**: Helper functions for common image processing tasks
- 🔌 **Extensible Design**: Easy to extend and customize for your needs

## 🚀 Getting Started

### Installation

```bash
pip install vlmrun
```

### Basic Usage

```python
from vlmrun.client import Client
from vlmrun.hub.schemas.document.invoice import Invoice

# Initialize the client
client = Client(api_key="your-api-key")

# Process an image
response = client.image.generate(
    image="https://example.com/invoice.jpg",
    model="vlm-1",
    domain="document.invoice",
    json_schema=Invoice.model_json_schema(),
)
```

<details>
<summary>Advanced Usage Examples</summary>

### Image Utilities

```python
from vlmrun.common.image import encode_image, download_image
from PIL import Image

# Convert image to base64 or binary
image = Image.open("image.jpg")
base64_str = encode_image(image, format="PNG")  # or format="JPEG"
binary_data = encode_image(image, format="binary")

# Download image from URL
image = download_image("https://example.com/image.jpg")
```

### Async Client

```python
from vlmrun.client import AsyncClient

async def process_image():
    client = AsyncClient(api_key="your-api-key")
    response = await client.image.generate(
        image="https://example.com/image.jpg",
        model="vlm-1",
        domain="document.invoice"
    )
    return response
```

</details>

## 📂 Directory Structure

```bash
vlmrun/
├── client/           # Client implementation
│   ├── sync.py      # Synchronous client
│   └── async.py     # Asynchronous client
├── common/          # Common utilities
│   ├── auth.py      # Authentication utilities
│   └── image.py     # Image processing utilities
└── types/           # Type definitions
    └── responses.py # API response types
```

## ✨ Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

## 🔗 Quick Links

* 💬 Need help? Email us at [support@vlm.run](mailto:support@vlm.run) or join our [Discord](https://discord.gg/4jgyECY4rq)
* 📚 Check out our [Documentation](https://docs.vlm.run/)
* 📣 Follow us on [Twitter](https://x.com/vlmrun) and [LinkedIn](https://www.linkedin.com/company/vlm-run)
