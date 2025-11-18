<div align="center">
<p align="center" style="width: 100%;">
    <img src="https://raw.githubusercontent.com/vlm-run/.github/refs/heads/main/profile/assets/vlm-black.svg" alt="VLM Run Logo" width="80" style="margin-bottom: -5px; color: #2e3138; vertical-align: middle; padding-right: 5px;"><br>
</p>
<h2>VLM Run Python SDK</h2>
<p align="center"><a href="https://docs.vlm.run"><b>Website</b></a> | <a href="https://app.vlm.run/"><b>Platform</b></a> | <a href="https://docs.vlm.run/"><b>Docs</b></a> | <a href="https://docs.vlm.run/blog"><b>Blog</b></a> | <a href="https://discord.gg/AMApC2UzVY"><b>Discord</b></a>
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

- OpenAI SDK integration (for chat completions API):
  ```bash
  pip install "vlmrun[openai]"
  ```

- All optional features:
  ```bash
  pip install "vlmrun[all]"
  ```

### Basic Usage

```python
from PIL import Image
from vlmrun.client import VLMRun
from vlmrun.common.utils import remote_image

# Initialize the client
client = VLMRun(api_key="<your-api-key>")

# Process an image using local file or remote URL
image: Image.Image = remote_image("https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.invoice/invoice_1.jpg")
response = client.image.generate(
    images=[image],
    domain="document.invoice"
)
print(response)

# Or process an image directly from URL
response = client.image.generate(
    urls=["https://storage.googleapis.com/vlm-data-public-prod/hub/examples/document.invoice/invoice_1.jpg"],
    domain="document.invoice"
)
print(response)
```

### OpenAI-Compatible Chat Completions

The VLM Run SDK provides OpenAI-compatible chat completions through the agent endpoint. This allows you to use the familiar OpenAI API with VLM Run's powerful vision-language models.

```python
from vlmrun import VLMRun

client = VLMRun(
    api_key="your-key",
    base_url="https://agent.vlm.run/v1"
)

response = client.agent.completions.create(
    model="vlm-1",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

For async support:

```python
import asyncio
from vlmrun import VLMRun

client = VLMRun(api_key="your-key", base_url="https://agent.vlm.run/v1")

async def main():
    response = await client.agent.async_completions.create(
        model="vlm-1",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

**Installation**: Install with OpenAI support using `pip install vlmrun[openai]`

## 🔗 Quick Links

* 💬 Need help? Email us at [support@vlm.run](mailto:support@vlm.run) or join our [Discord](https://discord.gg/AMApC2UzVY)
* 📚 Check out our [Documentation](https://docs.vlm.run/)
* 📣 Follow us on [Twitter](https://x.com/vlmrun) and [LinkedIn](https://www.linkedin.com/company/vlm-run)
