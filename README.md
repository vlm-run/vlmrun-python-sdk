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

- Visualization and notebook helpers (pandas, IPython):
  ```bash
  pip install "vlmrun[all]"
  ```

- System One typed decisions (`typesafe-sdk`):
  ```bash
  pip install "vlmrun[typesafe]"
  ```

- All optional features:
  ```bash
  pip install "vlmrun[all]"
  ```

The CLI and OpenAI-compatible gateway (`vlmrun gw chat`, `vlmrun chat`) work out of the box with `pip install vlmrun`.

### System One — typed decisions

`vlmrun gw systemone` answers named questions about text, JSON, images and PDFs with
calibrated probabilities. Answers are read off the model in one denoise step, so nothing
is generated and nothing is parsed — an answer can never be off-schema. The route speaks
TypeSafe's contract, so it is driven by the official `typesafe-sdk` client
(`pip install "vlmrun[typesafe]"`).

```bash
vlmrun gw systemone "Invoice #44 was charged twice, I need this fixed today" \
  --noul is_urgent="Is the customer asking for something time-sensitive?" \
  --choice department="billing|technical|sales" \
  --score frustration="Calm|Frustrated|Very angry"

# questions as inline JSON (or @file.json, or - for stdin)
vlmrun gw systemone ticket.txt --json -Q '[
  {"id": "is_urgent", "type": "noul"},
  {"id": "department", "type": "choice", "options": ["billing", "technical", "sales"]}
]'

# images and one PDF ride along; --detail sets the vision budget
vlmrun gw systemone invoice.pdf --detail high --choice kind="invoice|receipt|contract"
```

From Python:

```python
from vlmrun.client import VLMRun

client = VLMRun()
result = client.gateway.systemone.decide(
    state="Invoice #44 was charged twice, I need this fixed today",
    questions=[
        {"id": "is_urgent", "type": "noul", "instructions": "Is this time-sensitive?"},
        {"id": "department", "type": "choice", "options": ["billing", "technical", "sales"]},
    ],
)
result.nouls["is_urgent"].noul           # 0.91
result.choices["department"].choice      # "billing"
result.choices["department"].confidence  # 0.71
```

Three flags make a read scriptable:

```bash
# --gate sets the exit code: 0 all passed, 1 a gate failed, 2 the request failed
vlmrun gw s1 invoice.pdf --choice kind="invoice|receipt|contract" \
  --gate 'kind==invoice' --gate 'kind.confidence>0.9'

# --repeat sends the same request N times and reports mean and spread
vlmrun gw s1 ticket.txt --noul is_urgent --repeat 5

# --dry-run prints the request body without sending it (pipe it to curl)
vlmrun gw s1 scan.jpg --noul signed --dry-run
```

`vlmrun gw s1` is a shorthand for `vlmrun gw systemone`. See
`vlmrun gw systemone --help` for both question dialects, media rules and limits.

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
from vlmrun.client import VLMRun

client = VLMRun(
    api_key="your-key",
    base_url="https://api.vlm.run/v1"
)

response = client.agent.completions.create(
    model="vlmrun-orion-1",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

For async support:

```python
import asyncio
from vlmrun.client import VLMRun

client = VLMRun(api_key="your-key", base_url="https://api.vlm.run/v1")

async def main():
    response = await client.agent.async_completions.create(
        model="vlmrun-orion-1",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### CLI Chat with Skills

The `vlmrun chat` command supports **skills** — local directories containing a `SKILL.md` and optional assets that give the agent domain-specific expertise. Skills are sent inline with each request (no server-side upload required).

```bash
# Chat with an inline skill
vlmrun chat "Generate a youtube thumbnail for a video using the VLM Run brand colors" -k ./path/to/vlmrun-branding/

# Attach multiple skills (coming soon)
vlmrun chat "Analyze this invoice" -i invoice.pdf -k ./accounting-skills/ -k ./invoice-extraction/
```

To create a persistent server-side skill, use `vlmrun skills upload ./my-skill/`.

### Claude Code

Install the VLM Run CLI skill directly in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via the plugin marketplace in the [vlm-run/skills](https://github.com/vlm-run/skills) repository:

1. Register the repository as a plugin marketplace:

```
/plugin marketplace add vlm-run/skills
```

2. Install the skill:

```
/plugin install vlmrun-cli-skill@vlm-run/skills
```

3. Configure your API key and base URL using the CLI (get your key from [app.vlm.run](https://app.vlm.run)):

```bash
vlmrun config init
vlmrun config set --api-key <your-api-key>
vlmrun config show
```

4. Verify the skill is loaded by asking Claude Code (requires restart):

```
What skills are available in the /vlmrun-cli-skill?
```

## 🔗 Quick Links

* 💬 Need help? Email us at [support@vlm.run](mailto:support@vlm.run) or join our [Discord](https://discord.gg/AMApC2UzVY)
* 📚 Check out our [Documentation](https://docs.vlm.run/)
* 📣 Follow us on [Twitter](https://x.com/vlmrun) and [LinkedIn](https://www.linkedin.com/company/vlm-run)
