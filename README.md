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

`vlmrun gw systemone` answers named questions about text, JSON, images, PDFs and video
with calibrated probabilities. Answers are read off the model in one denoise step, so nothing
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

#### Video — a decision per frame

Pass a video and it is sampled into frames, each frame its own read. There is no video
model here and no streaming: the timeline is built client-side from single-frame
decisions, which is what keeps every frame's answer independent and comparable.
Needs the video extra (`pip install "vlmrun[video]"`).

```bash
# one read a second by default; --fps sets the rate
vlmrun gw s1 door.mp4 --noul is_open --fps 2

# answers come back as a series over time, not a mean
vlmrun gw s1 door.mp4 --choice state="open|closed|blocked" --fps 1
#   state  closed 0.0s-3.0s · open 4.0s-11.0s · blocked 12.0s-14.0s

# --gate-mode says how a gate reads the timeline:
#   any (default) · all · sustained:N · mean
vlmrun gw s1 door.mp4 --noul is_open --fps 4 \
  --gate 'is_open>0.9' --gate-mode sustained:8   # two seconds, not one bad frame
```

From Python, sampling and reading are separate pieces that compose.
`VideoReader.frames(fps=...)` is a plain iterator over a video at a given rate, and a
stream reads whatever you hand it:

```python
from vlmrun.common.video import VideoReader

with client.gateway.systemone.stream(
    questions=[{"id": "is_open", "type": "noul"}],
    state="Is the door open?",
    concurrency=8,
) as stream, VideoReader("door.mp4") as video:
    # Concurrent: reads overlap, results arrive in frame order.
    for decision in stream.map(video.frames(fps=2)):
        print(decision.timestamp_s, decision.response.nouls["is_open"].noul)
        if decision.timestamp_s > 10:
            break                      # queued reads are dropped

    # One at a time — the shape a camera or a queue wants.
    for frame in video.frames(fps=2):
        response = stream.send(frame)
```

Because `frames()` is just an iterator, it composes with the standard library —
`islice(video.frames(fps=2), 10)` for the first ten, or a generator expression to keep
only the frames you care about — and `map` accepts any iterable of paths, URLs, data
URLs or RGB arrays, not only frames. A reader is reusable: iterate it twice at different
rates off one open decoder.

`map` pulls lazily and keeps at most `concurrency` reads in flight, so a long clip
costs bounded memory and breaking out early stops the spend; results come back in input
order regardless of which answer arrived first. A sampled frame carries its index and
timestamp through, so the timeline survives. `send()` is the blocking single-read form:
simpler, but it does not overlap, so a `for` loop over `send()` is serial.

The stream is a context manager because it owns a worker pool, released on the way out
including when you stop early or raise.

The URLs are derived, not configured separately, so pointing the SDK at another
deployment moves everything with it:

```python
from vlmrun.constants import DEFAULT_GATEWAY_URL, gateway_base_url
from vlmrun.client.systemone import typesafe_base_url, typesafe_websocket_url

gateway_base_url()        # 'https://gateway.vlm.run/v1'  (VLMRUN_GATEWAY_BASE_URL wins)
typesafe_base_url()       # 'https://gateway.vlm.run/typesafe'
typesafe_websocket_url()  # 'wss://gateway.vlm.run/typesafe/ws'
```

`gateway_base_url()` takes an explicit argument first, then `VLMRUN_GATEWAY_BASE_URL`,
then the older `VLMRUN_GATEWAY_URL`, then `DEFAULT_GATEWAY_URL`. `TYPESAFE_BASE_URL`
overrides the `/typesafe` root on its own.

#### One session instead of a request per frame

`transport="ws"` (CLI: `--ws`) opens a single session on `/typesafe/ws` rather than a
request per read. The questions go once with the handshake and are cached server-side,
and reads are pipelined over one socket, correlated by id rather than by arrival order.
Needs `pip install "vlmrun[ws]"`.

```python
with client.gateway.systemone.stream(questions=Q, state="...", transport="ws") as stream:
    print(stream.limits["max_inflight"])            # what the server granted
    with VideoReader("door.mp4") as video:
        for decision in stream.map(video.frames(fps=2)):
            ...
print(stream.stats["cost"])                          # session totals, after closing
```

The surface is identical to the HTTP transport — `send`, `map`, the same
`FrameDecision`, the same `SystemOneResponse` — so nothing above it changes.

One socket avoids the per-request overhead of a read, which shows up as soon as the
session is allowed to pipeline. Measured on a 60s clip at 1 fps — 60 reads, best of three:

| reads in flight | HTTP | websocket |
|---|---|---|
| 4 | 2458 ms | **1525 ms** |
| 8 | 1969 ms | **904 ms** |

Token use and cost are identical either way; the saving is round trips, so it grows with
the number of frames.

`concurrency` is requested as the session's `max_inflight` during the handshake. This
matters: the route's own default is 2, well below this SDK's, so a session that does not
ask is throttled to a quarter of the width you asked for. The ack is still authoritative
— the server may grant less than requested, and the stream never outruns what it granted
— and the route refuses a request above 8.

One genuine difference: the state is session-scoped rather than per-request, so changing
it mid-stream drains the reads in flight first.

Every frame is a billed read, so `--fps` and the clip's length set the bill. `--max-frames`
(default 60) is checked against the duration before anything is sent, so an ask that would
cost more than you meant fails for free. `--concurrency` (default 4) sets how many reads
are in flight; results are always returned in timeline order. `--json` gives every frame,
each read tagged with its frame index and timestamp.

Several engines serve the route and there is no catch-all alias — `vlmrun gw s1 models`
lists what your gateway serves. Generative engines can think before answering:

```bash
vlmrun gw s1 models

vlmrun gw s1 ticket.txt -m google/gemma-4-26b-a4b-it \
  --reasoning-effort medium --choice dept="billing|tax|technical"
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
