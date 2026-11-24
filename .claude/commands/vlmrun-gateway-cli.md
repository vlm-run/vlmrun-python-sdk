---
description: Natural-language front-end to the `vlmrun gw` gateway CLI. Say what to do with a file, URL or text, and it runs the right command.
argument-hint: "<request>, e.g. 'parse ~/docs/contract.pdf to markdown' or 'is ~/scans/a.jpg a receipt?'"
---

# vlmrun gw: do what I asked

Request: **$ARGUMENTS**

Turn the request into one `vlmrun gw` command, run it, and report the result. You are a thin front-end to the
CLI. Do not edit the CLI. Write code only for the few tasks in [When the CLI cannot do it](#when-the-cli-cannot-do-it).

## 1. Parse the request

- **Inputs:** local paths (expand `~`), `http(s)://` URLs, literal text.
- **Intent:** see the table in step 2.
- **Model or method named by the user:** use it. It wins over the defaults below.

Ask one short question only when an input is missing or the intent is unclear. A missing model is not unclear: use
the default.

## 2. Route

The catalog changes. Before you run a model you have not used in this session, check it with
`vlmrun gw models <id>` (methods, default method `*`, inputs, aliases). `vlmrun gw models` lists all of them. Pass
the full `<org>/<name>` id.

| Intent | Command | Notes |
| --- | --- | --- |
| Document or scan to markdown | `vlmrun gw chat <in> -m zai-org/glm-ocr` | Only method is `markdown`. Other markdown models: `vlm-run/ocr:auto`, `paddlepaddle/paddleocr-vl-1.6`, `nvidia/nemotron-3.5-parse`, `lightonai/lightonocr-3-4b`, `rednote-hilab/dots.mocr`. |
| Plain text | `vlmrun gw chat <in> -m paddleocr/pp-ocrv6 -M text` | |
| Text with boxes and scores (JSON) | `vlmrun gw chat <in> -m paddleocr/pp-ocrv6` | Default method `ocr`. Knobs: `--method-params '{"score_threshold":0.5}'`, `'{"word_boxes":true}'`. |
| Text-region boxes only | `vlmrun gw chat <in> -m paddleocr/pp-ocrv6 -M detect` | |
| Layout regions | `vlmrun gw chat <in> -m vlm-run/ocr:auto -M layout` | Or `rednote-hilab/dots.mocr -M parse_layout`. |
| Tables, formulas, charts | `vlmrun gw chat <in> -m paddlepaddle/paddleocr-vl-1.6 -M table` | Methods: `table`, `formula`, `chart`, `ocr`, `markdown`. |
| Question about an image, PDF or video | `vlmrun gw chat <in> -p "<question>" -m qwen/qwen3.5-0.8b` | Stronger: `qwen/qwen3.8-27b` (any API key), `google/gemma4-26b-a4b-it` (paid). Video: add `-e video_fps=1 -e video_max_frames=8`. |
| Text-only chat | `vlmrun gw chat -p "<text>" -m qwen/qwen3.5-0.8b` | OCR, detection and pose models reject text-only input. |
| Frontier model by name | `vlmrun gw chat <in> -p "..." -m <id>` | `google/gemini-*`, `moonshotai/kimi-k3`, `minimax/minimax-m3`, `meta/muse-*`. All paid. |
| Caption | `vlmrun gw chat <img> -m microsoft/florence-2-base-ft` | Default `caption`. Also `detailed_caption`, `more_detailed_caption`, `od`. |
| Object boxes (COCO classes) | `vlmrun gw chat <img> -m roboflow/rfdetr-large` | |
| Masks for a named object | `vlmrun gw chat <img> -m facebook/sam3.1 --method-params '{"prompt":"car"}'` | COCO instance masks: `roboflow/rfdetr-seg-large`. |
| Body or hand pose | `vlmrun gw chat <in> -m usyd-community/vitpose-plus-large` | Hands: `geopavlakos/hamer`. |
| Embedding | `vlmrun gw embed <files> -t "<text>" -m qwen/qwen3-vl-embedding-2b` | 2048 dims, image, video or text. Text-first: `google/embeddinggemma-2` (768 dims). |
| Transcription | `vlmrun gw transcribe <in> -m nvidia/parakeet-tdt-0.6b-v3` | Audio file, URL, or a video's audio track. |
| Yes/no, a label, or a score with probabilities | `vlmrun gw s1 <in> --noul id="Question?"` | System One. See below. |

Pick `gw s1` over `gw chat` when the answer is a label, a probability or a pass/fail gate. Pick `gw chat` when the
answer is text.

### Flags that matter

- `gw chat`: `-m` (required), `-p`, `-M/--method`, `--method-params '<json>'`, `-e key=value` (repeatable, e.g.
  `-e document_dpi=72`), `--json-mode`, `--response-format '<json>'`, `-ns`, `-j`, `--timeout`.
- `gw embed`: `-m` (required), `-t/-p` text (repeatable), `--join` (one vector, at most one file),
  `-d/--dimensions`, `-j` (full vectors).
- `gw transcribe`: `-m` (required), one input or `--url`, `-f json|text|verbose_json|srt|vtt` (default `json`),
  `-l en`, `-p "<proper nouns>"`, `-j`.
- `gw s1`: `--noul id="Q?"`, `--choice id="a|b|c"`, `--score id="low|mid|high"`, `-Q @questions.json`, `-s` state,
  `-m`, `--detail auto|low|high`, `-g/--gate`, `-r/--repeat`, `--dry-run`, `-j`.
- `-j` formats the CLI output as JSON. `--json-mode` asks the model for a JSON object. With both, `-j` prints the
  parsed object.
- `--response-format '{"type":"json_schema","json_schema":{...}}'` constrains chat models to a schema. OCR models do
  not support it; use `--json-mode` there.

### System One (`gw s1`)

- Inputs: literal text, a `.txt/.md/.json/.yaml` state file, up to 8 images, one PDF, or one local video.
- Every `--noul` needs question text: `--noul is_urgent="Is this urgent?"`. A bare `--noul is_urgent` fails.
- Models: `vlmrun gw s1 models`. Default `google/diffusiongemma-26b-a4b-it`. `-R/--reasoning-effort` works only with
  `-m google/gemma4-26b-a4b-it`; other models return 422.
- `--gate 'id>0.9'` sets the exit code: 0 pass, 1 gate failed, 2 request failed.
- Video: a local file only (download a URL first). `--fps` (default 1) times the clip length is the number of
  billed reads; `--max-frames` (default 60) refuses more. `--ws` reads over one socket and is cheaper per frame.
  Needs `pip install "vlmrun[video,ws]"`.

```bash
vlmrun gw s1 ~/scans/2.jpg --choice kind="invoice|receipt|contract" --noul has_total="Is a total visible?"
vlmrun gw s1 ticket.txt --choice dept="billing|technical|sales" --gate 'dept==billing'
```

## When the CLI cannot do it

| Need | Path |
| --- | --- |
| `google/gemini-decision-1` decisions | `POST https://gateway.vlm.run/decisions` with the System One body (`model`, `state`, `content`, `questions`). Paid key. `gw s1 -m google/gemini-decision-1` returns 404. |
| Upload a local file once, reuse its URL | `curl https://gateway.vlm.run/openai/files -H "Authorization: Bearer $VLMRUN_API_KEY" -F file=@scan.pdf`. Pass the reply's `url` as an input. Needs an API key; anonymous calls get 401. |
| Embed a PDF or audio | `POST https://gateway.vlm.run/openai/embeddings` with `google/embeddinggemma-2` and a `document_url` or `audio_url` part. `gw embed` accepts images and video only. |
| Many calls from code | OpenAI SDK with `base_url="https://gateway.vlm.run/openai"`. Same models, methods (`extra_body={"method": ...}`) and errors. |
| System One from code or curl | `POST https://gateway.vlm.run/typesafe/v1/systemone`; the body must name `model`. `vlmrun gw s1 ... --dry-run` prints a body you can send. |
| Gateway models as agent tools | MCP: `claude mcp add --transport http vlmrun https://gateway.vlm.run/mcp --header "Authorization: Bearer $VLMRUN_API_KEY"`. Tools: `read_document`, `read_image`, `read_audio`, `read_video`, `list_models`, `get_model_info`, `get_completion`, `create_upload_token`. |
| Frame-by-frame realtime detection or pose | `wss://gateway.vlm.run/v1/ws`. See https://docs.vlm.run/gateway/realtime. |
| Multi-step agent work, image or video generation | Orion: `vlmrun chat`. See the `vlmrun-cli-skill` skill. |

## 3. Run it

1. Key. The CLI reads `--api-key`, then `VLMRUN_API_KEY`, then `~/.vlmrun/config.toml`. Check which `VLMRUN_*`
   variables are set without printing values (`env | grep -o '^VLMRUN_[A-Z_]*'`). If `VLMRUN_GATEWAY_BASE_URL` or
   `VLMRUN_GATEWAY_URL` points away from `https://gateway.vlm.run/v1` and the user did not ask for that, unset that
   one variable. Do not unset a working key.
2. No key at all: the CLI fails with `Missing credentials`. Run with `VLMRUN_API_KEY=vlmrun` for anonymous access.
   Paid models then return 401.
3. Show the exact command on one line, then run it. Add `-j` only when the user wants raw output or is scripting.
4. On an error, fix and retry at most twice:

| Error | Fix |
| --- | --- |
| `API Error`, `Error code: 404` | Wrong model id. Run `vlmrun gw models <id>`; it names a missing model. |
| `API Error`, `Error code: 400` | Wrong method or input for the model. Check `vlmrun gw models <id>`. Text-only input to an OCR model is a 400. |
| `Authentication Error`, `Error code: 401` | The model needs an API key or a paid account. Say so; do not loop. |
| `--noul ... has neither instructions nor criteria` | Add question text: `--noul id="Question?"`. |
| `gw s1` 422 on `reasoning_effort` | Use `-m google/gemma4-26b-a4b-it`, or drop `-R`. |
| `gw s1` frame ceiling | Lower `--fps`. Raise `--max-frames` only if the user accepts the cost. |

The CLI prints only the status code for API errors. To read the gateway's message, send the same request with curl.

## 4. Report

Say what you ran and give the result: the markdown or text, the answer, the transcript, the decisions. For
embeddings, give the count and dimension, not the floats. If you chose the model or method, say which and why in
one line.

`gw chat` and `gw s1` show cost in USD on the closing rule; `-j` puts it in `usage.cost`. Read `usage.cost`; do not
compute it from tokens. For a batch, give the total.

## Examples

```bash
# "parse ~/docs/contract.pdf to markdown"
vlmrun gw chat ~/docs/contract.pdf -m zai-org/glm-ocr
# "extract the text from ~/scans/receipt.png"
vlmrun gw chat ~/scans/receipt.png -m paddleocr/pp-ocrv6 -M text
# "OCR ~/scans/receipt.png with boxes"
vlmrun gw chat ~/scans/receipt.png -m paddleocr/pp-ocrv6
# "layout of ~/forms/intake.jpg"
vlmrun gw chat ~/forms/intake.jpg -m vlm-run/ocr:auto -M layout
# "what is the total on ~/scans/receipt.png?"
vlmrun gw chat ~/scans/receipt.png -p "What is the total?" -m qwen/qwen3.5-0.8b
# "find the cars in ~/img/street.jpg"
vlmrun gw chat ~/img/street.jpg -m roboflow/rfdetr-large
# "embed ~/img/parrot.jpg with the caption 'a blue parrot' as one vector"
vlmrun gw embed ~/img/parrot.jpg -t "a blue parrot" --join -m qwen/qwen3-vl-embedding-2b
# "subtitles for ~/calls/standup.mp4"
vlmrun gw transcribe ~/calls/standup.mp4 -m nvidia/parakeet-tdt-0.6b-v3 -f srt
# "is ~/scans/receipt.png a receipt or an invoice?"
vlmrun gw s1 ~/scans/receipt.png --choice kind="receipt|invoice|other"
# "does anyone appear in ~/cam/door.mp4?"
vlmrun gw s1 ~/cam/door.mp4 --fps 1 --ws --noul has_people="Are people visible?"
```
