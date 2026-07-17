---
description: Natural-language interface to the `vlmrun gw` gateway CLI — say what you want done to a file/URL/text and it runs the right command
argument-hint: "<request>, e.g. 'embed this image ~/data/image.jpg' or 'transcribe ~/clip.mp3'"
---

# vlmrun gateway — do what I asked

Request: **$ARGUMENTS**

Translate the request above into the correct `vlmrun gw` command, run it, and
show the result. You are a thin natural-language front-end to an existing CLI —
choose the subcommand, model, and method, execute, and report. Do not modify the
CLI or write code.

## 1. Parse the request

From the request text pull out:

- **Inputs** — local paths (`~/…`, `./…`, absolute), `http(s)://…` URLs, and any
  literal text to act on ("embed the phrase 'blue parrot'"). Expand `~`.
- **Intent** — what to do with them (see the mapping below).

If there is no input and the intent needs one, or the request is genuinely
ambiguous, ask one short clarifying question instead of guessing. A missing
model is not ambiguity — pick a sensible default.

## 2. Map intent → subcommand, model, method

Model ids drift, so confirm your choice against `vlmrun gw models` (and
`vlmrun gw models <model>` for its methods) before running. The defaults below
are current; if one is absent from the catalog, pick another model whose `task`
and `methods` fit. Always pass the full `<org>/<name>` id.

**Transcription** — audio, or "transcribe" a video (its audio track).
Triggers: transcribe, subtitles/captions, "what is said". Extensions: mp3, wav,
m4a, flac, ogg; or a video when the intent is transcription.
→ `vlmrun gw transcribe <input> -m nvidia/parakeet-tdt-0.6b-v3`
  - subtitles → `-f srt` (or `-f vtt`); timestamps/detail → `-f verbose_json`
  - a hosted URL → `--url <url>` (no positional file); language hint → `-l en`

**Embedding** — turn an image, text, or video into a vector.
Triggers: embed, vectorize, "embedding for", similarity/index.
→ `vlmrun gw embed <files…> -m qwen/qwen3-vl-embedding-2b`
  - literal text → `-t "…"` (repeatable); each file/`-t` is its own vector
  - "embed the image together with this caption" → add `--join` (one file max)
  - "give me N dims" → `--dimensions N`; scripting → `--json` for full vectors

**Document parsing** — a PDF/DOCX, or an image of a document/form/receipt.
Triggers: parse, read, "to markdown", extract text, OCR, layout/structure.
→ `vlmrun gw chat <file> -m <model> --method <method>`
  - to markdown / clean reading order → `-m zai-org/glm-ocr --method markdown`
  - text + confidence + polygons (JSON) → `-m paddleocr/pp-ocrv6 --method ocr`
  - just text-region boxes → `-m paddleocr/pp-ocrv6 --method detect`
  - layout blocks / sections → `-m rednote-hilab/dots.mocr --method parse_layout`
  - OCR knobs: `--method-params '{"lang":"en","score_threshold":0.5}'` (pp-ocrv6)

**Image understanding / VQA** — a free-form question or description of an image.
Triggers: describe, caption, "what/how many/what color/is there…", summarize.
→ `vlmrun gw chat <image> -p "<the question>" -m qwen/qwen3.5-0.8b`
  (Use the OCR models above only when the intent is text extraction, not a
  question about the scene.)

Notes that matter:

- Images are sent as `image_url`, documents as `document_url` — the CLI decides
  from the file's real bytes, so a mislabelled extension is fine; you don't set
  this.
- Only the OCR/document `chat` models are strict about needing a file. VQA takes
  `-p`; embedding takes `-t`.
- If the user names a model or method explicitly, honor it over the defaults.

## 3. Run it

Before executing, make sure a stale dev environment isn't shadowing the
configured key: `unset VLMRUN_API_KEY VLMRUN_BASE_URL VLMRUN_GATEWAY_URL` (the
CLI reads the key from `~/.vlmrun/config.toml`). Then:

1. Show the exact `vlmrun gw …` command you're about to run (one line).
2. Run it. Default to the human-readable panel; add `--json` only if the user
   asked for raw output or is clearly scripting.
3. If it fails: a `model not found` error prints the valid ids — pick the right
   one and retry once. An `Error: Unknown method …` means the method is wrong
   for that model — drop `--method` (use its default) or pick a listed one. A
   `model not found`/wrong-task pairing means you chose the wrong model for the
   task; re-map and retry. Don't loop more than a couple of times — if still
   stuck, report what you tried.

## 4. Report

State what you ran and give the result: the transcript, the extracted
text/markdown, the VQA answer, or for embeddings a one-line summary (how many
vectors, their dimension) rather than dumping raw floats unless asked. If you
picked a model/method the user didn't specify, say which and why in one line so
they can steer next time.

Chat/OCR responses carry `usage.cost` (USD) — the panel footer shows it, and
`--json` puts it under `usage.cost`. Report the cost when it's relevant, and for
a batch over many files sum it into a total.

## Examples

- `"embed this image ~/data/image.jpg"`
  → `vlmrun gw embed ~/data/image.jpg -m qwen/qwen3-vl-embedding-2b`
- `"what's in ~/photos/street.jpg?"`
  → `vlmrun gw chat ~/photos/street.jpg -p "What's in this image?" -m qwen/qwen3.5-0.8b`
- `"extract the text from ~/scans/receipt.png"`
  → `vlmrun gw chat ~/scans/receipt.png -m paddleocr/pp-ocrv6 --method ocr`
- `"parse ~/docs/contract.pdf to markdown"`
  → `vlmrun gw chat ~/docs/contract.pdf -m zai-org/glm-ocr --method markdown`
- `"get the layout blocks of ~/forms/intake.jpg"`
  → `vlmrun gw chat ~/forms/intake.jpg -m rednote-hilab/dots.mocr --method parse_layout`
- `"transcribe ~/calls/standup.mp4 as subtitles"`
  → `vlmrun gw transcribe ~/calls/standup.mp4 -m nvidia/parakeet-tdt-0.6b-v3 -f srt`
- `"embed the caption 'a blue parrot' with ~/img/parrot.jpg as one vector"`
  → `vlmrun gw embed ~/img/parrot.jpg -t "a blue parrot" --join -m qwen/qwen3-vl-embedding-2b`
