---
description: Explore, exercise, and improve the `vlmrun gateway` / `gw` CLI against the live gateway
argument-hint: "<objective>, e.g. 'improve the embed UX' or 'audit transcribe error handling'"
---

# vlmrun gateway — objective-driven improvement

Objective: **$ARGUMENTS**

You are working on the `vlmrun gateway` (aliased `vlmrun gw`) CLI in this repo —
`vlmrun/cli/_cli/gateway.py`, the SDK resource `vlmrun/client/gateway.py`, tests
`tests/test_gateway.py`, and docs `vlmrun/cli/README.md`. The gateway is an
OpenAI-compatible passthrough (`https://gateway.vlm.run/v1/openai`) to
third-party OCR / VLM / embedding / transcription models.

Pursue the objective above by running the CLI for real, not by reading code
alone. Follow this loop.

## 0. Ground yourself in what actually exists

Never hardcode a model list — it drifts. Discover it live:

```bash
vlmrun gw models            # task + methods per model
vlmrun gw models <model>    # methods, method_params, runnable example commands
vlmrun gw models --json     # raw catalog: id, aliases, methods, default_method,
                            # extra_body_help, capabilities.supported_input_types
```

Model ids are the full `<org>/<name>`; short aliases also work. Use the full
form in code, help text, and docs (aliases are a convenience, not the label).

## 1. Auth — check before assuming anonymous

The gateway allows anonymous calls, but the **SDK rejects an empty API key**, so
the CLI always needs one. First check the configured key works against *prod*:

```bash
python -c "from vlmrun.cli._cli.config import resolve_config; \
import urllib.request,urllib.error; k=resolve_config(api_key=None,base_url=None).api_key; \
req=urllib.request.Request('https://gateway.vlm.run/v1/openai/models',headers={'Authorization':f'Bearer {k}'}); \
print(urllib.request.urlopen(req,timeout=30).status)"
```

- `200` → run everything authenticated. `unset VLMRUN_API_KEY VLMRUN_BASE_URL
  VLMRUN_GATEWAY_URL` so config resolution isn't shadowed by a stale dev env.
- `403` → the configured key is a dev key the prod gateway rejects. Fall back to
  a local header-stripping proxy that forwards to `gateway.vlm.run` and drops the
  `Authorization` header (anonymous), pointing the CLI at it via
  `VLMRUN_GATEWAY_URL=http://127.0.0.1:<port>`. State clearly in your report that
  results are anonymous-only.

## 2. Exercise every relevant surface end-to-end

Run real inputs and confirm real output — a non-empty, *correct-shaped* result,
not just exit 0. Cover what the objective touches; for a broad objective, cover
all of it:

- **`chat`** — one run per model default method, then each `--method` from the
  catalog. Verify the methods actually differ (e.g. pp-ocrv6 `ocr` returns
  text+score+poly, `detect` poly-only, `markdown` plain text). Exercise
  `--method-params` and prove it changes output (e.g. sweep `score_threshold`).
  Try a PDF (streams per page) and an image (must not stream). Try a VQA model
  with `-p`.
- **`embed`** — text, image, batch (each input → its own vector), `--join`
  (one joint vector; rejects 2+ files), `--dimensions`. Sanity-check semantics:
  cosine(image, matching caption) should beat cosine(image, unrelated caption).
- **`transcribe`** — audio and a video's audio track; every `-f` format
  (json/text/verbose_json/srt/vtt — srt/vtt must carry timestamps);
  `--language`, `--prompt`, `--url`.

Use real files. `~/data/1-demo` has images, PDFs, and video with audio; extract
a short audio clip with `ffmpeg -i <video> -t 30 -vn -ar 16000 -ac 1 clip.mp3`.

Watch specifically for these known failure modes — they recur:

- **Silent-empty output.** A blank/`(empty response)` panel that reports token
  counts is a bug, not an empty result. Root causes seen: OCR wrapped in
  `<document>`/`<page>` tags eaten by the Markdown renderer; image/text-only
  requests streamed (the gateway returns a non-SSE body that an SSE reader
  drains to empty — only *documents* stream); a mislabelled data-URL MIME.
- **Extension lies.** A `.jpg` that is really WebP → wrong `data:` MIME → gateway
  misroute. MIME must be sniffed from magic bytes.
- **`file_url` vs `image_url`.** Images belong in `image_url`; `file_url` is
  routed through the document/PDF path and 400s on plain images.
- **Errors dressed as success.** An `{"error": ...}` body on a 200 (e.g. bad
  `--method`) must exit non-zero, not render as a Response.
- **Gateway-only params.** `method`, `method_params`, `document_dpi`,
  `image_resolution` are not OpenAI kwargs — they must ride in `extra_body`.

## 3. Read the help as a new user would

For every subcommand: `vlmrun gw <cmd> --help`. Flag anything that would mislead
someone who has never seen it:

- Group/command descriptions that no longer cover what the command does.
- Option help that contradicts current behaviour (e.g. an `--text` help that
  claims joint embedding after `--join` made it separate-by-default).
- Stale model ids in examples; misaligned example columns; missing mention of
  auth or of the `models`/`models <model>` discovery path.
- Errors that dump raw dict/JSON reprs where a sentence would do.

## 4. Fix, prove, keep in sync

For each change:

1. Make the smallest fix that serves the objective. Match the file's existing
   style (Rich panels, `typer.Exit(1)` + a red `[red]Error:[/]` line, helper
   functions near their peers).
2. Add a test in `tests/test_gateway.py`. Fakes must mirror **real** API
   payloads — the catalog shape from `models --json`, the nested embedding
   `input` (`[[part]]`, not `[part]`), an `{"error": ...}` body. A fake that
   accepts any kwarg or invents fields hides the bugs you're hunting.
3. Prove the test catches the bug: `git stash` the source, confirm the new test
   **fails**, `git stash pop`, confirm it passes.
4. Keep docs in sync (AGENTS.md requires this): CLI help strings, the
   `vlmrun/cli/README.md` gateway section, and the SDK docstrings in
   `client/gateway.py` must all agree with the code.

Then:

```bash
ruff format vlmrun/cli/_cli/gateway.py vlmrun/client/gateway.py tests/test_gateway.py
ruff check vlmrun/cli/_cli/gateway.py vlmrun/client/gateway.py tests/test_gateway.py
pytest -q tests/test_gateway.py        # then the full suite before finishing
```

Re-run the affected commands live one more time to confirm the fix in the real
CLI, not just in tests.

## 5. Report

Lead with what changed and the evidence. Give a compact table of
model × command/method × result (✅/⚠️/❌ with a one-line reason), noting which
were run authenticated vs anonymous. Separate **CLI fixes** (yours) from
**server/docs-team notes** (things only the gateway or docs can fix — e.g. the
docs' multimodal embedding example, a 500 on multi-image joint input, video
embedding being a constant no-op, streaming turning 400s into empty 200s).

Do not bump the package version or open/push a PR unless the objective asks for
it. Surface anything you could not verify (e.g. auth-gated paths) rather than
claiming it works.
