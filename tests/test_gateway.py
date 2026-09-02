"""Tests for the OpenAI-compatible gateway resource and `vlmrun gw` CLI."""

from __future__ import annotations

import json

import pytest
from rich.markdown import Markdown
from rich.text import Text
from typer.testing import CliRunner

from vlmrun.cli.cli import app
from vlmrun.cli._cli import gateway as gw
from vlmrun.client.gateway import Gateway
from vlmrun.constants import DEFAULT_GATEWAY_URL

# Minimal real file headers, so mime sniffing sees what it would in the wild.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4
MP4_BYTES = b"\x00\x00\x00\x20" + b"ftyp" + b"isom" + b"\x00" * 4

# ---------------------------------------------------------------------------
# Concrete fakes (per CLAUDE.md: no MagicMock)
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(
        self, prompt_tokens: int = 10, completion_tokens: int = 20, cost=None
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.cost = cost

    def model_dump(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
        }


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str, cost=None) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage(cost=cost)


class FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeStreamChoice:
    def __init__(self, content: str) -> None:
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content: str, usage=None) -> None:
        self.choices = [FakeStreamChoice(content)]
        self.usage = usage


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.content = "Hello world"
        self.cost = None

    def create(self, model, messages, stream=False, **kwargs):
        self.calls.append(
            {"model": model, "messages": messages, "stream": stream, **kwargs}
        )
        if stream:
            return iter(
                [
                    FakeChunk(self.content[:6]),
                    FakeChunk(self.content[6:], usage=FakeUsage(cost=self.cost)),
                ]
            )
        return FakeResponse(self.content, cost=self.cost)


class FakeModel:
    """Mimics an OpenAI ``Model`` object as the gateway actually returns it.

    Mirrors a real ``GET /v1/openai/models`` payload: methods/aliases/task and
    capabilities, and no pricing fields.
    """

    def __init__(self, id: str, **extra) -> None:
        self._data = {
            "id": id,
            "object": "model",
            "owned_by": "vlm-run",
            "aliases": [],
            "methods": [],
            "default_method": "",
            "extra_body_help": "",
            "capabilities": {"supported_input_types": []},
            "task": "chat",
            **extra,
        }

    def model_dump(self) -> dict:
        return dict(self._data)


class FakeEmbeddingItem:
    def __init__(self, index: int) -> None:
        self.object = "embedding"
        self.index = index
        self.embedding = [0.1, 0.2, 0.3, 0.4]


class FakeEmbeddingResponse:
    def __init__(self, n: int) -> None:
        self.data = [FakeEmbeddingItem(i) for i in range(n)]
        self.usage = FakeUsage(prompt_tokens=5, completion_tokens=0)

    def model_dump(self) -> dict:
        return {
            "data": [
                {"object": e.object, "index": e.index, "embedding": e.embedding}
                for e in self.data
            ]
        }


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, model, input, **kwargs):
        self.calls.append({"model": model, "input": input, **kwargs})
        return FakeEmbeddingResponse(len(input))


class FakeTranscription:
    def __init__(self, text: str) -> None:
        self.text = text

    def model_dump(self) -> dict:
        return {"text": self.text}


class FakeTranscriptions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, model, file, **kwargs):
        # `file` is a handle or tuple; record only what it resolves to.
        name = getattr(file, "name", None) or (
            file[0] if isinstance(file, tuple) else str(file)
        )
        self.calls.append({"model": model, "file": str(name), **kwargs})
        return FakeTranscription("hello from audio")


class FakeGateway:
    def __init__(self, healthy: bool = True) -> None:
        self.base_url = "https://gateway.vlm.run/v1"
        self._healthy = healthy
        self.completions = FakeCompletions()
        self.embeddings = FakeEmbeddings()
        self.transcriptions = FakeTranscriptions()

    def health(self) -> bool:
        return self._healthy

    def models(self) -> list:
        return [
            FakeModel(
                "zai-org/glm-ocr",
                aliases=["glm-ocr"],
                methods=["ocr", "markdown"],
                default_method="ocr",
                extra_body_help='{"method":"ocr"} | {"method":"markdown"}'
                " | document_url PDF (markdown per page)",
                capabilities={
                    "supported_input_types": ["text", "image_url", "document_url"],
                    "max_images": 1,
                },
            ),
            FakeModel(
                "paddleocr/pp-ocrv6",
                aliases=["pp-ocrv6"],
                methods=["ocr", "detect", "markdown"],
                default_method="ocr",
                extra_body_help='{"method":"ocr","method_params":'
                '{"lang":"en","score_threshold":0.5}}',
                capabilities={
                    "supported_input_types": ["text", "image_url", "document_url"],
                    "max_images": 1,
                },
            ),
        ]


class FakeClient:
    """Concrete stand-in for VLMRun used as the CLI context object."""

    def __init__(self, api_key=None, base_url=None, healthy: bool = True) -> None:
        self.api_key = api_key or "test-key"
        self.base_url = base_url or "https://api.vlm.run/v1"
        self.timeout = 120.0
        self.max_retries = 1
        self.gateway = FakeGateway(healthy=healthy)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patched_cli(monkeypatch):
    """Patch the CLI's VLMRun factory + credentials so ctx.obj is a FakeClient."""
    monkeypatch.setenv("VLMRUN_API_KEY", "test-key")
    holder = {}

    def _factory(api_key=None, base_url=None):
        healthy = holder.get("healthy", True)
        client = FakeClient(api_key=api_key, base_url=base_url, healthy=healthy)
        if "content" in holder:
            client.gateway.completions.content = holder["content"]
        if "cost" in holder:
            client.gateway.completions.cost = holder["cost"]
        holder["client"] = client
        return client

    monkeypatch.setattr("vlmrun.client.VLMRun", _factory)
    return holder


# ---------------------------------------------------------------------------
# Client resource: Gateway
# ---------------------------------------------------------------------------


class _MiniClient:
    def __init__(self, timeout=120.0) -> None:
        self.api_key = "sk-test"
        self.timeout = timeout
        self.max_retries = 3


class TestGatewayResource:
    def test_timeout_raises_floor_at_default(self):
        # The 120s default is bumped to 600s for slow gateway calls.
        assert Gateway(_MiniClient(timeout=120.0))._timeout() == 600.0

    def test_timeout_respects_explicit_short(self):
        # A user's fail-fast timeout must not be silently widened.
        assert Gateway(_MiniClient(timeout=5.0))._timeout() == 5.0

    def test_timeout_respects_explicit_long(self):
        assert Gateway(_MiniClient(timeout=900.0))._timeout() == 900.0

    def test_timeout_none_stays_none(self):
        assert Gateway(_MiniClient(timeout=None))._timeout() is None

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("VLMRUN_GATEWAY_URL", raising=False)
        g = Gateway(_MiniClient())
        assert g.base_url == DEFAULT_GATEWAY_URL
        assert g.openai_base_url == f"{DEFAULT_GATEWAY_URL}/openai"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VLMRUN_GATEWAY_URL", "https://gw.example.com/v1/")
        g = Gateway(_MiniClient())
        assert g.base_url == "https://gw.example.com/v1"

    def test_param_override(self, monkeypatch):
        monkeypatch.setenv("VLMRUN_GATEWAY_URL", "https://env.example.com/v1")
        g = Gateway(_MiniClient(), base_url="https://param.example.com/v1")
        assert g.base_url == "https://param.example.com/v1"

    def test_models_delegates_to_openai(self):
        g = Gateway(_MiniClient())

        class _Models:
            def list(self):
                return iter(["a", "b", "c"])

        class _OpenAI:
            models = _Models()

        # cached_property stored in instance __dict__ takes precedence.
        g.__dict__["_openai"] = _OpenAI()
        assert g.models() == ["a", "b", "c"]

    def test_health_dedicated_endpoint(self, monkeypatch):
        g = Gateway(_MiniClient())

        class _Resp:
            status_code = 200
            is_success = True

        monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
        assert g.health() is True

    def test_health_falls_back_to_models_on_404(self, monkeypatch):
        g = Gateway(_MiniClient())

        class _Resp:
            status_code = 404
            is_success = False

        monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())

        class _Models:
            def list(self):
                return iter([1])

        class _OpenAI:
            models = _Models()

        g.__dict__["_openai"] = _OpenAI()
        assert g.health() is True

    def test_health_false_on_connection_error(self, monkeypatch):
        g = Gateway(_MiniClient())

        def _boom(*a, **k):
            raise RuntimeError("no network")

        monkeypatch.setattr("requests.get", _boom)

        class _Models:
            def list(self):
                raise RuntimeError("still down")

        class _OpenAI:
            models = _Models()

        g.__dict__["_openai"] = _OpenAI()
        assert g.health() is False


# ---------------------------------------------------------------------------
# CLI helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_guess_mime(self, tmp_path):
        assert gw._guess_mime(tmp_path / "a.pdf") == "application/pdf"
        assert gw._guess_mime(tmp_path / "a.png") == "image/png"

    def test_content_part_type(self, tmp_path):
        assert gw._content_part_type(tmp_path / "a.pdf") == "document_url"
        assert gw._content_part_type(tmp_path / "a.docx") == "document_url"
        assert gw._content_part_type(tmp_path / "a.png") == "image_url"
        assert gw._content_part_type(tmp_path / "a.jpg") == "image_url"
        assert gw._content_part_type(tmp_path / "a.mp4") == "video_url"
        # Unidentifiable content still falls back to file_url.
        assert gw._content_part_type(tmp_path / "a.bin") == "file_url"

    def test_content_part_type_from_url(self):
        assert (
            gw._content_part_type_from_url("https://example.com/doc.pdf")
            == "document_url"
        )
        assert (
            gw._content_part_type_from_url("https://example.com/img.png") == "image_url"
        )
        assert (
            gw._content_part_type_from_url("https://example.com/clip.mp4?token=1")
            == "video_url"
        )

    def test_encode_url_part(self):
        part = gw._encode_url_part("https://example.com/scan.jpg")
        assert part == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/scan.jpg"},
        }

    def test_encode_url_part_document(self):
        part = gw._encode_url_part("https://example.com/report.pdf")
        assert part["type"] == "document_url"
        assert part["document_url"]["url"] == "https://example.com/report.pdf"

    def test_encode_document_part(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.7 fake")
        part = gw._encode_file_part(f)
        assert part["type"] == "document_url"
        url = part["document_url"]["url"]
        assert url.startswith("data:application/pdf;base64,")

    def test_encode_image_part(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(PNG_BYTES)
        part = gw._encode_file_part(f)
        # Images go as image_url: file_url is routed through the gateway's
        # document/PDF path and 400s on a plain image.
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_encode_part_sniffs_mislabelled_extension(self, tmp_path):
        # A WebP that claims to be a .jpg — the extension must not win, or the
        # gateway misroutes it and fails.
        f = tmp_path / "actually-webp.jpg"
        f.write_bytes(WEBP_BYTES)
        part = gw._encode_file_part(f)
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/webp;base64,")

    def test_encode_part_sniffs_pdf_without_extension(self, tmp_path):
        f = tmp_path / "nameless"
        f.write_bytes(b"%PDF-1.4 fake")
        part = gw._encode_file_part(f)
        assert part["type"] == "document_url"
        assert part["document_url"]["url"].startswith("data:application/pdf;base64,")

    def test_guess_mime_falls_back_to_extension(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"not-a-real-png")
        assert gw._guess_mime(f) == "image/png"

    def test_guess_mime_unknown(self, tmp_path):
        f = tmp_path / "mystery.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        assert gw._guess_mime(f) == "application/octet-stream"

    def test_sniff_mime_signatures(self):
        assert gw._sniff_mime(PNG_BYTES) == "image/png"
        assert gw._sniff_mime(WEBP_BYTES) == "image/webp"
        assert gw._sniff_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
        assert gw._sniff_mime(b"%PDF-1.4") == "application/pdf"
        assert gw._sniff_mime(b"nonsense") is None

    def test_build_messages_with_prompt(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(PNG_BYTES)
        messages = gw._build_messages([str(f)], "describe")
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[-1] == {"type": "text", "text": "describe"}

    def test_build_messages_mixed_files(self, tmp_path):
        img = tmp_path / "img.png"
        doc = tmp_path / "doc.pdf"
        img.write_bytes(PNG_BYTES)
        doc.write_bytes(b"%PDF fake")
        content = gw._build_messages([str(img), str(doc)], None)[0]["content"]
        assert [p["type"] for p in content] == ["image_url", "document_url"]

    def test_build_messages_with_url(self):
        content = gw._build_messages(["https://example.com/scan.jpg"], None)[0][
            "content"
        ]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/scan.jpg"

    def test_build_messages_mixed_file_and_url(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(PNG_BYTES)
        content = gw._build_messages([str(img), "https://example.com/doc.pdf"], None)[
            0
        ]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "document_url"
        assert content[1]["document_url"]["url"] == "https://example.com/doc.pdf"

    def test_parse_extra_json_and_string(self):
        parsed = gw._parse_extra(["temperature=0", "max_tokens=4096", "label=hello"])
        assert parsed == {"temperature": 0, "max_tokens": 4096, "label": "hello"}

    def test_parse_extra_invalid(self):
        with pytest.raises(Exception):
            gw._parse_extra(["nonsense"])

    def test_parse_response_format_shorthands(self):
        assert gw._parse_response_format("text") == {"type": "text"}
        assert gw._parse_response_format("json") == {"type": "json_object"}
        assert gw._parse_response_format("json_object") == {"type": "json_object"}

    def test_parse_response_format_json_object(self):
        schema = '{"type": "json_schema", "json_schema": {"name": "x"}}'
        assert gw._parse_response_format(schema) == {
            "type": "json_schema",
            "json_schema": {"name": "x"},
        }

    def test_parse_response_format_invalid(self):
        with pytest.raises(Exception):
            gw._parse_response_format("yaml")
        with pytest.raises(Exception):
            gw._parse_response_format('{"no": "type key"}')

    def test_openai_create_params_introspection(self):
        params = gw._openai_create_params()
        # Standard OpenAI fields are accepted by create() ...
        assert {"temperature", "max_tokens", "stream", "extra_body"} <= params
        # ... gateway-specific ones are not, and must ride in extra_body.
        assert not ({"method", "method_params", "document_dpi"} & params)

    def test_openai_create_params_missing_dep_raises_dependency_error(
        self, monkeypatch
    ):
        # A missing openai package must surface DependencyError (with install
        # hints), not a raw ImportError.
        from vlmrun.client.exceptions import DependencyError

        def _raise():
            raise DependencyError(
                message="OpenAI SDK is not installed",
                suggestion="pip install openai",
                error_type="missing_dependency",
            )

        monkeypatch.setattr(gw, "_require_openai", _raise)
        gw._openai_create_params.cache_clear()
        with pytest.raises(DependencyError):
            gw._openai_create_params()
        gw._openai_create_params.cache_clear()

    def test_split_create_kwargs_routes_gateway_fields(self):
        kwargs, body = gw._split_create_kwargs(
            {"temperature": 0, "method": "ocr", "document_dpi": 200}
        )
        assert kwargs == {"temperature": 0}
        assert body == {"method": "ocr", "document_dpi": 200}

    def test_split_create_kwargs_merges_explicit_extra_body(self):
        kwargs, body = gw._split_create_kwargs(
            {"extra_body": {"method": "ocr", "document_dpi": 72}, "method": "markdown"}
        )
        assert kwargs == {}
        # Routed keys win over the explicit extra_body payload.
        assert body == {"method": "markdown", "document_dpi": 72}

    def test_renderable_plain_text_for_markup_and_json(self):
        # OCR output is <document>-wrapped; Markdown would render it as HTML
        # and drop it entirely.
        assert isinstance(gw._renderable('<document pages="1">hi</document>'), Text)
        assert isinstance(gw._renderable('{"text": "hi"}'), Text)
        assert isinstance(gw._renderable("\n  <page>hi</page>"), Text)

    def test_renderable_markdown_for_prose(self):
        assert isinstance(gw._renderable("# Heading\n\nsome text"), Markdown)

    def test_format_cost(self):
        assert gw._format_cost(0.001508) == "$0.001508"
        assert gw._format_cost(0.0482) == "$0.0482"
        assert gw._format_cost(3.26e-06) == "$0.000003"
        assert gw._format_cost(1e-9) == "<$0.000001"  # real but tiny, never "$0"
        assert gw._format_cost(0) == "$0"
        assert gw._format_cost(None) is None
        assert gw._format_cost("nan-ish") is None

    def test_format_toks_per_sec(self):
        assert gw._format_toks_per_sec(16076, 14.51) == "1107 toks/s"
        assert gw._format_toks_per_sec(30, 1.0) == "30 toks/s"
        assert gw._format_toks_per_sec(0, 1.0) is None
        assert gw._format_toks_per_sec(10, 0) is None

    def test_parse_document_pages_from_content(self):
        content = (
            '<document pages="3">\n<page index="0">a</page>\n</document>'
        )
        assert gw._parse_document_pages_from_content(content) == 3
        assert gw._parse_document_pages_from_content("plain text") is None
        assert gw._parse_document_pages_from_content('<document pages="0">') is None
        glm_ocr_doc = (
            '<document file_name="tsla-8k.pdf" mimetype="application/pdf" '
            'num_pages="5" dpi="96">\n<page page_index="0">hi</page>\n</document>'
        )
        assert gw._parse_document_pages_from_content(glm_ocr_doc) == 5
        assert gw._parse_document_pages_from_content(
            "<document file_name='report.pdf' num_pages='4'>"
        ) == 4
        # URL-backed glm-ocr markdown can omit the count attribute; count <page> tags.
        assert gw._parse_document_pages_from_content(
            "<document file_name=\"tsla-8k.pdf\">\n"
            "<page page_index=\"0\">a</page>\n"
            "<page page_index=\"1\">b</page>\n"
            "</document>"
        ) == 2

    def test_build_chat_json_includes_pages_from_output(self):
        content = '<document pages="5">\n<page index="0">hi</page>\n</document>'
        out = gw._build_chat_json("glm-ocr", content, 2.5, FakeUsage())
        assert out["pages"] == 5
        assert out["pages_per_sec"] == 2.0
        assert "pages" not in gw._build_chat_json("glm-ocr", "plain", 1.0, FakeUsage())

    def test_format_pages_per_sec(self):
        assert gw._format_pages_per_sec(10, 5.0) == "2.00 pages/s"
        assert gw._format_pages_per_sec(1, 3.0) == "0.33 pages/s"
        assert gw._format_pages_per_sec(4, 0) is None

    def test_stats_parts_uses_toks_label_and_pages(self):
        usage = FakeUsage(prompt_tokens=5, completion_tokens=15)
        parts = gw._stats_parts("glm-ocr", 2.0, usage, pages=4)
        assert "T:20 toks" in parts[1]
        assert "7 toks/s" in parts[2]
        assert parts[3] == "4 pages"
        assert parts[4] == "2.00 pages/s"
        assert "2s" in parts

    def test_stats_markup_is_dim_white_not_bold(self):
        markup = gw._stats_markup(["glm-ocr", "2s"])
        assert "dim" in markup
        assert "white" in markup
        assert "not bold" in markup
        assert "[bold]" not in markup

    def test_stats_parts_skips_pages_for_images(self):
        usage = FakeUsage()
        parts = gw._stats_parts("glm-ocr", 1.0, usage, pages=None)
        assert all("pages" not in p for p in parts)

    def test_content_error_detects_error_payload(self):
        assert (
            gw._content_error('{"error": "Unknown method \'zzz\'"}')
            == "Unknown method 'zzz'"
        )

    def test_content_error_ignores_normal_output(self):
        # OCR JSON lines, document-wrapped text, and prose are not errors.
        assert gw._content_error('{"text": "Arizona", "score": 1.0}') is None
        assert gw._content_error('<document pages="1">hi</document>') is None
        assert gw._content_error("plain transcript text") is None
        # An object that merely contains an error key alongside data is not it.
        assert gw._content_error('{"error": "x", "text": "y"}') is None

    def test_format_methods_marks_default(self):
        out = gw._format_methods(
            {"methods": ["ocr", "detect"], "default_method": "ocr"}
        )
        assert "[bold]ocr[/bold]*" in out
        assert "detect" in out and "detect*" not in out

    def test_format_methods_empty(self):
        assert gw._format_methods({}) == "-"

    def test_format_inputs_lists_explicit_types(self):
        out = gw._format_inputs(
            {
                "capabilities": {
                    "supported_input_types": [
                        "text",
                        "image_url",
                        "document_url",
                        "video_url",
                    ]
                }
            }
        )
        assert out == "text, image_url, document_url, video_url"

    def test_format_inputs_empty(self):
        assert gw._format_inputs({}) == "-"

    def test_grouped_model_rows_orders_by_task_then_id(self):
        rows = [
            {"id": "z/chat-b", "task": "chat"},
            {"id": "a/embed", "task": "embed"},
            {"id": "m/transcribe", "task": "transcribe"},
            {"id": "a/chat-a", "task": "chat"},
        ]
        grouped = gw._grouped_model_rows(rows)
        assert [r.get("id") if r else None for r in grouped] == [
            "a/chat-a",
            "z/chat-b",
            None,
            "m/transcribe",
            None,
            "a/embed",
        ]

    def test_parse_extra_body_help_splits_json_and_prose(self):
        examples, notes = gw._parse_extra_body_help(
            '{"method":"ocr"} | {"method":"ocr","method_params":{"lang":"en"}}'
            " | document_url PDF (markdown per page)"
        )
        assert examples == [
            {"method": "ocr"},
            {"method": "ocr", "method_params": {"lang": "en"}},
        ]
        assert notes == ["document_url PDF (markdown per page)"]

    def test_parse_extra_body_help_empty(self):
        assert gw._parse_extra_body_help("") == ([], [])

    def test_example_command_renders_method_and_params(self):
        cmd = gw._example_command(
            "pp-ocrv6", {"method": "ocr", "method_params": {"lang": "en"}}, "doc.pdf"
        )
        assert cmd == (
            "vlmrun gw chat doc.pdf -m pp-ocrv6 --method ocr "
            '--method-params \'{"lang": "en"}\''
        )

    def test_example_command_routes_other_fields_to_extra(self):
        cmd = gw._example_command("glm-ocr", {"document_dpi": 200}, "doc.pdf")
        assert cmd == "vlmrun gw chat doc.pdf -m glm-ocr -e document_dpi=200"

    def test_sample_input_prefers_document(self):
        assert (
            gw._sample_input(
                {
                    "capabilities": {
                        "supported_input_types": ["image_url", "document_url"]
                    }
                }
            )
            == "doc.pdf"
        )
        assert (
            gw._sample_input({"capabilities": {"supported_input_types": ["image_url"]}})
            == "img.jpg"
        )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class TestGatewayCLI:
    @pytest.mark.parametrize("alias", ["gw", "gateway"])
    def test_both_aliases_registered(self, runner, patched_cli, alias):
        result = runner.invoke(app, [alias, "health"])
        assert result.exit_code == 0
        assert "healthy" in result.stdout.lower()

    def test_health_ok(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "health"])
        assert result.exit_code == 0
        assert "healthy" in result.stdout.lower()

    def test_health_unreachable(self, runner, patched_cli):
        patched_cli["healthy"] = False
        result = runner.invoke(app, ["gw", "health"])
        assert result.exit_code == 1
        assert "unreachable" in result.stdout.lower()

    def test_models_table(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models"])
        assert result.exit_code == 0
        assert "zai-org/glm-ocr" in result.stdout
        assert "paddleocr/pp-ocrv6" in result.stdout
        assert "image_url" in result.stdout
        assert "document_url" in result.stdout
        # Methods are listed, with the default marked.
        assert "detect" in result.stdout
        assert "ocr*" in result.stdout

    def test_models_json(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        ids = {m["id"] for m in data}
        assert ids == {"zai-org/glm-ocr", "paddleocr/pp-ocrv6"}

    def test_model_detail_by_alias(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "pp-ocrv6"])
        assert result.exit_code == 0
        assert "paddleocr/pp-ocrv6" in result.stdout
        # Detail view is scoped to the one model.
        assert "zai-org/glm-ocr" not in result.stdout
        assert "detect" in result.stdout

    def test_model_detail_by_id_shows_notes(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "zai-org/glm-ocr"])
        assert result.exit_code == 0
        # Prose fragments of extra_body_help surface as notes.
        assert "markdown per page" in result.stdout

    def test_model_detail_unknown_model(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_model_detail_json_emits_runnable_commands(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "pp-ocrv6", "--json"])
        assert result.exit_code == 0
        entry = json.loads(result.stdout)
        # Detail --json is a single object, not the catalog list.
        assert entry["id"] == "paddleocr/pp-ocrv6"
        assert entry["default_method"] == "ocr"
        assert entry["methods"] == ["ocr", "detect", "markdown"]
        assert entry["commands"] == [
            "vlmrun gw chat doc.pdf -m paddleocr/pp-ocrv6 --method ocr "
            '--method-params \'{"lang": "en", "score_threshold": 0.5}\''
        ]

    def test_models_list_json_still_returns_catalog(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "--json"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.stdout), list)

    def test_methods_command_is_gone(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "methods"])
        assert result.exit_code != 0


class TestGatewayEmbed:
    def test_embed_text(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "embed", "-t", "hello", "-m", "emb"])
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.embeddings.calls[-1]
        # Plain text rides as a bare string.
        assert call["input"] == ["hello"]

    def test_embed_image_nests_content_parts(self, runner, patched_cli, tmp_path):
        """Each item must be a *list* of parts; a flat parts list is rejected."""
        img = tmp_path / "a.png"
        img.write_bytes(PNG_BYTES)
        result = runner.invoke(app, ["gw", "embed", str(img), "-m", "emb"])
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.embeddings.calls[-1]
        assert len(call["input"]) == 1
        item = call["input"][0]
        assert isinstance(item, list)
        assert item[0]["type"] == "image_url"
        assert item[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_embed_video_uses_video_url_part(self, runner, patched_cli, tmp_path):
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(MP4_BYTES)
        result = runner.invoke(app, ["gw", "embed", str(vid), "-m", "emb"])
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.embeddings.calls[-1]
        assert call["input"][0][0]["type"] == "video_url"

    def test_embed_files_and_text_are_separate_vectors(
        self, runner, patched_cli, tmp_path
    ):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        a.write_bytes(PNG_BYTES)
        b.write_bytes(PNG_BYTES)
        result = runner.invoke(
            app, ["gw", "embed", str(a), str(b), "-t", "cap", "-m", "emb"]
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.embeddings.calls[-1]
        # Three independent items — batching two images into one item 500s.
        assert len(call["input"]) == 3
        assert call["input"][2] == "cap"

    def test_embed_join_combines_into_one_vector(self, runner, patched_cli, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(PNG_BYTES)
        result = runner.invoke(
            app, ["gw", "embed", str(img), "-t", "cap", "--join", "-m", "emb"]
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.embeddings.calls[-1]
        assert len(call["input"]) == 1
        assert [p["type"] for p in call["input"][0]] == ["image_url", "text"]

    def test_embed_join_rejects_multiple_files(self, runner, patched_cli, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        a.write_bytes(PNG_BYTES)
        b.write_bytes(PNG_BYTES)
        result = runner.invoke(
            app, ["gw", "embed", str(a), str(b), "--join", "-m", "emb"]
        )
        assert result.exit_code == 1
        assert "at most one file" in result.stdout

    def test_embed_requires_input(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "embed", "-m", "emb"])
        assert result.exit_code == 1
        assert "at least one file or --text" in result.stdout

    def test_embed_rejects_non_image_file(self, runner, patched_cli, tmp_path):
        # A PDF (or any non-image/video) must be rejected client-side rather
        # than sent as a mislabelled image_url the gateway would fail on.
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        result = runner.invoke(app, ["gw", "embed", str(doc), "-m", "emb"])
        assert result.exit_code == 1
        assert "images and video only" in result.stdout
        assert not patched_cli["client"].gateway.embeddings.calls

    def test_embed_dimensions_passed_through(self, runner, patched_cli):
        result = runner.invoke(
            app, ["gw", "embed", "-t", "hi", "-m", "emb", "--dimensions", "64"]
        )
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.embeddings.calls[-1]["dimensions"] == 64

    def test_embed_json(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "embed", "-t", "hi", "-m", "emb", "--json"])
        assert result.exit_code == 0, result.stdout
        assert len(json.loads(result.stdout)["data"][0]["embedding"]) == 4


class TestGatewayTranscribe:
    def test_transcribe_file(self, runner, patched_cli, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"fake-audio")
        result = runner.invoke(app, ["gw", "transcribe", str(audio), "-m", "asr"])
        assert result.exit_code == 0, result.stdout
        assert "hello from audio" in result.stdout
        call = patched_cli["client"].gateway.transcriptions.calls[-1]
        assert call["response_format"] == "json"

    def test_transcribe_format_and_hints(self, runner, patched_cli, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"fake-audio")
        result = runner.invoke(
            app,
            [
                "gw",
                "transcribe",
                str(audio),
                "-m",
                "asr",
                "-f",
                "srt",
                "-l",
                "en",
                "-p",
                "nouns",
            ],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.transcriptions.calls[-1]
        assert call["response_format"] == "srt"
        assert call["language"] == "en"
        assert call["prompt"] == "nouns"

    def test_transcribe_url_rides_in_extra_body(self, runner, patched_cli):
        result = runner.invoke(
            app, ["gw", "transcribe", "--url", "https://x/a.mp3", "-m", "asr"]
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.transcriptions.calls[-1]
        assert call["extra_body"] == {"url": "https://x/a.mp3"}

    def test_transcribe_requires_input(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "transcribe", "-m", "asr"])
        assert result.exit_code == 1
        assert "audio file or --url" in result.stdout

    def test_transcribe_rejects_file_and_url(self, runner, patched_cli, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"fake-audio")
        result = runner.invoke(
            app,
            ["gw", "transcribe", str(audio), "--url", "https://x/a.mp3", "-m", "asr"],
        )
        assert result.exit_code == 1
        assert "not both" in result.stdout

    def test_transcribe_bad_format(self, runner, patched_cli, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"fake-audio")
        result = runner.invoke(
            app, ["gw", "transcribe", str(audio), "-m", "asr", "-f", "bogus"]
        )
        assert result.exit_code == 1
        assert "unknown --format" in result.stdout.lower()

    def test_chat_requires_file(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "chat", "-m", "glm-ocr"])
        assert result.exit_code == 1
        assert "at least one input" in result.stdout.lower()

    def test_chat_rejects_missing_file(self, runner, patched_cli):
        result = runner.invoke(
            app, ["gw", "chat", "missing.pdf", "-m", "glm-ocr", "--no-stream"]
        )
        assert result.exit_code == 1
        assert "not a file" in result.stdout.lower()

    def test_chat_with_url(self, runner, patched_cli):
        url = "https://example.com/scan.jpg"
        result = runner.invoke(
            app,
            ["gw", "chat", url, "-m", "paddle-ocrv6", "--no-stream"],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.completions.calls[-1]
        content = call["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == url
        assert call["stream"] is False

    def test_chat_document_url_streams(self, runner, patched_cli):
        url = "https://example.com/report.pdf"
        result = runner.invoke(app, ["gw", "chat", url, "-m", "glm-ocr"])
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.completions.calls[-1]["stream"] is True

    def test_chat_with_file_json(self, runner, patched_cli, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        result = runner.invoke(
            app,
            ["gw", "chat", str(f), "-m", "glm-ocr", "--no-stream", "--json"],
        )
        assert result.exit_code == 0, result.stdout
        out = json.loads(result.stdout)
        assert out["model"] == "glm-ocr"
        assert out["content"] == "Hello world"

    def test_chat_streaming_default(self, runner, patched_cli, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(app, ["gw", "chat", str(f), "-m", "paddle-ocrv6"])
        assert result.exit_code == 0, result.stdout
        assert "Hello world" in result.stdout

    def test_chat_sends_document_and_image_urls(self, runner, patched_cli, tmp_path):
        pdf = tmp_path / "doc.pdf"
        img = tmp_path / "scan.png"
        pdf.write_bytes(b"%PDF fake")
        img.write_bytes(PNG_BYTES)
        result = runner.invoke(
            app,
            ["gw", "chat", str(pdf), str(img), "-m", "glm-ocr", "--no-stream"],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.completions.calls[-1]
        content = call["messages"][0]["content"]
        assert content[0]["type"] == "document_url"
        assert content[0]["document_url"]["url"].startswith(
            "data:application/pdf;base64,"
        )
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_chat_image_streams_by_default(self, runner, patched_cli, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(PNG_BYTES)
        result = runner.invoke(app, ["gw", "chat", str(img), "-m", "pp-ocrv6"])
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.completions.calls[-1]["stream"] is True
        assert "Hello world" in result.stdout

    def test_chat_video_streams_by_default(self, runner, patched_cli, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8)
        result = runner.invoke(app, ["gw", "chat", str(video), "-m", "pp-ocrv6"])
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.completions.calls[-1]["stream"] is True
        assert "Hello world" in result.stdout

    def test_chat_text_only_streams_by_default(self, runner, patched_cli):
        result = runner.invoke(
            app, ["gw", "chat", "-m", "qwen/qwen3.5-0.8b", "-p", "hi"]
        )
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.completions.calls[-1]["stream"] is True

    def test_chat_document_streams_by_default(self, runner, patched_cli, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF fake")
        result = runner.invoke(app, ["gw", "chat", str(pdf), "-m", "glm-ocr"])
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.completions.calls[-1]["stream"] is True

    def test_chat_document_no_stream_flag_respected(
        self, runner, patched_cli, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF fake")
        result = runner.invoke(app, ["gw", "chat", str(pdf), "-m", "glm-ocr", "-ns"])
        assert result.exit_code == 0, result.stdout
        assert patched_cli["client"].gateway.completions.calls[-1]["stream"] is False

    def test_drain_stream_collects_content_reasoning_usage(self):
        """The stream drainer accumulates content and reasoning separately,
        keeps the trailing usage, and fires on_update once per text chunk."""

        class _Delta:
            def __init__(self, content=None, reasoning=None):
                self.content = content
                self.reasoning_content = reasoning

        class _Choice:
            def __init__(self, delta):
                self.delta = delta

        class _Chunk:
            def __init__(self, delta=None, usage=None):
                self.choices = [_Choice(delta)] if delta is not None else []
                self.usage = usage

        usage = FakeUsage()
        stream = [
            _Chunk(_Delta(reasoning="think ")),
            _Chunk(_Delta(reasoning="more")),
            _Chunk(_Delta(content="Hello ")),
            _Chunk(_Delta(content="world")),
            _Chunk(usage=usage),  # usage-only trailing chunk, no text
        ]
        updates: list = []
        content, reasoning, got_usage = gw._drain_stream(
            iter(stream), on_update=lambda c, r: updates.append((c, r))
        )
        assert content == "Hello world"
        assert reasoning == "think more"
        assert got_usage is usage
        # one update per text-bearing chunk (4); the usage-only chunk is silent
        assert len(updates) == 4
        assert updates[-1] == ("Hello world", "think more")

    def test_drain_stream_reasoning_field_fallback(self):
        """`delta.reasoning` is honored when `reasoning_content` is absent."""

        class _Delta:
            def __init__(self, reasoning):
                self.content = None
                self.reasoning = reasoning  # no reasoning_content attribute

        class _Choice:
            def __init__(self, delta):
                self.delta = delta

        class _Chunk:
            def __init__(self, delta):
                self.choices = [_Choice(delta)]
                self.usage = None

        content, reasoning, _ = gw._drain_stream(iter([_Chunk(_Delta("hm"))]))
        assert content == ""
        assert reasoning == "hm"

    def test_chat_multiple_files_and_extra(self, runner, patched_cli, tmp_path):
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"%PDF a")
        f2.write_bytes(b"%PDF b")
        result = runner.invoke(
            app,
            [
                "gw",
                "chat",
                str(f1),
                str(f2),
                "-m",
                "paddle-ocrv6",
                "-e",
                "temperature=0",
                "--no-stream",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stdout
        out = json.loads(result.stdout)
        assert out["content"] == "Hello world"

    def test_chat_method_and_params_sent_via_extra_body(
        self, runner, patched_cli, tmp_path
    ):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app,
            [
                "gateway",
                "chat",
                str(f),
                "-m",
                "pp-ocrv6",
                "--method",
                "ocr",
                "--method-params",
                '{"lang": "en", "score_threshold": 0.9}',
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.completions.calls[-1]
        assert call["extra_body"] == {
            "method": "ocr",
            "method_params": {"lang": "en", "score_threshold": 0.9},
        }
        # method must not leak into create()'s own kwargs.
        assert "method" not in call

    def test_chat_response_format_sent_as_top_level_kwarg(
        self, runner, patched_cli, tmp_path
    ):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app,
            [
                "gw",
                "chat",
                str(f),
                "-m",
                "pp-ocrv6",
                "--response-format",
                "json_object",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.completions.calls[-1]
        # A standard OpenAI field, so it rides top-level, not in extra_body.
        assert call["response_format"] == {"type": "json_object"}
        assert "response_format" not in call.get("extra_body", {})

    def test_chat_json_mode_sent_as_top_level_kwarg(
        self, runner, patched_cli, tmp_path
    ):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app,
            ["gw", "chat", str(f), "-m", "pp-ocrv6", "--json-mode", "--no-stream"],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.completions.calls[-1]
        assert call["response_format"] == {"type": "json_object"}
        assert "response_format" not in call.get("extra_body", {})

    def test_chat_json_mode_and_response_format_mutually_exclusive(
        self, runner, patched_cli, tmp_path
    ):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app,
            [
                "gw",
                "chat",
                str(f),
                "-m",
                "pp-ocrv6",
                "--json-mode",
                "--response-format",
                "json_object",
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.stdout.lower()

    def test_chat_response_format_invalid(self, runner, patched_cli, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app,
            ["gw", "chat", str(f), "-m", "pp-ocrv6", "--response-format", "yaml"],
        )
        assert result.exit_code == 1
        assert "response-format" in result.stdout.lower()

    def test_chat_extra_routes_gateway_field_to_extra_body(
        self, runner, patched_cli, tmp_path
    ):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app,
            [
                "gw",
                "chat",
                str(f),
                "-m",
                "pp-ocrv6",
                "-e",
                "document_dpi=200",
                "-e",
                "temperature=0",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0, result.stdout
        call = patched_cli["client"].gateway.completions.calls[-1]
        assert call["extra_body"] == {"document_dpi": 200}
        assert call["temperature"] == 0

    def test_chat_no_extra_body_when_unused(self, runner, patched_cli, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--no-stream"]
        )
        assert result.exit_code == 0, result.stdout
        assert "extra_body" not in patched_cli["client"].gateway.completions.calls[-1]

    def test_chat_invalid_method_params(self, runner, patched_cli, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--method-params", "nope"]
        )
        assert result.exit_code == 1
        assert "valid json" in result.stdout.lower()

    def test_chat_method_params_must_be_object(self, runner, patched_cli, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--method-params", "[1, 2]"]
        )
        assert result.exit_code == 1
        assert "json object" in result.stdout.lower()

    def test_chat_renders_document_wrapped_ocr_output(
        self, runner, patched_cli, tmp_path
    ):
        """OCR output must survive rendering (regression: Markdown ate the tags)."""
        patched_cli["content"] = (
            '<document pages="1">\n<page index="0">\nDLN B58471293\n</page>\n</document>'
        )
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--no-stream"]
        )
        assert result.exit_code == 0, result.stdout
        assert "DLN B58471293" in result.stdout
        assert "<document" in result.stdout

    def test_chat_error_payload_fails_loudly(self, runner, patched_cli, tmp_path):
        """An {"error": ...} body (e.g. bad --method) must exit non-zero."""
        patched_cli["content"] = '{"error": "Unknown method \'bogus\'"}'
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--no-stream"]
        )
        assert result.exit_code == 1
        assert "Unknown method" in result.stdout

    def test_chat_error_payload_nonzero_in_json(self, runner, patched_cli, tmp_path):
        patched_cli["content"] = '{"error": "boom"}'
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--no-stream", "--json"]
        )
        # Still emits the JSON (for scripts) but signals failure.
        assert result.exit_code == 1
        assert json.loads(result.stdout)["content"] == '{"error": "boom"}'

    def test_chat_panel_shows_cost(self, runner, patched_cli, tmp_path):
        patched_cli["cost"] = 0.001508
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--no-stream"]
        )
        assert result.exit_code == 0, result.stdout
        assert "$0.001508" in result.stdout
        assert " toks/s" in result.stdout
        assert " toks" in result.stdout
        assert " tokens" not in result.stdout

    def test_chat_panel_shows_document_pages(self, runner, patched_cli, tmp_path):
        patched_cli["content"] = (
            '<document pages="5">\n<page index="0">\nhello\n</page>\n</document>'
        )
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        result = runner.invoke(
            app, ["gw", "chat", str(doc), "-m", "glm-ocr", "--no-stream"]
        )
        assert result.exit_code == 0, result.stdout
        assert "5 pages" in result.stdout
        assert "pages/s" in result.stdout

    def test_chat_document_url_shows_pages_streaming(self, runner, patched_cli):
        patched_cli["content"] = (
            '<document file_name="report.pdf" mimetype="application/pdf" '
            'num_pages="5" dpi="96">\n<page page_index="0">hello</page>\n</document>'
        )
        url = "https://example.com/report.pdf"
        result = runner.invoke(app, ["gw", "chat", url, "-m", "glm-ocr"])
        assert result.exit_code == 0, result.stdout
        assert "5 pages" in result.stdout
        assert "pages/s" in result.stdout
        content = patched_cli["client"].gateway.completions.calls[-1]["messages"][0][
            "content"
        ]
        assert content[0]["type"] == "document_url"
        assert content[0]["document_url"]["url"] == url

    def test_chat_document_url_shows_pages_no_stream(self, runner, patched_cli):
        patched_cli["content"] = (
            '<document file_name="report.pdf" num_pages="4">\n'
            '<page page_index="0">hello</page>\n</document>'
        )
        url = "https://storage.example.com/hub/examples/report.pdf"
        result = runner.invoke(
            app, ["gw", "chat", url, "-m", "glm-ocr", "--no-stream"]
        )
        assert result.exit_code == 0, result.stdout
        assert "4 pages" in result.stdout
        assert "pages/s" in result.stdout

    def test_chat_json_carries_cost(self, runner, patched_cli, tmp_path):
        patched_cli["cost"] = 0.0042
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        result = runner.invoke(
            app, ["gw", "chat", str(f), "-m", "pp-ocrv6", "--no-stream", "--json"]
        )
        assert result.exit_code == 0, result.stdout
        assert json.loads(result.stdout)["usage"]["cost"] == 0.0042

    def test_chat_json_includes_pages_from_output_no_stream(
        self, runner, patched_cli, tmp_path
    ):
        patched_cli["content"] = (
            '<document pages="4">\n<page index="0">\nhello\n</page>\n</document>'
        )
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF fake")
        result = runner.invoke(
            app, ["gw", "chat", str(doc), "-m", "glm-ocr", "--no-stream", "--json"]
        )
        assert result.exit_code == 0, result.stdout
        out = json.loads(result.stdout)
        assert out["pages"] == 4
        assert out["pages_per_sec"] == round(4 / out["latency_s"], 2)

    def test_chat_json_includes_pages_from_output_stream(
        self, runner, patched_cli, tmp_path
    ):
        patched_cli["content"] = (
            '<document pages="3">\n<page index="0">\nhello\n</page>\n</document>'
        )
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF fake")
        result = runner.invoke(
            app, ["gw", "chat", str(doc), "-m", "glm-ocr", "--json"]
        )
        assert result.exit_code == 0, result.stdout
        out = json.loads(result.stdout)
        assert out["pages"] == 3
        assert out["pages_per_sec"] == round(3 / out["latency_s"], 2)
