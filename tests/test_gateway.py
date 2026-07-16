"""Tests for the OpenAI-compatible gateway resource and `vlmrun gw` CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from vlmrun.cli.cli import app
from vlmrun.cli._cli import gateway as gw
from vlmrun.client.gateway import Gateway
from vlmrun.constants import DEFAULT_GATEWAY_URL

# ---------------------------------------------------------------------------
# Concrete fakes (per CLAUDE.md: no MagicMock)
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def model_dump(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


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

    def create(self, model, messages, stream=False, **kwargs):
        self.calls.append(
            {"model": model, "messages": messages, "stream": stream, **kwargs}
        )
        if stream:
            return iter(
                [
                    FakeChunk("Hello "),
                    FakeChunk("world", usage=FakeUsage()),
                ]
            )
        return FakeResponse("Hello world")


class FakeModel:
    """Mimics an OpenAI ``Model`` object with gateway pricing extras."""

    def __init__(self, id: str, owned_by: str, pricing: dict) -> None:
        self._data = {"id": id, "owned_by": owned_by, "pricing": pricing}

    def model_dump(self) -> dict:
        return dict(self._data)


class FakeGateway:
    def __init__(self, healthy: bool = True) -> None:
        self.base_url = "https://gateway.vlm.run/v1"
        self._healthy = healthy
        self.completions = FakeCompletions()

    def health(self) -> bool:
        return self._healthy

    def models(self) -> list:
        return [
            FakeModel("glm-ocr", "zhipu", {"input": 0.1, "output": 0.2}),
            FakeModel("paddle-ocrv6", "paddle", {}),
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
        holder["client"] = client
        return client

    monkeypatch.setattr("vlmrun.cli.cli.VLMRun", _factory)
    return holder


# ---------------------------------------------------------------------------
# Client resource: Gateway
# ---------------------------------------------------------------------------


class _MiniClient:
    def __init__(self) -> None:
        self.api_key = "sk-test"
        self.timeout = 120.0
        self.max_retries = 3


class TestGatewayResource:
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

        monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
        assert g.health() is True

    def test_health_falls_back_to_models_on_404(self, monkeypatch):
        g = Gateway(_MiniClient())

        class _Resp:
            status_code = 404
            is_success = False

        monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())

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

        monkeypatch.setattr("httpx.get", _boom)

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
        assert gw._content_part_type(tmp_path / "a.png") == "file_url"
        assert gw._content_part_type(tmp_path / "a.jpg") == "file_url"

    def test_encode_document_part(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.7 fake")
        part = gw._encode_file_part(f)
        assert part["type"] == "document_url"
        url = part["document_url"]["url"]
        assert url.startswith("data:application/pdf;base64,")

    def test_encode_image_part(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        part = gw._encode_file_part(f)
        assert part["type"] == "file_url"
        url = part["file_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_build_messages_with_prompt(self, tmp_path):
        f = tmp_path / "img.png"
        f.write_bytes(b"fakepng")
        messages = gw._build_messages([f], "describe")
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content[0]["type"] == "file_url"
        assert content[-1] == {"type": "text", "text": "describe"}

    def test_build_messages_mixed_files(self, tmp_path):
        img = tmp_path / "img.png"
        doc = tmp_path / "doc.pdf"
        img.write_bytes(b"fakepng")
        doc.write_bytes(b"%PDF fake")
        content = gw._build_messages([img, doc], None)[0]["content"]
        assert [p["type"] for p in content] == ["file_url", "document_url"]

    def test_parse_extra_json_and_string(self):
        parsed = gw._parse_extra(["temperature=0", "max_tokens=4096", "label=hello"])
        assert parsed == {"temperature": 0, "max_tokens": 4096, "label": "hello"}

    def test_parse_extra_invalid(self):
        with pytest.raises(Exception):
            gw._parse_extra(["nonsense"])

    def test_extract_pricing_nested(self):
        i, o = gw._extract_pricing({"pricing": {"input": 0.1, "output": 0.2}})
        assert i == "$0.1"
        assert o == "$0.2"

    def test_extract_pricing_per_token(self):
        i, o = gw._extract_pricing(
            {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            }
        )
        assert i == "$1"
        assert o == "$2"

    def test_extract_pricing_missing(self):
        i, o = gw._extract_pricing({"id": "x"})
        assert i == "-"
        assert o == "-"


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class TestGatewayCLI:
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
        assert "glm-ocr" in result.stdout
        assert "paddle-ocrv6" in result.stdout

    def test_models_json(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "models", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        ids = {m["id"] for m in data}
        assert ids == {"glm-ocr", "paddle-ocrv6"}

    def test_chat_requires_file(self, runner, patched_cli):
        result = runner.invoke(app, ["gw", "chat", "-m", "glm-ocr"])
        assert result.exit_code == 1
        assert "at least one input file" in result.stdout.lower()

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

    def test_chat_sends_document_and_file_urls(self, runner, patched_cli, tmp_path):
        pdf = tmp_path / "doc.pdf"
        img = tmp_path / "scan.png"
        pdf.write_bytes(b"%PDF fake")
        img.write_bytes(b"fakepng")
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
        assert content[1]["type"] == "file_url"
        assert content[1]["file_url"]["url"].startswith("data:image/png;base64,")

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
