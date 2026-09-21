"""Tests for ``vlmrun gw systemone``.

The command is exercised end to end through the Typer runner with
:meth:`SystemOne.decide` replaced by a recorder, so the assertions cover the
CLI's own work: flag parsing, positional resolution and rendering.
"""

from __future__ import annotations

import base64
import json

import pytest

from tests.conftest import strip_ansi
from vlmrun.cli._cli.gateway_systemone import (
    _answer_row,
    _bar,
    _classify,
    _questions_from_flags,
    _resolve_inputs,
)
from vlmrun.cli.cli import app
from vlmrun.client.systemone import SystemOne

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n"


class Answer:
    """A stand-in for one decoded answer object."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class Usage:
    def __init__(self, input_tokens=149, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class Recorded:
    """A stand-in SystemOneResponse that also records the call that made it."""

    def __init__(self, calls, answers=None):
        self.calls = calls
        self.model = "google/diffusiongemma-26b-a4b-it"
        self.usage = Usage()
        self.answers = answers or {
            "is_urgent": Answer(type="noul", noul=0.91),
            "dept": Answer(
                type="choice",
                choice="billing",
                probabilities={"billing": 0.9, "sales": 0.1},
                confidence=0.7,
            ),
            "mood": Answer(
                type="score",
                score=0.9,
                legend={"0": "calm", "1": "angry"},
                probabilities={"0": 0.1, "1": 0.9},
                confidence=0.5,
            ),
        }

    @property
    def raw_http_response(self):
        raise RuntimeError("no raw response in tests")

    def model_dump_json(self):
        return json.dumps(
            {"model": self.model, "answers": {"is_urgent": {"noul": 0.91}}}
        )


@pytest.fixture
def decide(monkeypatch):
    """Capture the arguments the CLI hands to the SDK."""
    calls: list = []

    def fake_decide(self, state, questions, **kwargs):
        calls.append({"state": state, "questions": questions, **kwargs})
        return Recorded(calls)

    monkeypatch.setattr(SystemOne, "decide", fake_decide)
    return calls


class TestQuestionFlags:
    def test_noul_with_and_without_instructions(self):
        questions = _questions_from_flags(["a", 'b="Is it urgent?"'], None, None)
        assert questions[0] == {"id": "a", "type": "noul"}
        assert questions[1] == {
            "id": "b",
            "type": "noul",
            "instructions": '"Is it urgent?"',
        }

    def test_choice_labels_and_descriptions(self):
        questions = _questions_from_flags(None, ["dept=billing:Charges|sales"], None)
        assert questions[0]["options"] == [
            {"name": "billing", "description": "Charges"},
            "sales",
        ]

    def test_score_levels(self):
        questions = _questions_from_flags(None, None, ["mood=calm|angry"])
        assert questions[0] == {
            "id": "mood",
            "type": "score",
            "levels": ["calm", "angry"],
        }

    def test_order_is_noul_then_choice_then_score(self):
        questions = _questions_from_flags(["a"], ["b=x|y"], ["c=l|h"])
        assert [q["id"] for q in questions] == ["a", "b", "c"]


class TestClassifyInputs:
    def test_plain_text_is_state(self):
        assert _classify("just some text") == "text"

    def test_image_and_pdf_and_text_file(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(PDF_BYTES)
        note = tmp_path / "a.txt"
        note.write_text("hello")
        assert _classify(str(image)) == "image"
        assert _classify(str(pdf)) == "document"
        assert _classify(str(note)) == "state_file"

    def test_remote_urls(self):
        assert _classify("https://example.com/a.pdf") == "document"
        assert _classify("https://example.com/a.jpg") == "image"

    def test_json_state_file_is_parsed(self, tmp_path):
        state_file = tmp_path / "s.json"
        state_file.write_text('{"message": "hi"}')
        state_parts, images, document = _resolve_inputs([str(state_file)], False)
        assert state_parts == [{"message": "hi"}]
        assert images == [] and document is None

    def test_two_pdfs_are_rejected(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(PDF_BYTES)
        with pytest.raises(Exception):
            _resolve_inputs([str(pdf), str(pdf)], False)


class TestRenderHelpers:
    def test_bar_is_monotonic(self):
        assert _bar(0.0).strip() == ""
        assert len(_bar(1.0).strip()) == 9
        assert len(_bar(0.5).strip()) <= 5

    def test_answer_rows(self):
        shown, value, metric, _ = _answer_row(Answer(type="noul", noul=0.91))
        assert (shown, value, metric) == ("yes", 0.91, "P(yes)")
        shown, value, metric, probabilities = _answer_row(
            Answer(
                type="choice",
                choice="billing",
                probabilities={"billing": 0.9, "sales": 0.1},
                confidence=0.7,
            )
        )
        assert shown == "billing" and value == 0.7
        assert probabilities.startswith("billing 0.90")
        shown, _, _, probabilities = _answer_row(
            Answer(
                type="score",
                score=0.9,
                legend={"0": "calm", "1": "angry"},
                probabilities={"0": 0.1, "1": 0.9},
                confidence=0.5,
            )
        )
        assert shown == "0.9 angry"
        assert probabilities.startswith("0 calm 0.10")


class TestCommand:
    def test_inline_flags_build_questions(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "the state",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
            ],
        )
        assert result.exit_code == 0, result.stdout
        call = decide[0]
        assert call["state"] == "the state"
        assert call["questions"] == [
            {"id": "is_urgent", "type": "noul"},
            {"id": "dept", "type": "choice", "options": ["billing", "sales"]},
        ]
        out = strip_ansi(result.stdout)
        assert "is_urgent" in out and "0.91" in out and "billing" in out

    def test_questions_json_list(self, runner, decide, config_file):
        spec = '[{"id": "a", "type": "noul"}]'
        result = runner.invoke(app, ["gw", "systemone", "x", "-Q", spec])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["questions"] == [{"id": "a", "type": "noul"}]

    def test_questions_wrapper_and_flag_override(self, runner, decide, config_file):
        spec = '{"questions": {"a": {"type": "noul"}}}'
        result = runner.invoke(
            app, ["gw", "systemone", "x", "-Q", spec, "--choice", "a=x|y"]
        )
        assert result.exit_code == 0, result.stdout
        assert decide[0]["questions"]["a"]["type"] == "choice"

    def test_body_supplies_state_and_questions(self, runner, decide, config_file):
        body = json.dumps(
            {
                "state": "from body",
                "questions": [{"id": "a", "type": "noul"}],
                "samples": 4,
            }
        )
        result = runner.invoke(app, ["gw", "systemone", "--body", body])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["state"] == "from body"
        assert decide[0]["extra_body"] == {"samples": 4}

    def test_positional_state_overrides_body(self, runner, decide, config_file):
        body = json.dumps(
            {"state": "from body", "questions": [{"id": "a", "type": "noul"}]}
        )
        result = runner.invoke(app, ["gw", "systemone", "from the cli", "--body", body])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["state"] == "from the cli"

    def test_media_positionals(self, runner, decide, config_file, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(PDF_BYTES)
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                str(image),
                str(pdf),
                "-s",
                "does it match?",
                "--detail",
                "high",
                "--noul",
                "matches",
            ],
        )
        assert result.exit_code == 0, result.stdout
        call = decide[0]
        assert call["images"] == [str(image)]
        assert call["document"] == str(pdf)
        assert call["detail"] == "high"
        assert call["state"] == "does it match?"

    def test_state_file_is_read(self, runner, decide, config_file, tmp_path):
        ticket = tmp_path / "ticket.txt"
        ticket.write_text("charged twice")
        result = runner.invoke(app, ["gw", "systemone", str(ticket), "--noul", "a"])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["state"] == "charged twice"

    def test_steps_and_samples(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            ["gw", "systemone", "x", "--noul", "a", "--steps", "2", "--samples", "4"],
        )
        assert result.exit_code == 0, result.stdout
        assert decide[0]["steps"] == 2 and decide[0]["samples"] == 4

    def test_json_output(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "x", "--noul", "a", "--json"])
        assert result.exit_code == 0, result.stdout
        assert "0.91" in strip_ansi(result.stdout)

    def test_no_questions_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "x"])
        assert result.exit_code == 1
        assert "no questions" in strip_ansi(result.stdout)

    def test_no_state_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "--noul", "a"])
        assert result.exit_code == 1
        assert "no state" in strip_ansi(result.stdout)

    def test_bad_detail_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(
            app, ["gw", "systemone", "x", "--noul", "a", "--detail", "medium"]
        )
        assert result.exit_code == 1
        assert "--detail must be" in strip_ansi(result.stdout)

    def test_choice_needs_options(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "x", "--choice", "dept"])
        assert result.exit_code == 1
        assert "needs options" in strip_ansi(result.stdout)

    def test_bad_questions_json(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "x", "-Q", "{not json"])
        assert result.exit_code == 1
        assert "valid JSON" in strip_ansi(result.stdout)

    def test_help_documents_both_dialects(self, runner):
        result = runner.invoke(app, ["gw", "systemone", "--help"])
        assert result.exit_code == 0
        out = strip_ansi(result.stdout)
        assert "List form" in out and "Map form" in out
        assert "options" in out and "levels" in out
