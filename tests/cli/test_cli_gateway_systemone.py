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
    EXIT_ERROR,
    EXIT_GATE_FAILED,
    _aggregate,
    _answer_row,
    _bar,
    _classify,
    _parse_gate,
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
        assert result.exit_code == EXIT_ERROR
        assert "no questions" in strip_ansi(result.stdout)

    def test_no_state_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "--noul", "a"])
        assert result.exit_code == EXIT_ERROR
        assert "no state" in strip_ansi(result.stdout)

    def test_bad_detail_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(
            app, ["gw", "systemone", "x", "--noul", "a", "--detail", "medium"]
        )
        assert result.exit_code == EXIT_ERROR
        assert "--detail must be" in strip_ansi(result.stdout)

    def test_choice_needs_options(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "x", "--choice", "dept"])
        assert result.exit_code == EXIT_ERROR
        assert "needs options" in strip_ansi(result.stdout)

    def test_bad_questions_json(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "systemone", "x", "-Q", "{not json"])
        assert result.exit_code == EXIT_ERROR
        assert "valid JSON" in strip_ansi(result.stdout)

    def test_s1_is_the_same_command(self, runner, decide, config_file):
        result = runner.invoke(app, ["gw", "s1", "x", "--noul", "a"])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["questions"] == [{"id": "a", "type": "noul"}]

    def test_s1_is_hidden_from_the_group_help(self, runner):
        result = runner.invoke(app, ["gw", "--help"])
        out = strip_ansi(result.stdout)
        assert "systemone" in out
        assert "\n│ s1 " not in out

    def test_help_documents_both_dialects(self, runner):
        result = runner.invoke(app, ["gw", "systemone", "--help"])
        assert result.exit_code == 0
        out = strip_ansi(result.stdout)
        assert "List form" in out and "Map form" in out
        assert "options" in out and "levels" in out


class TestGates:
    def test_parses_each_operator(self):
        assert _parse_gate("a>0.8") == ("a", ">", 0.8)
        assert _parse_gate("a.confidence>=0.5") == ("a.confidence", ">=", 0.5)
        assert _parse_gate("dept==billing") == ("dept", "==", "billing")
        assert _parse_gate("mood<1.5") == ("mood", "<", 1.5)

    def test_rejects_a_non_comparison(self, runner):
        with pytest.raises(Exception):
            _parse_gate("is_urgent")

    def test_passing_gates_exit_zero(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "is_urgent>0.8",
                "-g",
                "dept==billing",
                "-g",
                "dept.confidence>=0.5",
            ],
        )
        assert result.exit_code == 0, result.stdout
        out = strip_ansi(result.stdout)
        assert "PASS" in out and "FAIL" not in out

    def test_failing_gate_exits_one(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "is_urgent<0.5",
            ],
        )
        assert result.exit_code == EXIT_GATE_FAILED
        assert "FAIL" in strip_ansi(result.stdout)

    def test_label_gate_on_choice(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "dept==sales",
            ],
        )
        assert result.exit_code == EXIT_GATE_FAILED

    def test_probabilities_selector(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "dept.probabilities.billing>0.8",
            ],
        )
        assert result.exit_code == 0, result.stdout

    def test_unknown_question_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "nope>0.5",
            ],
        )
        assert result.exit_code == EXIT_ERROR
        assert "not one of the questions asked" in strip_ansi(result.stdout)

    def test_numeric_operator_on_a_label_is_an_error(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "dept>billing",
            ],
        )
        assert result.exit_code == EXIT_ERROR


class TestGateSelectorValidation:
    """A selector that cannot apply is a broken gate, not a passing one."""

    def test_choice_selector_on_a_noul_is_rejected(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "-g",
                "is_urgent.choice!=yes",
            ],
        )
        assert result.exit_code == EXIT_ERROR
        out = strip_ansi(result.stdout)
        assert "not valid for a noul question" in out

    def test_confidence_selector_on_a_noul_is_rejected(
        self, runner, decide, config_file
    ):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "is_urgent.confidence>0.5",
            ],
        )
        assert result.exit_code == EXIT_ERROR
        assert "not valid for a noul question" in strip_ansi(result.stdout)

    def test_noul_selector_on_a_choice_is_rejected(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "dept.noul>0.5",
            ],
        )
        assert result.exit_code == EXIT_ERROR
        assert "not valid for a choice question" in strip_ansi(result.stdout)

    def test_valid_selectors_still_work(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--choice",
                "dept=billing|sales",
                "--score",
                "mood=calm|angry",
                "-g",
                "is_urgent>0.5",
                "-g",
                "dept.confidence>0.5",
                "-g",
                "mood.score<2",
                "-g",
                "dept==billing",
            ],
        )
        assert result.exit_code == 0, result.stdout


class TestGatePreflight:
    def test_a_broken_gate_costs_no_request(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "approved",
                "-g",
                "approved.choice!=yes",
            ],
        )
        assert result.exit_code == EXIT_ERROR
        assert (
            decide == []
        ), "a malformed gate must be caught before the read is paid for"

    def test_unknown_question_is_caught_before_the_read(
        self, runner, decide, config_file
    ):
        result = runner.invoke(
            app, ["gw", "systemone", "x", "--noul", "a", "-g", "nope>0.5"]
        )
        assert result.exit_code == EXIT_ERROR
        assert decide == []

    def test_dry_run_still_validates_gates(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "a",
                "-g",
                "a.confidence>0.5",
                "--dry-run",
            ],
        )
        assert result.exit_code == EXIT_ERROR


class TestBodyPrecedence:
    """Documented precedence is --body < -Q < flags."""

    def test_samples_flag_beats_body(self, runner, decide, config_file):
        body = json.dumps(
            {"state": "x", "questions": [{"id": "a", "type": "noul"}], "samples": 32}
        )
        result = runner.invoke(
            app, ["gw", "systemone", "--body", body, "--samples", "1"]
        )
        assert result.exit_code == 0, result.stdout
        assert decide[0]["samples"] == 1
        assert "samples" not in (decide[0]["extra_body"] or {})

    def test_steps_flag_beats_body(self, runner, decide, config_file):
        body = json.dumps(
            {"state": "x", "questions": [{"id": "a", "type": "noul"}], "steps": 8}
        )
        result = runner.invoke(app, ["gw", "systemone", "--body", body, "--steps", "2"])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["steps"] == 2
        assert "steps" not in (decide[0]["extra_body"] or {})

    def test_body_value_survives_without_the_flag(self, runner, decide, config_file):
        body = json.dumps(
            {"state": "x", "questions": [{"id": "a", "type": "noul"}], "samples": 32}
        )
        result = runner.invoke(app, ["gw", "systemone", "--body", body])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["extra_body"]["samples"] == 32

    def test_media_positionals_beat_body_content(
        self, runner, decide, config_file, tmp_path
    ):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        body = json.dumps(
            {
                "state": "x",
                "questions": [{"id": "a", "type": "noul"}],
                "content": [{"type": "text", "text": "from the body"}],
            }
        )
        result = runner.invoke(app, ["gw", "systemone", str(image), "--body", body])
        assert result.exit_code == 0, result.stdout
        assert decide[0]["images"] == [str(image)]
        assert "content" not in (decide[0]["extra_body"] or {})


class TestRepeat:
    def test_sends_n_requests_and_reports_spread(self, runner, decide, config_file):
        result = runner.invoke(
            app, ["gw", "systemone", "x", "--noul", "a", "--repeat", "3"]
        )
        assert result.exit_code == 0, result.stdout
        assert len(decide) == 3
        out = strip_ansi(result.stdout)
        assert "3 reads" in out and "±" in out

    def test_json_repeat_is_an_array(self, runner, decide, config_file):
        result = runner.invoke(
            app, ["gw", "systemone", "x", "--noul", "a", "--repeat", "2", "--json"]
        )
        assert result.exit_code == 0, result.stdout
        assert strip_ansi(result.stdout).lstrip().startswith("[")

    def test_gates_use_the_mean(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            [
                "gw",
                "systemone",
                "x",
                "--noul",
                "is_urgent",
                "--repeat",
                "3",
                "-g",
                "is_urgent>0.9",
            ],
        )
        assert result.exit_code == 0, result.stdout

    def test_aggregate_of_one_run_is_the_answer(self):
        folded = _aggregate([Recorded([])])
        assert folded["is_urgent"]["mean"] == 0.91
        assert folded["is_urgent"]["stdev"] == 0.0
        assert folded["dept"]["label"] == "billing"
        assert folded["mood"]["mean"] == 0.9


class TestUsageFooter:
    def test_cost_and_cached_tokens_are_shown(
        self, runner, decide, config_file, monkeypatch
    ):
        monkeypatch.setattr(
            "vlmrun.cli._cli.gateway_systemone.usage_of",
            lambda run: {
                "input_tokens": 368,
                "input_tokens_details": {"cached_tokens": 256},
                "reads": 4,
                "cost": 5.2e-05,
            },
        )
        result = runner.invoke(app, ["gw", "systemone", "x", "--noul", "a"])
        assert result.exit_code == 0, result.stdout
        out = strip_ansi(result.stdout)
        assert "368 tok (256 cached)" in out
        assert "4 reads" in out
        assert "$0.000052" in out


class TestDryRun:
    def test_prints_the_body_and_sends_nothing(self, runner, decide, config_file):
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
                "--samples",
                "4",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert decide == []
        body = json.loads(strip_ansi(result.stdout))
        assert body == {
            "state": "the state",
            "model": "google/diffusiongemma-26b-a4b-it",
            "questions": {
                "is_urgent": {"type": "noul"},
                "dept": {
                    "type": "choice",
                    "criteria": {"billing": None, "sales": None},
                },
            },
            "samples": 4,
        }

    def test_bad_questions_still_fail(self, runner, decide, config_file):
        result = runner.invoke(
            app,
            ["gw", "systemone", "x", "-Q", '[{"id":"a","type":"nope"}]', "--dry-run"],
        )
        assert result.exit_code == EXIT_ERROR
