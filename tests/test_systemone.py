"""Tests for the System One resource and its question dialects.

The question normalizer and media assembly are pure, so they run everywhere.
The tests that exercise a request end to end need ``typesafe-sdk`` (the
``typesafe`` extra) and drive it through an ``httpx2.MockTransport``, so no
network or API key is involved.
"""

from __future__ import annotations

import base64
import json

import pytest

from vlmrun.client.exceptions import InputError
from vlmrun.client.systemone import (
    SYSTEMONE_MODEL,
    SystemOne,
    build_content,
    normalize_questions,
    typesafe_base_url,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n"


class TestTypesafeBaseUrl:
    def test_strips_v1_and_appends_prefix(self):
        assert (
            typesafe_base_url("https://gateway.vlm.run/v1")
            == "https://gateway.vlm.run/typesafe"
        )

    def test_local_gateway(self):
        assert (
            typesafe_base_url("http://localhost:8021/v1/")
            == "http://localhost:8021/typesafe"
        )

    def test_url_without_v1(self):
        assert (
            typesafe_base_url("https://gateway.vlm.run")
            == "https://gateway.vlm.run/typesafe"
        )

    def test_default(self):
        assert typesafe_base_url().endswith("/typesafe")


class TestNormalizeQuestions:
    def test_list_form_uses_id_as_key(self):
        questions = normalize_questions(
            [
                {"id": "is_urgent", "type": "noul", "instructions": "Time-sensitive?"},
                {"id": "dept", "type": "choice", "options": ["billing", "technical"]},
            ]
        )
        assert list(questions) == ["is_urgent", "dept"]
        assert questions["is_urgent"] == {
            "type": "noul",
            "instructions": "Time-sensitive?",
        }

    def test_wrapper_is_unwrapped(self):
        bare = normalize_questions([{"id": "a", "type": "noul"}])
        wrapped = normalize_questions({"questions": [{"id": "a", "type": "noul"}]})
        assert bare == wrapped

    def test_map_form_passes_through(self):
        questions = normalize_questions({"a": {"type": "noul"}})
        assert questions == {"a": {"type": "noul"}}

    def test_map_form_rejects_mismatched_id(self):
        with pytest.raises(InputError, match="but the key is"):
            normalize_questions({"a": {"id": "b", "type": "noul"}})

    def test_options_become_criteria(self):
        questions = normalize_questions(
            [
                {
                    "id": "dept",
                    "type": "choice",
                    "options": [{"name": "billing", "description": "Charges"}, "sales"],
                }
            ]
        )
        assert questions["dept"]["criteria"] == {"billing": "Charges", "sales": None}

    def test_options_as_mapping(self):
        questions = normalize_questions(
            [{"id": "dept", "type": "choice", "options": {"a": "A", "b": None}}]
        )
        assert questions["dept"]["criteria"] == {"a": "A", "b": None}

    def test_levels_become_criteria(self):
        questions = normalize_questions(
            [{"id": "mood", "type": "score", "levels": ["calm", "angry"]}]
        )
        assert questions["mood"]["criteria"] == ["calm", "angry"]

    def test_noul_yes_no_aliases(self):
        questions = normalize_questions(
            [{"id": "a", "type": "noul", "criteria": {"yes": "Y", "no": "N"}}]
        )
        assert questions["a"]["criteria"] == {"true": "Y", "false": "N"}

    def test_missing_id_in_list(self):
        with pytest.raises(InputError, match='needs an "id"'):
            normalize_questions([{"type": "noul"}])

    def test_duplicate_id(self):
        with pytest.raises(InputError, match="duplicate question id"):
            normalize_questions(
                [{"id": "a", "type": "noul"}, {"id": "a", "type": "noul"}]
            )

    def test_unknown_type(self):
        with pytest.raises(InputError, match="type must be one of"):
            normalize_questions([{"id": "a", "type": "choise", "options": ["x", "y"]}])

    def test_wrong_alias_for_type(self):
        with pytest.raises(InputError, match='A score question uses "levels"'):
            normalize_questions([{"id": "a", "type": "score", "options": ["x", "y"]}])

    def test_choice_needs_two_options(self):
        with pytest.raises(InputError, match="2-128 options"):
            normalize_questions([{"id": "a", "type": "choice", "options": ["x"]}])

    def test_score_level_bounds(self):
        with pytest.raises(InputError, match="2-10 levels"):
            normalize_questions([{"id": "a", "type": "score", "levels": ["x"] * 11}])

    def test_options_and_criteria_conflict(self):
        with pytest.raises(InputError, match="not both"):
            normalize_questions(
                [
                    {
                        "id": "a",
                        "type": "choice",
                        "options": ["x", "y"],
                        "criteria": {"x": None, "y": None},
                    }
                ]
            )

    def test_empty_spec(self):
        with pytest.raises(InputError, match="no questions"):
            normalize_questions([])


class TestBuildContent:
    def test_no_media_returns_none(self):
        assert build_content() is None

    def test_local_image_is_inlined(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        part = build_content(images=[image], detail="high")[0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")
        assert part["image_url"]["detail"] == "high"

    def test_image_is_validated_by_magic_bytes_not_extension(self, tmp_path):
        bogus = tmp_path / "a.png"
        bogus.write_bytes(b"not an image at all")
        with pytest.raises(InputError, match="is not a recognized image"):
            build_content(images=[bogus])

    def test_unsupported_image_type_is_rejected(self, tmp_path):
        bitmap = tmp_path / "a.bmp"
        bitmap.write_bytes(b"BM" + b"\x00" * 32)
        with pytest.raises(InputError, match="is not supported"):
            build_content(images=[bitmap])

    def test_local_pdf_is_inlined_with_filename(self, tmp_path):
        pdf = tmp_path / "invoice.pdf"
        pdf.write_bytes(PDF_BYTES)
        part = build_content(document=pdf)[0]
        assert part["type"] == "file"
        assert part["file"]["filename"] == "invoice.pdf"
        assert part["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_remote_pdf_rides_as_url(self):
        part = build_content(document="https://example.com/a.pdf")[0]
        assert part["file"]["file_data"] == "https://example.com/a.pdf"
        assert "filename" not in part["file"]

    def test_non_pdf_document_is_rejected(self, tmp_path):
        doc = tmp_path / "a.pdf"
        doc.write_bytes(b"still not a pdf")
        with pytest.raises(InputError, match="is not a PDF"):
            build_content(document=doc)

    def test_too_many_images(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        with pytest.raises(InputError, match="the limit is 8"):
            build_content(images=[image] * 9)

    def test_document_leads_then_images_then_text(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(PDF_BYTES)
        parts = build_content(images=[image], document=pdf, text="extra")
        assert [part["type"] for part in parts] == ["file", "image_url", "text"]


typesafe_sdk = pytest.importorskip("typesafe_sdk", reason="requires vlmrun[typesafe]")
import httpx2  # noqa: E402 - available with the typesafe extra


class FakeClient:
    """The slice of the VLM Run client a resource uses."""

    def __init__(self, api_key: str = "test-key") -> None:
        self.api_key = api_key
        self.base_url = "https://api.vlm.run/v1"
        self.timeout = 120.0
        self.max_retries = 5


ANSWERS = {
    "model": SYSTEMONE_MODEL,
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.91},
        "dept": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "sales": 0.1},
            "confidence": 0.7,
        },
        "mood": {
            "type": "score",
            "score": 0.9,
            "legend": {"0": "calm", "1": "angry"},
            "probabilities": {"0": 0.1, "1": 0.9},
            "confidence": 0.5,
        },
    },
    "usage": {"input_tokens": 149, "output_tokens": 0},
}


def resource(handler, **kwargs) -> SystemOne:
    """A SystemOne whose TypeSafe client talks to ``handler`` instead of the network."""
    system_one = SystemOne(FakeClient(), gateway_url="http://gw.test/v1", **kwargs)
    system_one.__dict__["client"] = typesafe_sdk.TypeSafeClient(
        api_key="test-key",
        base_url=system_one.base_url,
        model=system_one.model,
        transport=httpx2.MockTransport(handler),
    )
    return system_one


def capture(sent: list, payload=ANSWERS, status: int = 200):
    """A MockTransport handler that records each request body."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(
            {
                "url": str(request.url),
                "body": json.loads(request.content) if request.content else None,
            }
        )
        return httpx2.Response(
            status,
            json=payload,
            headers={"x-typesafe-request-id": "req_test"},
        )

    return handler


class TestDecide:
    def test_posts_the_wire_body(self):
        sent: list = []
        response = resource(capture(sent)).decide(
            "the state",
            [
                {"id": "is_urgent", "type": "noul"},
                {"id": "dept", "type": "choice", "options": ["billing", "sales"]},
            ],
        )
        assert sent[0]["url"] == "http://gw.test/typesafe/v1/systemone"
        assert sent[0]["body"] == {
            "state": "the state",
            "model": SYSTEMONE_MODEL,
            "questions": {
                "is_urgent": {"type": "noul"},
                "dept": {
                    "type": "choice",
                    "criteria": {"billing": None, "sales": None},
                },
            },
        }
        assert response.nouls["is_urgent"].noul == 0.91
        assert response.choices["dept"].choice == "billing"
        assert response.scores["mood"].score == 0.9
        assert response.usage.input_tokens == 149

    def test_no_content_key_without_media(self):
        sent: list = []
        resource(capture(sent)).decide("x", [{"id": "a", "type": "noul"}])
        assert "content" not in sent[0]["body"]

    def test_media_and_extensions_ride_extra_body(self, tmp_path):
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        sent: list = []
        resource(capture(sent)).decide(
            "x",
            [{"id": "a", "type": "noul"}],
            images=[image],
            detail="high",
            steps=2,
            samples=4,
        )
        body = sent[0]["body"]
        assert body["steps"] == 2 and body["samples"] == 4
        assert body["content"][0]["type"] == "image_url"
        assert body["content"][0]["image_url"]["detail"] == "high"

    def test_extra_body_wins(self):
        sent: list = []
        resource(capture(sent)).decide(
            "x", [{"id": "a", "type": "noul"}], samples=4, extra_body={"samples": 8}
        )
        assert sent[0]["body"]["samples"] == 8

    def test_model_override(self):
        sent: list = []
        resource(capture(sent)).decide(
            "x", [{"id": "a", "type": "noul"}], model="jev-latest"
        )
        assert sent[0]["body"]["model"] == "jev-latest"

    def test_bad_questions_never_reach_the_wire(self):
        sent: list = []
        with pytest.raises(InputError):
            resource(capture(sent)).decide("x", [{"id": "a", "type": "nope"}])
        assert sent == []

    def test_unknown_model_raises_not_found(self):
        detail = {"detail": {"error_type": "not_found_error", "message": "nope"}}
        with pytest.raises(typesafe_sdk.TypeSafeNotFoundError):
            resource(capture([], detail, status=404)).decide(
                "x", [{"id": "a", "type": "noul"}]
            )

    def test_validation_error_carries_the_detail_list(self):
        detail = {
            "detail": [
                {"type": "value_error", "loc": ["body", "questions"], "msg": "bad"}
            ]
        }
        with pytest.raises(typesafe_sdk.TypeSafeUnprocessableEntityError) as excinfo:
            resource(capture([], detail, status=422), model=SYSTEMONE_MODEL).decide(
                "x", [{"id": "a", "type": "noul"}]
            )
        assert excinfo.value.body["detail"][0]["msg"] == "bad"

    def test_models_listing(self):
        payload = {
            "models": [
                {
                    "name": SYSTEMONE_MODEL,
                    "description": "DiffusionGemma",
                    "release_date": "2026-09-01",
                }
            ]
        }
        sent: list = []
        models = resource(capture(sent, payload)).models()
        assert sent[0]["url"] == "http://gw.test/typesafe/v1/models"
        assert models.models[0].name == SYSTEMONE_MODEL


class TestResourceWiring:
    def test_gateway_exposes_systemone(self, monkeypatch):
        monkeypatch.setenv("VLMRUN_GATEWAY_URL", "http://gw.test/v1")
        monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
        from vlmrun.client.gateway import Gateway

        gateway = Gateway(FakeClient())
        assert gateway.systemone.base_url == "http://gw.test/typesafe"
        assert gateway.systemone.model == SYSTEMONE_MODEL

    def test_typesafe_base_url_env_wins(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_BASE_URL", "http://elsewhere/typesafe")
        assert (
            SystemOne(FakeClient(), gateway_url="http://gw.test/v1").base_url
            == "http://elsewhere/typesafe"
        )

    def test_missing_api_key_falls_back(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        client = FakeClient(api_key=None)
        assert SystemOne(client)._api_key() == "EMPTY"

    def test_is_a_context_manager(self):
        sent: list = []
        with resource(capture(sent)) as system_one:
            system_one.decide("x", [{"id": "a", "type": "noul"}])
        assert "client" not in system_one.__dict__
