"""Tests for inline skills support in types and CLI."""

import base64
import hashlib
import io
import zipfile

import pytest

from vlmrun.client.types import AgentSkill, InlineSkillSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill_zip(skill_md_content: str = "# Test Skill\n", extra_files: dict | None = None) -> str:
    """Create a base64-encoded zip bundle with a SKILL.md and optional extra files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SKILL.md", skill_md_content)
        if extra_files:
            for name, content in extra_files.items():
                zf.writestr(name, content)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# InlineSkillSource
# ---------------------------------------------------------------------------


class TestInlineSkillSource:
    """Tests for the InlineSkillSource model."""

    def test_defaults(self):
        src = InlineSkillSource(data="abc123")
        assert src.type == "base64"
        assert src.media_type == "application/zip"
        assert src.data == "abc123"

    def test_custom_fields(self):
        src = InlineSkillSource(type="base64", media_type="application/zip", data="xyz")
        assert src.data == "xyz"

    def test_data_required(self):
        with pytest.raises(Exception):
            InlineSkillSource()

    def test_serialization_roundtrip(self):
        src = InlineSkillSource(data="roundtrip")
        dumped = src.model_dump()
        restored = InlineSkillSource(**dumped)
        assert restored.data == "roundtrip"
        assert restored.type == "base64"


# ---------------------------------------------------------------------------
# AgentSkill — referenced skills (backward compat)
# ---------------------------------------------------------------------------


class TestAgentSkillReferenced:
    """Tests for AgentSkill with type='skill_reference'."""

    def test_minimal_referenced_skill(self):
        skill = AgentSkill(skill_id="pillow")
        assert skill.type == "skill_reference"
        assert skill.skill_id == "pillow"
        assert skill.version == "latest"
        assert skill.is_inline is False

    def test_referenced_with_skill_name(self):
        skill = AgentSkill(skill_name="invoice-parsing")
        assert skill.skill_name == "invoice-parsing"
        assert skill.is_inline is False

    def test_effective_skill_id_prefers_skill_id(self):
        skill = AgentSkill(skill_id="abc", skill_name="fallback")
        assert skill.effective_skill_id == "abc"

    def test_effective_skill_id_falls_back_to_skill_name(self):
        skill = AgentSkill(skill_name="my-skill")
        assert skill.effective_skill_id == "my-skill"

    def test_effective_name_referenced(self):
        skill = AgentSkill(skill_name="my-skill", skill_id="id-123")
        assert skill.effective_name == "my-skill"

    def test_inline_data_is_none_for_referenced(self):
        skill = AgentSkill(skill_id="pillow")
        assert skill.inline_data is None

    def test_effective_description_empty_when_unset(self):
        skill = AgentSkill(skill_id="x")
        assert skill.effective_description == ""


# ---------------------------------------------------------------------------
# AgentSkill — inline skills
# ---------------------------------------------------------------------------


class TestAgentSkillInline:
    """Tests for AgentSkill with type='inline'."""

    def test_inline_with_source(self):
        bundle = _make_skill_zip()
        skill = AgentSkill(
            type="inline",
            name="csv-insights",
            description="Summarize CSV files.",
            source=InlineSkillSource(data=bundle),
        )
        assert skill.is_inline is True
        assert skill.inline_data == bundle
        assert skill.effective_name == "csv-insights"
        assert skill.effective_description == "Summarize CSV files."

    def test_inline_with_legacy_bundle(self):
        bundle = _make_skill_zip()
        skill = AgentSkill(
            type="inline",
            name="legacy-skill",
            bundle=bundle,
        )
        assert skill.is_inline is True
        assert skill.inline_data == bundle

    def test_inline_source_preferred_over_bundle(self):
        src_data = _make_skill_zip("# Source\n")
        bundle_data = _make_skill_zip("# Bundle\n")
        skill = AgentSkill(
            type="inline",
            name="both",
            source=InlineSkillSource(data=src_data),
            bundle=bundle_data,
        )
        assert skill.inline_data == src_data

    def test_effective_skill_id_inline_deterministic(self):
        bundle = _make_skill_zip()
        skill = AgentSkill(
            type="inline",
            name="test",
            source=InlineSkillSource(data=bundle),
        )
        expected_hash = hashlib.sha256(bundle.encode()).hexdigest()[:16]
        assert skill.effective_skill_id == f"inline_{expected_hash}"

    def test_effective_skill_id_inline_different_bundles_differ(self):
        b1 = _make_skill_zip("# A\n")
        b2 = _make_skill_zip("# B\n")
        s1 = AgentSkill(type="inline", name="a", source=InlineSkillSource(data=b1))
        s2 = AgentSkill(type="inline", name="b", source=InlineSkillSource(data=b2))
        assert s1.effective_skill_id != s2.effective_skill_id

    def test_effective_name_falls_back_to_inline(self):
        bundle = _make_skill_zip()
        skill = AgentSkill(type="inline", source=InlineSkillSource(data=bundle))
        assert skill.effective_name == "inline"

    def test_effective_description_empty_when_unset_inline(self):
        bundle = _make_skill_zip()
        skill = AgentSkill(type="inline", source=InlineSkillSource(data=bundle))
        assert skill.effective_description == ""

    def test_model_dump_excludes_none(self):
        bundle = _make_skill_zip()
        skill = AgentSkill(
            type="inline",
            name="dump-test",
            source=InlineSkillSource(data=bundle),
        )
        dumped = skill.model_dump(exclude_none=True)
        assert "skill_id" not in dumped
        assert "skill_name" not in dumped
        assert dumped["type"] == "inline"
        assert dumped["name"] == "dump-test"
        assert "source" in dumped
        assert dumped["source"]["data"] == bundle

    def test_inline_skill_serialization_roundtrip(self):
        bundle = _make_skill_zip("# Round-trip\n", {"schema.json": '{"type": "object"}'})
        skill = AgentSkill(
            type="inline",
            name="rt-skill",
            description="Round-trip test",
            source=InlineSkillSource(data=bundle),
        )
        dumped = skill.model_dump()
        restored = AgentSkill(**dumped)
        assert restored.is_inline is True
        assert restored.inline_data == bundle
        assert restored.effective_name == "rt-skill"
        assert restored.effective_description == "Round-trip test"


# ---------------------------------------------------------------------------
# AgentSkill — edge cases
# ---------------------------------------------------------------------------


class TestAgentSkillEdgeCases:
    """Edge cases and validation for AgentSkill."""

    def test_not_inline_without_source_or_bundle(self):
        """type='inline' but no source/bundle => is_inline is False."""
        skill = AgentSkill(type="inline")
        assert skill.is_inline is False

    def test_default_type_is_skill_reference(self):
        skill = AgentSkill(skill_id="x")
        assert skill.type == "skill_reference"

    def test_version_default(self):
        skill = AgentSkill(skill_id="x")
        assert skill.version == "latest"
