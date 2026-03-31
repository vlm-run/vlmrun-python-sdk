"""Tests for inline skills support in types, CLI, and library utilities."""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Optional

import pytest

from vlmrun.client.types import AgentSkill, InlineSkillSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill_zip(
    skill_md_content: str = "# Test Skill\n", extra_files: Optional[dict] = None
) -> str:
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
        bundle = _make_skill_zip(
            "# Round-trip\n", {"schema.json": '{"type": "object"}'}
        )
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


# ---------------------------------------------------------------------------
# Library utilities — parse_skill_frontmatter
# ---------------------------------------------------------------------------

_FRONTMATTER_SKILL_MD = """\
---
name: my-skill
description: Does cool things
---
# My Skill

Body content.
"""


class TestParseSkillFrontmatter:
    """Tests for parse_skill_frontmatter."""

    def test_parses_name_and_description(self, tmp_path: Path):
        from vlmrun.client.skills import parse_skill_frontmatter

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(_FRONTMATTER_SKILL_MD)
        name, desc = parse_skill_frontmatter(skill_md)
        assert name == "my-skill"
        assert desc == "Does cool things"

    def test_missing_frontmatter_returns_none(self, tmp_path: Path):
        from vlmrun.client.skills import parse_skill_frontmatter

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("# No Frontmatter\n\nJust body.\n")
        name, desc = parse_skill_frontmatter(skill_md)
        assert name is None
        assert desc is None

    def test_partial_frontmatter(self, tmp_path: Path):
        from vlmrun.client.skills import parse_skill_frontmatter

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: only-name\n---\n# Body\n")
        name, desc = parse_skill_frontmatter(skill_md)
        assert name == "only-name"
        assert desc is None


# ---------------------------------------------------------------------------
# Library utilities — bundle_from_directory
# ---------------------------------------------------------------------------


class TestBundleFromDirectory:
    """Tests for bundle_from_directory."""

    def test_produces_valid_base64_zip(self, tmp_path: Path):
        from vlmrun.client.skills import bundle_from_directory

        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        bundle = bundle_from_directory(tmp_path)

        zip_bytes = base64.b64decode(bundle)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert "SKILL.md" in zf.namelist()

    def test_includes_nested_files(self, tmp_path: Path):
        from vlmrun.client.skills import bundle_from_directory

        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "logo.png").write_bytes(b"\x89PNG")

        bundle = bundle_from_directory(tmp_path)
        zip_bytes = base64.b64decode(bundle)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "SKILL.md" in names
            assert "assets/logo.png" in names

    def test_deterministic(self, tmp_path: Path):
        from vlmrun.client.skills import bundle_from_directory

        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        assert bundle_from_directory(tmp_path) == bundle_from_directory(tmp_path)


# ---------------------------------------------------------------------------
# Library utilities — hash_directory
# ---------------------------------------------------------------------------


class TestHashDirectory:
    """Tests for hash_directory."""

    def test_deterministic(self, tmp_path: Path):
        from vlmrun.client.skills import hash_directory

        (tmp_path / "SKILL.md").write_text("content")
        assert hash_directory(tmp_path) == hash_directory(tmp_path)

    def test_different_content_different_hash(self, tmp_path: Path):
        from vlmrun.client.skills import hash_directory

        d1 = tmp_path / "a"
        d1.mkdir()
        (d1 / "SKILL.md").write_text("aaa")

        d2 = tmp_path / "b"
        d2.mkdir()
        (d2 / "SKILL.md").write_text("bbb")

        assert hash_directory(d1) != hash_directory(d2)

    def test_returns_hex_string(self, tmp_path: Path):
        from vlmrun.client.skills import hash_directory

        (tmp_path / "f.txt").write_text("x")
        h = hash_directory(tmp_path)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# AgentSkill.from_directory classmethod
# ---------------------------------------------------------------------------


class TestAgentSkillFromDirectory:
    """Tests for AgentSkill.from_directory() classmethod."""

    def test_returns_inline_agent_skill(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        skill = AgentSkill.from_directory(tmp_path)
        assert isinstance(skill, AgentSkill)
        assert skill.type == "inline"
        assert skill.is_inline is True
        assert skill.name == "my-skill"
        assert skill.description == "Does cool things"
        assert skill.source is not None
        assert skill.source.data  # non-empty bundle

    def test_uses_directory_name_as_fallback(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text("---\ndescription: no name\n---\n# Body\n")
        skill = AgentSkill.from_directory(tmp_path)
        assert skill.name == tmp_path.name

    def test_missing_skill_md_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="SKILL.md"):
            AgentSkill.from_directory(tmp_path)

    def test_bundle_is_valid_zip(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        (tmp_path / "helper.py").write_text("print('hi')")

        skill = AgentSkill.from_directory(tmp_path)
        zip_bytes = base64.b64decode(skill.source.data)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "SKILL.md" in names
            assert "helper.py" in names

    def test_name_override(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        skill = AgentSkill.from_directory(tmp_path, name="custom-name")
        assert skill.name == "custom-name"

    def test_description_override(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        skill = AgentSkill.from_directory(tmp_path, description="Custom desc")
        assert skill.description == "Custom desc"

    def test_accepts_string_path(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        skill = AgentSkill.from_directory(str(tmp_path))
        assert isinstance(skill, AgentSkill)
        assert skill.type == "inline"

    def test_no_frontmatter_uses_directory_name(self, tmp_path: Path):
        (tmp_path / "SKILL.md").write_text("# No Frontmatter\n\nJust body.\n")
        skill = AgentSkill.from_directory(tmp_path)
        assert skill.name == tmp_path.name
        assert skill.description == ""


# ---------------------------------------------------------------------------
# Library utilities — inline_skill_from_directory (delegates to from_directory)
# ---------------------------------------------------------------------------


class TestInlineSkillFromDirectory:
    """Tests for inline_skill_from_directory (thin wrapper)."""

    def test_returns_inline_agent_skill(self, tmp_path: Path):
        from vlmrun.client.skills import inline_skill_from_directory

        (tmp_path / "SKILL.md").write_text(_FRONTMATTER_SKILL_MD)
        skill = inline_skill_from_directory(tmp_path)
        assert isinstance(skill, AgentSkill)
        assert skill.type == "inline"
        assert skill.is_inline is True
        assert skill.name == "my-skill"
        assert skill.description == "Does cool things"
        assert skill.source is not None
        assert skill.source.data  # non-empty bundle

    def test_missing_skill_md_raises(self, tmp_path: Path):
        from vlmrun.client.skills import inline_skill_from_directory

        with pytest.raises(FileNotFoundError, match="SKILL.md"):
            inline_skill_from_directory(tmp_path)
