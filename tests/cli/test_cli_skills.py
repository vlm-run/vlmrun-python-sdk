"""Tests for the skills CLI subcommand."""

import json
import pytest
from typer.testing import CliRunner

from vlmrun.cli.cli import app
from vlmrun.cli._cli.skills import (
    _parse_skill_frontmatter,
    _looks_like_uuid,
    _fmt_size,
)
from tests.conftest import strip_ansi


# ---------------------------------------------------------------------------
# skills list
# ---------------------------------------------------------------------------


def test_list_skills(runner, mock_client, config_file):
    """skills list shows a table of skills."""
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "invoice-parsing" in out
    assert "receipt-parsing" in out
    assert "Skills" in out
    assert "2 skill(s)" in out


def test_list_skills_columns(runner, mock_client, config_file):
    """skills list table has expected column headers."""
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "NAME" in out
    assert "VERSION" in out
    assert "DESCRIPT" in out  # may be truncated to DESCRIPT… in narrow terminals
    assert "PUB" in out
    assert "CREATED" in out


def test_list_skills_json(runner, mock_client, config_file):
    """skills list --json emits valid JSON."""
    result = runner.invoke(app, ["skills", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["name"] == "invoice-parsing"


def test_list_skills_empty(runner, mock_client, config_file, monkeypatch):
    """skills list prints a warning when no skills are returned."""
    monkeypatch.setattr(mock_client.skills, "list", lambda **kw: [])
    monkeypatch.setattr("vlmrun.cli.cli.VLMRun", lambda **kw: mock_client)
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "No skills found" in result.stdout


def test_list_skills_limit(runner, mock_client, config_file):
    """skills list --limit is accepted without error."""
    result = runner.invoke(app, ["skills", "list", "--limit", "5"])
    assert result.exit_code == 0


def test_list_skills_grouped(runner, mock_client, config_file):
    """skills list --grouped is accepted without error."""
    result = runner.invoke(app, ["skills", "list", "--grouped"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# skills get
# ---------------------------------------------------------------------------


def test_get_skill_by_uuid(runner, mock_client, config_file):
    """skills get with a UUID-like ID returns skill details."""
    result = runner.invoke(
        app, ["skills", "get", "fe5f8791-ec9e-4c3b-a904-4ec14a9d172c"]
    )
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "Skill" in out
    assert "invoice-parsing" in out


def test_get_skill_by_name(runner, mock_client, config_file):
    """skills get with a name (non-UUID) resolves by name."""
    result = runner.invoke(app, ["skills", "get", "invoice-parsing"])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "invoice-parsing" in out


def test_get_skill_by_name_and_version(runner, mock_client, config_file):
    """skills get with --version pin is accepted."""
    result = runner.invoke(
        app, ["skills", "get", "invoice-parsing", "--version", "20260101-abcd1234"]
    )
    assert result.exit_code == 0
    assert result.exit_code == 0


def test_get_skill_json(runner, mock_client, config_file):
    """skills get --json emits valid JSON."""
    result = runner.invoke(app, ["skills", "get", "invoice-parsing", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["name"] == "invoice-parsing"
    assert "id" in data
    assert "version" in data


# ---------------------------------------------------------------------------
# skills create
# ---------------------------------------------------------------------------


def test_create_skill_no_source(runner, mock_client, config_file):
    """skills create with no source prints an error."""
    result = runner.invoke(app, ["skills", "create"])
    assert result.exit_code == 1
    assert "Provide exactly one" in result.stdout


def test_create_skill_multiple_sources(runner, mock_client, config_file):
    """skills create with more than one source prints an error."""
    result = runner.invoke(
        app,
        ["skills", "create", "--prompt", "some prompt", "--file-id", "file-abc"],
    )
    assert result.exit_code == 1
    assert "Provide exactly one" in result.stdout


def test_create_skill_prompt_and_prompt_file(runner, mock_client, config_file, tmp_path):
    """skills create rejects --prompt and --prompt-file together."""
    pf = tmp_path / "prompt.txt"
    pf.write_text("Extract invoice data")
    result = runner.invoke(
        app,
        ["skills", "create", "--prompt", "direct text", "--prompt-file", str(pf)],
    )
    assert result.exit_code == 1
    assert "--prompt or --prompt-file" in result.stdout


def test_create_skill_with_prompt(runner, mock_client, config_file):
    """skills create --prompt succeeds and shows skill panel."""
    result = runner.invoke(
        app, ["skills", "create", "--prompt", "Extract invoice fields"]
    )
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "Skill" in out


def test_create_skill_with_prompt_file(runner, mock_client, config_file, tmp_path):
    """skills create --prompt-file reads prompt from file."""
    pf = tmp_path / "prompt.txt"
    pf.write_text("Extract receipt data")
    result = runner.invoke(app, ["skills", "create", "--prompt-file", str(pf)])
    assert result.exit_code == 0


def test_create_skill_with_file_id(runner, mock_client, config_file):
    """skills create --file-id + --name succeeds."""
    result = runner.invoke(
        app,
        ["skills", "create", "--file-id", "file-abc123", "--name", "my-skill"],
    )
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "Skill" in out


def test_create_skill_json(runner, mock_client, config_file):
    """skills create --json emits valid JSON."""
    result = runner.invoke(
        app,
        ["skills", "create", "--prompt", "Extract data", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "id" in data
    assert "name" in data


# ---------------------------------------------------------------------------
# skills upload
# ---------------------------------------------------------------------------


def test_upload_skill(runner, mock_client, config_file, tmp_path):
    """skills upload zips the directory, uploads, and creates the skill."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill\n---\n# My Skill\n"
    )
    (skill_dir / "tools.md").write_text("# Tools\n")

    result = runner.invoke(app, ["skills", "upload", str(skill_dir)])
    assert result.exit_code == 0
    out = strip_ansi(result.stdout)
    assert "Zipped" in out
    assert "Uploaded" in out
    assert "Created skill" in out
    assert "Skill" in out


def test_upload_skill_missing_skillmd(runner, mock_client, config_file, tmp_path):
    """skills upload fails when SKILL.md is missing."""
    skill_dir = tmp_path / "no-skillmd"
    skill_dir.mkdir()
    (skill_dir / "tools.md").write_text("# Tools\n")

    result = runner.invoke(app, ["skills", "upload", str(skill_dir)])
    assert result.exit_code == 1
    assert "SKILL.md not found" in result.stdout


def test_upload_skill_no_name_in_frontmatter(runner, mock_client, config_file, tmp_path):
    """skills upload fails when SKILL.md has no name and --name is not given."""
    skill_dir = tmp_path / "unnamed-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# No frontmatter here\n")

    result = runner.invoke(app, ["skills", "upload", str(skill_dir)])
    assert result.exit_code == 1
    assert "Could not determine skill name" in result.stdout


def test_upload_skill_name_override(runner, mock_client, config_file, tmp_path):
    """--name overrides the name in SKILL.md frontmatter."""
    skill_dir = tmp_path / "base-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: original-name\ndescription: desc\n---\n"
    )

    result = runner.invoke(
        app, ["skills", "upload", str(skill_dir), "--name", "override-name"]
    )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Helper unit tests (no CLI runner needed)
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text("---\nname: my-skill\ndescription: Does stuff\n---\n# Body\n")
        name, desc = _parse_skill_frontmatter(f)
        assert name == "my-skill"
        assert desc == "Does stuff"

    def test_quoted_values(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text('---\nname: "my-skill"\ndescription: \'Quoted desc\'\n---\n')
        name, desc = _parse_skill_frontmatter(f)
        assert name == "my-skill"
        assert desc == "Quoted desc"

    def test_no_frontmatter(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text("# Just a heading\nSome content.\n")
        name, desc = _parse_skill_frontmatter(f)
        assert name is None
        assert desc is None

    def test_missing_description(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text("---\nname: my-skill\n---\n")
        name, desc = _parse_skill_frontmatter(f)
        assert name == "my-skill"
        assert desc is None


class TestLooksLikeUuid:
    def test_standard_uuid(self):
        assert _looks_like_uuid("fe5f8791-ec9e-4c3b-a904-4ec14a9d172c") is True

    def test_uuid_no_dashes(self):
        assert _looks_like_uuid("fe5f8791ec9e4c3ba9044ec14a9d172c") is True

    def test_short_name(self):
        assert _looks_like_uuid("invoice-parsing") is False

    def test_empty(self):
        assert _looks_like_uuid("") is False

    def test_version_string(self):
        assert _looks_like_uuid("20260101-abcd1234") is False


class TestFmtSize:
    def test_bytes(self):
        assert _fmt_size(512) == "512B"

    def test_kilobytes(self):
        assert _fmt_size(2048) == "2KB"

    def test_megabytes(self):
        assert _fmt_size(2 * 1024 * 1024) == "2MB"

    def test_fractional(self):
        result = _fmt_size(1536)  # 1.5 KB
        assert "1.5KB" in result
