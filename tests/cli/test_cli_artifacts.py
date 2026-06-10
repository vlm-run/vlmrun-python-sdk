"""Test artifacts subcommand."""

from __future__ import annotations

from vlmrun.cli.cli import app


def test_get_artifact_by_session_id(runner, mock_client, config_file):
    """Test getting an artifact with --session-id."""
    result = runner.invoke(
        app,
        [
            "artifacts",
            "get",
            "test-object-456",
            "--session-id",
            "550e8400-e29b-41d4-a716-446655440000",
            "--raw",
        ],
    )
    assert result.exit_code == 0
    assert "bytes" in result.stdout.lower() or "Received" in result.stdout


def test_get_artifact_by_execution_id(runner, mock_client, config_file):
    """Test getting an artifact with --execution-id."""
    result = runner.invoke(
        app,
        [
            "artifacts",
            "get",
            "test-object-456",
            "--execution-id",
            "exec-001",
            "--raw",
        ],
    )
    assert result.exit_code == 0


def test_get_artifact_to_output_file(runner, mock_client, config_file, tmp_path):
    """Test saving an artifact to a specific output path."""
    out = tmp_path / "artifact.bin"
    result = runner.invoke(
        app,
        [
            "artifacts",
            "get",
            "test-object-456",
            "--session-id",
            "550e8400-e29b-41d4-a716-446655440000",
            "--raw",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert out.read_bytes() == b"mock artifact content"


def test_get_artifact_missing_ids(runner, mock_client, config_file):
    """Test that omitting both --session-id and --execution-id fails."""
    result = runner.invoke(
        app,
        ["artifacts", "get", "test-object-456"],
    )
    assert result.exit_code == 1
    assert "session-id" in result.stdout.lower() or "Error" in result.stdout


def test_get_artifact_both_ids(runner, mock_client, config_file):
    """Test that providing both --session-id and --execution-id fails."""
    result = runner.invoke(
        app,
        [
            "artifacts",
            "get",
            "test-object-456",
            "--session-id",
            "sess-001",
            "--execution-id",
            "exec-001",
        ],
    )
    assert result.exit_code == 1
    assert "only one" in result.stdout.lower() or "Error" in result.stdout


def test_artifacts_no_args(runner, mock_client, config_file):
    """Test that running `vlmrun artifacts` with no args shows help."""
    result = runner.invoke(app, ["artifacts"])
    # Typer returns exit code 0 or 2 for no_args_is_help
    assert result.exit_code in (0, 2)
