# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM Run Python SDK - Official Python SDK for the VLM Run API platform. Provides structured data extraction from images, documents, videos, and audio using vision-language models.

## Development Commands

```bash
# Install for development
pip install -e ".[test]"

# Run all tests
pytest -sv tests

# Run a single test file
pytest -sv tests/test_client.py

# Run a specific test
pytest -sv tests/test_client.py::test_client_init

# Lint and format code
make lint
# Or directly: pre-commit run --all-files

# Build package
make dist
```

## Architecture

### Client Structure (`vlmrun/client/`)

The SDK uses a resource-based pattern where `VLMRun` is the main client that exposes specialized resources:

- **VLMRun** (`client.py`) - Main entry point, initializes all resources on construction after validating the API key via `/health` endpoint
- **APIRequestor** (`base_requestor.py`) - Handles HTTP requests with retry logic (tenacity), timeout handling, and error mapping to custom exception types
- **Resource Classes** - Each API domain has its own resource class:
  - `ImagePredictions`, `DocumentPredictions`, `VideoPredictions`, `AudioPredictions` (`predictions.py`)
  - `Agent` (`agent.py`) - Also provides OpenAI-compatible completions interface via `client.agent.completions`
  - `Files`, `Hub`, `Models`, `Datasets`, `Finetuning`, `Feedback`, `Executions`, `Artifacts`

### Predictions Pattern

`FilePredictions(route)` is a factory function that creates prediction classes for document/audio/video. These inherit from `SchemaCastMixin` which handles automatic type casting of responses to Pydantic models based on domain schemas.

Key methods: `generate()` (use domain), `execute()` (use named model), `schema()` (auto-generate schema).

### CLI Structure (`vlmrun/cli/`)

Built with Typer. Entry point is `vlmrun/cli/cli.py`. Subcommands are in `vlmrun/cli/_cli/`:
- `chat.py` - Interactive chat with Orion agent
- `config.py` - API key and base URL configuration
- `generate.py`, `files.py`, `predictions.py`, `hub.py`, `models.py`

### Type System (`vlmrun/client/types.py`)

All API responses are typed with Pydantic models. Key types:
- `PredictionResponse` - Standard response for all predictions
- `GenerationConfig` - Configuration for generation (prompt, response_model, json_schema, temperature, etc.)
- `MarkdownDocument/MarkdownPage/MarkdownTable` - Structured document extraction results

### Common Utilities (`vlmrun/common/`)

- `image.py` - Image encoding, EXIF handling
- `video.py` - Video frame extraction
- `pdf.py` - PDF processing
- `utils.py` - Helper functions like `remote_image()`

## Configuration

Environment variables:
- `VLMRUN_API_KEY` - API key (required)
- `VLMRUN_BASE_URL` - Override default API URL (default: `https://api.vlm.run/v1`)
- `VLMRUN_CACHE_DIR` - Override cache directory (default: `~/.vlmrun/cache`)

## Optional Dependencies

Install based on needed functionality:
- `pip install vlmrun[cli]` - CLI with Typer/Rich
- `pip install vlmrun[video]` - Video processing (numpy)
- `pip install vlmrun[doc]` - PDF processing (pypdfium2)
- `pip install vlmrun[openai]` - OpenAI SDK for chat completions API
- `pip install vlmrun[all]` - All optional dependencies

## Testing

Tests use pytest with fixtures in `tests/conftest.py`. Tests require `VLMRUN_API_KEY` environment variable. Test files mirror the client structure (e.g., `test_predictions.py`, `test_agent.py`).

## Code Style

Uses ruff for linting and black for formatting. Pre-commit hooks enforce style on commit. Configuration is in `pyproject.toml` and `.pre-commit-config.yaml`.
