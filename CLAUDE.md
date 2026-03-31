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

---

## Code Quality & Developer Experience Guidelines

The following guidelines ensure high-quality, maintainable, and idiomatic Python code in this SDK. All contributors (human and AI) should adhere to these principles.

### Object-Oriented Design

- **Prefer class-based implementations** for resources, services, and API abstractions. This SDK already follows a resource-based class pattern (e.g., `Agent`, `Predictions`, `Files`) — extend it consistently.
- **Use `@staticmethod`** for utility methods that don't depend on instance state (e.g., `ImagePredictions._handle_images_or_urls`).
- **Use `@classmethod`** for alternative constructors and model validators (e.g., `SkillInfo._normalize_version`).
- **Use `@property` and `@cached_property`** for computed attributes (e.g., `VLMRun.version`, `VLMRun.requestor`, `Agent.completions`).
- **Use mixins** for shared cross-cutting behavior (e.g., `SchemaCastMixin` for response type casting).
- When adding new functionality, always ask: _"Should this be a class (stateful, composable, testable) or a standalone function?"_ Default to class-based when the feature manages state, holds configuration, or represents an API resource.

### Context Managers

- **Use context managers** (`with` / `async with`) when managing resources that require setup/teardown (e.g., HTTP sessions, file handles, temporary directories).
- For new client or session patterns, prefer implementing `__enter__`/`__exit__` or `__aenter__`/`__aexit__` to ensure proper cleanup.
- Example: if adding connection pooling or streaming responses, wrap them in context managers.

### Use Advanced Libraries

- **Use the OpenAI SDK** for chat completions instead of raw HTTP requests — this pattern is already established via `Agent.completions` and `Agent.async_completions`.
- **Use `tenacity`** for retry logic instead of hand-rolled loops (already used in `APIRequestor`).
- **Use `pydantic`** for all data models and validation — never use raw dicts for API responses.
- **Use `cachetools`** for caching (already used for schema caching in `predictions.py`).
- When a well-maintained, widely-adopted library exists for a task (HTTP, retries, validation, caching, etc.), prefer it over a custom implementation.

### Documentation & Spec Maintenance

- When implementation deviates from or extends the written API spec, **update the corresponding documentation** (README.md, docstrings, type annotations).
- All public methods must have **docstrings** with Args, Returns, and Raises sections (follow the existing Google-style docstring convention in this repo).
- Keep `vlmrun/client/types.py` in sync with the API — if a new field is added to an API response, add it to the corresponding Pydantic model.

### Functional vs Class-Oriented

- Before implementing significant new features, consider whether a **functional or class-oriented approach** is cleaner:
  - Class-oriented: API resources, stateful clients, configuration objects, anything that benefits from inheritance or composition.
  - Functional: Pure transformations, one-off utilities, simple data processing pipelines.
- The existing `FilePredictions(route)` factory pattern is a good example of mixing both — a function that returns a class.

### Testing Requirements

- **Always write pytest tests** for new features and bug fixes. Do not implement significant new functionality without corresponding tests.
- **Do not use `unittest.mock.MagicMock`**. Instead, use:
  - Concrete mock classes (as in `tests/conftest.py`'s `MockVLMRun` pattern) that mirror the real API surface.
  - `monkeypatch` for environment variable and attribute patching.
  - `unittest.mock.patch` only for targeted patching of specific methods (e.g., `APIRequestor.request`).
- Test files should mirror the client structure: `test_predictions.py`, `test_agent.py`, `test_client.py`, etc.
- Tests requiring a live API should be guarded with `@pytest.mark.skipif` checks for `VLMRUN_API_KEY`.

### Modern Python (3.10+)

- **Use `X | None` instead of `Optional[X]`** for type hints (PEP 604). Example: `api_key: str | None = None` instead of `api_key: Optional[str] = None`.
- **Use `list[T]` instead of `List[T]`**, `dict[K, V]` instead of `Dict[K, V]`**, `tuple[T, ...]` instead of `Tuple[T, ...]` (PEP 585 built-in generics).
- Use **`from __future__ import annotations`** at the top of files for forward reference support (already used in some files).
- Prefer **`match` / `case`** (structural pattern matching, Python 3.10+) over long if/elif chains where appropriate.
- Use **f-strings** for all string formatting (no `%` or `.format()`).
- Use **`|` for union types** in all new code: `str | int` not `Union[str, int]`.

### Error Handling

- Use the existing exception hierarchy in `vlmrun/client/exceptions.py`. Do not raise generic `Exception` or `ValueError` for API-related errors.
- Every custom exception should include a helpful `suggestion` field guiding the user toward resolution.
- Preserve the pattern: `VLMRunError` → `APIError` / `ClientError` → specific error types.

### SDK Patterns to Follow

- **Resource initialization**: Resources take a `VLMRunProtocol` client instance and create their own `APIRequestor`.
- **Response typing**: Always return typed Pydantic models, never raw dicts.
- **Deprecation**: Use `warnings.warn(..., DeprecationWarning)` for deprecated parameters; provide migration guidance in the message.
- **Optional dependencies**: Gate optional imports behind try/except and raise `DependencyError` with install instructions (see `Agent.completions` pattern).
