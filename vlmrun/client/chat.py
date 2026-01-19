"""Reusable chat functions for VLM Run API.

This module provides `chat` and `chat_stream` functions that can be used
by the CLI, MCP server, or any other client code.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from vlmrun.client import VLMRun

from vlmrun.client.types import FileResponse
from vlmrun.constants import SUPPORTED_INPUT_FILETYPES


# Available models
AVAILABLE_MODELS = [
    "vlmrun-orion-1:fast",
    "vlmrun-orion-1:auto",
    "vlmrun-orion-1:pro",
]

DEFAULT_MODEL = "vlmrun-orion-1:auto"


@dataclass
class ChatResponse:
    """Response from a non-streaming chat completion."""

    content: str
    session_id: Optional[str]
    model: str
    latency_s: float
    usage: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    artifact_refs: List[str] = field(default_factory=list)


@dataclass
class ChatStreamChunk:
    """A chunk from a streaming chat completion."""

    content: str
    session_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    is_final: bool = False


class ChatError(Exception):
    """Error raised during chat operations."""

    pass


def extract_artifact_refs(response_content: str) -> List[str]:
    """Extract artifact reference IDs from response content.

    Looks for patterns like img_XXXXXX, aud_XXXXXX, vid_XXXXXX, doc_XXXXXX,
    recon_XXXXXX, arr_XXXXXX, url_XXXXXX in the response text.

    Args:
        response_content: The response text to search for artifact references.

    Returns:
        A sorted list of unique artifact reference IDs found in the content.
    """
    patterns = [
        r"\bimg_\w{6}\b",  # ImageRef
        r"\baud_\w{6}\b",  # AudioRef
        r"\bvid_\w{6}\b",  # VideoRef
        r"\bdoc_\w{6}\b",  # DocumentRef
        r"\brecon_\w{6}\b",  # ReconRef
        r"\barr_\w{6}\b",  # ArrayRef
        r"\burl_\w{6}\b",  # UrlRef
    ]

    refs: Set[str] = set()
    for pattern in patterns:
        matches = re.findall(pattern, response_content)
        refs.update(matches)

    return sorted(list(refs))


def _validate_model(model: str) -> None:
    """Validate that the model is supported.

    Args:
        model: The model name to validate.

    Raises:
        ChatError: If the model is not supported.
    """
    if model not in AVAILABLE_MODELS:
        raise ChatError(
            f"Invalid model '{model}'. Available models: {', '.join(AVAILABLE_MODELS)}"
        )


def _validate_inputs(inputs: Optional[List[Path]]) -> None:
    """Validate that input files have supported file types.

    Args:
        inputs: List of input file paths to validate.

    Raises:
        ChatError: If any file has an unsupported file type.
    """
    if not inputs:
        return

    for file_path in inputs:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_INPUT_FILETYPES:
            raise ChatError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: {', '.join(SUPPORTED_INPUT_FILETYPES)}"
            )


def _upload_files(
    client: "VLMRun", files: List[Path], max_workers: int = 4
) -> List[FileResponse]:
    """Upload files concurrently and return their file responses.

    Args:
        client: VLMRun client instance.
        files: List of file paths to upload.
        max_workers: Maximum number of concurrent uploads.

    Returns:
        List of FileResponse objects for the uploaded files.

    Raises:
        ChatError: If any file upload fails.
    """
    file_responses: List[FileResponse] = []

    with ThreadPoolExecutor(max_workers=min(len(files), max_workers)) as executor:
        futures = {
            executor.submit(client.files.upload, file=f, purpose="assistants"): f
            for f in files
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                file_response = future.result()
                file_responses.append(file_response)
            except Exception as e:
                raise ChatError(f"Error uploading {file_path.name}: {e}") from e

    return file_responses


def _build_messages(
    prompt: str, file_responses: Optional[List[FileResponse]] = None
) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with optional file attachments.

    Args:
        prompt: The text prompt for the chat.
        file_responses: Optional list of FileResponse objects for file attachments.

    Returns:
        List of message dictionaries in OpenAI format.
    """
    content: List[Dict[str, Any]] = [
        {"type": "input_file", "file_id": file_response.id}
        for file_response in file_responses or []
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_usage_data(usage: Any) -> Optional[Dict[str, Any]]:
    """Extract usage data from a response or chunk.

    Args:
        usage: The usage object from the API response.

    Returns:
        Dictionary with usage data or None if not available.
    """
    if not usage:
        return None

    usage_data = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }

    if hasattr(usage, "credits_used") and usage.credits_used is not None:
        usage_data["credits_used"] = usage.credits_used
    elif hasattr(usage, "credits") and usage.credits is not None:
        usage_data["credits_used"] = usage.credits

    return usage_data


def chat(
    client: "VLMRun",
    prompt: str,
    inputs: Optional[List[Path]] = None,
    model: str = DEFAULT_MODEL,
    session_id: Optional[str] = None,
) -> ChatResponse:
    """Perform a non-streaming chat completion.

    This function uploads any input files, sends the prompt to the VLM Run API,
    and returns the complete response.

    Args:
        client: VLMRun client instance.
        prompt: The text prompt for the chat.
        inputs: Optional list of input file paths (images, videos, documents).
        model: The model to use for the chat. Defaults to "vlmrun-orion-1:auto".
        session_id: Optional session ID for persisting chat history.

    Returns:
        ChatResponse containing the response content, session ID, usage data, etc.

    Raises:
        ChatError: If the model is invalid, file types are unsupported, or API call fails.

    Example:
        ```python
        from vlmrun.client import VLMRun
        from vlmrun.client.chat import chat
        from pathlib import Path

        client = VLMRun(api_key="your-api-key")
        response = chat(
            client=client,
            prompt="Describe this image",
            inputs=[Path("image.jpg")],
        )
        print(response.content)
        ```
    """
    _validate_model(model)
    _validate_inputs(inputs)

    start_time = time.time()

    file_responses: List[FileResponse] = []
    if inputs:
        file_responses = _upload_files(client, inputs)

    messages = _build_messages(prompt, file_responses if file_responses else None)

    response = client.agent.completions.create(
        model=model,
        messages=messages,
        stream=False,
        session_id=session_id,
    )

    latency_s = time.time() - start_time
    response_content = response.choices[0].message.content or ""
    response_id = getattr(response, "session_id", None)
    usage_data = _extract_usage_data(response.usage)
    artifact_refs = extract_artifact_refs(response_content)

    return ChatResponse(
        content=response_content,
        session_id=response_id,
        model=model,
        latency_s=latency_s,
        usage=usage_data,
        id=response.id,
        artifact_refs=artifact_refs,
    )


def chat_stream(
    client: "VLMRun",
    prompt: str,
    inputs: Optional[List[Path]] = None,
    model: str = DEFAULT_MODEL,
    session_id: Optional[str] = None,
) -> Iterator[ChatStreamChunk]:
    """Perform a streaming chat completion.

    This function uploads any input files, sends the prompt to the VLM Run API,
    and yields response chunks as they arrive.

    Args:
        client: VLMRun client instance.
        prompt: The text prompt for the chat.
        inputs: Optional list of input file paths (images, videos, documents).
        model: The model to use for the chat. Defaults to "vlmrun-orion-1:auto".
        session_id: Optional session ID for persisting chat history.

    Yields:
        ChatStreamChunk objects containing response content chunks.
        The final chunk will have is_final=True and may contain usage data.

    Raises:
        ChatError: If the model is invalid, file types are unsupported, or API call fails.

    Example:
        ```python
        from vlmrun.client import VLMRun
        from vlmrun.client.chat import chat_stream
        from pathlib import Path

        client = VLMRun(api_key="your-api-key")
        for chunk in chat_stream(
            client=client,
            prompt="Describe this image",
            inputs=[Path("image.jpg")],
        ):
            print(chunk.content, end="", flush=True)
            if chunk.is_final:
                print(f"\\nUsage: {chunk.usage}")
        ```
    """
    _validate_model(model)
    _validate_inputs(inputs)

    file_responses: List[FileResponse] = []
    if inputs:
        file_responses = _upload_files(client, inputs)

    messages = _build_messages(prompt, file_responses if file_responses else None)

    stream = client.agent.completions.create(
        model=model,
        messages=messages,
        stream=True,
        session_id=session_id,
    )

    response_session_id: Optional[str] = None
    chunks_content: List[str] = []

    for chunk in stream:
        if not response_session_id and hasattr(chunk, "session_id"):
            response_session_id = chunk.session_id

        content = ""
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            chunks_content.append(content)

        usage_data = None
        is_final = False

        if hasattr(chunk, "usage") and chunk.usage:
            usage_data = _extract_usage_data(chunk.usage)
            is_final = True

        yield ChatStreamChunk(
            content=content,
            session_id=response_session_id,
            usage=usage_data,
            is_final=is_final,
        )


def collect_stream(stream: Iterator[ChatStreamChunk]) -> ChatResponse:
    """Collect all chunks from a streaming response into a single ChatResponse.

    This is a convenience function for cases where you want to use streaming
    internally but return a complete response.

    Args:
        stream: An iterator of ChatStreamChunk objects from chat_stream().

    Returns:
        ChatResponse containing the complete response.

    Example:
        ```python
        from vlmrun.client.chat import chat_stream, collect_stream

        stream = chat_stream(client, "Hello")
        response = collect_stream(stream)
        print(response.content)
        ```
    """
    chunks: List[str] = []
    session_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    start_time = time.time()

    for chunk in stream:
        if chunk.content:
            chunks.append(chunk.content)
        if chunk.session_id:
            session_id = chunk.session_id
        if chunk.usage:
            usage = chunk.usage

    latency_s = time.time() - start_time
    content = "".join(chunks)
    artifact_refs = extract_artifact_refs(content)

    return ChatResponse(
        content=content,
        session_id=session_id,
        model=DEFAULT_MODEL,
        latency_s=latency_s,
        usage=usage,
        artifact_refs=artifact_refs,
    )
