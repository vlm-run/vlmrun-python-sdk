"""VLM Run Agent Completions API - Chat-style interface for agent interactions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Literal, Optional, Union
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, Field

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import (
    AgentExecutionConfig,
    AgentExecutionResponse,
    CreditUsage,
    FileResponse,
)
from vlmrun.types.abstract import VLMRunProtocol

# Status types for streaming
StreamEventType = Literal[
    "status", "chunk", "artifact", "usage", "error", "complete"
]


class StreamEvent(BaseModel):
    """Event emitted during streaming execution."""

    type: StreamEventType
    data: Any = None
    message: Optional[str] = None


class CompletionUsage(BaseModel):
    """Usage information for a completion."""

    credits_used: Optional[int] = None
    elements_processed: Optional[int] = None
    element_type: Optional[str] = None


class Artifact(BaseModel):
    """An artifact produced by the agent."""

    id: str
    url: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size: Optional[int] = None


class CompletionResponse(BaseModel):
    """Response from an agent completion."""

    id: str
    model: str
    content: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    artifacts: List[Artifact] = Field(default_factory=list)
    usage: Optional[CompletionUsage] = None
    status: str = "completed"
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def text(self) -> Optional[str]:
        """Get the text content of the response."""
        if self.content:
            return self.content
        if self.response:
            # Try common text fields
            for key in ["text", "message", "content", "answer", "result"]:
                if key in self.response and isinstance(self.response[key], str):
                    return self.response[key]
        return None


class CompletionChunk(BaseModel):
    """A chunk of a streaming completion response."""

    id: str
    model: str
    delta: Optional[str] = None
    status: Optional[str] = None
    finished: bool = False
    usage: Optional[CompletionUsage] = None
    artifacts: List[Artifact] = Field(default_factory=list)


class Completions:
    """Agent Completions API - Chat-style interface for VLM Run agents.

    This provides a clean, chat-completions-style interface for interacting
    with VLM Run's Orion visual AI agent.

    Example:
        ```python
        from vlmrun import VLMRun

        client = VLMRun()

        # Simple completion
        response = client.agent.completions.create(
            prompt="What's in this image?",
            files=["photo.jpg"],
            model="vlmrun-orion-1:auto",
        )
        print(response.text)

        # Streaming completion
        for chunk in client.agent.completions.create(
            prompt="Describe this video",
            files=["video.mp4"],
            stream=True,
        ):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
        ```
    """

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Completions resource.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client, timeout=120)

    def _upload_file(self, file: Union[Path, str]) -> FileResponse:
        """Upload a file and return the file response."""
        if isinstance(file, str):
            file = Path(file)

        return self._client.files.upload(file, purpose="assistants")

    def _get_file_url(self, file_response: FileResponse) -> str:
        """Get the public URL for a file."""
        if file_response.public_url:
            return file_response.public_url
        if file_response.id:
            return self._client.files.get_public_url(file_response.id)
        raise ValueError("File response has no ID or public URL")

    def _extract_artifacts(self, response: Dict[str, Any]) -> List[Artifact]:
        """Extract artifacts from an agent response."""
        artifacts = []

        if not response:
            return artifacts

        def find_artifacts(obj: Any) -> None:
            if isinstance(obj, dict):
                # Check for explicit artifacts field
                if "artifacts" in obj and isinstance(obj["artifacts"], list):
                    for artifact in obj["artifacts"]:
                        if isinstance(artifact, dict) and "url" in artifact:
                            artifacts.append(Artifact(
                                id=artifact.get("id", uuid.uuid4().hex[:12]),
                                url=artifact["url"],
                                filename=artifact.get("filename"),
                                content_type=artifact.get("content_type"),
                                size=artifact.get("size"),
                            ))

                # Check for common artifact URL fields
                for key in ["file_url", "output_url", "download_url", "result_url"]:
                    if key in obj and isinstance(obj[key], str):
                        url = obj[key]
                        if url.startswith(("http://", "https://")):
                            artifacts.append(Artifact(
                                id=uuid.uuid4().hex[:12],
                                url=url,
                                filename=obj.get("filename"),
                            ))

                for value in obj.values():
                    find_artifacts(value)

            elif isinstance(obj, list):
                for item in obj:
                    find_artifacts(item)

        find_artifacts(response)

        # Deduplicate by URL
        seen_urls = set()
        unique = []
        for artifact in artifacts:
            if artifact.url not in seen_urls:
                seen_urls.add(artifact.url)
                unique.append(artifact)

        return unique

    def _extract_text_content(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract text content from an agent response."""
        if not response:
            return None

        # Try common text fields
        for key in ["text", "message", "content", "answer", "result", "output"]:
            if key in response and isinstance(response[key], str):
                return response[key]

        return None

    def _wait_with_events(
        self,
        execution_id: str,
        model: str,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ) -> Generator[CompletionChunk, None, CompletionResponse]:
        """Wait for execution with streaming events.

        Yields CompletionChunks with status updates, then returns the final response.
        """
        start_time = time.time()
        last_status = None
        current_interval = poll_interval

        while True:
            try:
                response = self._client.executions.get(execution_id)

                # Emit status update if changed
                if response.status != last_status:
                    last_status = response.status
                    yield CompletionChunk(
                        id=execution_id,
                        model=model,
                        status=response.status,
                        finished=response.status in ("completed", "failed"),
                    )

                if response.status == "completed":
                    # Build final response
                    artifacts = []
                    text_content = None

                    if response.response:
                        artifacts = self._extract_artifacts(response.response)
                        text_content = self._extract_text_content(response.response)

                    usage = None
                    if response.usage:
                        usage = CompletionUsage(
                            credits_used=response.usage.credits_used,
                            elements_processed=response.usage.elements_processed,
                            element_type=response.usage.element_type,
                        )

                    return CompletionResponse(
                        id=execution_id,
                        model=model,
                        content=text_content,
                        response=response.response,
                        artifacts=artifacts,
                        usage=usage,
                        status="completed",
                        created_at=response.created_at.isoformat() if response.created_at else None,
                        completed_at=response.completed_at.isoformat() if response.completed_at else None,
                    )

                if response.status == "failed":
                    error_msg = "Execution failed"
                    if response.response:
                        error_msg = str(response.response)

                    yield CompletionChunk(
                        id=execution_id,
                        model=model,
                        status="failed",
                        delta=error_msg,
                        finished=True,
                    )

                    return CompletionResponse(
                        id=execution_id,
                        model=model,
                        content=error_msg,
                        status="failed",
                    )

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Execution did not complete within {timeout} seconds. "
                        f"Last status: {response.status}"
                    )

                time.sleep(current_interval)
                # Gradually increase poll interval
                current_interval = min(current_interval * 1.2, 10.0)

            except TimeoutError:
                raise
            except Exception as e:
                yield CompletionChunk(
                    id=execution_id,
                    model=model,
                    status="error",
                    delta=str(e),
                    finished=True,
                )
                raise

    def create(
        self,
        prompt: str,
        files: Optional[List[Union[Path, str]]] = None,
        file_urls: Optional[List[str]] = None,
        model: str = "vlmrun-orion-1:auto",
        stream: bool = False,
        timeout: int = 300,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[CompletionResponse, Generator[CompletionChunk, None, CompletionResponse]]:
        """Create a completion with the Orion visual AI agent.

        Args:
            prompt: The prompt/instruction for the agent.
            files: Local file paths to upload and process (images, videos, documents).
            file_urls: Pre-uploaded file URLs to process.
            model: The model to use. Options:
                - "vlmrun-orion-1:fast" - Optimized for speed
                - "vlmrun-orion-1:auto" - Automatically selects best approach (default)
                - "vlmrun-orion-1:pro" - Most capable, for complex workflows
            stream: If True, yield CompletionChunks as the execution progresses.
            timeout: Maximum time to wait for completion (seconds).
            metadata: Optional metadata to attach to the request.

        Returns:
            If stream=False: CompletionResponse with the final result.
            If stream=True: Generator yielding CompletionChunks, finally returning CompletionResponse.

        Examples:
            ```python
            # Simple completion
            response = client.agent.completions.create(
                prompt="What's in this image?",
                files=["photo.jpg"],
            )
            print(response.text)

            # With streaming
            for chunk in client.agent.completions.create(
                prompt="Analyze this document",
                files=["document.pdf"],
                stream=True,
            ):
                print(f"Status: {chunk.status}")

            # Using pre-uploaded files
            response = client.agent.completions.create(
                prompt="Compare these images",
                file_urls=["https://...", "https://..."],
            )
            ```
        """
        # Upload files if provided
        all_urls: List[str] = []

        if files:
            for file in files:
                file_response = self._upload_file(file)
                url = self._get_file_url(file_response)
                all_urls.append(url)

        if file_urls:
            all_urls.extend(file_urls)

        # Build inputs
        inputs: Optional[Dict[str, Any]] = None
        if all_urls:
            inputs = {"files": all_urls}

        # Build execution config
        config = AgentExecutionConfig(prompt=prompt)

        # Execute the agent
        execution_response = self._client.agent.execute(
            name=model,
            inputs=inputs,
            batch=True,
            config=config,
        )

        execution_id = execution_response.id

        if stream:
            return self._wait_with_events(
                execution_id=execution_id,
                model=model,
                timeout=timeout,
            )
        else:
            # Wait for completion and return final response
            result = self._client.executions.wait(execution_id, timeout=timeout)

            artifacts = []
            text_content = None

            if result.response:
                artifacts = self._extract_artifacts(result.response)
                text_content = self._extract_text_content(result.response)

            usage = None
            if result.usage:
                usage = CompletionUsage(
                    credits_used=result.usage.credits_used,
                    elements_processed=result.usage.elements_processed,
                    element_type=result.usage.element_type,
                )

            return CompletionResponse(
                id=execution_id,
                model=model,
                content=text_content,
                response=result.response,
                artifacts=artifacts,
                usage=usage,
                status=result.status,
                created_at=result.created_at.isoformat() if result.created_at else None,
                completed_at=result.completed_at.isoformat() if result.completed_at else None,
            )

    def download_artifact(
        self,
        artifact: Union[Artifact, str],
        output_dir: Union[Path, str],
        filename: Optional[str] = None,
    ) -> Path:
        """Download an artifact to a local directory.

        Args:
            artifact: The Artifact object or URL string to download.
            output_dir: Directory to save the artifact.
            filename: Optional filename override.

        Returns:
            Path to the downloaded file.
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(artifact, str):
            url = artifact
            artifact_filename = filename
        else:
            url = artifact.url
            artifact_filename = filename or artifact.filename

        # Determine filename from URL if not provided
        if not artifact_filename:
            from urllib.parse import urlparse
            import os

            parsed = urlparse(url)
            artifact_filename = os.path.basename(parsed.path) or f"artifact_{uuid.uuid4().hex[:8]}"

        output_path = output_dir / artifact_filename

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path
