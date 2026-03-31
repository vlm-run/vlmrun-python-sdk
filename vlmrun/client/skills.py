"""VLM Run API Skills resource."""

from __future__ import annotations

import base64
import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import (
    AgentSkill,
    InlineSkillSource,
    SkillDownloadResponse,
    SkillInfo,
)
from vlmrun.types.abstract import VLMRunProtocol

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def parse_skill_frontmatter(skill_md: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract name and description from a SKILL.md YAML frontmatter.

    Args:
        skill_md: Path to SKILL.md file.

    Returns:
        Tuple of (name, description), either of which may be ``None``
        if the frontmatter is missing or does not contain the field.
    """
    text = skill_md.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    name: Optional[str] = None
    description: Optional[str] = None
    if match:
        for line in match.group(1).splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip("\"'")
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("\"'")
    return name, description


def bundle_from_directory(directory: Path) -> str:
    """Zip a skill directory into a base64-encoded bundle string.

    Walks *directory* recursively, adding every file with paths relative
    to *directory*.  The result is ready to be passed as
    ``InlineSkillSource.data``.

    Args:
        directory: Path to a skill folder.

    Returns:
        Base64-encoded zip bundle string.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(directory))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def hash_directory(directory: Path) -> str:
    """Compute a stable SHA-256 hex digest over all file contents in a directory."""
    h = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(directory).as_posix()
            h.update(rel.encode())
            h.update(file_path.read_bytes())
    return h.hexdigest()


def inline_skill_from_directory(directory: Path) -> AgentSkill:
    """Build an inline :class:`AgentSkill` from a local skill directory.

    Zips the directory contents into memory, base64-encodes the result,
    and returns an ``AgentSkill`` with ``type="inline"`` that can be sent
    directly in a chat completion request.

    Args:
        directory: Path to a skill folder containing at least a ``SKILL.md``.

    Returns:
        AgentSkill with type="inline" and the base64-encoded zip bundle.

    Raises:
        FileNotFoundError: If ``SKILL.md`` is missing from *directory*.
    """
    skill_md = directory / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {directory}")

    name, description = parse_skill_frontmatter(skill_md)
    bundle = bundle_from_directory(directory)

    return AgentSkill(
        type="inline",
        name=name or directory.name,
        description=description or "",
        source=InlineSkillSource(data=bundle),
    )


class Skills:
    """Skills resource for VLM Run API.

    Provides methods to list, lookup, create, update, and download skills.
    """

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Skills resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(
        self,
        limit: int = 25,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
        grouped: bool = False,
    ) -> List[SkillInfo]:
        """List available skills.

        Args:
            limit: Max items to return (1-1000).
            offset: Number of items to skip.
            order_by: Sort field (created_at, updated_at, name).
            descending: Sort direction.
            grouped: If True, return only the latest version per skill name.

        Returns:
            List of SkillInfo objects
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order_by": order_by,
            "descending": descending,
            "grouped": grouped,
        }
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="skills",
            params=params,
        )

        if not isinstance(response, list):
            raise TypeError("Expected list response")

        return [SkillInfo(**skill) for skill in response]

    def get(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        skill_version: Optional[str] = None,
        version: Optional[str] = None,
    ) -> SkillInfo:
        """Lookup a skill by name, ID, or name + version.

        If `id` is provided, fetches the skill directly by ID via GET /v1/skills/{skill_id}.
        Otherwise, looks up by `name` (and optional `skill_version`) via POST /v1/skills/lookup.

        Args:
            name: Skill name for lookup
            id: Skill ID for direct retrieval
            skill_version: Skill version (used with name)
            version: DEPRECATED — use ``skill_version`` instead.

        Returns:
            SkillInfo: Skill information

        Raises:
            ValueError: If neither name nor id is provided
        """
        effective_version = skill_version or version
        if id and not name:
            response, status_code, headers = self._requestor.request(
                method="GET",
                url=f"skills/{id}",
            )
        elif name:
            data: Dict[str, Any] = {"name": name}
            if effective_version:
                data["skill_version"] = effective_version
            response, status_code, headers = self._requestor.request(
                method="POST",
                url="skills/lookup",
                data=data,
            )
        else:
            raise ValueError("Either `name` or `id` must be provided.")

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return SkillInfo(**response)

    def create(
        self,
        prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        file_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SkillInfo:
        """Create a new skill.

        Skills can be created from a prompt (with optional JSON schema),
        from a chat session, or from an uploaded skill zip file.

        Args:
            prompt: Text prompt to auto-generate SKILL.md and schema
            json_schema: Optional JSON schema (used with prompt)
            session_id: Chat session ID to generate skill from
            file_id: Uploaded skill zip file ID
            name: Skill name (used with file_id)
            description: Skill description (used with file_id)

        Returns:
            SkillInfo: Created skill information
        """
        data: Dict[str, Any] = {}

        if prompt is not None:
            data["prompt"] = prompt
        if json_schema is not None:
            data["json_schema"] = json_schema
        if session_id is not None:
            data["session_id"] = session_id
        if file_id is not None:
            data["file_id"] = file_id
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description

        response, status_code, headers = self._requestor.request(
            method="POST",
            url="skills/create",
            data=data,
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return SkillInfo(**response)

    def update(
        self,
        skill_id: str,
        file_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SkillInfo:
        """Update an existing skill (creates a new version).

        Args:
            skill_id: ID of the skill to update
            file_id: New skill zip file ID
            description: Updated description

        Returns:
            SkillInfo: Updated skill information
        """
        data: Dict[str, Any] = {}

        if file_id is not None:
            data["file_id"] = file_id
        if description is not None:
            data["description"] = description

        response, status_code, headers = self._requestor.request(
            method="POST",
            url=f"skills/{skill_id}/update",
            data=data,
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return SkillInfo(**response)

    def download(self, skill_id: str) -> SkillDownloadResponse:
        """Get a presigned download URL for a skill zip.

        Args:
            skill_id: ID of the skill to download

        Returns:
            SkillDownloadResponse: Download URL and expiry information
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"skills/{skill_id}/download",
        )

        if not isinstance(response, dict):
            raise TypeError("Expected dict response")

        return SkillDownloadResponse(**response)

    def create_from_directory(
        self,
        directory: Path,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> AgentSkill:
        """Upload a local skill directory and create a server-side skill.

        Zips the directory, uploads the archive via the files API, and
        creates a new skill pointing to it.  Returns a referenced
        :class:`AgentSkill` that can be sent in a chat completion request.

        Args:
            directory: Path to a skill folder containing at least a ``SKILL.md``.
            name: Override skill name (defaults to SKILL.md frontmatter or directory name).
            description: Override skill description (defaults to SKILL.md frontmatter).

        Returns:
            AgentSkill with ``type="skill_reference"`` pointing to the created skill.

        Raises:
            FileNotFoundError: If ``SKILL.md`` is missing from *directory*.
        """
        skill_md = directory / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {directory}")

        fm_name, fm_description = parse_skill_frontmatter(skill_md)
        skill_name = name or fm_name or directory.name
        skill_description = description or fm_description

        archive_dir = Path.home() / ".vlmrun" / "skill_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)

        short_hash = hash_directory(directory)[:8]
        zip_path = archive_dir / f"{skill_name}_{short_hash}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(directory.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(directory))

        file_response = self._client.files.upload(file=zip_path, purpose="assistants")
        skill_info = self.create(
            file_id=file_response.id,
            name=skill_name,
            description=skill_description,
        )

        return AgentSkill(
            type="skill_reference",
            skill_id=skill_info.id,
            skill_name=skill_info.name,
        )
