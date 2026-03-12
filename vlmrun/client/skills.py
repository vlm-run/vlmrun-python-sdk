"""VLM Run API Skills resource."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.types.abstract import VLMRunProtocol
from vlmrun.client.types import SkillInfo, SkillDownloadResponse


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

    def list(self) -> List[SkillInfo]:
        """List all available skills.

        Returns:
            List of SkillInfo objects
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="skills",
        )

        if not isinstance(response, list):
            raise TypeError("Expected list response")

        return [SkillInfo(**skill) for skill in response]

    def get(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> SkillInfo:
        """Lookup a skill by name, ID, or name + version.

        If `id` is provided, fetches the skill directly by ID via GET /v1/skills/{skill_id}.
        Otherwise, looks up by `name` (and optional `version`) via POST /v1/skills/lookup.

        Args:
            name: Skill name for lookup
            id: Skill ID for direct retrieval
            version: Skill version (used with name)

        Returns:
            SkillInfo: Skill information

        Raises:
            ValueError: If neither name nor id is provided
        """
        if id and not name:
            response, status_code, headers = self._requestor.request(
                method="GET",
                url=f"skills/{id}",
            )
        elif name:
            data: Dict[str, Any] = {"name": name}
            if version:
                data["version"] = version
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
