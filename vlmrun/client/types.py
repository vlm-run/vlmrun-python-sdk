"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Literal, Optional, Type, List, Tuple

from pydantic import BaseModel, Field, model_validator
from pydantic.dataclasses import dataclass
from datetime import datetime
from vlmrun.hub.utils import jsonschema_to_model
import math
import pandas as pd

JobStatus = Literal["enqueued", "pending", "running", "completed", "failed", "paused"]

# AgentToolset type - tool categories available for agent execution
AgentToolset = Literal[
    "core",
    "image",
    "image-gen",
    "world-gen",
    "viz",
    "document",
    "video",
    "web",
]


@dataclass
class APIError(Exception):
    message: str
    http_status: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


class FileResponse(BaseModel):
    id: Optional[str]
    filename: Optional[str]
    bytes: int
    purpose: Literal[
        "fine-tune",
        "assistants",
        "assistants_output",
        "batch",
        "batch_output",
        "vision",
        "datasets",
    ]
    created_at: datetime
    object: str = "file"
    public_url: Optional[str] = None


class PresignedUrlRequest(BaseModel):
    filename: str
    purpose: Optional[str] = None


class PresignedUrlResponse(BaseModel):
    id: Optional[str]
    url: Optional[str]
    filename: Optional[str]
    content_type: Optional[str]
    upload_method: Optional[str]
    public_url: Optional[str]
    created_at: datetime
    expiration: Optional[int] = None
    method: Optional[str] = None


class CreditUsage(BaseModel):
    elements_processed: Optional[int] = None
    element_type: Optional[Literal["image", "page", "video", "audio"]] = None
    credits_used: Optional[int] = None


class PredictionResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    response: Optional[Any] = None
    status: JobStatus
    usage: CreditUsage
    domain: Optional[str] = None


class ModelInfo(BaseModel):
    model: str
    domain: str


class DomainInfo(BaseModel):
    domain: str


class SchemaResponse(BaseModel):
    domain: str
    description: Optional[str] = None
    schema_version: str
    schema_hash: str
    gql_stmt: str
    json_schema: Dict[str, Any]

    @property
    def response_model(self) -> Type[BaseModel]:
        return jsonschema_to_model(self.json_schema)


class HubInfoResponse(BaseModel):
    version: str


class HubDomainInfo(BaseModel):
    domain: str


class HubSchemaResponse(SchemaResponse): ...


class DatasetResponse(BaseModel):
    id: str
    dataset_type: str
    dataset_name: str
    domain: str
    file_id: Optional[str] = None
    message: Optional[str] = None

    wandb_base_url: Optional[str] = None
    wandb_project_name: Optional[str] = None
    wandb_url: Optional[str] = None

    created_at: datetime
    completed_at: Optional[datetime] = None
    status: JobStatus
    usage: CreditUsage


class FinetuningResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: JobStatus
    message: str
    model: str
    suffix: Optional[str] = None
    wandb_url: Optional[str] = None
    wandb_base_url: Optional[str] = None
    wandb_project_name: Optional[str] = None
    usage: CreditUsage


class FinetuningProvisionResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    model: str
    message: str


class FeedbackSubmitRequest(BaseModel):
    """Request model for submitting feedback."""

    request_id: str = Field(..., description="The unique identifier for the request.")
    response: Optional[Dict[str, Any]] = Field(
        None, description="The updated response for the request."
    )
    notes: Optional[str] = Field(None, description="The notes for the feedback.")


class FeedbackItem(BaseModel):
    """Individual feedback item."""

    id: str = Field(..., description="The unique identifier for the feedback.")
    created_at: datetime = Field(
        ...,
        description="Date and time when the feedback was submitted (in UTC timezone)",
    )
    response: Optional[Dict[str, Any]] = Field(
        None, description="The updated response for the request."
    )
    notes: Optional[str] = Field(None, description="The notes for the feedback.")


class FeedbackSubmitResponse(BaseModel):
    """Response model for feedback operations."""

    id: str = Field(..., description="The unique identifier for the feedback.")
    request_id: str = Field(
        ..., description="The request ID for which the feedback is being submitted."
    )
    created_at: datetime = Field(
        ...,
        description="Date and time when the feedback was submitted (in UTC timezone)",
    )


class FeedbackListResponse(BaseModel):
    """Response model for feedback operations."""

    request_id: str = Field(
        ..., description="The request ID for which the feedback is being submitted."
    )
    items: List[FeedbackItem] = Field(
        ..., description="The list of feedback items for the request."
    )


class AgentExecutionResponse(BaseModel):
    """Response to the agent execution request."""

    id: str = Field(..., description="ID of the agent")
    name: str = Field(..., description="Name of the agent")
    created_at: datetime = Field(
        ..., description="Date and time when the agent was created (in UTC timezone)"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Date and time when the agent was completed (in UTC timezone)"
    )
    response: Optional[Dict[str, Any]] = Field(
        None, description="The response from the agent"
    )
    status: JobStatus = Field(..., description="The status of the agent")
    usage: CreditUsage = Field(..., description="The usage of the agent")


class AgentExecutionOrCreationConfig(BaseModel):
    prompt: Optional[str] = Field(default=None, description="The prompt to the agent")
    response_model: Optional[Type[BaseModel]] = Field(
        default=None, description="The response model of the agent"
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="The JSON schema response model of the agent"
    )
    skills: Optional[List["AgentSkill"]] = Field(
        default=None,
        description="List of agent skills to enable for this execution. Skills provide domain-specific expertise and capabilities.",
    )

    @model_validator(mode="after")
    def validate_config(self):
        """Validate the config."""
        if self.response_model and self.json_schema:
            raise ValueError(
                "`response_model` and `json_schema` cannot be used together, please provide only one."
            )
        return self

    def model_dump(self, **kwargs) -> dict:
        """Dump the config as a dictionary, converting response_model to json_schema if present."""
        data = super().model_dump(**kwargs)

        if self.response_model and self.json_schema:
            raise ValueError(
                "`response_model` and `json_schema` cannot be used together, please provide only one."
            )

        if self.response_model is not None:
            assert (
                self.json_schema is None
            ), "`response_model` and `json_schema` cannot be used together"
            json_schema = self.response_model.model_json_schema()
            data["json_schema"] = json_schema
            data.pop("response_model", None)

        return data


class AgentCreationConfig(AgentExecutionOrCreationConfig):
    """Configuration for the agent creation request."""

    prompt: Optional[str] = Field(
        default=None, description="The prompt to guide the creation of the agent."
    )


class AgentExecutionConfig(AgentExecutionOrCreationConfig):
    """Configuration for the agent execution request."""

    prompt: Optional[str] = Field(
        default=None, description="The prompt to guide the execution of the agent."
    )


class AgentInfo(BaseModel):
    id: str = Field(..., description="ID of the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    prompt: str = Field(..., description="The prompt of the agent")
    json_schema: Optional[Dict[str, Any]] = Field(
        None, description="The JSON schema of the agent's response"
    )
    json_sample: Optional[Dict[str, Any]] = Field(
        None, description="The sample response JSON from the agent"
    )
    created_at: datetime = Field(
        ..., description="Date and time when the agent was created (in UTC timezone)"
    )
    updated_at: datetime = Field(
        ..., description="Date and time when the agent was updated (in UTC timezone)"
    )
    status: JobStatus = Field(..., description="The status of the agent")


class AgentCreationResponse(BaseModel):
    id: str = Field(..., description="ID of the agent")
    name: str = Field(..., description="Name of the agent")
    created_at: datetime = Field(
        ..., description="Date and time when the agent was created (in UTC timezone)"
    )
    updated_at: datetime = Field(
        ..., description="Date and time when the agent was updated (in UTC timezone)"
    )
    status: JobStatus = Field(..., description="The status of the agent")


class SkillInfo(BaseModel):
    """Skill information response."""

    id: str = Field(..., description="ID of the skill")
    name: str = Field(..., description="Name of the skill")
    description: Optional[str] = Field(None, description="Description of the skill")
    skill_version: Optional[str] = Field(None, description="Version of the skill")
    version: Optional[str] = Field(
        None,
        description="DEPRECATED: Use 'skill_version' instead. Version of the skill.",
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_version(cls, values: Any) -> Any:
        """Accept 'version' as an alias for 'skill_version' for backward compatibility."""
        if isinstance(values, dict):
            if values.get("version") and not values.get("skill_version"):
                values["skill_version"] = values["version"]
        return values

    created_at: Optional[datetime] = Field(
        None, description="Date and time when the skill was created (in UTC timezone)"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Date and time when the skill was updated (in UTC timezone)"
    )
    status: Optional[JobStatus] = Field(None, description="The status of the skill")
    is_public: Optional[bool] = Field(None, description="Whether the skill is public")


class SkillDownloadResponse(BaseModel):
    """Response for skill download URL."""

    download_url: str = Field(
        ..., description="Presigned download URL for the skill zip"
    )
    expires_in: Optional[int] = Field(None, description="Seconds until the URL expires")


class InlineSkillSource(BaseModel):
    """Source payload for an inline skill bundle.

    Follows the format::

        {
            "type": "base64",
            "media_type": "application/zip",
            "data": "<base64-encoded-zip>"
        }
    """

    type: str = Field(
        default="base64",
        description="Encoding type for the inline skill data. Currently only 'base64' is supported.",
    )
    media_type: str = Field(
        default="application/zip",
        description="MIME type of the skill bundle. Must be 'application/zip'.",
    )
    data: str = Field(
        ...,
        description="Base64-encoded zip bundle containing the skill files.",
    )


class AgentSkill(BaseModel):
    """A modular capability that extends the agent's functionality.

    Two modes are supported:

    1. **Referenced skills** (``type="skill_reference"``) -- Provide ``skill_id``
       (UUID or name) and optionally ``skill_version``.

       .. code-block:: python

           AgentSkill(type="skill_reference", skill_id="pillow")

    2. **Inline skills** (``type="inline"``) -- Supply ``name``, ``description``,
       and a ``source`` object containing the base64-encoded zip bundle.

       .. code-block:: python

           AgentSkill(
               type="inline",
               name="csv-insights",
               description="Summarize CSV files.",
               source=InlineSkillSource(data="<base64-zip>"),
           )
    """

    type: str = Field(
        default="skill_reference",
        description=(
            "The type of the skill. "
            "Use 'skill_reference' for DB-stored skills referenced by id/name. "
            "Use 'inline' to provide the skill as a base64-encoded zip bundle."
        ),
    )

    # -- Referenced skill fields -------------------------------------------
    skill_id: Optional[str] = Field(
        default=None,
        description="The unique identifier of the skill -- a UUID or a name string.",
    )
    skill_name: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable skill name for lookup. "
            "Alternative to skill_id. Deprecated in favour of skill_id."
        ),
    )
    skill_version: Optional[str] = Field(
        default="latest",
        description="The version of the skill -- an integer or 'latest'.",
    )
    version: Optional[str] = Field(
        default=None,
        description="DEPRECATED: Use 'skill_version' instead. The version of the skill.",
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_version(cls, values: Any) -> Any:
        """Accept 'version' as an alias for 'skill_version' for backward compatibility."""
        if isinstance(values, dict):
            if values.get("version") and not values.get("skill_version"):
                values["skill_version"] = values["version"]
        return values

    # -- Inline skill fields -----------------------------------------------
    name: Optional[str] = Field(
        default=None,
        description="Human-readable name for the inline skill.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short description of what the inline skill does.",
    )
    source: Optional[InlineSkillSource] = Field(
        default=None,
        description=(
            "Source payload for inline skills. Contains the base64-encoded zip "
            "bundle with type, media_type, and data fields."
        ),
    )

    # -- Legacy / backward-compat ------------------------------------------
    bundle: Optional[str] = Field(
        default=None,
        description=(
            "DEPRECATED: Use 'source.data' instead. "
            "Base64-encoded zip bundle containing the skill files (inline skills only)."
        ),
    )

    @property
    def is_inline(self) -> bool:
        """Return True if this is an inline skill (base64 bundle)."""
        return self.type == "inline" and (
            self.source is not None or self.bundle is not None
        )

    @property
    def inline_data(self) -> Optional[str]:
        """Return the base64-encoded zip data for inline skills.

        Prefers ``source.data`` over the legacy ``bundle`` field.
        Returns ``None`` for referenced skills.
        """
        if self.source is not None:
            return self.source.data
        return self.bundle

    @property
    def effective_skill_id(self) -> Optional[str]:
        """Return the identifier to use for resolution.

        For referenced skills, ``skill_id`` takes precedence over ``skill_name``.
        For inline skills, a deterministic hash of the bundle is returned.
        """
        if self.is_inline:
            import hashlib

            data = self.inline_data or ""
            return f"inline_{hashlib.sha256(data.encode()).hexdigest()[:16]}"
        return self.skill_id or self.skill_name

    @property
    def effective_name(self) -> Optional[str]:
        """Return the best available name for this skill."""
        if self.is_inline:
            return self.name or "inline"
        return self.skill_name or self.skill_id

    @property
    def effective_description(self) -> Optional[str]:
        """Return the description for this skill."""
        return self.description or ""

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "AgentSkill":
        """Build an inline :class:`AgentSkill` from a local skill directory.

        Zips the directory contents, base64-encodes the result, and returns
        an ``AgentSkill`` with ``type="inline"`` that can be sent directly
        in a chat completion or agent execution request.

        Args:
            directory: Path to a skill folder containing at least a ``SKILL.md``.
            name: Override skill name (defaults to SKILL.md frontmatter or directory name).
            description: Override skill description (defaults to SKILL.md frontmatter).

        Returns:
            AgentSkill with ``type="inline"`` and the base64-encoded zip bundle.

        Raises:
            FileNotFoundError: If ``SKILL.md`` is missing from *directory*.

        Example::

            skill = AgentSkill.from_directory(Path("./my-skill"))
            response = client.agent.completions.create(
                model="vlmrun-orion-1:auto",
                messages=[...],
                skills=[skill],
            )
        """
        from vlmrun.client.skills import (
            bundle_from_directory,
            parse_skill_frontmatter,
        )

        directory = Path(directory)
        skill_md = directory / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {directory}")

        fm_name, fm_description = parse_skill_frontmatter(skill_md)

        return cls(
            type="inline",
            name=name or fm_name or directory.name,
            description=description or fm_description or "",
            source=InlineSkillSource(data=bundle_from_directory(directory)),
        )


class GenerationConfig(BaseModel):
    prompt: Optional[str] = Field(default=None)
    response_model: Optional[Type[BaseModel]] = Field(default=None)
    json_schema: Optional[Dict[str, Any]] = Field(default=None)
    gql_stmt: Optional[str] = Field(default=None)
    max_retries: int = Field(default=3)
    max_tokens: int = Field(default=65535)
    temperature: float = Field(default=0.0)

    detail: Literal["auto", "lo", "hi"] = Field(default="auto")
    confidence: bool = Field(default=False)
    grounding: bool = Field(default=False)
    skills: Optional[List["AgentSkill"]] = Field(
        default=None,
        description="List of agent skills to enable for this request. Skills provide domain-specific expertise and capabilities.",
    )

    def model_dump(self, **kwargs) -> dict:
        """Dump the config as a dictionary, converting response_model to json_schema if present."""
        data = super().model_dump(**kwargs)

        if self.response_model and self.json_schema:
            raise ValueError(
                "`response_model` and `json_schema` cannot be used together"
            )

        if self.response_model is not None:
            assert (
                self.json_schema is None
            ), "`response_model` and `json_schema` cannot be used together"
            json_schema = self.response_model.model_json_schema()
            data["json_schema"] = json_schema
            data.pop("response_model", None)

        return data


class RequestMetadata(BaseModel):
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    session_id: Optional[str] = Field(default=None)
    allow_training: bool = Field(default=True)


class MarkdownPageMetadata(BaseModel):
    page_number: Optional[int] = Field(
        None, description="Page number of the document, starting from 0."
    )


def replace_nan_recursive_fast(obj):
    """Replace NaN values with empty strings recursively."""
    _isnan = math.isnan
    _float = float
    _dict = dict
    _list = list
    _isinstance = isinstance

    def _replace(o):
        if _isinstance(o, _dict):
            return {k: _replace(v) for k, v in o.items()}
        elif _isinstance(o, _list):
            return [_replace(v) for v in o]
        elif _isinstance(o, _float) and _isnan(o):
            return ""
        return o

    return _replace(obj)


class BoxCoords(BaseModel):
    """Bounding box coordinates."""

    xywh: List[float]

    @property
    def xyxy(self) -> List[float]:
        x, y, w, h = self.xywh
        return [x, y, x + w, y + h]


class TableCell(BaseModel):
    """A table cell with layout information."""

    content: str
    bbox: Optional[BoxCoords] = None
    index: Tuple[int, int]
    span: Optional[Tuple[Optional[int], Optional[int]]] = None


class TableHeader(BaseModel):
    """A table header with column information."""

    id: str
    column: int
    name: str
    dtype: Optional[
        Literal[
            "str",
            "int",
            "float",
            "bool",
            "date",
            "datetime",
            "time",
            "timedelta",
            "duration",
            "currency",
            "percentage",
            "enum",
        ]
    ] = None


class TableMetadata(BaseModel):
    """Metadata for a table."""

    title: Optional[str] = None
    caption: Optional[str] = None
    notes: Optional[str] = None


class MarkdownTable(BaseModel):
    """A table in the document."""

    metadata: Optional[TableMetadata] = Field(default_factory=TableMetadata)
    content: Optional[str] = None
    headers: List[TableHeader]
    data: Optional[List[dict]] = None
    bbox: Optional[BoxCoords] = None
    cells: Optional[List[TableCell]] = None

    def __str__(self):
        """Return a string representation of the markdown table."""
        return self.to_dataframe(header="name").to_markdown()

    @model_validator(mode="after")
    def validate_metadata(self):
        """Ensure metadata is not None."""
        if self.metadata is None:
            self.metadata = TableMetadata()
        return self

    def to_dataframe(
        self, header: Literal["id", "name", "none"] = "id"
    ) -> pd.DataFrame:
        """Convert the table to a pandas DataFrame."""
        try:
            self.data = replace_nan_recursive_fast(self.data)

            if not self.data:
                return pd.DataFrame()

            df = pd.DataFrame.from_records(self.data)

            if header == "id":
                df.columns = [h.id for h in self.headers]
            elif header == "name":
                df.columns = [h.name for h in self.headers]
                if len(df.columns) != len(set(df.columns)):
                    return self.to_dataframe(header="id")
            elif header == "none":
                df.columns = [h.id for h in self.headers]
            else:
                raise ValueError(f"Invalid header type: {header}")
            return df
        except Exception:
            return pd.DataFrame()

    def render(self) -> str:
        """Render the table as a markdown table."""
        return self.to_dataframe(header="name").to_markdown()


class MarkdownFigure(BaseModel):
    """A figure such as an image, bar chart, line chart, pie chart, etc."""

    id: int
    title: Optional[str] = None
    caption: Optional[str] = None
    content: Optional[str] = None

    def render(self) -> str:
        """Replace all <Figure> blocks with their rendered markdown content."""
        return f"""<Figure id="fg-{self.id}"/>\n\n{self.content or ""}"""


class MarkdownPage(BaseModel):
    """A markdown document page."""

    content: str
    markdown_content: Optional[str] = None
    tables: Optional[List[MarkdownTable]] = None
    figures: Optional[List[MarkdownFigure]] = None

    def __str__(self):
        """Return a string representation of the markdown page."""
        content = self.markdown_content or self.render()
        return content

    @model_validator(mode="after")
    def render_markdown(self):
        """Render the markdown content of the document page."""
        self.markdown_content = self.render()
        return self

    def render(self) -> str:
        """Replace all <Table> and <Figure> blocks with their rendered markdown content."""
        rendered_content = self.content

        if self.tables:
            for id, table in enumerate(self.tables):
                table_placeholder = f'<Table id="tb-{id}"/>'
                if table_placeholder in rendered_content:
                    rendered_content = rendered_content.replace(
                        table_placeholder, table.render()
                    )
                else:
                    rendered_content = rendered_content + table.render()

        if self.figures:
            for figure in self.figures:
                figure_placeholder = f'<Figure id="fg-{figure.id}"/>'
                if figure_placeholder in rendered_content:
                    rendered_content = rendered_content.replace(
                        figure_placeholder, figure.render()
                    )

        return rendered_content


class MarkdownDocument(BaseModel):
    """A markdown document containing multiple pages."""

    pages: List[MarkdownPage]

    def __str__(self):
        """Return a string representation of the markdown document."""
        return "\n\n---PAGE BREAK---\n\n".join(str(page) for page in self.pages)

    def __getitem__(self, index: int) -> MarkdownPage:
        """Get a page by index."""
        return self.pages[index]

    def __len__(self) -> int:
        """Get the number of pages."""
        return len(self.pages)

    @property
    def content(self) -> str:
        """Get all content concatenated."""
        return "\n\n---PAGE BREAK---\n\n".join(page.content for page in self.pages)

    @property
    def all_tables(self) -> List[MarkdownTable]:
        """Get all tables from all pages."""
        tables = []
        for page in self.pages:
            if page.tables:
                tables.extend(page.tables)
        return tables

    @property
    def all_figures(self) -> List[MarkdownFigure]:
        """Get all figures from all pages."""
        figures = []
        for page in self.pages:
            if page.figures:
                figures.extend(page.figures)
        return figures
