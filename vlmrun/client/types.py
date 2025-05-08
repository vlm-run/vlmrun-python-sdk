"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Literal, Optional, Type, List
from vlmrun.hub.utils import jsonschema_to_model
from textwrap import dedent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from loguru import logger

JobStatus = Literal["enqueued", "pending", "running", "completed", "failed", "paused"]


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


class FeedbackSubmitResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str
    response: Any


class GenerationConfig(BaseModel):
    prompt: Optional[str] = Field(default=None)
    response_model: Optional[Type[BaseModel]] = Field(default=None)
    json_schema: Optional[Dict[str, Any]] = Field(default=None)
    gql_stmt: Optional[str] = Field(default=None)
    max_retries: int = Field(default=3)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.0)

    detail: Literal["auto", "lo", "hi"] = Field(default="auto")
    confidence: bool = Field(default=False)
    grounding: bool = Field(default=False)


class RequestMetadata(BaseModel):
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    session_id: Optional[str] = Field(default=None)
    allow_training: bool = Field(default=True)


class MarkdownPageMetadata(BaseModel):
    page_number: Optional[int] = Field(
        None, description="Page number of the document, starting from 0."
    )


class MarkdownTable(BaseModel):
    """A table in the document."""

    id: int = Field(
        ..., description="The reference id of the table, starting from 'tb-<number>'."
    )
    data: List[List[str]] = Field(
        ..., description="The table data as a 2D array of strings"
    )
    headers: List[str] = Field(..., description="The table headers")

    def render(self) -> str:
        """Render the table as a markdown table."""
        # Create header row
        header = "| " + " | ".join(self.headers) + " |"
        # Create separator row
        separator = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        # Create data rows
        rows = ["| " + " | ".join(row) + " |" for row in self.data]
        # Combine all rows
        return "\n".join([header, separator] + rows)


class MarkdownFigure(BaseModel):
    """A figure such as an image, bar chart, line chart, pie chart, etc."""

    id: int = Field(
        ..., description="The reference id of the figure, starting from 'fg-<number>'."
    )
    title: Optional[str] = Field(
        default=None, description="Title of the figure, if any, otherwise None."
    )
    caption: Optional[str] = Field(
        default=None, description="Caption of the figure, if any, otherwise None."
    )
    content: Optional[str] = Field(
        default=None, description="The best-attempt at describing the figure."
    )

    def render(self) -> str:
        """Replace all <Figure> blocks with their rendered markdown content."""
        return dedent(f"""<Figure id="fg-{self.id}"/>\n\n{self.content or ''}""")


class MarkdownBlock(BaseModel):
    """A markdown block such as a paragraph, heading, entire list, etc."""

    content: str = Field(..., description="The content of the markdown block.")


class MarkdownCode(BaseModel):
    """A code block."""

    id: int = Field(
        ...,
        description="The reference id of the code block, starting from 'cd-<number>'.",
    )
    language: str = Field(
        ...,
        description="The language of the code block (e.g. 'python', 'javascript', 'bash', 'sql', etc.).",
    )
    content: str = Field(..., description="The content of the code block.")


class MarkdownDiagram(BaseModel):
    """A diagram block represented as a Mermaid block."""

    id: int = Field(
        ..., description="The reference id of the diagram, starting from 'dg-<number>'."
    )
    content: str = Field(..., description="The content of the diagram.")


class MarkdownPage(BaseModel):
    """Represents a markdown document page with its content and metadata."""

    content: str = Field(
        ..., description="The Github-Flavored markdown content of the document page"
    )
    markdown_content: Optional[str] = Field(
        None, description="The rendered markdown content of the document page"
    )
    tables: Optional[list[MarkdownTable]] = Field(
        None, description="List of tables in the document, if any, otherwise None"
    )
    figures: Optional[list[MarkdownFigure]] = Field(
        None,
        description="List of figures, images, charts, or diagrams in the document, if any, otherwise None. Do NOT include tables in this list",
    )

    def __str__(self):
        """Return a string representation of the markdown page."""
        # Use the rendered markdown content if available, otherwise render it
        content = self.markdown_content or self.render()

        # Format the content with rich formatting if available
        console = Console(record=True, width=240)
        md = Markdown(content)
        panel = Panel(md, border_style="blue", title="Markdown Content")
        console.print(panel)
        return console.export_text()

    def render(self) -> str:
        """Replace all <Table> and <Figure> blocks with their rendered markdown content."""
        rendered_content = self.content

        if self.tables:
            for id, table in enumerate(self.tables):
                table_placeholder = f'<Table id="tb-{id}"/>'
                # Check if the table placeholder exists in the markdown content
                if table_placeholder in rendered_content:
                    rendered_content = rendered_content.replace(
                        table_placeholder, table.render()
                    )
                else:
                    logger.warning(
                        f"Table placeholder {table_placeholder} not found in the markdown content."
                    )
                    rendered_content = rendered_content + table.render()

        if self.figures:
            for figure in self.figures:
                figure_placeholder = f'<Figure id="fg-{figure.id}"/>'
                # Check if the figure placeholder exists in the markdown content
                if figure_placeholder in rendered_content:
                    rendered_content = rendered_content.replace(
                        figure_placeholder, figure.render()
                    )
                else:
                    logger.warning(
                        f"Figure placeholder {figure_placeholder} not found in the markdown content."
                    )

        return rendered_content
