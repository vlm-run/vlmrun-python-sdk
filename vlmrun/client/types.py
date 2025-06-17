"""Type definitions for VLM Run API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from pydantic.dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Literal, Optional, Type, List, Tuple
from vlmrun.hub.utils import jsonschema_to_model
import math
import pandas as pd

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


class PresignedUrlResponse(BaseModel):
    id: Optional[str]
    url: Optional[str]
    filename: Optional[str]
    expiration: Optional[int]
    method: Optional[str]
    content_type: Optional[str]
    created_at: datetime


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


class FeedbackCreateParams(BaseModel):
    response: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str
    response: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class FeedbackListResponse(BaseModel):
    data: List["FeedbackResponse"]
    count: int
    limit: int
    offset: int


class FeedbackSubmitResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str
    response: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


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
        return f"""<Figure id="fg-{self.id}"/>\n\n{self.content or ''}"""


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
