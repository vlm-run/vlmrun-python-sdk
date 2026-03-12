"""Files API commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, List

import typer
from rich import box
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from vlmrun.client import VLMRun
    from vlmrun.client.types import FileResponse

app = typer.Typer(
    help="File operations",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


@app.command()
def list(ctx: typer.Context) -> None:
    """List all files."""
    client: VLMRun = ctx.obj

    files: List[FileResponse] = client.files.list()
    table = Table(
        show_header=True,
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("ID", style="bold cyan", min_width=40)
    table.add_column("FILENAME")
    table.add_column("SIZE", style="dim")
    table.add_column("CREATED", style="dim")
    table.add_column("PURPOSE", style="dim")
    for file in files:
        table.add_row(
            file.id,
            file.filename,
            f"{file.bytes / 1024 / 1024:.2f} MB",
            file.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            str(file.purpose),
        )

    console.print(
        Panel(
            table,
            title="[bold]Files[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(files)} file(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


@app.command()
def upload(
    ctx: typer.Context,
    file: Path = typer.Argument(..., help="File to upload", exists=True, readable=True),
    purpose: str = typer.Option(
        "fine-tune",
        help="Purpose of the file (one of: datasets, fine-tune, assistants, assistants_output, batch, batch_output, vision)",
    ),
) -> None:
    """Upload a file."""
    client: VLMRun = ctx.obj
    result = client.files.upload(str(file), purpose=purpose)
    console.print(f"Uploaded file {result.filename} with ID: {result.id}")


@app.command()
def delete(
    ctx: typer.Context,
    file_id: str = typer.Argument(..., help="ID of the file to delete"),
) -> None:
    """Delete a file."""
    client: VLMRun = ctx.obj
    client.files.delete(file_id)
    console.print(f"Deleted file {file_id}")


@app.command()
def get(
    ctx: typer.Context,
    file_id: str = typer.Argument(..., help="ID of the file to retrieve"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Get file content."""
    client: VLMRun = ctx.obj
    content = client.files.get_content(file_id)
    if output:
        output.write_bytes(content)
        console.print(f"File content written to {output}")
    else:
        console.print(content.decode())
