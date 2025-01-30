"""Files API commands."""

from pathlib import Path
from typing import Optional, List

import typer
from rich.table import Table
from rich.console import Console
from rich import print as rprint
from vlmrun.client.types import FileResponse

app = typer.Typer(
    help="File operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def list(ctx: typer.Context) -> None:
    """List all files."""
    client = ctx.obj

    files: List[FileResponse] = client.files.list()
    console = Console()
    table = Table(show_header=True)
    table.add_column("File ID")
    table.add_column("Filename")
    table.add_column("Size")
    table.add_column("Created At")
    table.add_column("Purpose")
    for file in files:
        table.add_row(
            file.id,
            file.filename,
            f"{file.bytes / 1024 / 1024:.2f} MB",
            file.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            str(file.purpose),
        )

    console.print(table)


@app.command()
def upload(
    ctx: typer.Context,
    file: Path = typer.Argument(..., help="File to upload", exists=True, readable=True),
    purpose: str = typer.Option("fine-tune", help="Purpose of the file"),
) -> None:
    """Upload a file."""
    client = ctx.obj
    result = client.files.upload(str(file), purpose=purpose)
    rprint(f"Uploaded file {result.filename} with ID: {result.id}")


@app.command()
def delete(
    ctx: typer.Context,
    file_id: str = typer.Argument(..., help="ID of the file to delete"),
) -> None:
    """Delete a file."""
    client = ctx.obj
    client.files.delete(file_id)
    rprint(f"Deleted file {file_id}")


@app.command()
def get(
    ctx: typer.Context,
    file_id: str = typer.Argument(..., help="ID of the file to retrieve"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Get file content."""
    client = ctx.obj
    content = client.files.get_content(file_id)
    if output:
        output.write_bytes(content)
        rprint(f"File content written to {output}")
    else:
        rprint(content.decode())
