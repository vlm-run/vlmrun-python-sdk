"""Files API commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.console import Console
from rich import print as rprint

app = typer.Typer(help="File operations")


@app.command()
def list(ctx: typer.Context) -> None:
    """List all files."""
    client = ctx.obj

    files = client.list_files()
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("File ID")
    table.add_column("Filename")
    table.add_column("Size")
    table.add_column("Created At")

    for file in files:
        table.add_row(
            file["id"],
            file["filename"],
            str(file["size"]),
            file["created_at"],
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
    result = client.upload_file(str(file), purpose=purpose)
    rprint(f"Uploaded file {result['filename']} with ID: {result['id']}")


@app.command()
def delete(
    ctx: typer.Context,
    file_id: str = typer.Argument(..., help="ID of the file to delete"),
) -> None:
    """Delete a file."""
    client = ctx.obj
    client.delete_file(file_id)
    rprint(f"Deleted file {file_id}")


@app.command()
def get(
    ctx: typer.Context,
    file_id: str = typer.Argument(..., help="ID of the file to retrieve"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
) -> None:
    """Get file content."""
    client = ctx.obj
    content = client.get_file(file_id)
    if output:
        output.write_bytes(content)
        rprint(f"File content written to {output}")
    else:
        rprint(content.decode())
