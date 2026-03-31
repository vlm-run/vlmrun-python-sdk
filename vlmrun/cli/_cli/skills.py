"""Skills API commands."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from vlmrun.client import VLMRun
    from vlmrun.client.types import SkillDownloadResponse, SkillInfo

app = typer.Typer(
    help="Create, list, lookup, update, and download skills.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


# ---------------------------------------------------------------------------
# vlmrun skills list
# ---------------------------------------------------------------------------


@app.command("list")
def list_skills(
    ctx: typer.Context,
    limit: int = typer.Option(
        25, "--limit", "-n", help="Max items to return (1-1000)."
    ),
    offset: int = typer.Option(0, "--offset", help="Number of items to skip."),
    order_by: str = typer.Option(
        "created_at",
        "--order-by",
        help="Sort field: created_at, updated_at, or name.",
    ),
    descending: bool = typer.Option(True, "--desc/--asc", help="Sort direction."),
    grouped: bool = typer.Option(
        False,
        "--grouped",
        "-g",
        help="Show only the latest version of each skill name.",
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
) -> None:
    """List available skills."""
    client: VLMRun = ctx.obj
    skills = client.skills.list(
        limit=limit,
        offset=offset,
        order_by=order_by,
        descending=descending,
        grouped=grouped,
    )

    if not skills:
        console.print("[yellow]No skills found.[/]")
        return

    if output_json:
        print(json.dumps([s.model_dump(mode="json") for s in skills], indent=2))
        return

    table = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("NAME", style="bold cyan", no_wrap=True)
    table.add_column("VERSION", style="dim", max_width=20)
    table.add_column("DESCRIPTION", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("PUB", justify="center", max_width=3)
    table.add_column("CREATED", style="dim", max_width=16)

    for skill in skills:
        created = (
            skill.created_at.strftime("%Y-%m-%d %H:%M") if skill.created_at else ""
        )
        public_marker = "[green]y[/]" if skill.is_public else "n"
        table.add_row(
            skill.name,
            skill.skill_version or "",
            skill.description or "",
            public_marker,
            created,
        )

    console.print(
        Panel(
            table,
            title="[bold]Skills[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(skills)} skill(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# vlmrun skills get
# ---------------------------------------------------------------------------


@app.command("get")
def get_skill(
    ctx: typer.Context,
    name_or_id: str = typer.Argument(..., help="Skill name or UUID."),
    version: Optional[str] = typer.Option(
        None, "--version", "-V", help="Pin a specific version (used with name)."
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
) -> None:
    """Get details for a skill by name or ID."""
    client: VLMRun = ctx.obj
    skill = _resolve_skill(client, name_or_id, version)

    if output_json:
        print(json.dumps(skill.model_dump(mode="json"), indent=2))
        return

    _print_skill_detail(skill)


# ---------------------------------------------------------------------------
# vlmrun skills create
# ---------------------------------------------------------------------------


@app.command("create")
def create_skill(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Text prompt to auto-generate the skill."
    ),
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file",
        "-f",
        help="Read prompt from a file.",
        exists=True,
        readable=True,
    ),
    schema_file: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Path to a JSON schema file.",
        exists=True,
        readable=True,
    ),
    file_id: Optional[str] = typer.Option(
        None, "--file-id", help="Pre-uploaded skill zip file ID."
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Skill name (required for --file-id, optional otherwise).",
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Skill description."
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON."),
) -> None:
    """Create a new skill from a prompt or file.

    \b
    Two creation modes (provide exactly one source):
      --prompt / --prompt-file   Auto-generate SKILL.md (+ optional --schema)
      --file-id                  Use a pre-uploaded skill zip
    """
    client: VLMRun = ctx.obj

    # Resolve prompt from file if provided
    resolved_prompt = prompt
    if prompt_file:
        if prompt:
            console.print("[red]Error:[/] Provide --prompt or --prompt-file, not both.")
            raise typer.Exit(1)
        resolved_prompt = prompt_file.read_text().strip()

    # Validate that exactly one source is given
    sources = [resolved_prompt is not None, file_id is not None]
    if sum(sources) != 1:
        console.print(
            "[red]Error:[/] Provide exactly one of --prompt/--prompt-file or --file-id."
        )
        raise typer.Exit(1)

    # Load JSON schema if provided
    json_schema = None
    if schema_file:
        try:
            json_schema = json.loads(schema_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"[red]Error reading schema file:[/] {e}")
            raise typer.Exit(1)

    skill = client.skills.create(
        prompt=resolved_prompt,
        json_schema=json_schema,
        file_id=file_id,
        name=name,
        description=description,
    )

    if output_json:
        print(json.dumps(skill.model_dump(mode="json"), indent=2))
        return

    _print_skill_detail(skill)


@app.command("upload")
def upload_skill(
    ctx: typer.Context,
    directory: Path = typer.Argument(
        ...,
        help="Path to the skill folder (must contain SKILL.md).",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Skill name (overrides SKILL.md frontmatter)."
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Skill description (overrides SKILL.md frontmatter).",
    ),
) -> None:
    """Zip and upload a local skill folder, then create the skill.

    \b
    The directory must contain a SKILL.md file. The skill name and
    description are parsed from its YAML frontmatter automatically.
    Use --name / --description to override.

    Archives are stored under ~/.vlmrun/skill_archives/.

    \b
    EXAMPLE:
      vlmrun skills upload ./my-skill
    """
    client: VLMRun = ctx.obj

    skill_md = directory / "SKILL.md"
    if not skill_md.exists():
        console.print(f"[red]Error:[/] SKILL.md not found in {directory}")
        raise typer.Exit(1)

    fm_name, fm_desc = _parse_skill_frontmatter(skill_md)

    name = name or fm_name
    if not name:
        console.print(
            "[red]Error:[/] Could not determine skill name from SKILL.md frontmatter. Use --name."
        )
        raise typer.Exit(1)
    description = description or fm_desc

    archive_dir = Path.home() / ".vlmrun" / "skill_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _hash_directory(directory)
    short_hash = content_hash[:8]
    zip_filename = f"{name}_{short_hash}.zip"
    zip_path = archive_dir / zip_filename
    if zip_path.exists():
        console.print(f"[yellow]Warning:[/] Skill zip file already exists: {zip_path}")

    # -- Step 1: zip --------------------------------------------------------
    with console.status("[bold blue]Zipping skill folder…"):
        file_count = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(directory))
                    file_count += 1
        zip_size = zip_path.stat().st_size

    console.print(
        f"  [green]\u2713[/green] [white]Zipped {file_count} file(s) ({_fmt_size(zip_size)} \u2192 {zip_path})[/white]"
    )

    # -- Step 2: upload -----------------------------------------------------
    with console.status("[bold blue]Uploading zip…"):
        file_response = client.files.upload(file=zip_path, purpose="assistants")

    console.print(
        f"  [green]\u2713[/green] [white]Uploaded (file_id={file_response.id})[/white]"
    )

    # -- Step 3: create skill -----------------------------------------------
    with console.status("[bold blue]Creating skill…"):
        skill = client.skills.create(
            file_id=file_response.id,
            name=name,
            description=description,
        )

    console.print("  [green]\u2713[/green] [white]Created skill[/white]")

    _print_skill_with_tree(skill, directory, subtitle_path=zip_path)


# ---------------------------------------------------------------------------
# vlmrun skills download
# ---------------------------------------------------------------------------


@app.command("download")
def download_skill(
    ctx: typer.Context,
    name_or_id: str = typer.Argument(..., help="Skill name or UUID."),
    version: Optional[str] = typer.Option(
        None, "--version", "-V", help="Pin a specific version (used with name)."
    ),
    output_dir: Path = typer.Option(
        Path.home() / ".vlmrun" / "skills",
        "--output",
        "-o",
        help="Extract skill into this directory (creates <dir>/<skill-name>/). [default: ~/.vlmrun/skills/]",
    ),
) -> None:
    """Download a skill zip and extract it to a local folder."""
    import io
    import requests

    client: VLMRun = ctx.obj

    with console.status("[bold blue]Fetching download URL…"):
        skill = _resolve_skill(client, name_or_id, version)
        resp: SkillDownloadResponse = client.skills.download(skill.id)

    # Download and extract
    with console.status("[bold blue]Downloading zip…"):
        r = requests.get(resp.download_url, timeout=120)
        r.raise_for_status()

    skill_dir = output_dir / skill.name
    skill_dir.mkdir(parents=True, exist_ok=True)

    with console.status("[bold blue]Extracting…"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(skill_dir)

    _print_skill_with_tree(skill, skill_dir, subtitle_path=skill_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_skill_frontmatter(skill_md: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract name and description from SKILL.md YAML frontmatter.

    Delegates to :func:`vlmrun.client.skills.parse_skill_frontmatter`.
    """
    from vlmrun.client.skills import parse_skill_frontmatter

    return parse_skill_frontmatter(skill_md)


def _print_skill_with_tree(skill: SkillInfo, folder: Path, subtitle_path: Path) -> None:
    """Print a combined tree + metadata panel for a skill."""
    files = [f for f in folder.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)

    tree = Tree(f"[bold]{folder.name}/[/bold]")
    for item in sorted(files):
        rel = item.relative_to(folder)
        size = _fmt_size(item.stat().st_size)
        tree.add(f"{rel}  [dim]{size}[/dim]")

    meta = Table(show_header=False, box=None, padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column()
    meta.add_row("ID", skill.id)
    meta.add_row("Name", skill.name)
    meta.add_row("Version", skill.skill_version or "\u2014")
    if skill.description:
        meta.add_row("Description", skill.description)
    if skill.is_public is not None:
        meta.add_row("Public", "y" if skill.is_public else "n")
    if skill.created_at:
        meta.add_row("Created", skill.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"))

    body = Group(tree, Text(), meta)

    subtitle = f"{len(files)} file(s), {_fmt_size(total_size)} \u2022 {subtitle_path}"
    console.print(
        Panel(
            body,
            title="Skill",
            title_align="left",
            subtitle=f"[dim]{subtitle}[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(1, 2),
        )
    )


def _hash_directory(directory: Path) -> str:
    """Compute a stable SHA-256 hash over all file contents in a directory.

    Delegates to :func:`vlmrun.client.skills.hash_directory`.
    """
    from vlmrun.client.skills import hash_directory

    return hash_directory(directory)


def _fmt_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return (
                f"{size_bytes:.1f}{unit}"
                if size_bytes != int(size_bytes)
                else f"{int(size_bytes)}{unit}"
            )
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _resolve_skill(
    client: VLMRun, name_or_id: str, version: Optional[str] = None
) -> SkillInfo:
    """Resolve a skill by name-or-UUID, with optional version."""
    if _looks_like_uuid(name_or_id):
        return client.skills.get(id=name_or_id)
    return client.skills.get(name=name_or_id, version=version)


def _looks_like_uuid(value: str) -> bool:
    """Heuristic: UUIDs are 32+ hex chars (with optional dashes)."""
    stripped = value.replace("-", "")
    return len(stripped) >= 32 and all(
        c in "0123456789abcdef" for c in stripped.lower()
    )


def _print_skill_detail(skill: SkillInfo) -> None:
    """Pretty-print a single skill's details."""
    meta = Table(show_header=False, box=None, padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column()

    meta.add_row("ID", skill.id)
    meta.add_row("Name", skill.name)
    meta.add_row("Version", skill.skill_version or "—")
    if skill.description:
        meta.add_row("Description", skill.description)
    if skill.is_public is not None:
        meta.add_row("Public", "y" if skill.is_public else "n")
    if skill.created_at:
        meta.add_row("Created", skill.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    if skill.updated_at:
        meta.add_row("Updated", skill.updated_at.strftime("%Y-%m-%d %H:%M:%S UTC"))

    console.print(
        Panel(
            meta, title="Skill", title_align="left", border_style="blue", padding=(1, 2)
        )
    )
