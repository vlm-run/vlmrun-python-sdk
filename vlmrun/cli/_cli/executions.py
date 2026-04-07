"""Executions CLI commands — list and retrieve agent execution results."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

if TYPE_CHECKING:
    from vlmrun.client import VLMRun
    from vlmrun.client.types import AgentExecutionResponse

app = typer.Typer(
    help="List and retrieve agent execution results.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def _status_style(status: str) -> str:
    return {
        "completed": "green",
        "running": "yellow",
        "failed": "red",
        "pending": "dim",
        "enqueued": "dim",
        "paused": "yellow",
    }.get(status, "white")


def _compute_duration(created_at, completed_at, usage) -> str:
    if usage and usage.duration_seconds:
        return f"{usage.duration_seconds}s"
    if completed_at and created_at:
        delta = (completed_at - created_at).total_seconds()
        if delta > 0:
            return f"{int(delta)}s"
    return "-"


def _format_row(
    id_val: str,
    name_val: str,
    status_val: str,
    status_style: str,
    created_val: str,
    dur_val: str,
    name_width: int,
) -> Text:
    name_trunc = name_val[:name_width].ljust(name_width)
    line = Text()
    line.append(f" {id_val}  ", style="bold cyan")
    line.append(f"{name_trunc}  ", style="white")
    line.append(f"{status_val:<10s}  ", style=status_style)
    line.append(f"{created_val}  ", style="dim")
    line.append(f"{dur_val:>6s}", style="dim")
    return line


@app.command()
def list(
    ctx: typer.Context,
    skip: int = typer.Option(0, help="Skip the first N executions"),
    limit: int = typer.Option(10, help="Maximum number of executions to list"),
    status: Optional[str] = typer.Option(
        None,
        help="Filter by status (enqueued, pending, running, completed, failed, paused)",
    ),
    since: Optional[str] = typer.Option(
        None, help="Show executions since date (YYYY-MM-DD)"
    ),
    until: Optional[str] = typer.Option(
        None, help="Show executions until date (YYYY-MM-DD)"
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format. Use 'json' for JSON output.",
    ),
) -> None:
    """List agent executions."""
    client: VLMRun = ctx.obj
    output_json = output_format and output_format.lower() == "json"

    executions = client.executions.list(skip=skip, limit=limit)

    if status:
        executions = [e for e in executions if e.status == status]

    if since:
        try:
            from datetime import timezone

            since_date = datetime.strptime(since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            executions = [e for e in executions if e.created_at >= since_date]
        except ValueError:
            console.print(
                "[red]Error:[/] Invalid date format for --since. Use YYYY-MM-DD"
            )
            raise typer.Exit(1)

    if until:
        try:
            from datetime import timezone

            until_date = datetime.strptime(until, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            executions = [e for e in executions if e.created_at <= until_date]
        except ValueError:
            console.print(
                "[red]Error:[/] Invalid date format for --until. Use YYYY-MM-DD"
            )
            raise typer.Exit(1)

    if not executions:
        if output_json:
            print("[]")
        else:
            console.print("[yellow]No executions found[/]")
        return

    if output_json:
        print(
            json.dumps(
                [e.model_dump(mode="json") for e in executions],
                indent=2,
                default=str,
            )
        )
        return

    # Layout: pad(1) + ID(36) + gap(2) + name(flex) + gap(2) + status(10) + gap(2) + created(16) + gap(2) + dur(6)
    # Fixed cols = 1+36+2 + 2+10+2+16+2+6 = 77
    panel_w = min(console.width, 150)
    inner_w = panel_w - 2
    name_w = max(inner_w - 85, 10)

    header = Text()
    header.append(f" {'ID':<36s}  ", style="bold")
    header.append(f"{'NAME':<{name_w}s}  ", style="bold")
    header.append(f"{'STATUS':<10s}  ", style="bold")
    header.append(f"{'CREATED':<16s}  ", style="bold")
    header.append(f"{'DURATION':>6s}", style="bold")

    sep = Text(f" {'─' * (inner_w - 1)}")

    rows: list[Text] = [header, sep]
    for ex in executions:
        st = _status_style(ex.status)
        usage = ex.usage
        dur = _compute_duration(ex.created_at, ex.completed_at, usage)
        rows.append(
            _format_row(
                ex.id,
                ex.name or "-",
                ex.status,
                st,
                ex.created_at.strftime("%Y-%m-%d %H:%M"),
                dur,
                name_w,
            )
        )

    console.print(
        Panel(
            Group(*rows),
            title="[bold]Executions[/bold]",
            title_align="left",
            subtitle=f"[dim]{len(executions)} execution(s)[/dim]",
            subtitle_align="right",
            border_style="blue",
            padding=(0, 0),
            width=panel_w,
        )
    )


@app.command()
def get(
    ctx: typer.Context,
    execution_id: str = typer.Argument(..., help="ID of the execution to retrieve"),
    wait: bool = typer.Option(False, "--wait", help="Wait for execution to complete"),
    timeout: int = typer.Option(
        300, "--timeout", help="Timeout in seconds when waiting"
    ),
    poll_interval: int = typer.Option(
        5, "--poll-interval", help="Seconds between status checks"
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format. Use 'json' for JSON output.",
    ),
) -> None:
    """Get execution details by ID, with optional wait."""
    client: VLMRun = ctx.obj
    output_json = output_format and output_format.lower() == "json"

    try:
        if wait:
            if not output_json:
                with Status(
                    f"Waiting for execution [cyan]{execution_id}[/cyan]...",
                    console=console,
                    spinner="dots",
                ):
                    execution = client.executions.wait(
                        execution_id, timeout=timeout, sleep=poll_interval
                    )
            else:
                execution = client.executions.wait(
                    execution_id, timeout=timeout, sleep=poll_interval
                )
        else:
            execution = client.executions.get(execution_id)

        if output_json:
            print(
                json.dumps(
                    execution.model_dump(mode="json"), indent=2, default=str
                )
            )
            return

        console.print("\nExecution Details:\n", style="white")

        details = {
            "ID": execution.id,
            "Name": execution.name,
            "Status": execution.status,
        }

        if execution.created_at:
            details["Created"] = execution.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if execution.completed_at:
            details["Completed"] = execution.completed_at.strftime("%Y-%m-%d %H:%M:%S")

        if execution.usage:
            if execution.usage.elements_processed is not None:
                details["Elements Processed"] = execution.usage.elements_processed
            if execution.usage.element_type is not None:
                details["Element Type"] = execution.usage.element_type
            if execution.usage.credits_used is not None:
                details["Credits Used"] = execution.usage.credits_used
            if execution.usage.steps is not None:
                details["Steps"] = execution.usage.steps

        dur = _compute_duration(
            execution.created_at, execution.completed_at, execution.usage
        )
        if dur != "-":
            details["Duration"] = dur

        for key, value in details.items():
            console.print(f"{key}: {value}", style="white")

        if execution.response:
            console.print("\nResponse:", style="white")
            if isinstance(execution.response, dict):
                console.print(
                    json.dumps(execution.response, indent=2, default=str),
                    style="white",
                )
            else:
                console.print(execution.response, style="white")

    except TimeoutError as e:
        console.print(f"\n[yellow]Timeout:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        raise typer.Exit(1) from e
