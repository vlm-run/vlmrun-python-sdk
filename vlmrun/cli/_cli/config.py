from pathlib import Path
from typing import Optional
import typer
import tomli
from rich import print as rprint

CONFIG_FILE = Path.home() / ".vlmrun.toml"


def get_config() -> dict:
    """Load configuration from ~/.vlmrun.toml"""
    if not CONFIG_FILE.exists():
        return {}

    try:
        with CONFIG_FILE.open("rb") as f:
            return tomli.load(f)
    except (PermissionError, OSError) as e:
        rprint(f"[red]Error reading config file:[/] {e}")
        return {}
    except tomli.TOMLDecodeError:
        rprint("[red]Error:[/] Invalid TOML format in config file")
        return {}


def save_config(config: dict) -> None:
    """Save configuration to ~/.vlmrun.toml"""
    config = {k: v for k, v in config.items() if v is not None}

    try:
        with CONFIG_FILE.open("w") as f:
            for key, value in config.items():
                escaped_value = str(value).replace('"', '\\"')
                f.write(f'{key} = "{escaped_value}"\n')
    except (PermissionError, OSError) as e:
        rprint(f"[red]Error saving config file:[/] {e}")
        raise typer.Exit(1)


app = typer.Typer(
    help="Configuration operations",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="show")
def show_config() -> None:
    """Show current configuration."""
    config = get_config()

    if not config:
        rprint("[yellow]No configuration values set[/]")
        return

    display_config = config.copy()
    if "api_key" in display_config:
        key = display_config["api_key"]
        display_config["api_key"] = f"{key[:4]}...{key[-4:]}"

    for key, value in display_config.items():
        rprint(f"{key}: {value}")


@app.command(name="set")
def set_value(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="VLM Run API key"),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="VLM Run API base URL"
    ),
) -> None:
    """Set configuration values."""
    config = get_config()

    if api_key:
        config["api_key"] = api_key
        rprint("[green]✓[/] Set API key")

    if base_url:
        config["base_url"] = base_url
        rprint(f"[green]✓[/] Set base URL: {base_url}")

    if not (api_key or base_url):
        rprint("[yellow]No values provided to set[/]")
        return

    save_config(config)


@app.command()
def unset(
    api_key: bool = typer.Option(False, "--api-key", help="Unset API key"),
    base_url: bool = typer.Option(False, "--base-url", help="Unset base URL"),
) -> None:
    """Remove configuration values."""
    config = get_config()

    if api_key and "api_key" in config:
        del config["api_key"]
        rprint("[green]✓[/] Removed API key")

    if base_url and "base_url" in config:
        del config["base_url"]
        rprint("[green]✓[/] Removed base URL")

    if not (api_key or base_url):
        rprint("[yellow]No values specified to unset[/]")
        return

    save_config(config)
