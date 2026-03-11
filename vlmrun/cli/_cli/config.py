from pathlib import Path
from typing import Optional
import sys
import typer
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text
from pydantic import Field
from pydantic.dataclasses import dataclass
from dataclasses import asdict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_DIR = Path.home() / ".vlmrun"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_DEFAULT_CONFIG_TEMPLATE = """\
# VLM Run CLI configuration
# Docs: https://docs.vlm.run

# Your VLM Run API key (required)
# Get one at: https://app.vlm.run/dashboard/settings/api-keys
api_key = ""

# API base URL (optional, defaults to https://api.vlm.run/v1)
# base_url = "https://api.vlm.run/v1"
"""


@dataclass
class Config:
    """VLM Run configuration."""

    api_key: Optional[str] = Field(default=None, description="VLM Run API key")
    base_url: Optional[str] = Field(default=None, description="VLM Run API base URL")


def get_config() -> Config:
    """Load configuration from ~/.vlmrun/config.toml.

    Empty string values are treated as unset (None).
    """
    if not CONFIG_FILE.exists():
        return Config()

    try:
        with CONFIG_FILE.open("rb") as f:
            data = tomllib.load(f)
            filtered = {k: v for k, v in data.items() if v}
            return Config(**filtered)
    except (PermissionError, OSError) as e:
        rprint(f"[red]Error reading config file:[/] {e}")
        return Config()
    except tomllib.TOMLDecodeError:
        rprint("[red]Error:[/] Invalid TOML format in config file")
        return Config()


def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
) -> Config:
    """Resolve configuration with priority: CLI flag > env var > TOML > default.

    Typer already handles CLI flag > env var, so `api_key` / `base_url` here
    are the result of that first resolution.  This function layers in the
    TOML file as the final fallback.
    """
    toml_cfg = get_config()
    return Config(
        api_key=api_key or toml_cfg.api_key,
        base_url=base_url or toml_cfg.base_url,
    )


def save_config(config: Config) -> None:
    """Save configuration to ~/.vlmrun/config.toml"""
    config_dict = {k: v for k, v in asdict(config).items() if v is not None}

    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with CONFIG_FILE.open("w") as f:
            for key, value in config_dict.items():
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


@app.command(name="init")
def init_config(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config file"
    ),
) -> None:
    """Initialize a default configuration file at ~/.vlmrun/config.toml."""
    if CONFIG_FILE.exists() and not force:
        rprint(f"[yellow]Config file already exists:[/] {CONFIG_FILE}")
        rprint("Use [green]--force[/] to overwrite it.")
        raise typer.Exit(1)

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(_DEFAULT_CONFIG_TEMPLATE)
    except (PermissionError, OSError) as e:
        rprint(f"[red]Error writing config file:[/] {e}")
        raise typer.Exit(1)

    rprint(f"[green]✓[/] Config file created at [bold]{CONFIG_FILE}[/]\n")
    rprint(
        Panel(
            Text.from_markup(
                ""
                "1. Create an API key at:\n"
                "   [blue link=https://app.vlm.run/dashboard/settings/api-keys]"
                "https://app.vlm.run/dashboard/settings/api-keys[/]\n\n"
                "2. Add your key to the config:\n"
                "   [green]vlmrun config set --api-key 'your-api-key'[/]\n\n"
                "   Or edit the file directly:\n"
                f"   [green]{CONFIG_FILE}[/]"
            ),
            title="Getting Started",
            title_align="left",
            border_style="blue",
        )
    )


@app.command(name="show")
def show_config() -> None:
    """Show current configuration."""
    config = get_config()

    if not any(getattr(config, field) for field in asdict(config).keys()):
        rprint("[yellow]No configuration values set[/]")
        return

    display_config = asdict(config)
    if config.api_key:
        key = config.api_key
        display_config["api_key"] = f"{key[:4]}...{key[-4:]}"

    for key, value in display_config.items():
        if value is not None:
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
        config.api_key = api_key
        rprint("[green]✓[/] Set API key")

    if base_url:
        config.base_url = base_url
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

    if api_key and config.api_key:
        config.api_key = None
        rprint("[green]✓[/] Removed API key")

    if base_url and config.base_url:
        config.base_url = None
        rprint("[green]✓[/] Removed base URL")

    if not (api_key or base_url):
        rprint("[yellow]No values specified to unset[/]")
        return

    save_config(config)
