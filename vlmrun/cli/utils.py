"""Utility functions for the CLI."""
import typer
from vlmrun.client import Client

def get_context_client(ctx: typer.Context) -> Client:
    """Get the client from the context."""
    return ctx.obj
