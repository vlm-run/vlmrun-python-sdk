import typer
from typing import List
from vlmrun.cli.utils import get_context_client

app = typer.Typer(help="Dataset operations")

@app.command()
def generate(
    ctx: typer.Context,
    domain: str = typer.Argument(..., help="Domain for dataset"),
    urls: List[str] = typer.Option(..., help="List of URLs"),
    url_type: str = typer.Option(
        ...,
        help="Type of URLs ('yt_playlist' or 'yt_video')",
        case_sensitive=False
    ),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_format: str = typer.Option(
        "json",
        help="Format of the dataset ('json' or 'messages')"
    ),
    max_frames_per_video: int = typer.Option(
        ...,
        help="Maximum number of frames to generate per video",
        min=1
    ),
    max_samples: int = typer.Option(
        ...,
        help="Maximum number of samples to generate (between 10 and 100,000)",
        min=10,
        max=100_000
    )
) -> None:
    """
    Generate a dataset by calling /v1/dataset/generate.

    This command allows you to generate datasets from YouTube playlists or videos.
    The generated dataset will be saved according to the specified format and parameters.
    """
    client = get_context_client(ctx)
    
    # Validate url_type
    valid_url_types = ["yt_playlist", "yt_video"]
    if url_type.lower() not in [t.lower() for t in valid_url_types]:
        typer.echo(f"Error: url_type must be one of {valid_url_types}", err=True)
        raise typer.Exit(1)
    
    # Make the request to the endpoint
    response = client.api_request(
        method="POST",
        path="/v1/dataset/generate",
        json={
            "domain": domain,
            "urls": urls,
            "url_type": url_type,
            "dataset_name": dataset_name,
            "dataset_format": dataset_format,
            "max_frames_per_video": max_frames_per_video,
            "max_samples": max_samples
        }
    )
    
    # Extract dataset_uri from response and display it
    if isinstance(response, dict) and "dataset_uri" in response:
        typer.echo(f"Dataset generated successfully!")
        typer.echo(f"Dataset URI: {response['dataset_uri']}")
        if "elapsed_s" in response:
            typer.echo(f"Generation time: {response['elapsed_s']:.2f} seconds")
    else:
        typer.echo(response)
