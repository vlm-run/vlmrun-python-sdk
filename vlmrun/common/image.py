"""Image utilities for VLMRun."""

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Literal, Union

from PIL import Image

from vlmrun.constants import SUPPORTED_VIDEO_FILETYPES


def encode_video(path: Union[Path, str]) -> str:
    """Convert a video file to a base64 string with data URI prefix.

    Args:
        path: Path to video file

    Returns:
        Base64 encoded string with data URI prefix

    Raises:
        FileNotFoundError: If video file doesn't exist
        IOError: If file is not a video or too large
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_FILETYPES:
        raise IOError(f"File is not a video: {path}")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise IOError(
            f"File is too large, only videos up to 50MB are supported. [size={path.stat().st_size/1024/1024:.2f}MB]"
        )

    # Read the video file
    with open(path, "rb") as f:
        video_bytes = f.read()
        video_b64 = b64encode(video_bytes).decode()
        # Get video mime type from extension
        mime_type = f"video/{path.suffix.lower()[1:]}"
        return f"data:{mime_type};base64,{video_b64}"


def encode_image(
    image: Union[Image.Image, str, Path],
    format: Literal["PNG", "JPEG", "binary"] = "PNG",
    quality: int = 90,
) -> Union[str, bytes]:
    """Convert an image to a base64 string or binary format.

    Args:
        image: PIL Image, path to image, or Path object
        format: Output format ("PNG", "JPEG", or "binary")
        quality: JPEG quality (1-100, default 90). Only used for JPEG format.

    Returns:
        Base64 encoded string or binary bytes

    Raises:
        FileNotFoundError: If image path doesn't exist
        ValueError: If image type is invalid or quality is out of range
    """
    if not (1 <= quality <= 100):
        raise ValueError(f"Quality must be between 1 and 100, got {quality}")

    if isinstance(image, (str, Path)):
        if not Path(image).exists():
            raise FileNotFoundError(f"File not found {image}")
        image = Image.open(str(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image type: {type(image)}")

    buffered = BytesIO()
    if format == "binary":
        image.save(buffered, format="PNG")
        buffered.seek(0)
        return buffered.getvalue()

    image_format = image.format or format
    if image_format.upper() not in ["PNG", "JPEG"]:
        raise ValueError(f"Unsupported format: {image_format}")

    save_params = {
        "format": image_format,
        **(
            {"quality": quality, "subsampling": 0}
            if image_format.upper() == "JPEG"
            else {}
        ),
    }

    try:
        image.save(buffered, **save_params)
        img_str = b64encode(buffered.getvalue()).decode()
        return f"data:image/{image_format.lower()};base64,{img_str}"
    except Exception as e:
        raise ValueError(f"Failed to save image in {image_format} format") from e
