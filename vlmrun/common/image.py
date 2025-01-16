"""Image utilities for VLMRun."""

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Literal, Union

from PIL import Image


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
