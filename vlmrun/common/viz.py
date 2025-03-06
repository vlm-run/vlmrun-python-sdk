from typing import Union, List, Dict, Any, Optional, Tuple, Literal
from PIL import Image
import pandas as pd
from IPython.display import HTML
import json
from pydantic import BaseModel
import cv2
import numpy as np
import io
import base64
from pathlib import Path

DEFAULT_BOX_COLOR = (255, 0, 0)
DEFAULT_BOX_THICKNESS = 2
DEFAULT_IMAGE_FORMAT = "PNG"
VALID_RENDER_TYPES = ["default", "bboxes"]
DEFAULT_IMAGE_WIDTH = 800

EXCLUDED_FIELDS = {
    "vector",
    "image_uri",
    "embedding",
    "features",
    "image_bytes",
    "raw_bytes",
    "binary_data",
    "tensor",
}

Coordinates4 = Tuple[float, float, float, float]
BoundingBox = Coordinates4  # (x1, y1, x2, y2)
XYWHBox = Coordinates4  # (x, y, width, height)
ImageType = Union[str, Path, Image.Image]
ResultType = Union[Dict[str, Any], BaseModel]
ImageInfoType = Dict[str, Any]


class DisplayOptions(BaseModel):
    """Configuration options for result display."""

    render_type: str = "default"
    image_width: Optional[int] = None
    fields: Optional[List[str]] = None
    as_json: bool = False
    limit: Optional[int] = None
    box_color: Tuple[int, int, int] = DEFAULT_BOX_COLOR
    box_thickness: int = DEFAULT_BOX_THICKNESS
    table_style: Optional[str] = None
    show_content: bool = False
    show_confidence: bool = False

    @property
    def is_valid_render_type(self) -> bool:
        return self.render_type in VALID_RENDER_TYPES

    def validate_image_width(self) -> None:
        if self.image_width is not None and (
            not isinstance(self.image_width, int) or self.image_width <= 0
        ):
            raise ValueError("image_width must be a positive integer")


def xywh_to_xyxy(box: XYWHBox) -> BoundingBox:
    """Convert bounding box from (x, y, width, height) to (x1, y1, x2, y2) format.

    Args:
        box: Tuple of (x, y, width, height) coordinates

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    x, y, w, h = box
    return (x, y, x + w, y + h)


def extract_bbox(value: Union[Dict, List]) -> Optional[BoundingBox]:
    """Extract bounding box from various formats.

    Handles:
    - Direct bbox list/tuple
    - Dict with 'bbox' key
    - Dict with 'xywh' key
    - Dict with 'bbox' containing nested 'xywh'

    Args:
        value: Value containing bounding box information

    Returns:
        Bounding box as (x1, y1, x2, y2) or None if no valid bbox found
    """
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(value)

    if isinstance(value, dict):
        if "bbox" in value:
            bbox = value["bbox"]
            if isinstance(bbox, dict) and "xywh" in bbox:
                return xywh_to_xyxy(tuple(bbox["xywh"]))
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return tuple(bbox)
        elif "xywh" in value:
            return xywh_to_xyxy(tuple(value["xywh"]))

    return None


def get_boxes_from_response(response: Union[Dict, Any]) -> List[Dict[str, BoundingBox]]:
    """Extract bounding boxes from VLM Run response.

    Handles various response formats including:
    - Direct bounding_boxes/boxes list
    - Metadata fields containing bounding boxes
    - Both xyxy and xywh formats
    - Nested metadata fields (e.g. in address object)

    Args:
        response: Raw response dictionary or object with response attribute

    Returns:
        List of dictionaries containing bounding box coordinates in (x1, y1, x2, y2) format
    """
    if hasattr(response, "response"):
        response = response.response

    if not isinstance(response, dict):
        return []

    boxes = []

    def process_metadata(metadata: Dict, field_name: str) -> None:
        if isinstance(metadata, dict):
            bbox = extract_bbox(metadata)
            if bbox:
                boxes.append(
                    {
                        "bbox": bbox,
                        "field": field_name,
                        "content": metadata.get("bbox_content"),
                        "confidence": metadata.get("confidence"),
                    }
                )

    def process_dict(d: Dict, prefix: str = "") -> None:
        for key, value in d.items():
            if key.endswith("_metadata"):
                field = key.replace("_metadata", "")
                field = field if not prefix else f"{prefix}.{field}"
                process_metadata(value, field)
            elif isinstance(value, dict):
                process_dict(value, key)

    # Handle direct bounding boxes
    for key in ["bounding_boxes", "boxes"]:
        if key in response and isinstance(response[key], list):
            for box in response[key]:
                bbox = extract_bbox(box)
                if bbox:
                    boxes.append({"bbox": bbox})

    if "bbox" in response:
        bbox = extract_bbox(response["bbox"])
        if bbox:
            boxes.append({"bbox": bbox})

    # Process all metadata fields, including nested ones
    process_dict(response)

    return boxes


def ensure_image(image: ImageType) -> Image.Image:
    """Ensure input is converted to PIL Image.

    Args:
        image: Input image as string path, Path object, or PIL Image

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    if isinstance(image, (str, Path)):
        try:
            return Image.open(image)
        except Exception as e:
            raise ValueError(f"Failed to load image from path: {e}")

    if isinstance(image, Image.Image):
        return image

    raise ValueError("Image must be a path string, Path object, or PIL Image")


def to_dict(obj: Union[Dict, BaseModel]) -> Dict:
    """Convert object to dictionary.

    Args:
        obj: Dictionary or Pydantic model

    Returns:
        Dictionary representation
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def render_bbox_image(
    image: ImageType,
    response: Union[Dict, BaseModel, Any],
    width: Optional[int] = None,
    return_base64: bool = False,
    box_color: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
    box_thickness: int = DEFAULT_BOX_THICKNESS,
    show_content: bool = False,
    show_confidence: bool = False,
) -> Union[str, Image.Image]:
    """Render image with bounding boxes from VLM Run response.

    Args:
        image: Input image (path or PIL Image)
        response: VLM Run response (can be raw dict, Pydantic model, or response object)
        width: Optional width to resize image
        return_base64: If True, returns HTML img tag with base64 data
        box_color: BGR color tuple for bounding boxes
        box_thickness: Thickness of bounding box lines
        show_content: Whether to show bbox_content as label
        show_confidence: Whether to show confidence scores

    Returns:
        Either HTML string (if return_base64=True) or PIL Image with boxes

    Raises:
        ValueError: If image is invalid or bounding box coordinates are invalid
    """
    image = ensure_image(image)
    response_dict = to_dict(response)
    boxes = get_boxes_from_response(response_dict)

    # Convert PIL to cv2 image
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_height, img_width = img.shape[:2]

    # Draw boxes
    for box in boxes:
        try:
            x1, y1, x2, y2 = box["bbox"]
            if not all(isinstance(coord, (int, float)) for coord in (x1, y1, x2, y2)):
                raise ValueError(f"Invalid box coordinates: {box['bbox']}")

            # Convert normalized coordinates to pixel coordinates
            x1_px = int(x1 * img_width)
            y1_px = int(y1 * img_height)
            x2_px = int(x2 * img_width)
            y2_px = int(y2 * img_height)

            # Draw rectangle
            cv2.rectangle(img, (x1_px, y1_px), (x2_px, y2_px), box_color, box_thickness)

            # Draw label if enabled and available
            if (show_content and "content" in box and box["content"]) or (
                show_confidence and "confidence" in box
            ):
                # Build label text
                label_parts = []
                if show_content and box.get("content"):
                    label_parts.append(box["content"])
                if show_confidence and box.get("confidence") is not None:
                    label_parts.append(f"({box['confidence']:.2f})")

                if label_parts:  # Only draw if we have something to show
                    label = " ".join(label_parts)

                    # Get text size and calculate background rectangle
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )

                    # Draw background rectangle for text
                    margin = 2
                    cv2.rectangle(
                        img,
                        (x1_px, y1_px - text_height - 2 * margin),
                        (x1_px + text_width + 2 * margin, y1_px),
                        box_color,
                        -1,
                    )  # Filled rectangle

                    # Draw text
                    cv2.putText(
                        img,
                        label,
                        (x1_px + margin, y1_px - margin),
                        font,
                        font_scale,
                        (0, 0, 0),  # Black text
                        thickness,
                    )

        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid bounding box format: {e}")

    # Convert back to PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    if width:
        if not isinstance(width, int) or width <= 0:
            raise ValueError("Width must be a positive integer")
        ratio = width / img.width
        height = int(img.height * ratio)
        img = img.resize((width, height))

    if return_base64:
        buffer = io.BytesIO()
        img.save(buffer, format=DEFAULT_IMAGE_FORMAT)
        image_str = base64.b64encode(buffer.getvalue()).decode()
        return (
            f'<img src="data:image/{DEFAULT_IMAGE_FORMAT.lower()};base64,{image_str}"/>'
        )

    return img


def render_image(image: ImageType, width: Optional[int] = None) -> str:
    """Render image as HTML.

    Args:
        image: Input image (path or PIL Image)
        width: Optional width to resize image

    Returns:
        HTML img tag with base64-encoded image data
    """
    image = ensure_image(image)

    if width:
        if not isinstance(width, int) or width <= 0:
            raise ValueError("Width must be a positive integer")
        ratio = width / image.width
        height = int(image.height * ratio)
        image = image.resize((width, height))

    buffer = io.BytesIO()
    image.save(buffer, format=DEFAULT_IMAGE_FORMAT)
    image_str = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/{DEFAULT_IMAGE_FORMAT.lower()};base64,{image_str}"/>'


def get_nested_value(obj: Dict, path: str) -> Any:
    """Get nested dictionary values using dot notation."""
    try:
        for key in path.split("."):
            obj = obj[key]
        return obj
    except (KeyError, TypeError, AttributeError) as e:
        raise KeyError(f"Failed to get value at path '{path}': {e}")


def filter_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out large binary fields from response data.

    Args:
        data: Response data dictionary

    Returns:
        Filtered dictionary without binary/large fields
    """
    if not isinstance(data, dict):
        return data

    return {
        k: filter_response_data(v) if isinstance(v, dict) else v
        for k, v in data.items()
        if k not in EXCLUDED_FIELDS
    }


def format_json_html(data: Dict[str, Any], indent: int = 2) -> str:
    """Format JSON data as HTML with proper styling.

    Args:
        data: Data to format as JSON
        indent: Number of spaces for indentation

    Returns:
        HTML-formatted JSON string with styling
    """
    json_str = json.dumps(data, indent=indent).replace("\n", "<br>")
    return f'<pre style="margin: 0; white-space: pre-wrap;">{json_str}</pre>'


def show_results(
    result: Union[ResultType, List[ResultType]],
    image: Union[ImageType, List[ImageType]],
    image_info: Optional[Union[ImageInfoType, List[ImageInfoType]]] = None,
    *,
    render_type: Literal["default", "bboxes"] = "default",
    fields: Optional[List[str]] = None,
    image_width: Optional[int] = DEFAULT_IMAGE_WIDTH,
    as_json: bool = False,
    limit: Optional[int] = None,
    box_color: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
    box_thickness: int = DEFAULT_BOX_THICKNESS,
    table_style: Optional[str] = None,
    show_content: bool = False,
    show_confidence: bool = False,
) -> HTML:
    """Display VLM Run results with images in a tabular format.

    This function renders VLM Run results alongside their corresponding images in a
    structured table format. It supports both single and batch results, custom field
    selection, bounding box visualization, and JSON formatting.

    Args:
        result: VLM Run response or list of responses. Can be:
            - Raw dictionary from API response
            - Pydantic model instance
            - List of either of the above

        image: Image or list of images to display. Can be:
            - Path string or Path object to image file
            - PIL Image object
            - List of any of the above
            - Must match length of results

        image_info: Optional metadata for images. Can be:
            - Dictionary of metadata for single image
            - List of dictionaries for multiple images
            - Must match length of images if provided

    Keyword Args:
        render_type: Visualization mode for images. Options:
            - 'default': Simple image display
            - 'bboxes': Draws bounding boxes from response

        fields: Optional list of fields to display from response.
            - Supports nested fields using dot notation (e.g., 'address.street')
            - If None, displays all fields
            - Ignored if as_json=True

        image_width: Optional width to resize images to.
            - Maintains aspect ratio
            - Applied to all images
            - Must be a positive integer

        as_json: If True, displays full JSON response instead of individual fields.
            - JSON is properly formatted with indentation
            - Useful for debugging or detailed inspection

        limit: Maximum number of results to display.
            - If None, shows all results
            - Useful for previewing large result sets

        box_color: BGR color tuple for bounding boxes when render_type='bboxes'
            - Defaults to green (0, 255, 0)

        box_thickness: Line thickness for bounding boxes
            - Must be a positive integer

        table_style: Optional CSS styling for the output table
            - Applied to the DataFrame HTML output
            - Can be used to customize table appearance

        show_content: Whether to show bbox_content as label

        show_confidence: Whether to show confidence scores

    Returns:
        IPython.display.HTML: HTML representation of the results table.
        Can be displayed directly in Jupyter notebooks.

    Examples:
        # Basic usage with single result
        show_results(response, image)

        # Display with bounding boxes and custom style
        show_results(
            response,
            image,
            render_type='bboxes',
            image_width=600,
            box_color=(255, 0, 0),  # Red boxes
            table_style='table { border-collapse: collapse; }'
        )

        # Show specific fields with metadata
        show_results(
            response,
            image,
            image_info={'timestamp': '2024-03-14'},
            fields=['description', 'confidence']
        )

        # Display full JSON with limit
        show_results(
            responses,
            images,
            as_json=True,
            limit=5
        )
    """
    options = DisplayOptions(
        render_type=render_type,
        image_width=image_width,
        fields=fields,
        as_json=as_json,
        limit=limit,
        box_color=box_color,
        box_thickness=box_thickness,
        table_style=table_style,
        show_content=show_content,
        show_confidence=show_confidence,
    )

    # Validate options
    if not options.is_valid_render_type:
        raise ValueError(
            f"Invalid render_type: {render_type}. Must be one of {VALID_RENDER_TYPES}"
        )
    options.validate_image_width()

    # Input validation
    if result is None or image is None:
        raise ValueError("Both result and image must be provided")

    results = [result] if not isinstance(result, list) else result
    images = [image] if not isinstance(image, list) else image

    if len(results) != len(images):
        raise ValueError(
            f"Number of results ({len(results)}) must match number of images ({len(images)})"
        )

    if image_info is not None:
        info_list = [image_info] if not isinstance(image_info, list) else image_info
        if len(info_list) != len(images):
            raise ValueError(
                f"Number of image_info items ({len(info_list)}) must match number of images ({len(images)})"
            )

    if options.limit is not None:
        results = results[: options.limit]
        images = images[: options.limit]
        if image_info is not None:
            info_list = info_list[: options.limit]

    def to_dict(obj: Union[Dict, BaseModel]) -> Dict:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return obj

    results = [filter_response_data(to_dict(r)) for r in results]

    if options.as_json:
        data = [
            {
                "Image": (
                    render_bbox_image(
                        img,
                        res,
                        width=options.image_width,
                        return_base64=True,
                        box_color=options.box_color,
                        box_thickness=options.box_thickness,
                        show_content=options.show_content,
                        show_confidence=options.show_confidence,
                    )
                    if options.render_type == "bboxes"
                    else render_image(img, width=options.image_width)
                ),
                "Response": format_json_html(res),
            }
            for img, res in zip(images, results)
        ]

        if image_info:
            for d, info in zip(data, info_list):
                d["Image Info"] = format_json_html(info)

    else:
        data = []
        for idx, (img, res) in enumerate(zip(images, results)):
            row = {
                "Image": (
                    render_bbox_image(
                        img,
                        res,
                        width=options.image_width,
                        return_base64=True,
                        box_color=options.box_color,
                        box_thickness=options.box_thickness,
                        show_content=options.show_content,
                        show_confidence=options.show_confidence,
                    )
                    if options.render_type == "bboxes"
                    else render_image(img, width=options.image_width)
                )
            }

            if options.fields:
                # Use provided fields
                for field in options.fields:
                    try:
                        row[field] = get_nested_value(res, field)
                    except KeyError:
                        row[field] = None
            else:
                filtered_res = {
                    k: v for k, v in res.items() if k not in EXCLUDED_FIELDS
                }
                row.update(filtered_res)

            if image_info:
                row["Image Info"] = format_json_html(info_list[idx])

            data.append(row)

    df = pd.DataFrame(data)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)

    default_style = """
        table { border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; border: 1px solid #ddd; }
        th { background-color: #f5f5f5; }
        pre { max-height: 400px; overflow-y: auto; }
    """
    html_output = df.to_html(escape=False)
    style = options.table_style if options.table_style else default_style
    html_output = f"<style>{style}</style>{html_output}"

    return HTML(html_output)
