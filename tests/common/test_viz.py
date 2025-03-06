import pytest
from PIL import Image
from vlmrun.common.viz import (
    DisplayOptions,
    xywh_to_xyxy,
    extract_bbox,
    get_boxes_from_response,
    ensure_image,
    render_bbox_image,
    render_image,
    get_nested_value,
    filter_response_data,
    format_json_html,
    show_results,
)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new("RGB", (100, 100), color="white")
    return img


@pytest.fixture
def sample_response():
    """Create a sample response with various bounding box formats."""
    return {
        "bounding_boxes": [
            {"bbox": [0.1, 0.2, 0.3, 0.4]},
            {"bbox": {"xywh": [0.1, 0.2, 0.2, 0.2]}},
        ],
        "boxes": [
            {"bbox": [0.5, 0.5, 0.7, 0.8]},
        ],
        "object_metadata": {
            "bbox": [0.2, 0.3, 0.4, 0.5],
            "bbox_content": "Object",
            "confidence": 0.95,
        },
    }


def test_display_options_validation():
    """Test DisplayOptions validation."""
    opts = DisplayOptions(render_type="default")
    assert opts.is_valid_render_type

    opts = DisplayOptions(render_type="bboxes")
    assert opts.is_valid_render_type

    opts = DisplayOptions(render_type="invalid")
    assert not opts.is_valid_render_type

    opts = DisplayOptions(image_width=100)
    opts.validate_image_width()

    with pytest.raises(ValueError):
        DisplayOptions(image_width=-1).validate_image_width()

    with pytest.raises(ValueError):
        DisplayOptions(image_width=0).validate_image_width()


def test_xywh_to_xyxy():
    """Test conversion from XYWH to XYXY format."""
    xywh = (0.1, 0.2, 0.3, 0.4)
    xyxy = xywh_to_xyxy(xywh)
    expected = (0.1, 0.2, 0.4, 0.6)
    assert all(pytest.approx(a) == b for a, b in zip(xyxy, expected))

    xywh = (10, 20, 30, 40)
    xyxy = xywh_to_xyxy(xywh)
    expected = (10, 20, 40, 60)
    assert all(pytest.approx(a) == b for a, b in zip(xyxy, expected))


def test_extract_bbox():
    """Test bounding box extraction from various formats."""
    result = extract_bbox([0.1, 0.2, 0.3, 0.4])
    expected = (0.1, 0.2, 0.3, 0.4)
    assert all(pytest.approx(a) == b for a, b in zip(result, expected))

    result = extract_bbox({"bbox": [0.1, 0.2, 0.3, 0.4]})
    expected = (0.1, 0.2, 0.3, 0.4)
    assert all(pytest.approx(a) == b for a, b in zip(result, expected))

    result = extract_bbox({"xywh": [0.1, 0.2, 0.3, 0.4]})
    expected = (0.1, 0.2, 0.4, 0.6)
    assert all(pytest.approx(a) == b for a, b in zip(result, expected))

    result = extract_bbox({"bbox": {"xywh": [0.1, 0.2, 0.3, 0.4]}})
    expected = (0.1, 0.2, 0.4, 0.6)
    assert all(pytest.approx(a) == b for a, b in zip(result, expected))

    assert extract_bbox({"invalid": "format"}) is None
    assert extract_bbox([1, 2, 3]) is None


def test_get_boxes_from_response(sample_response):
    """Test extraction of bounding boxes from response."""
    boxes = get_boxes_from_response(sample_response)
    assert len(boxes) == 4

    expected = (0.1, 0.2, 0.3, 0.4)
    assert all(pytest.approx(a) == b for a, b in zip(boxes[0]["bbox"], expected))

    expected = (0.1, 0.2, 0.3, 0.4)
    assert all(pytest.approx(a) == b for a, b in zip(boxes[1]["bbox"], expected))

    expected = (0.5, 0.5, 0.7, 0.8)
    assert all(pytest.approx(a) == b for a, b in zip(boxes[2]["bbox"], expected))

    expected = (0.2, 0.3, 0.4, 0.5)
    assert all(pytest.approx(a) == b for a, b in zip(boxes[3]["bbox"], expected))
    assert boxes[3]["field"] == "object"
    assert boxes[3]["content"] == "Object"
    assert pytest.approx(boxes[3]["confidence"]) == 0.95


def test_ensure_image(sample_image, tmp_path):
    """Test image loading and validation."""
    assert ensure_image(sample_image) == sample_image

    img_path = tmp_path / "test.png"
    sample_image.save(img_path)
    loaded_img = ensure_image(img_path)
    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size == (100, 100)

    with pytest.raises(ValueError):
        ensure_image(tmp_path / "nonexistent.png")

    with pytest.raises(ValueError):
        ensure_image(123)


def test_render_bbox_image(sample_image, sample_response):
    """Test bounding box rendering on image."""
    result = render_bbox_image(sample_image, sample_response)
    assert isinstance(result, Image.Image)

    result = render_bbox_image(sample_image, sample_response, return_base64=True)
    assert isinstance(result, str)
    assert result.startswith('<img src="data:image/png;base64,')

    result = render_bbox_image(
        sample_image, sample_response, show_content=True, show_confidence=True
    )
    assert isinstance(result, Image.Image)


def test_render_image(sample_image):
    """Test image rendering to HTML."""
    result = render_image(sample_image)
    assert isinstance(result, str)
    assert result.startswith('<img src="data:image/png;base64,')

    result = render_image(sample_image, width=50)
    assert isinstance(result, str)
    assert result.startswith('<img src="data:image/png;base64,')

    with pytest.raises(ValueError):
        render_image(sample_image, width=-1)


def test_get_nested_value():
    """Test nested dictionary value retrieval."""
    data = {"a": {"b": {"c": 1}}}

    assert get_nested_value(data, "a.b.c") == 1

    with pytest.raises(KeyError):
        get_nested_value(data, "a.b.d")

    with pytest.raises(KeyError):
        get_nested_value(data, "x.y.z")


def test_filter_response_data():
    """Test filtering of response data."""
    data = {
        "text": "hello",
        "vector": [1, 2, 3],
        "nested": {"text": "world", "embedding": [4, 5, 6]},
    }

    filtered = filter_response_data(data)
    assert "text" in filtered
    assert "vector" not in filtered
    assert "nested" in filtered
    assert "text" in filtered["nested"]
    assert "embedding" not in filtered["nested"]


def test_format_json_html():
    """Test JSON formatting to HTML."""
    data = {"a": 1, "b": [2, 3]}
    result = format_json_html(data)
    assert isinstance(result, str)
    assert result.startswith("<pre")
    assert "style=" in result
    assert '"a": 1' in result
    assert '"b": [' in result


def test_show_results(sample_image, sample_response):
    """Test full results display functionality."""
    result = show_results(sample_response, sample_image)
    assert result is not None

    result = show_results(
        sample_response,
        sample_image,
        render_type="bboxes",
        show_content=True,
        show_confidence=True,
    )
    assert result is not None

    result = show_results(
        [sample_response, sample_response], [sample_image, sample_image]
    )
    assert result is not None

    result = show_results(
        sample_response, sample_image, image_info={"timestamp": "2024-03-14"}
    )
    assert result is not None

    result = show_results(sample_response, sample_image, fields=["text", "confidence"])
    assert result is not None

    with pytest.raises(ValueError):
        show_results(None, sample_image)

    with pytest.raises(ValueError):
        show_results(sample_response, None)

    with pytest.raises(ValueError):
        show_results([sample_response], [sample_image, sample_image])
