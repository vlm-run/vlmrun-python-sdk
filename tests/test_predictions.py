"""Tests for predictions operations."""

import pytest
from pydantic import BaseModel
from PIL import Image
from vlmrun.client.types import PredictionResponse, GenerationConfig, SchemaResponse


class MockInvoiceSchema(BaseModel):
    """Mock invoice schema for testing."""

    invoice_number: str
    total_amount: float


def test_list_predictions(mock_client):
    """Test listing predictions."""
    client = mock_client
    response = client.predictions.list()
    assert isinstance(response, list)
    assert all(isinstance(pred, PredictionResponse) for pred in response)
    assert len(response) > 0
    assert response[0].id == "prediction1"
    assert response[0].status == "running"


def test_get_prediction(mock_client):
    """Test getting prediction by ID."""
    client = mock_client
    response = client.predictions.get("prediction1")
    assert isinstance(response, PredictionResponse)
    assert response.id == "prediction1"
    assert response.status == "running"


def test_wait_prediction(mock_client):
    """Test waiting for prediction completion."""
    client = mock_client
    response = client.predictions.wait("prediction1", timeout=1)
    assert isinstance(response, PredictionResponse)
    assert response.id == "prediction1"
    assert response.status == "completed"


def test_image_generate(mock_client, tmp_path):
    """Test generating image prediction with local file."""
    # Create a dummy image for testing
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    client = mock_client
    response = client.image.generate(
        domain="test-domain",
        images=[img_path],
    )
    assert isinstance(response, PredictionResponse)
    assert response.id is not None


def test_image_generate_with_url(mock_client):
    """Test generating predictions with image URL."""
    client = mock_client
    response = client.image.generate(
        domain="test-domain",
        urls=["https://example.com/image.jpg"],
    )
    assert isinstance(response, PredictionResponse)
    assert response.id is not None


def test_image_generate_validation(mock_client):
    """Test validation of image generate parameters."""
    client = mock_client

    # Test missing both images and urls
    with pytest.raises(ValueError, match="Either `images` or `urls` must be provided"):
        client.image.generate(domain="test-domain")

    # Test providing both images and urls
    img = Image.new("RGB", (100, 100), color="red")
    with pytest.raises(
        ValueError, match="Only one of `images` or `urls` can be provided"
    ):
        client.image.generate(
            domain="test-domain",
            images=[img],
            urls=["https://example.com/image.jpg"],
        )

    # Test empty urls list
    with pytest.raises(ValueError, match="Either `images` or `urls` must be provided"):
        client.image.generate(domain="test-domain", urls=[])


def test_document_generate(mock_client, tmp_path):
    """Test generating document prediction."""
    doc_path = tmp_path / "test.pdf"
    doc_path.write_bytes(b"test content")

    client = mock_client
    response = client.document.generate(
        file_or_url=doc_path,
        model="test-model",
        domain="test-domain",
        json_schema={"type": "object"},
    )
    assert isinstance(response, PredictionResponse)


def test_video_generate(mock_client, tmp_path):
    """Test generating video prediction."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"test content")

    client = mock_client
    response = client.video.generate(
        file_or_url=video_path,
        model="test-model",
        domain="test-domain",
        json_schema={"type": "object"},
    )
    assert isinstance(response, PredictionResponse)


def test_audio_generate(mock_client, tmp_path):
    """Test generating audio prediction."""
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"test content")

    client = mock_client
    response = client.audio.generate(
        file_or_url=audio_path,
        model="test-model",
        domain="test-domain",
        json_schema={"type": "object"},
    )
    assert isinstance(response, PredictionResponse)


def test_schema_casting_with_domain(mock_client):
    """Test response casting using domain schema."""

    def mock_get_schema(*args, **kwargs):
        return SchemaResponse(
            domain="document.invoice",
            schema_version="1.0.0",
            schema_hash="1234567890",
            gql_stmt="",
            json_schema=MockInvoiceSchema.model_json_schema(),
        )

    mock_client.get_schema = mock_get_schema

    response = mock_client.image.generate(
        domain="document.invoice", urls=["https://example.com/test.jpg"]
    )
    assert isinstance(response.response, BaseModel)


def test_schema_casting_with_custom_schema(mock_client):
    """Test response casting using custom schema from GenerationConfig."""
    response = mock_client.image.generate(
        domain="document.invoice",
        urls=["https://example.com/test.jpg"],
        config=GenerationConfig(json_schema=MockInvoiceSchema.model_json_schema()),
    )

    assert response.response.invoice_number == "INV-001"
    assert response.response.total_amount == 100.0


@pytest.mark.parametrize("prediction_type", ["image", "document", "video", "audio"])
def test_schema_casting_across_prediction_types(mock_client, prediction_type):
    """Test schema casting works consistently across different prediction types."""

    def mock_get_schema(domain):
        return SchemaResponse(
            domain="document.invoice",
            schema_version="1.0.0",
            schema_hash="1234567890",
            gql_stmt="",
            json_schema=MockInvoiceSchema.model_json_schema(),
        )

    mock_client.get_schema = mock_get_schema

    pred_client = getattr(mock_client, prediction_type)

    if prediction_type == "image":
        response = pred_client.generate(
            domain="document.invoice", urls=["https://example.com/test.jpg"]
        )
    else:
        response = pred_client.generate(
            domain="document.invoice", url="https://example.com/test.file"
        )

    assert isinstance(response.response, BaseModel)
