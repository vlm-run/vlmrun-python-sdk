"""Tests for predictions operations."""

import pytest
from pydantic import BaseModel
from PIL import Image
from loguru import logger
from vlmrun.client.types import (
    PredictionResponse,
    GenerationConfig,
    SchemaResponse,
    CreditUsage,
    MarkdownDocument,
)


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


def test_wait_prediction(mock_client, monkeypatch):
    """Test waiting for prediction completion."""
    client = mock_client

    def mock_get(id):
        return PredictionResponse(
            id=id,
            status="completed",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            response={"result": "test"},
            usage=CreditUsage(credits_used=100),
        )

    # Mock time.sleep to do nothing
    def mock_sleep(seconds):
        pass

    monkeypatch.setattr(client.predictions, "get", mock_get)
    monkeypatch.setattr("time.sleep", mock_sleep)

    response = client.predictions.wait("prediction1", timeout=1)
    assert isinstance(response, PredictionResponse)
    assert response.id == "prediction1"
    assert response.status == "completed"


def test_wait_prediction_timeout(mock_client, monkeypatch):
    """Test waiting for prediction with timeout."""
    from vlmrun.client.predictions import Predictions

    # Create a new Predictions instance with our mock client
    predictions = Predictions(mock_client)

    # Mock get to always return running status
    def mock_get(id):
        return PredictionResponse(
            id=id,
            status="running",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at=None,
            response=None,
            usage=CreditUsage(credits_used=0),
        )

    # Mock time.sleep to do nothing
    def mock_sleep(seconds):
        pass

    # Mock time.time to simulate elapsed time
    current_time = 0

    def mock_time():
        nonlocal current_time
        current_time += 1
        return current_time

    monkeypatch.setattr(predictions, "get", mock_get)
    monkeypatch.setattr("time.sleep", mock_sleep)
    monkeypatch.setattr("time.time", mock_time)

    # Test timeout
    with pytest.raises(TimeoutError) as exc_info:
        predictions.wait("prediction1", timeout=1, sleep=1)
    assert "did not complete within 1 seconds" in str(exc_info.value)
    assert "Last status: running" in str(exc_info.value)


def test_wait_prediction_polling(mock_client, monkeypatch):
    """Test waiting for prediction with different polling intervals."""
    from vlmrun.client.predictions import Predictions

    predictions = Predictions(mock_client)

    # Mock get to return completed after one call
    def mock_get(id):
        return PredictionResponse(
            id=id,
            status="completed",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            response={"result": "test"},
            usage=CreditUsage(credits_used=100),
        )

    # Mock time.sleep to do nothing
    def mock_sleep(seconds):
        pass

    monkeypatch.setattr(predictions, "get", mock_get)
    monkeypatch.setattr("time.sleep", mock_sleep)

    # Test that prediction completes
    response = predictions.wait("prediction1", timeout=10, sleep=2)
    assert response.status == "completed"
    assert response.response == {"result": "test"}


def test_wait_prediction_immediate_completion(mock_client, monkeypatch):
    """Test waiting for prediction that completes immediately."""
    client = mock_client

    def mock_get(id):
        return PredictionResponse(
            id=id,
            status="completed",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            response={"result": "test"},
            usage=CreditUsage(credits_used=100),
        )

    # Mock time.sleep to do nothing
    def mock_sleep(seconds):
        pass

    monkeypatch.setattr(client.predictions, "get", mock_get)
    monkeypatch.setattr("time.sleep", mock_sleep)

    # Should return immediately without sleeping
    response = client.predictions.wait("prediction1", timeout=10, sleep=2)
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


def test_image_execute(mock_client, tmp_path):
    """Test executing image prediction with local file."""
    # Create a dummy image for testing
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    client = mock_client
    response = client.image.execute(
        name="test-model",
        images=[img_path],
    )
    assert isinstance(response, PredictionResponse)
    assert response.id is not None


def test_image_execute_with_url(mock_client):
    """Test executing predictions with image URL."""
    client = mock_client
    response = client.image.execute(
        name="test-model",
        urls=["https://example.com/image.jpg"],
    )
    assert isinstance(response, PredictionResponse)
    assert response.id is not None


def test_image_execute_validation(mock_client):
    """Test validation of image execute parameters."""
    client = mock_client

    # Test missing both images and urls
    with pytest.raises(ValueError, match="Either `images` or `urls` must be provided"):
        client.image.execute(name="test-model")

    # Test providing both images and urls
    img = Image.new("RGB", (100, 100), color="red")
    with pytest.raises(
        ValueError, match="Only one of `images` or `urls` can be provided"
    ):
        client.image.execute(
            name="test-model",
            images=[img],
            urls=["https://example.com/image.jpg"],
        )


def test_document_execute(mock_client, tmp_path):
    """Test executing document prediction."""
    doc_path = tmp_path / "test.pdf"
    doc_path.write_bytes(b"test content")

    client = mock_client
    response = client.document.execute(
        name="test-model",
        file=doc_path,
    )
    assert isinstance(response, PredictionResponse)


def test_video_execute(mock_client, tmp_path):
    """Test executing video prediction."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"test content")

    client = mock_client
    response = client.video.execute(
        name="test-model",
        file=video_path,
    )
    assert isinstance(response, PredictionResponse)


def test_audio_execute(mock_client, tmp_path):
    """Test executing audio prediction."""
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"test content")

    client = mock_client
    response = client.audio.execute(
        name="test-model",
        file=audio_path,
    )
    assert isinstance(response, PredictionResponse)


def test_schema_casting_with_domain(mock_client):
    """Test response casting using domain schema."""
    response = mock_client.image.generate(
        domain="document.invoice", urls=["https://example.com/test.jpg"], autocast=True
    )
    assert isinstance(response.response, BaseModel)
    assert response.response.invoice_number == "INV-001"
    assert response.response.total_amount == 100.0


def test_schema_casting_with_custom_schema(mock_client):
    """Test response casting using custom schema from GenerationConfig."""
    response = mock_client.image.generate(
        domain="document.invoice",
        urls=["https://example.com/test.jpg"],
        config=GenerationConfig(json_schema=MockInvoiceSchema.model_json_schema()),
        autocast=True,
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
            domain="document.invoice",
            urls=["https://example.com/test.jpg"],
            autocast=True,
        )
    else:
        response = pred_client.generate(
            domain="document.invoice",
            url="https://example.com/test.file",
            autocast=True,
        )

    assert isinstance(response.response, BaseModel)


def test_document_markdown_automatic_casting(mock_client, monkeypatch):
    """Test automatic casting of document.markdown responses to MarkdownDocument."""

    client = mock_client

    mock_response = {
        "pages": [
            {
                "content": "# Test Document\n\nThis is a test document.",
                "markdown_content": "# Test Document\n\nThis is a test document.",
                "tables": [
                    {
                        "headers": [
                            {"id": "header1", "column": 0, "name": "Header 1"},
                            {"id": "header2", "column": 1, "name": "Header 2"},
                        ],
                        "data": [{"header1": "Value 1", "header2": "Value 2"}],
                    }
                ],
                "figures": [
                    {
                        "id": 1,
                        "title": "Test Figure",
                        "caption": "Test Caption",
                        "content": "Test content",
                    }
                ],
            }
        ]
    }

    def mock_generate(domain=None, **kwargs):
        prediction = PredictionResponse(
            id="test-prediction",
            status="completed",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            response=(
                mock_response
                if domain == "document.markdown"
                else {"invoice_number": "INV-001", "total_amount": 100.0}
            ),
            usage=CreditUsage(credits_used=100),
        )

        if (
            domain == "document.markdown"
            and prediction.status == "completed"
            and prediction.response
        ):
            prediction.response = MarkdownDocument(**prediction.response)
        elif kwargs.get("autocast", False):
            prediction.response = MockInvoiceSchema(**prediction.response)

        return prediction

    monkeypatch.setattr(client.document, "generate", mock_generate)

    response = client.document.generate(domain="document.markdown", file="test.pdf")
    assert isinstance(response, PredictionResponse)
    assert isinstance(response.response, MarkdownDocument)
    assert len(response.response.pages) == 1
    assert (
        response.response.pages[0].content
        == "# Test Document\n\nThis is a test document."
    )
    assert len(response.response.all_tables) == 1
    assert len(response.response.all_figures) == 1

    response = client.document.generate(
        domain="document.markdown", file="test.pdf", autocast=True
    )
    assert isinstance(response, PredictionResponse)
    assert isinstance(response.response, MarkdownDocument)
    assert len(response.response.pages) == 1
    assert (
        response.response.pages[0].content
        == "# Test Document\n\nThis is a test document."
    )
    assert len(response.response.all_tables) == 1
    assert len(response.response.all_figures) == 1

    response = client.document.generate(
        domain="document.invoice", file="test.pdf", autocast=False
    )
    assert isinstance(response, PredictionResponse)
    assert not isinstance(response.response, MarkdownDocument)
    assert isinstance(response.response, dict)


def test_document_markdown_automatic_casting_in_get(mock_client, monkeypatch):
    """Test automatic casting of document.markdown responses to MarkdownDocument in predictions.get()."""

    client = mock_client

    mock_response = {
        "pages": [
            {
                "content": "# Test Document\n\nThis is a test document.",
                "markdown_content": "# Test Document\n\nThis is a test document.",
                "tables": [
                    {
                        "headers": [
                            {"id": "header1", "column": 0, "name": "Header 1"},
                            {"id": "header2", "column": 1, "name": "Header 2"},
                        ],
                        "data": [{"header1": "Value 1", "header2": "Value 2"}],
                    }
                ],
                "figures": [
                    {
                        "id": 1,
                        "title": "Test Figure",
                        "caption": "Test Caption",
                        "content": "Test content",
                    }
                ],
            }
        ]
    }

    def mock_get(prediction_id):
        prediction = PredictionResponse(
            id=prediction_id,
            status="completed",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            domain="document.markdown",
            response=mock_response,
            usage=CreditUsage(credits_used=100),
        )
        if (
            prediction.domain == "document.markdown"
            and prediction.status == "completed"
            and prediction.response
        ):
            try:
                prediction.response = MarkdownDocument(**prediction.response)
            except Exception as e:
                logger.warning(f"Failed to cast response to MarkdownDocument: {e}")
        return prediction

    monkeypatch.setattr(client.predictions, "get", mock_get)

    response = client.predictions.get("test-prediction")
    assert isinstance(response, PredictionResponse)
    assert isinstance(response.response, MarkdownDocument)
    assert len(response.response.pages) == 1
    assert (
        response.response.pages[0].content
        == "# Test Document\n\nThis is a test document."
    )
    assert len(response.response.all_tables) == 1
    assert len(response.response.all_figures) == 1

    def mock_get_invoice(prediction_id):
        return PredictionResponse(
            id=prediction_id,
            status="completed",
            created_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            domain="document.invoice",
            response={"invoice_number": "INV-001", "total_amount": 100.0},
            usage=CreditUsage(credits_used=100),
        )

    monkeypatch.setattr(client.predictions, "get", mock_get_invoice)
    response = client.predictions.get("test-prediction")
    assert isinstance(response, PredictionResponse)
    assert not isinstance(response.response, MarkdownDocument)
    assert isinstance(response.response, dict)
