"""Tests for predictions operations."""

from PIL import Image
from vlmrun.client.types import PredictionResponse


def test_list_predictions(mock_client):
    """Test listing predictions."""
    client = mock_client
    response = client.prediction.list()
    assert isinstance(response, list)
    assert all(isinstance(pred, PredictionResponse) for pred in response)
    assert len(response) > 0
    assert response[0].id == "prediction1"
    assert response[0].status == "running"


def test_get_prediction(mock_client):
    """Test getting prediction by ID."""
    client = mock_client
    response = client.prediction.get("prediction1")
    assert isinstance(response, PredictionResponse)
    assert response.id == "prediction1"
    assert response.status == "running"


def test_wait_prediction(mock_client):
    """Test waiting for prediction completion."""
    client = mock_client
    response = client.prediction.wait("prediction1", timeout=1)
    assert isinstance(response, PredictionResponse)
    assert response.id == "prediction1"
    assert response.status == "completed"


def test_image_generate(mock_client, tmp_path):
    """Test generating image prediction."""
    # Create a dummy image for testing
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    client = mock_client
    response = client.image.generate(
        image=img_path,
        model="test-model",
        domain="test-domain",
        json_schema={"type": "object"},
    )
    assert isinstance(response, PredictionResponse)


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
