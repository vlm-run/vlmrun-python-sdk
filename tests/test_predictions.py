"""Tests for predictions operations."""

import pytest
from pydantic import BaseModel
from PIL import Image
from vlmrun.client.types import PredictionResponse, GenerationConfig
from vlmrun.common.gql import create_pydantic_model_from_gql


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

    def mock_get_schema(domain):
        return MockInvoiceSchema

    mock_client.hub.get_pydantic_model = mock_get_schema

    response = mock_client.image.generate(
        domain="document.invoice", urls=["https://example.com/test.jpg"]
    )

    assert isinstance(response.response, MockInvoiceSchema)


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
        return MockInvoiceSchema

    mock_client.hub.get_pydantic_model = mock_get_schema

    pred_client = getattr(mock_client, prediction_type)

    if prediction_type == "image":
        response = pred_client.generate(
            domain="document.invoice", urls=["https://example.com/test.jpg"]
        )
    else:
        response = pred_client.generate(
            domain="document.invoice", url="https://example.com/test.file"
        )

    assert isinstance(response.response, MockInvoiceSchema)


class AddressSchema(BaseModel):
    """Mock address schema for testing."""

    street: str
    city: str
    state: str
    zip: str


class NestedInvoiceSchema(BaseModel):
    """Mock invoice schema for testing."""

    invoice_id: str
    total_amount: float
    period_start: str
    period_end: str
    address: AddressSchema


def test_create_pydantic_model_from_gql_basic():
    """Test basic field filtering with GQL."""
    gql_query = """
    {
        invoice_id
        total_amount
    }
    """

    FilteredModel = create_pydantic_model_from_gql(NestedInvoiceSchema, gql_query)
    assert set(FilteredModel.model_fields.keys()) == {"invoice_id", "total_amount"}


def test_create_pydantic_model_from_gql_nested():
    """Test nested field filtering with GQL."""
    gql_query = """
    {
        invoice_id
        address {
            state
            zip
        }
    }
    """

    FilteredModel = create_pydantic_model_from_gql(NestedInvoiceSchema, gql_query)

    assert set(FilteredModel.model_fields.keys()) == {"invoice_id", "address"}

    AddressModel = FilteredModel.model_fields["address"].annotation
    assert set(AddressModel.model_fields.keys()) == {"state", "zip"}


def test_create_pydantic_model_from_gql_invalid_field():
    """Test handling of invalid field in GQL query."""
    gql_query = """
    {
        invoice_id
        nonexistent_field
    }
    """

    with pytest.raises(
        ValueError, match="Field 'nonexistent_field' not found in model"
    ):
        create_pydantic_model_from_gql(NestedInvoiceSchema, gql_query)


def test_create_pydantic_model_from_gql_invalid_nested_field():
    """Test handling of invalid nested field in GQL query."""
    gql_query = """
    {
        invoice_id
        address {
            nonexistent_field
        }
    }
    """

    with pytest.raises(
        ValueError, match="Field 'nonexistent_field' not found in nested model"
    ):
        create_pydantic_model_from_gql(NestedInvoiceSchema, gql_query)


def test_create_pydantic_model_from_gql_malformed():
    """Test handling of malformed GQL query."""
    malformed_query = """
    {
        invoice_id
        address {
    """

    with pytest.raises(Exception, match="Syntax Error"):
        create_pydantic_model_from_gql(NestedInvoiceSchema, malformed_query)


def test_create_pydantic_model_from_gql_nested_scalar():
    """Test handling of attempting to query nested fields of a scalar."""
    gql_query = """
    {
        invoice_id {
            nested_field
        }
    }
    """

    with pytest.raises(ValueError, match="Cannot query nested fields of scalar type"):
        create_pydantic_model_from_gql(NestedInvoiceSchema, gql_query)


def test_create_pydantic_model_from_gql_all_fields():
    """Test requesting all fields."""
    gql_query = """
    {
        invoice_id
        total_amount
        period_start
        period_end
        address {
            street
            city
            state
            zip
        }
    }
    """

    FilteredModel = create_pydantic_model_from_gql(NestedInvoiceSchema, gql_query)

    assert set(FilteredModel.model_fields.keys()) == {
        "invoice_id",
        "total_amount",
        "period_start",
        "period_end",
        "address",
    }

    AddressModel = FilteredModel.model_fields["address"].annotation
    assert set(AddressModel.model_fields.keys()) == {"street", "city", "state", "zip"}
