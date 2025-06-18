"""Tests for feedback operations."""

from pydantic import BaseModel
from vlmrun.client.types import FeedbackSubmitResponse, FeedbackItem


class TestLabel(BaseModel):
    """Test label model."""

    score: int
    comment: str


def test_submit_feedback(mock_client):
    """Test submitting feedback for a prediction."""
    response = mock_client.feedback.submit(
        request_id="prediction1", 
        response={"score": 5, "comment": "Great prediction!"}, 
        notes="Test feedback"
    )
    assert isinstance(response, FeedbackSubmitResponse)
    assert response.request_id == "prediction1"
    assert len(response.items) > 0


def test_list_feedback(mock_client):
    """Test listing feedback for a prediction."""
    response = mock_client.feedback.list("prediction1", limit=5, offset=0)
    assert isinstance(response, FeedbackSubmitResponse)
    assert response.request_id == "prediction1"
    assert len(response.items) >= 0
