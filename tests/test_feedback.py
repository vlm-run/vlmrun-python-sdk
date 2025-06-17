"""Tests for feedback operations."""

from pydantic import BaseModel
from vlmrun.client.types import FeedbackSubmitResponse, FeedbackResponse, FeedbackListResponse


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
    assert isinstance(response, FeedbackResponse)
    assert response.id == "feedback1"


def test_list_feedback(mock_client):
    """Test listing feedback for a prediction."""
    response = mock_client.feedback.list("prediction1", limit=5, offset=0)
    assert isinstance(response, FeedbackListResponse)
    assert len(response.data) >= 0
    assert response.count >= 0


def test_get_feedback(mock_client):
    """Test getting feedback by ID (alias for list)."""
    response = mock_client.feedback.get("prediction1")
    assert isinstance(response, FeedbackListResponse)
    assert len(response.data) >= 0
