"""Tests for feedback operations."""

from pydantic import BaseModel
from vlmrun.client.types import FeedbackSubmitResponse


class TestLabel(BaseModel):
    """Test label model."""

    score: int
    comment: str


def test_submit_feedback(mock_client):
    """Test submitting feedback for a prediction."""
    label = TestLabel(score=5, comment="Great prediction!")
    response = mock_client.feedback.submit(
        id="prediction1", label=label, notes="Test feedback", flag=False
    )
    assert isinstance(response, FeedbackSubmitResponse)
    assert response.id == "feedback1"


def test_get_feedback(mock_client):
    """Test getting feedback by ID."""
    response = mock_client.feedback.get("prediction1")
    assert isinstance(response, FeedbackSubmitResponse)
    assert response.id == "feedback1"
