"""Tests for evaluations operations."""

from vlmrun.client.types import (
    EvaluationMetricsResponse,
    EvaluationPreviewResponse,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    EvaluationSummaryStatsResponse,
    EvaluationUniqueSourcesResponse,
)


def test_list_evaluations(mock_client):
    """Test listing evaluation runs."""
    response = mock_client.evaluations.list(limit=10, offset=0)
    assert isinstance(response, EvaluationRunListResponse)
    assert response.count == 1
    assert len(response.data) == 1
    assert response.data[0].id == "eval-run-1"
    assert response.data[0].source_type == "skill"


def test_get_evaluation(mock_client):
    """Test getting a specific evaluation run."""
    response = mock_client.evaluations.get("eval-run-1")
    assert isinstance(response, EvaluationRunResponse)
    assert response.id == "eval-run-1"
    assert response.status == "completed"
    assert response.accuracy == 0.95


def test_preview_evaluation(mock_client):
    """Test getting evaluation preview data."""
    response = mock_client.evaluations.preview(
        source_type="skill",
        source_id="skill-123",
    )
    assert isinstance(response, EvaluationPreviewResponse)
    assert response.total_items == 100
    assert response.total_with_feedback == 50
    assert response.feedback_with_corrections == 30


def test_evaluation_metrics(mock_client):
    """Test getting evaluation metrics."""
    response = mock_client.evaluations.metrics(limit=10)
    assert isinstance(response, EvaluationMetricsResponse)
    assert response.avg_accuracy == 0.92
    assert isinstance(response.accuracy_trend, list)


def test_evaluation_summary_stats(mock_client):
    """Test getting evaluation summary stats."""
    response = mock_client.evaluations.summary_stats()
    assert isinstance(response, EvaluationSummaryStatsResponse)
    assert response.total_runs == 10


def test_evaluation_unique_sources(mock_client):
    """Test getting unique evaluation sources."""
    response = mock_client.evaluations.unique_sources()
    assert isinstance(response, EvaluationUniqueSourcesResponse)
    assert isinstance(response.sources, list)


def test_delete_evaluation(mock_client):
    """Test deleting an evaluation run."""
    result = mock_client.evaluations.delete("eval-run-1")
    assert result is None
