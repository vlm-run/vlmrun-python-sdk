"""VLM Run API Evaluations resource."""

from __future__ import annotations

from typing import Any

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import (
    EvaluationRunResponse,
    EvaluationRunListResponse,
    EvaluationPreviewResponse,
    EvaluationMetricsResponse,
    EvaluationSummaryStatsResponse,
    EvaluationUniqueSourcesResponse,
)
from vlmrun.types.abstract import VLMRunProtocol


class Evaluations:
    """Evaluations resource for VLM Run API."""

    def __init__(self, client: "VLMRunProtocol") -> None:
        """Initialize Evaluations resource with VLMRun instance.

        Args:
            client: VLM Run API instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def list(
        self,
        limit: int = 30,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
        created_at_gte: str | None = None,
        created_at_lte: str | None = None,
    ) -> EvaluationRunListResponse:
        """List evaluation runs with pagination and filtering.

        Args:
            limit: Maximum number of results to return (default: 30)
            offset: Number of results to skip (default: 0)
            order_by: Field to order by (default: "created_at")
            descending: Sort in descending order (default: True)
            created_at_gte: Filter runs created at or after this datetime (ISO format)
            created_at_lte: Filter runs created at or before this datetime (ISO format)

        Returns:
            EvaluationRunListResponse: Paginated list of evaluation runs
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order_by": order_by,
            "descending": descending,
        }
        if created_at_gte:
            params["created_at__gte"] = created_at_gte
        if created_at_lte:
            params["created_at__lte"] = created_at_lte

        response, status_code, headers = self._requestor.request(
            method="GET", url="evaluations", params=params
        )
        return EvaluationRunListResponse(**response)

    def get(self, run_id: str) -> EvaluationRunResponse:
        """Get a specific evaluation run by ID.

        Args:
            run_id: The evaluation run ID

        Returns:
            EvaluationRunResponse: The evaluation run details
        """
        response, status_code, headers = self._requestor.request(
            method="GET", url=f"evaluations/{run_id}"
        )
        return EvaluationRunResponse(**response)

    def preview(
        self,
        source_type: str,
        source_id: str,
        data_from: str | None = None,
        data_to: str | None = None,
    ) -> EvaluationPreviewResponse:
        """Get a preview of available data for an evaluation source.

        Args:
            source_type: Type of evaluation source
            source_id: ID of the source
            data_from: Optional start datetime for data range (ISO format)
            data_to: Optional end datetime for data range (ISO format)

        Returns:
            EvaluationPreviewResponse: Preview of available evaluation data
        """
        params: dict[str, str] = {
            "source_type": source_type,
            "source_id": source_id,
        }
        if data_from:
            params["data_from"] = data_from
        if data_to:
            params["data_to"] = data_to

        response, status_code, headers = self._requestor.request(
            method="GET", url="evaluations/preview", params=params
        )
        return EvaluationPreviewResponse(**response)

    def metrics(
        self,
        limit: int = 20,
        source_type: str | None = None,
        source_label: str | None = None,
    ) -> EvaluationMetricsResponse:
        """Get aggregated metrics across evaluation runs.

        Args:
            limit: Number of recent runs to aggregate over (default: 20)
            source_type: Optional filter by source type
            source_label: Optional filter by source label

        Returns:
            EvaluationMetricsResponse: Aggregated evaluation metrics
        """
        params: dict[str, Any] = {"limit": limit}
        if source_type:
            params["source_type"] = source_type
        if source_label:
            params["source_label"] = source_label

        response, status_code, headers = self._requestor.request(
            method="GET", url="evaluations/metrics", params=params
        )
        return EvaluationMetricsResponse(**response)

    def summary_stats(self) -> EvaluationSummaryStatsResponse:
        """Get summary statistics for evaluation runs.

        Returns:
            EvaluationSummaryStatsResponse: Summary stats including total runs and source type counts
        """
        response, status_code, headers = self._requestor.request(
            method="GET", url="evaluations/summary-stats"
        )
        return EvaluationSummaryStatsResponse(**response)

    def unique_sources(self) -> EvaluationUniqueSourcesResponse:
        """Get unique evaluation sources for filtering.

        Returns:
            EvaluationUniqueSourcesResponse: Unique (type, label) pairs
        """
        response, status_code, headers = self._requestor.request(
            method="GET", url="evaluations/unique-sources"
        )
        return EvaluationUniqueSourcesResponse(**response)

    def delete(self, run_id: str) -> None:
        """Delete an evaluation run.

        Args:
            run_id: The evaluation run ID to delete
        """
        self._requestor.request(method="DELETE", url=f"evaluations/{run_id}")
