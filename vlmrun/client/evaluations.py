"""VLM Run API Evaluations resource."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import (
    EvaluationRunResponse,
    EvaluationRunListResponse,
    EvaluationPreviewResponse,
    EvaluationMetricsResponse,
    EvaluationSummaryStatsResponse,
    EvaluationUniqueSourcesResponse,
    OptimizeSkillResponse,
    RerunSkillResponse,
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
        created_at_gte: Optional[str] = None,
        created_at_lte: Optional[str] = None,
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
        params: Dict[str, Any] = {
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

    def run(
        self,
        source_type: str,
        source_id: str,
        source_label: str,
        execution_ids: Optional[List[str]] = None,
        request_ids: Optional[List[str]] = None,
        skill_ids: Optional[List[str]] = None,
        data_from: Optional[str] = None,
        data_to: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        infer_corrections: Optional[bool] = None,
    ) -> EvaluationRunResponse:
        """Trigger a new evaluation run.

        Args:
            source_type: Type of evaluation source ("agent", "request_domain", or "skill")
            source_id: ID of the source to evaluate
            source_label: Display label for the source
            execution_ids: Optional list of execution IDs to evaluate
            request_ids: Optional list of request IDs to evaluate
            skill_ids: Optional list of skill IDs to evaluate
            data_from: Optional start datetime for data range (ISO format)
            data_to: Optional end datetime for data range (ISO format)
            evaluators: Optional list of evaluator types to use
            infer_corrections: Whether to infer corrections from feedback

        Returns:
            EvaluationRunResponse: The created evaluation run
        """
        data: Dict[str, Any] = {
            "source_type": source_type,
            "source_id": source_id,
            "source_label": source_label,
        }
        if execution_ids is not None:
            data["execution_ids"] = execution_ids
        if request_ids is not None:
            data["request_ids"] = request_ids
        if skill_ids is not None:
            data["skill_ids"] = skill_ids
        if data_from is not None:
            data["data_from"] = data_from
        if data_to is not None:
            data["data_to"] = data_to
        if evaluators is not None:
            data["evaluators"] = evaluators
        if infer_corrections is not None:
            data["infer_corrections"] = infer_corrections

        response, status_code, headers = self._requestor.request(
            method="POST", url="evaluations/run", data=data
        )
        return EvaluationRunResponse(**response)

    def preview(
        self,
        source_type: str,
        source_id: str,
        data_from: Optional[str] = None,
        data_to: Optional[str] = None,
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
        params: Dict[str, str] = {
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
        source_type: Optional[str] = None,
        source_label: Optional[str] = None,
    ) -> EvaluationMetricsResponse:
        """Get aggregated metrics across evaluation runs.

        Args:
            limit: Number of recent runs to aggregate over (default: 20)
            source_type: Optional filter by source type
            source_label: Optional filter by source label

        Returns:
            EvaluationMetricsResponse: Aggregated evaluation metrics
        """
        params: Dict[str, Any] = {"limit": limit}
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

    def optimize_skill(
        self,
        skill_id: str,
        skill_ids: Optional[List[str]] = None,
        data_from: Optional[str] = None,
        data_to: Optional[str] = None,
        infer_corrections: bool = True,
        n_samples: Optional[int] = None,
    ) -> OptimizeSkillResponse:
        """Optimize a skill based on evaluation data.

        Args:
            skill_id: The skill ID to optimize
            skill_ids: Optional list of additional skill IDs
            data_from: Optional start datetime for data range (ISO format)
            data_to: Optional end datetime for data range (ISO format)
            infer_corrections: Whether to infer corrections (default: True)
            n_samples: Optional number of samples to use

        Returns:
            OptimizeSkillResponse: The optimization result
        """
        data: Dict[str, Any] = {"skill_id": skill_id}
        if skill_ids is not None:
            data["skill_ids"] = skill_ids
        if data_from is not None:
            data["data_from"] = data_from
        if data_to is not None:
            data["data_to"] = data_to
        data["infer_corrections"] = infer_corrections
        if n_samples is not None:
            data["n_samples"] = n_samples

        response, status_code, headers = self._requestor.request(
            method="POST", url="evaluations/optimize", data=data
        )
        return OptimizeSkillResponse(**response)

    def rerun_skill(
        self,
        evaluation_id: str,
        skill_id: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        infer_corrections: bool = True,
        n_samples: Optional[int] = None,
    ) -> RerunSkillResponse:
        """Re-run an evaluation with different parameters.

        Args:
            evaluation_id: The evaluation run ID to re-run
            skill_id: Optional skill ID to use for the re-run
            evaluators: Optional list of evaluator types to use
            infer_corrections: Whether to infer corrections (default: True)
            n_samples: Optional number of samples to use

        Returns:
            RerunSkillResponse: The re-run result
        """
        data: Dict[str, Any] = {"evaluation_id": evaluation_id}
        if skill_id is not None:
            data["skill_id"] = skill_id
        if evaluators is not None:
            data["evaluators"] = evaluators
        data["infer_corrections"] = infer_corrections
        if n_samples is not None:
            data["n_samples"] = n_samples

        response, status_code, headers = self._requestor.request(
            method="POST", url="evaluations/rerun", data=data
        )
        return RerunSkillResponse(**response)
