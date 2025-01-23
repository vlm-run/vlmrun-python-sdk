"""VLM Run API Fine-tuning resource."""

from __future__ import annotations

from typing import Dict, List

from vlmrun.client.base_requestor import APIRequestor
from vlmrun.client.types import FinetuningJobResponse
from vlmrun.types.abstract import Client


class Finetuning:
    """Fine-tuning resource for VLM Run API."""

    def __init__(self, client: "Client") -> None:
        """Initialize FineTuning resource with client.

        Args:
            client: VLM Run API client instance
        """
        self._client = client
        self._requestor = APIRequestor(client)

    def create(
        self,
        training_file: str,
        validation_file: str,
        model: str,
        n_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        use_lora: bool = True,
        track: bool = True,
        wandb_project: str | None = None,
        **kwargs,
    ) -> FinetuningJobResponse:
        """Create a fine-tuning job.

        Args:
            training_file: File ID for training data
            model: Base model to fine-tune
            n_epochs: Number of epochs (default: 1)
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            **kwargs: Additional fine-tuning parameters

        Returns:
            FinetuningJobResponse: Created fine-tuning job
        """
        response, status_code, headers = self._requestor.request(
            method="POST",
            url="fine_tuning/jobs/create",
            json={
                "training_file": training_file,
                "validation_file": validation_file,
                "model": model,
                "num_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_lora": use_lora,
                "track": track,
                "wandb_project": wandb_project,
            },
        )
        return FinetuningJobResponse(**response)

    def list(self, skip: int = 0, limit: int = 10) -> list[FinetuningJobResponse]:
        """List all fine-tuning jobs.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[FinetuningJobResponse]: List of fine-tuning jobs
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="fine_tuning/jobs",
            params={"skip": skip, "limit": limit},
        )
        return [FinetuningJobResponse(**job) for job in response]

    def list_models(self, skip: int = 0, limit: int = 10) -> List[str]:
        """List all fine-tuning models.

        Args:
            skip: Number of items to skip
            limit: Maximum number of items to return

        Returns:
            List[str]: List of fine-tuning models
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url="fine_tuning/models",
            params={"skip": skip, "limit": limit},
        )
        return response

    def get(self, job_id: str) -> FinetuningJobResponse:
        """Get fine-tuning job details.

        Args:
            job_id: ID of job to retrieve

        Returns:
            FinetuningJobResponse: Fine-tuning job details
        """
        response, status_code, headers = self._requestor.request(
            method="GET",
            url=f"fine_tuning/jobs/{job_id}",
        )
        return FinetuningJobResponse(**response)

    def cancel(self, job_id: str) -> Dict:
        """Cancel a fine-tuning job.

        Args:
            job_id: ID of job to cancel

        Returns:
            Dict: Cancelled job details
        """
        raise NotImplementedError("Not implemented")
