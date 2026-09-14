from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

REPORT_SCHEMA = {
    "generation_id": pl.Utf8,
    "scenario": pl.Utf8,
    "report": pl.Utf8,
    "model": pl.Utf8,
    "timestamp": pl.Datetime,
}


class BasePipeline(ABC):
    """Base class for synthetic medical-report generation pipelines.

    Provides shared building blocks (LLM wrappers, generation loop, Parquet
    persistence) and defines the interface that each concrete pipeline must
    implement.
    """

    name: str = "base"

    def __init__(
        self,
        config: dict,
        prompt_workflow: str = "one_stage",
    ) -> None:
        self.config = config
        if prompt_workflow not in {"one_stage", "two_stage"}:
            raise ValueError(
                "prompt_workflow doit valoir 'one_stage' ou 'two_stage'"
            )
        self.prompt_workflow = prompt_workflow

    @abstractmethod
    def check_data(self) -> None:
        """Verify that source data is present and prepare it if needed."""

    @abstractmethod
    def load_data(self) -> dict[str, pl.LazyFrame]:
        """Load prepared data as LazyFrames."""

    @abstractmethod
    def get_fictive(self, data: dict[str, pl.LazyFrame], **kwargs) -> pl.DataFrame:
        """Generate fictitious hospital stays from loaded data."""

    @abstractmethod
    def get_scenario(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform fictitious stays into text scenarios for the LLM."""
