"""Evaluator contract.

A score is only ever produced by running a fixed, versioned grading procedure
against the candidate artifact. Nothing a discovery agent *says* about its own
quality is admissible evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..domain.models import EvaluatorStatus


@dataclass(frozen=True)
class EvaluationResult:
    status: EvaluatorStatus
    score: float
    duration_s: float
    result_digest: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is EvaluatorStatus.SUCCESS


class Evaluator(Protocol):
    """Fixed grading procedure. ``spec_digest`` pins the exact grader version."""

    @property
    def spec_digest(self) -> str: ...

    def evaluate(self, candidate_dir: Path, *, invocation_id: str) -> EvaluationResult: ...


__all__ = ["EvaluationResult", "Evaluator"]
