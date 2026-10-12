"""Shared adapter contract for discovery and policy-development agents.

A backend produces *artifacts* (files in an isolated workspace) plus a
sanitized summary. It never produces a score: scoring belongs to the fixed
evaluator alone.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..runtime.context import ContinuationContext


@dataclass(frozen=True)
class TaskSpec:
    """The fixed problem statement handed to a discovery worker."""

    task_id: str
    entry_filename: str
    instructions: str
    seed_workspace: Path
    is_paper_benchmark: bool = False

    @property
    def spec_digest(self) -> str:
        material = json.dumps(
            {
                "task_id": self.task_id,
                "entry_filename": self.entry_filename,
                "instructions": self.instructions,
                "is_paper_benchmark": self.is_paper_benchmark,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HistoryEntry:
    """Public, sanitized view of one finished world (safe to hand to an agent)."""

    outer_t: int
    world_digest: str
    node_count: int
    best_score: float
    policy_version_id: str
    summary: str = ""

    # Complete public world export as canonical JSON, never a hidden current tree.
    contents_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_t": self.outer_t,
            "world_digest": self.world_digest,
            "node_count": self.node_count,
            "best_score": self.best_score,
            "policy_version_id": self.policy_version_id,
            "summary": self.summary,
            "contents_json": self.contents_json,
        }


@dataclass(frozen=True)
class GeneratedAttempt:
    invocation_id: str
    backend_id: str
    model: str
    workspace_path: Path
    entry_file_digest: str | None
    succeeded: bool
    duration_s: float
    error_type: str | None = None
    summary: str = ""
    session_id: str | None = None
    raw_response_digest: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    generation_record: Mapping[str, Any] = field(default_factory=dict)


class DiscoveryBackend(Protocol):
    """Generates one candidate workspace from a parent workspace."""

    backend_id: str
    is_live: bool

    def generate(
        self,
        *,
        task: TaskSpec,
        parent_workspace: Path,
        history: Sequence[HistoryEntry],
        output_workspace: Path,
        invocation_id: str,
        context: ContinuationContext | None = None,
    ) -> GeneratedAttempt: ...


def digest_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "ContinuationContext",
    "DiscoveryBackend",
    "GeneratedAttempt",
    "HistoryEntry",
    "TaskSpec",
    "digest_file",
    "digest_text",
]
