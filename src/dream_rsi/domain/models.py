"""Persisted domain DTOs for the strict-§3 Dream-RSI runtime.

Frozen contract: worker modules must not edit this file. Everything persisted
carries ``schema_version``. Deserialization is explicit field-by-field — the
runtime never unpickles untrusted data.

Source of truth for field meaning: ``contracts/SCHEMAS.md``.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from math import isfinite
from typing import Any, TypeVar

SCHEMA_VERSION = 1

T = Any


def utcnow_iso() -> str:
    """Timezone-aware UTC timestamp; stable, sortable, JSON-safe."""
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class RunMode(str, Enum):
    """A fixture run must never be merged into live statistics."""

    OFFLINE_FIXTURE = "offline_fixture"
    LIVE = "live"


class AlgorithmMode(str, Enum):
    STRICT_S3 = "strict_s3"


class RunState(str, Enum):
    CREATED = "CREATED"
    ONLINE_RUNNING = "ONLINE_RUNNING"
    WORLD_SEALED = "WORLD_SEALED"
    DREAM_RUNNING = "DREAM_RUNNING"
    POLICY_SELECTED = "POLICY_SELECTED"
    COMMITTED = "COMMITTED"
    COMPLETE = "COMPLETE"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
    FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
    FAILED_FINAL = "FAILED_FINAL"


class WorldState(str, Enum):
    BUILDING = "BUILDING"
    SEALED_SCOREABLE = "SEALED_SCOREABLE"
    SEALED_UNSCOREABLE = "SEALED_UNSCOREABLE"
    QUARANTINED = "QUARANTINED"


class AttemptState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    GENERATION_DONE = "GENERATION_DONE"
    EVALUATING = "EVALUATING"
    FINISHED = "FINISHED"
    ERROR_RECORDED = "ERROR_RECORDED"
    CANCELLED = "CANCELLED"


class PolicyState(str, Enum):
    DRAFT = "DRAFT"
    STATIC_VALID = "STATIC_VALID"
    SANDBOX_VALID = "SANDBOX_VALID"
    REPLAYED = "REPLAYED"
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"


class EvaluatorStatus(str, Enum):
    """Truthful outcome of the fixed evaluator for one node.

    ``FAILED`` means the evaluator ran and the candidate genuinely scored badly.
    ``ERROR`` means the evaluator could not run; the task protocol supplies the
    mapped score and the failure is recorded, never hidden.
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ERROR = "ERROR"


class StopReason(str, Enum):
    STOP_EMPTY_BATCH = "STOP_EMPTY_BATCH"
    MAX_ROUNDS = "MAX_ROUNDS"
    FULLY_REVEALED = "FULLY_REVEALED"
    MAX_PARENTS = "MAX_PARENTS"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


_T = TypeVar("_T")


def _load(cls: type[_T], data: Mapping[str, Any],
          normalizers: Mapping[str, Callable[[Any], Any]] | None = None) -> _T:
    """Explicit, non-eval field loading. Unknown keys are ignored on purpose."""
    norm = normalizers or {}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        if value is not None and f.name in norm:
            value = norm[f.name](value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _pairs(value: Any) -> tuple[tuple[int, float], ...]:
    return tuple((int(n), float(a)) for n, a in value)


def _strs(value: Any) -> tuple[str, ...]:
    return tuple(str(v) for v in value)


def _batches(value: Any) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(x) for x in batch) for batch in value)


class Serializable:
    """Mixin giving every DTO a versioned, JSON-safe dict form."""

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]
        data["schema_version"] = SCHEMA_VERSION
        return data


@dataclass(frozen=True)
class Run(Serializable):
    run_id: str
    mode: RunMode
    algorithm_mode: AlgorithmMode
    task_spec_digest: str
    provider_id: str
    agent_model: str
    fixed_evaluator_digest: str
    cfg_digest: str
    state: RunState = RunState.CREATED
    created_at: str = field(default_factory=utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    status_note: str | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Run:
        return _load(cls, data, {"mode": RunMode, "algorithm_mode": AlgorithmMode,
                                 "state": RunState})


@dataclass(frozen=True)
class World(Serializable):
    world_id: str
    outer_t: int
    run_id: str
    root_id: str
    policy_version_id: str
    state: WorldState = WorldState.BUILDING
    world_digest: str | None = None
    b_count: int = 0          # B_i = |T_i| - 1, valid once sealed
    s_best: float = 0.0       # S_i = max non-root score
    created_at: str = field(default_factory=utcnow_iso)
    sealed_at: str | None = None
    seal_note: str | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> World:
        return _load(cls, data, {"state": WorldState})


@dataclass(frozen=True)
class Node(Serializable):
    node_id: str
    world_id: str
    parent_id: str | None
    creation_index: int
    round_index: int
    branch_index: int
    depth: int
    workspace_snapshot_digest: str
    observation_digest: str
    score: float
    evaluator_status: EvaluatorStatus
    agent_session_digest: str | None = None
    evaluator_result_digest: str | None = None
    error_type: str | None = None
    summary: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("node score must be finite (NaN/Inf is a protocol bug)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Node:
        return _load(cls, data, {"evaluator_status": EvaluatorStatus})


@dataclass(frozen=True)
class Attempt(Serializable):
    attempt_id: str
    world_id: str
    parent_id: str
    round_index: int
    invocation_id: str
    state: AttemptState = AttemptState.QUEUED
    workspace_snapshot_digest: str | None = None
    failure_reason: str | None = None
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Attempt:
        return _load(cls, data, {"state": AttemptState})


@dataclass(frozen=True)
class PolicyVersion(Serializable):
    version_id: str
    run_id: str
    iteration_t: int
    m: int
    parent_version: str | None
    source_sha256: str
    source_artifact_id: str
    origin: str                       # "incumbent" | "revision"
    state: PolicyState = PolicyState.DRAFT
    static_validation_status: str = "PENDING"
    sandbox_validation_status: str = "PENDING"
    failure_reason: str | None = None
    created_at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyVersion:
        return _load(cls, data, {"state": PolicyState})


@dataclass(frozen=True)
class ReplayEpisode(Serializable):
    """One (world, policy, rho) replay. Counters are hard evidence of isolation."""

    episode_id: str
    world_id: str
    policy_version_id: str
    rho: float
    n: int
    kstar: int
    best_score: float
    attained_quality: float
    selected_batches: tuple[tuple[str, ...], ...]
    revealed_order: tuple[str, ...]
    stop_reason: StopReason
    replay_digest: str
    no_progress_count: int = 0
    discovery_agent_calls: int = 0
    evaluator_calls: int = 0
    created_at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayEpisode:
        return _load(cls, data, {"stop_reason": StopReason,
                                 "selected_batches": _batches,
                                 "revealed_order": _strs})


@dataclass(frozen=True)
class ScoreRecord(Serializable):
    policy_version_id: str
    world_id: str
    b_count: int
    s_best: float
    five_points: tuple[tuple[int, float], ...]
    frontier: tuple[tuple[int, float], ...]
    hv: float
    parallel_penalty: float
    lambda_weight: float
    value: float
    score_formula_version: str = "eq1-v2"
    created_at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoreRecord:
        return _load(cls, data, {"five_points": _pairs, "frontier": _pairs})


@dataclass(frozen=True)
class Revision(Serializable):
    iteration_t: int
    m: int
    input_policy_id: str
    output_policy_id: str | None
    status: str
    feedback_digest: str
    proposal_digest: str | None = None
    failure_reason: str | None = None
    created_at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Revision:
        return _load(cls, data)


@dataclass(frozen=True)
class Event(Serializable):
    run_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=utcnow_iso)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Event:
        return _load(cls, data, {"payload": dict})


__all__ = [
    "SCHEMA_VERSION",
    "AlgorithmMode",
    "Attempt",
    "AttemptState",
    "EvaluatorStatus",
    "Event",
    "Node",
    "PolicyState",
    "PolicyVersion",
    "ReplayEpisode",
    "Revision",
    "Run",
    "RunMode",
    "RunState",
    "ScoreRecord",
    "Serializable",
    "StopReason",
    "World",
    "WorldState",
    "new_id",
    "utcnow_iso",
]
