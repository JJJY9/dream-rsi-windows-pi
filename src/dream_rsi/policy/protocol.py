"""The one policy interface shared by online rollout and offline replay (§3).

Frozen contract: worker modules must not edit this file.

Information barrier
-------------------
A policy sees **only** the prefix that has already been observed, plus the
public budget knob ``rho``. It never receives the full world, hidden node
metadata, ``B_i``/``S_i``, future child counts, or hidden creation indices.
``legal_actions`` expresses *parent eligibility* (root ∪ current leaves) and
deliberately does **not** reveal whether a legal parent actually has an
unrevealed historical child.

This is the single selection interface used by both modes; only the effect of
the returned batch differs (online: really execute; offline: look up history).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PublicNode:
    """A node as visible to a policy — sanitized, observation-only."""

    node_id: str
    parent_id: str | None
    creation_index: int
    depth: int
    score: float
    evaluator_status: str
    error_type: str | None
    summary: str
    observation_json: str = ""

    @property
    def public_observation(self) -> dict[str, Any]:
        """A fresh public-data mapping; mutating it cannot change the stored world."""
        value = json.loads(self.observation_json) if self.observation_json else {}
        return dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "creation_index": self.creation_index,
            "depth": self.depth,
            "score": self.score,
            "evaluator_status": self.evaluator_status,
            "error_type": self.error_type,
            "summary": self.summary,
            "observation_json": self.observation_json,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PublicNode:
        return cls(
            node_id=str(data["node_id"]),
            parent_id=None if data.get("parent_id") is None else str(data["parent_id"]),
            creation_index=int(data["creation_index"]),
            depth=int(data["depth"]),
            score=float(data["score"]),
            evaluator_status=str(data["evaluator_status"]),
            error_type=None if data.get("error_type") is None else str(data["error_type"]),
            summary=str(data.get("summary", "")),
            observation_json=str(data.get("observation_json", "")),
        )


@dataclass(frozen=True)
class PrefixObservation:
    """Everything a policy is allowed to know at decision time."""

    nodes: tuple[PublicNode, ...]
    legal_actions: tuple[str, ...]
    completed_rounds: int
    max_parallelism: int
    node_ids: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.node_ids:
            object.__setattr__(self, "node_ids", tuple(n.node_id for n in self.nodes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "legal_actions": list(self.legal_actions),
            "completed_rounds": self.completed_rounds,
            "max_parallelism": self.max_parallelism,
            "node_ids": list(self.node_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrefixObservation:
        return cls(
            nodes=tuple(PublicNode.from_dict(n) for n in data.get("nodes", ())),
            legal_actions=tuple(str(x) for x in data.get("legal_actions", ())),
            completed_rounds=int(data.get("completed_rounds", 0)),
            max_parallelism=int(data.get("max_parallelism", 1)),
            node_ids=tuple(str(x) for x in data.get("node_ids", ())),
        )


@runtime_checkable
class ExplorationPolicy(Protocol):
    """A policy maps an observed prefix to a batch of parent node ids.

    Returning ``[]`` stops the current rollout. Returning anything that is not
    a sequence of already-observed legal ids is rejected by the controller; the
    policy is never trusted to police itself.
    """

    def select(self, observation: PrefixObservation, rho: float) -> Sequence[str]:
        ...


__all__ = ["ExplorationPolicy", "PrefixObservation", "PublicNode"]
