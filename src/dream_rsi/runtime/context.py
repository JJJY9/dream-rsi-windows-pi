"""Versioned discovery context, separate from a policy's visible tree prefix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.errors import IntegrityError
from ..domain.models import Node
from ..storage.cas import ContentAddressedStore

CONTEXT_FORMAT = "dream-rsi-context-v2"
OBSERVATION_FORMAT = "dream-rsi-observation-v2"


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


@dataclass(frozen=True)
class ContinuationContext:
    """Controller-owned public data; no private grading payloads or future nodes."""

    parent_id: str
    branch_observations: tuple[str, ...] = ()
    completed_worlds: tuple[str, ...] = ()
    directory: Path | None = None

    @staticmethod
    def workspace_path(digest: str) -> str:
        return f"ws/{digest[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CONTEXT_FORMAT,
            "parent_id": self.parent_id,
            "branch": [json.loads(value) for value in self.branch_observations],
            "completed_worlds": [json.loads(value) for value in self.completed_worlds],
            "artifact_paths": {
                "workspace": "ws/<first 24 digest characters>",
                "generation": "gen/<first 24 digest characters>.json",
            },
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical(self.to_dict()).encode()).hexdigest()

    def materialize(self, store: ContentAddressedStore) -> None:
        """Export exact public history for reading with discovery tools.

        This is an input contract, not an OS sandbox. Discovery treats these
        files as read-only; the framework reads its own hash-verified CAS copy.
        """
        if self.directory is None:
            raise IntegrityError("context directory is required")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "index.json").write_text(canonical(self.to_dict()), encoding="utf-8")
        observations = [json.loads(value) for value in self.branch_observations]
        for value in self.completed_worlds:
            observations.extend(node["observation"] for node in json.loads(value)["nodes"])
        exported: set[str] = set()
        for observation in observations:
            for key in ("workspace_snapshot_digest", "generation_digest"):
                digest = observation.get(key)
                if not digest or digest in exported:
                    continue
                exported.add(digest)
                if key == "workspace_snapshot_digest":
                    store.restore_tree(digest, self.directory / self.workspace_path(digest))
                else:
                    path = self.directory / "gen" / f"{digest[:24]}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(store.get_bytes(digest))


def load_observation(store: ContentAddressedStore, digest: str) -> dict[str, Any]:
    value = store.get_json(digest)
    if not isinstance(value, dict) or value.get("format") != OBSERVATION_FORMAT:
        raise IntegrityError("node observation is legacy or incomplete; cannot inherit it")
    return value


def branch_chain(
    nodes: Sequence[Node], parent_id: str, store: ContentAddressedStore
) -> tuple[str, ...]:
    by_id = {node.node_id: node for node in nodes}
    chain: list[str] = []
    current: str | None = parent_id
    while current is not None:
        node = by_id[current]
        value = load_observation(store, node.observation_digest)
        if value["node_id"] != node.node_id or value["parent_id"] != node.parent_id:
            raise IntegrityError("observation identity disagrees with node")
        chain.append(canonical(value))
        current = node.parent_id
    return tuple(reversed(chain))


def validate_world_artifacts(store: ContentAddressedStore, nodes: Sequence[Node],
                             previous_worlds: Sequence[dict[str, Any]]) -> None:
    """Audit actual blobs and inheritance, not just the presence of digest strings."""
    by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        observation = load_observation(store, node.observation_digest)
        if (observation.get("node_id") != node.node_id
                or observation.get("world_id") != node.world_id
                or observation.get("parent_id") != node.parent_id
                or observation.get("workspace_snapshot_digest") != node.workspace_snapshot_digest):
            raise IntegrityError("observation identity/snapshot mismatch")
        store.tree_entries(node.workspace_snapshot_digest)
        if node.parent_id is None:
            if observation["public"]["score"] != 0:
                raise IntegrityError("root observation score must be zero")
            if node.observation_format == 2 and canonical(observation["public"]) != node.public_observation_json:
                raise IntegrityError("root public observation mismatch")
            continue
        if node.observation_format != 2:
            raise IntegrityError("legacy node has no complete observation contract")
        public = observation["public"]
        if (canonical(public) != node.public_observation_json or public["score"] != node.score
                or public["status"] != node.evaluator_status.value):
            raise IntegrityError("public observation differs from measured node")
        if observation["parent_observation_digest"] != by_id[node.parent_id].observation_digest:
            raise IntegrityError("parent observation reference mismatch")
        context = store.get_json(observation["context_digest"])
        expected_branch = [json.loads(value) for value in branch_chain(nodes, node.parent_id, store)]
        if (context.get("format") != CONTEXT_FORMAT or context.get("parent_id") != node.parent_id
                or context.get("branch") != expected_branch
                or context.get("completed_worlds") != list(previous_worlds)):
            raise IntegrityError("attempt context has incomplete/incorrect branch or completed history")
        generation = store.get_json(node.agent_session_digest or "")
        if (generation.get("format") != "dream-rsi-generation-v2"
                or generation.get("context_digest") != observation["context_digest"]
                or observation.get("generation_digest") != node.agent_session_digest):
            raise IntegrityError("generation/context reference mismatch")
        evaluation = store.get_json(node.evaluator_result_digest or "")
        if (evaluation.get("format") != "dream-rsi-evaluation-v2"
                or evaluation.get("score") != node.score
                or evaluation.get("public_detail") != public["diagnostics"]):
            raise IntegrityError("stored evaluator result differs from public observation")


def export_world(store: ContentAddressedStore, nodes: Sequence[Node], *, world_id: str,
                 outer_t: int, world_digest: str | None) -> dict[str, Any]:
    return {"format": "dream-rsi-history-v2", "world_id": world_id,
            "outer_t": outer_t, "world_digest": world_digest,
            "nodes": [{"node_id": node.node_id, "parent_id": node.parent_id,
                       "creation_index": node.creation_index,
                       "observation": load_observation(store, node.observation_digest)}
                      for node in nodes]}
