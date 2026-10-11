"""Online discovery tree (§3) and the sealed, immutable full world.

Frozen contract: the runtime calls exactly the public API below.  A
:class:`DiscoveryTree` is mutable and only the controller mutates it.  A
:class:`SealedWorld` is built once from persisted :class:`Node` records and
re-validates every §3 structural invariant before it may be used for replay.

Paper semantics
---------------
``T`` is a rooted tree.  ``root`` is the fixed initial workspace
(``parent_id=None``, ``creation_index=0``, ``score=0.0``).  Every non-root node
has exactly one parent and **at most one** recorded child; ``root`` may have
unlimited children.  ``creation_index`` is unique per world, contiguous from
``0``, and a parent's index is always smaller than its child's.

The action set is ``A(T) = {root} + leaves(R)`` where ``R`` is the *revealed*
subtree: a revealed node counts as a leaf when none of its children has been
revealed yet.  ``next_child(v, revealed)`` is the child of ``v`` with the
smallest ``creation_index`` that is not yet revealed, else ``None`` — for a
non-root that is its single recorded child (P-09), for ``root`` the
earliest-created unrevealed direct child (P-10), and at most one root child is
revealed per call.

Nothing in :meth:`DiscoveryTree.observation` may leak a hidden node, a hidden
child count or a hidden score: only revealed :class:`PublicNode` records are
emitted and ``legal_actions`` expresses *parent eligibility*, never future
child existence.  The emitted prefix is byte-for-byte the same information the
reference oracle's ``FullWorld.prefix`` exposes.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from ..policy.protocol import PrefixObservation, PublicNode
from .errors import InvalidWorld, TreeInvariantViolation
from .models import EvaluatorStatus, Node

__all__ = ["DiscoveryTree", "SealedWorld"]

#: Version tag of the canonical serialisation used by ``world_digest``.  Bump
#: it whenever the record format changes so old digests can never collide with
#: new ones.
_WORLD_DIGEST_FORMAT = "dream-rsi.world.v1"


def _status_value(status: EvaluatorStatus) -> str:
    """Return the plain string token of an evaluator status.

    ``Node`` is *typed* as carrying an ``EvaluatorStatus``, but a record rebuilt
    by hand or by a loose deserialiser may carry the raw string; read the value
    defensively instead of assuming the enum instance.
    """
    return status.value if isinstance(status, EvaluatorStatus) else str(status)


def _canonical_record(node: Node) -> dict[str, Any]:
    """One node in the canonical, process-independent digest form.

    ``created_at`` is deliberately excluded: it is wall-clock metadata, so
    including it would make two runs that built the same world disagree.  The
    world id lives in the digest header instead of on every record.
    """
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "creation_index": node.creation_index,
        "round_index": node.round_index,
        "branch_index": node.branch_index,
        "depth": node.depth,
        "workspace_snapshot_digest": node.workspace_snapshot_digest,
        "observation_digest": node.observation_digest,
        "score": node.score,
        "evaluator_status": _status_value(node.evaluator_status),
        "agent_session_digest": node.agent_session_digest,
        "evaluator_result_digest": node.evaluator_result_digest,
        "error_type": node.error_type,
        "summary": node.summary,
    }


def _world_digest(world_id: str, root_id: str, ordered: Sequence[Node]) -> str:
    """sha256 over the canonical serialisation of a whole world.

    Deterministic across runs and processes: sorted keys, compact separators,
    ASCII-only JSON, UTF-8 encoding, no wall-clock fields.
    """
    payload = {
        "format": _WORLD_DIGEST_FORMAT,
        "world_id": world_id,
        "root_id": root_id,
        "node_count": len(ordered),
        "nodes": [_canonical_record(n) for n in ordered],
    }
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _build_observation(
    ordered: Sequence[Node],
    children: Mapping[str, Sequence[str]],
    root_id: str,
    revealed: Collection[str],
    completed_rounds: int,
    max_parallelism: int,
) -> PrefixObservation:
    """Build the exact prefix a policy may see — the oracle's ``prefix()``.

    ``ordered`` must already be in ``creation_index`` order.  Ids in
    ``revealed`` that are not part of this world are ignored, exactly like the
    reference oracle: the revealed set is controller-owned, and a stray id can
    only ever hide information, never leak it.
    """
    shown = set(revealed)
    selected = tuple(n for n in ordered if n.node_id in shown)
    leaves = tuple(
        n.node_id for n in selected if not any(child in shown for child in children[n.node_id])
    )
    nodes = tuple(
        PublicNode(
            node_id=n.node_id,
            parent_id=n.parent_id,
            creation_index=n.creation_index,
            depth=n.depth,
            score=n.score,
            evaluator_status=_status_value(n.evaluator_status),
            error_type=n.error_type,
            summary=n.summary,
        )
        for n in selected
    )
    return PrefixObservation(
        nodes=nodes,
        legal_actions=tuple(dict.fromkeys((root_id, *leaves))),
        completed_rounds=completed_rounds,
        max_parallelism=max_parallelism,
    )


def _root_record(
    world_id: str,
    root_id: str,
    *,
    workspace_snapshot_digest: str = "",
    observation_digest: str = "",
    summary: str = "",
) -> Node:
    """The fixed initial workspace: parentless, index 0, score 0.0.

    Its evaluator status is ``SUCCESS``: the root is the trusted incumbent
    workspace, not an evaluated candidate.
    """
    return Node(
        node_id=root_id,
        world_id=world_id,
        parent_id=None,
        creation_index=0,
        round_index=0,
        branch_index=0,
        depth=0,
        workspace_snapshot_digest=workspace_snapshot_digest,
        observation_digest=observation_digest,
        score=0.0,
        evaluator_status=EvaluatorStatus.SUCCESS,
        summary=summary,
    )


class DiscoveryTree:
    """Mutable ONLINE tree. Only the controller mutates it.

    Invariants enforced on every mutation: a non-root node has exactly one
    parent and at most one recorded child; ``root`` may have unlimited
    children; ``creation_index`` is assigned by the tree as ``len(self)`` and is
    therefore unique, contiguous from 0 and always greater than its parent's;
    ``depth`` is assigned as ``parent.depth + 1``.
    """

    def __init__(self, world_id: str, root_id: str = "root") -> None:
        self._world_id = world_id
        self._root_id = root_id
        root = _root_record(world_id, root_id)
        self._nodes: list[Node] = [root]
        self._by_id: dict[str, Node] = {root_id: root}
        self._children: dict[str, list[str]] = {root_id: []}

    @classmethod
    def new_root(
        cls,
        *,
        world_id: str,
        workspace_snapshot_digest: str,
        observation_digest: str,
        root_id: str = "root",
        summary: str = "",
    ) -> DiscoveryTree:
        """Start a fresh tree whose root carries the fixed initial workspace."""
        tree = cls(world_id, root_id)
        root = _root_record(
            world_id,
            root_id,
            workspace_snapshot_digest=workspace_snapshot_digest,
            observation_digest=observation_digest,
            summary=summary,
        )
        tree._nodes[0] = root
        tree._by_id[root_id] = root
        return tree

    # ------------------------------------------------------------------ identity

    @property
    def world_id(self) -> str:
        return self._world_id

    @property
    def root_id(self) -> str:
        return self._root_id

    # ------------------------------------------------------------------ mutation

    def add_child(
        self,
        *,
        parent_id: str,
        node_id: str,
        round_index: int,
        branch_index: int,
        workspace_snapshot_digest: str,
        observation_digest: str,
        score: float,
        evaluator_status: EvaluatorStatus,
        summary: str = "",
        agent_session_digest: str | None = None,
        evaluator_result_digest: str | None = None,
        error_type: str | None = None,
    ) -> Node:
        """Append one persisted attempt result as a child of ``parent_id``.

        The tree assigns ``creation_index`` (= ``len(self)``) and ``depth``
        (= parent depth + 1).  The call is all-or-nothing: every rejection
        raises :class:`TreeInvariantViolation` before anything is recorded.

        A non-root parent may receive at most one child — ever.  Once it has
        one it is internal and can never be selected again, which is exactly
        the paper's "non-root has at most one recorded child" (P-09).
        """
        parent = self._by_id.get(parent_id)
        if parent is None:
            raise TreeInvariantViolation(f"unknown parent_id {parent_id!r}")
        if node_id in self._by_id:
            raise TreeInvariantViolation(f"node_id {node_id!r} already exists in this world")
        if parent_id != self._root_id and self._children[parent_id]:
            raise TreeInvariantViolation(
                f"parent {parent_id!r} already has a recorded child and is no longer a leaf"
            )
        if not math.isfinite(score):
            raise TreeInvariantViolation(
                f"score must be finite for node {node_id!r}, got {score!r}"
            )
        if round_index < 0:
            raise TreeInvariantViolation(f"round_index must be >= 0, got {round_index!r}")
        if branch_index < 0:
            raise TreeInvariantViolation(f"branch_index must be >= 0, got {branch_index!r}")

        index = len(self._nodes)
        node = Node(
            node_id=node_id,
            world_id=self._world_id,
            parent_id=parent_id,
            creation_index=index,
            round_index=round_index,
            branch_index=branch_index,
            depth=parent.depth + 1,
            workspace_snapshot_digest=workspace_snapshot_digest,
            observation_digest=observation_digest,
            score=score,
            evaluator_status=evaluator_status,
            agent_session_digest=agent_session_digest,
            evaluator_result_digest=evaluator_result_digest,
            error_type=error_type,
            summary=summary,
        )
        self._nodes.append(node)
        self._by_id[node_id] = node
        self._children[parent_id].append(node_id)
        self._children[node_id] = []
        return node

    # ------------------------------------------------------------------ lookups

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and node_id in self._by_id

    def node(self, node_id: str) -> Node:
        """The record for ``node_id``; ``KeyError`` when it is not in the world."""
        node = self._by_id.get(node_id)
        if node is None:
            raise KeyError(node_id)
        return node

    def get(self, node_id: str) -> Node | None:
        return self._by_id.get(node_id)

    def nodes(self) -> tuple[Node, ...]:
        """Every node, in ``creation_index`` order (root first)."""
        return tuple(self._nodes)

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self._nodes)

    def children_of(self, node_id: str) -> tuple[str, ...]:
        """Direct children in creation order; ``KeyError`` for an unknown node."""
        kids = self._children.get(node_id)
        if kids is None:
            raise KeyError(node_id)
        return tuple(kids)

    def leaves(self) -> tuple[str, ...]:
        """Revealed-subtree leaves in creation order, excluding ``root``.

        Online the whole tree is revealed, so this is every non-root node with
        no recorded child.  ``root`` is excluded because ``legal_actions`` adds
        it explicitly, exactly like the reference action set
        ``A(T) = {root} + leaves``.
        """
        return tuple(
            nid for nid in self.node_ids() if nid != self._root_id and not self._children[nid]
        )

    def legal_actions(self) -> tuple[str, ...]:
        """``(root, *leaves)``, deduped, in a stable order."""
        return tuple(dict.fromkeys((self._root_id, *self.leaves())))

    def is_leaf(self, node_id: str) -> bool:
        """Whether ``node_id`` is one of :meth:`leaves`; ``KeyError`` if unknown."""
        children = self._children.get(node_id)
        if children is None:
            raise KeyError(node_id)
        return node_id != self._root_id and not children

    def depth_of(self, node_id: str) -> int:
        return self.node(node_id).depth

    # ------------------------------------------------------------------ replay view

    def next_child(self, node_id: str, revealed: Collection[str]) -> str | None:
        """The child of ``node_id`` with the smallest unrevealed creation index.

        ``None`` means "no such child"; the child is never invented.  For a
        non-root this is its single recorded child (P-09); for ``root`` it is
        the earliest-created unrevealed direct child (P-10), so at most one
        root child is revealed per call.
        """
        shown = set(revealed)
        for child in self.children_of(node_id):
            if child not in shown:
                return child
        return None

    def observation(
        self,
        revealed: Collection[str],
        completed_rounds: int,
        max_parallelism: int,
    ) -> PrefixObservation:
        """The prefix a policy may see: revealed nodes only, oracle-exact."""
        return _build_observation(
            self._nodes, self._children, self._root_id, revealed, completed_rounds, max_parallelism
        )

    # ------------------------------------------------------------------ sealing

    def world_digest(self) -> str:
        """Content digest of the world so far (see :func:`_world_digest`)."""
        return _world_digest(self._world_id, self._root_id, self._nodes)

    def seal(self) -> tuple[int, float]:
        """``(B, S)``: ``B = |T| - 1`` and ``S = max non-root score`` (0.0 if none).

        ``S`` is reported honestly — a non-positive value stays negative so the
        controller can mark the world ``not_scoreable`` instead of silently
        substituting a fake denominator.
        """
        non_root = self._nodes[1:]
        best = max((n.score for n in non_root), default=0.0)
        return len(non_root), best


def _validate_world(nodes: Sequence[Node], root_id: str) -> tuple[Node, ...]:
    """Re-validate every §3 invariant; return the nodes in creation order."""
    records = tuple(nodes)
    for record in records:
        if not isinstance(record, Node):
            raise InvalidWorld("world records must be dream_rsi.domain.models.Node instances")
    if not records:
        raise InvalidWorld("missing root: world has no nodes")

    by_id: dict[str, Node] = {}
    for record in records:
        if record.node_id in by_id:
            raise InvalidWorld(f"duplicate node_id {record.node_id!r}")
        by_id[record.node_id] = record

    root = by_id.get(root_id)
    if root is None:
        raise InvalidWorld(f"missing root {root_id!r}")

    if any(not math.isfinite(n.score) for n in records):
        raise InvalidWorld("every score must be finite (NaN/Inf is a protocol bug)")
    if root.parent_id is not None:
        raise InvalidWorld("root must not have a parent")
    if root.creation_index != 0:
        raise InvalidWorld("root creation_index must be 0")
    if root.score != 0.0:
        raise InvalidWorld("root score must be 0.0")

    if len({n.creation_index for n in records}) != len(records):
        raise InvalidWorld("duplicate creation_index")
    if any(n.creation_index < 0 for n in records):
        raise InvalidWorld("negative creation_index")
    ordered = tuple(sorted(records, key=lambda n: n.creation_index))
    if tuple(n.creation_index for n in ordered) != tuple(range(len(ordered))):
        raise InvalidWorld("creation indices must be contiguous from 0 (exactly 0..N-1)")

    parents: dict[str, Node] = {}
    children: dict[str, list[str]] = {n.node_id: [] for n in ordered}
    for n in ordered:
        if n.node_id == root_id:
            continue
        parent = None if n.parent_id is None else by_id.get(n.parent_id)
        if parent is None:
            raise InvalidWorld(f"parent {n.parent_id!r} of {n.node_id!r} is not in the world")
        if parent.creation_index >= n.creation_index:
            raise InvalidWorld(
                f"parent {parent.node_id!r} must be created before child {n.node_id!r}"
            )
        parents[n.node_id] = parent
        children[parent.node_id].append(n.node_id)

    for parent_id, kids in children.items():
        if parent_id != root_id and len(kids) > 1:
            raise InvalidWorld(
                f"non-root node {parent_id!r} has {len(kids)} recorded children; at most 1 allowed"
            )

    # ``depth`` is derived (parent depth + 1) and is shown to policies, so a
    # corrupted record must not slip through.  The parent-before-child rule
    # above guarantees the parent's depth is already known in creation order.
    for n in ordered:
        if n.node_id == root_id:
            if n.depth != 0:
                raise InvalidWorld("root depth must be 0")
            continue
        if n.depth != parents[n.node_id].depth + 1:
            raise InvalidWorld(
                f"depth of {n.node_id!r} must be parent depth + 1 "
                f"({parents[n.node_id].depth + 1}), got {n.depth}"
            )

    # Defense in depth: with the checks above a cycle is already impossible,
    # so this only ever fires for a world whose records are internally
    # inconsistent in more than one way at once.
    seen = {root_id}
    todo = [root_id]
    while todo:
        for kid in children[todo.pop()]:
            if kid in seen:
                raise InvalidWorld("duplicate/cyclic reachability")
            seen.add(kid)
            todo.append(kid)
    if len(seen) != len(ordered):
        raise InvalidWorld("node unreachable from root")

    return ordered


class SealedWorld:
    """Immutable full world built from persisted Nodes; RE-VALIDATES every invariant.

    Construction goes through :meth:`from_nodes`, which raises
    :class:`InvalidWorld` for: a missing root; a root with a parent, a
    ``creation_index`` other than 0, or a non-zero score; a duplicate
    ``node_id``; a duplicate or non-contiguous ``creation_index``; a non-root
    whose parent is absent or was created at or after it; a non-root with two
    or more recorded children; a node unreachable from the root; a non-finite
    score; and a ``depth`` inconsistent with the parent chain.
    """

    def __init__(self, world_id: str, root_id: str, nodes: Sequence[Node]) -> None:
        """Validate and freeze ``nodes``; prefer :meth:`from_nodes`."""
        ordered = _validate_world(nodes, root_id)
        children: dict[str, list[str]] = {n.node_id: [] for n in ordered}
        for n in ordered:
            if n.node_id == root_id:
                continue
            parent_id = n.parent_id
            if parent_id is None:  # unreachable: validated above
                raise InvalidWorld(f"non-root node {n.node_id!r} has no parent")
            children[parent_id].append(n.node_id)
        self._world_id = world_id
        self._root_id = root_id
        self._nodes: tuple[Node, ...] = ordered
        self._by_id: dict[str, Node] = {n.node_id: n for n in ordered}
        self._children = {k: tuple(v) for k, v in children.items()}
        self._b_count = len(ordered) - 1
        self._s_best = max((n.score for n in ordered[1:]), default=0.0)

    @classmethod
    def from_nodes(
        cls,
        world_id: str,
        nodes: Sequence[Node],
        root_id: str = "root",
    ) -> SealedWorld:
        """Build a sealed world from persisted records, or raise ``InvalidWorld``."""
        return cls(world_id, root_id, nodes)

    # ------------------------------------------------------------------ identity

    @property
    def world_id(self) -> str:
        return self._world_id

    @property
    def root_id(self) -> str:
        return self._root_id

    @property
    def b_count(self) -> int:
        """``B = |T| - 1``; zero for a root-only world (not scoreable)."""
        return self._b_count

    @property
    def s_best(self) -> float:
        """``S = max non-root score``; 0.0 when there are no non-root nodes."""
        return self._s_best

    # ------------------------------------------------------------------ lookups

    def node(self, node_id: str) -> Node:
        node = self._by_id.get(node_id)
        if node is None:
            raise KeyError(node_id)
        return node

    def get(self, node_id: str) -> Node | None:
        return self._by_id.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> tuple[Node, ...]:
        """Every node, in ``creation_index`` order (root first)."""
        return self._nodes

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self._nodes)

    def children_of(self, node_id: str) -> tuple[str, ...]:
        kids = self._children.get(node_id)
        if kids is None:
            raise KeyError(node_id)
        return kids

    # ------------------------------------------------------------------ replay view

    def next_child(self, node_id: str, revealed: Collection[str]) -> str | None:
        """The child of ``node_id`` with the smallest unrevealed creation index.

        Identical semantics to :meth:`DiscoveryTree.next_child`: the child is
        never invented, P-09 applies to a non-root, P-10 to ``root``.
        """
        shown = set(revealed)
        for child in self.children_of(node_id):
            if child not in shown:
                return child
        return None

    def observation(
        self,
        revealed: Collection[str],
        completed_rounds: int,
        max_parallelism: int,
    ) -> PrefixObservation:
        """The prefix a policy may see: revealed nodes only, oracle-exact."""
        return _build_observation(
            self._nodes, self._children, self._root_id, revealed, completed_rounds, max_parallelism
        )

    # ------------------------------------------------------------------ sealing

    def world_digest(self) -> str:
        """Same content digest a :class:`DiscoveryTree` with these nodes yields."""
        return _world_digest(self._world_id, self._root_id, self._nodes)

    def seal(self) -> tuple[int, float]:
        """``(B, S)``, already validated at construction."""
        return self._b_count, self._s_best
