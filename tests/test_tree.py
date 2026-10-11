"""Tree / sealed-world tests for the strict-§3 domain module (TR01-TR12).

The reference oracle in ``reference/oracle.py`` is imported *here only*: it is
the independent specification of the prefix/revelation semantics, so every
observation assertion below is checked field by field against
``FullWorld.prefix`` and every child lookup against ``FullWorld.next_child``.

TR05 (``W`` cap), TR09 (zero backend/evaluator calls) and TR11 (K2 forced
stop) belong to the policy-validation / replay modules and are covered there;
the tree-side parts of TR12 (exactly one root; ``root`` first and unique in
``legal_actions``) are asserted here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for _path in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from reference.oracle import (  # noqa: E402
    FullWorld,
    Prefix,
)
from reference.oracle import Node as OracleNode  # noqa: E402

from dream_rsi.domain.errors import InvalidWorld, TreeInvariantViolation  # noqa: E402
from dream_rsi.domain.models import EvaluatorStatus, Node  # noqa: E402
from dream_rsi.domain.tree import DiscoveryTree, SealedWorld  # noqa: E402
from dream_rsi.policy.protocol import PrefixObservation  # noqa: E402

WORLD = "w-tree-test"
ROOT = "root"


# --------------------------------------------------------------------------- helpers


def _root_tree(world_id: str = WORLD) -> DiscoveryTree:
    return DiscoveryTree.new_root(
        world_id=world_id,
        workspace_snapshot_digest="ws-root",
        observation_digest="obs-root",
        summary="initial workspace",
    )


def _add(
    tree: DiscoveryTree,
    parent_id: str,
    node_id: str,
    *,
    score: float = 0.0,
    round_index: int = 0,
    branch_index: int = 0,
    status: EvaluatorStatus = EvaluatorStatus.SUCCESS,
    summary: str | None = None,
) -> Node:
    return tree.add_child(
        parent_id=parent_id,
        node_id=node_id,
        round_index=round_index,
        branch_index=branch_index,
        workspace_snapshot_digest=f"ws-{node_id}",
        observation_digest=f"obs-{node_id}",
        score=score,
        evaluator_status=status,
        summary=f"node {node_id}" if summary is None else summary,
    )


def _tree_ab() -> DiscoveryTree:
    """Root -> A -> A1 and Root -> B (the TR07 shape)."""
    tree = _root_tree()
    _add(tree, ROOT, "A", score=0.4)
    _add(tree, "A", "A1", score=0.5, round_index=1)
    _add(tree, ROOT, "B", score=0.7, branch_index=1)
    return tree


def _tree_abc() -> DiscoveryTree:
    """Root's three direct children A, B, C in creation order (TR01)."""
    tree = _root_tree()
    for branch, node_id in enumerate(("A", "B", "C")):
        _add(tree, ROOT, node_id, branch_index=branch)
    return tree


def _oracle_world(nodes: tuple[Node, ...]) -> FullWorld:
    """The oracle view of the same records.

    The oracle's free-text ``observation`` field is the counterpart of the
    node's human-readable ``summary`` (the digest fields are not public data).
    """
    return FullWorld(
        [
            OracleNode(n.node_id, n.parent_id, n.creation_index, n.score, n.summary)
            for n in nodes
        ]
    )


def _assert_same_prefix(observed: PrefixObservation, prefix: Prefix) -> None:
    assert [n.node_id for n in observed.nodes] == [n.id for n in prefix.nodes]
    assert [n.parent_id for n in observed.nodes] == [n.parent for n in prefix.nodes]
    assert [n.creation_index for n in observed.nodes] == [n.created for n in prefix.nodes]
    assert [n.score for n in observed.nodes] == [n.score for n in prefix.nodes]
    assert [n.summary for n in observed.nodes] == [n.observation for n in prefix.nodes]
    assert observed.legal_actions == prefix.legal
    assert observed.completed_rounds == prefix.rounds
    assert observed.max_parallelism == prefix.workers
    assert observed.node_ids == tuple(n.node_id for n in observed.nodes)


def _mk(
    node_id: str,
    parent_id: str | None,
    creation_index: int,
    depth: int,
    score: float = 0.0,
    *,
    round_index: int = 0,
    branch_index: int = 0,
    status: EvaluatorStatus = EvaluatorStatus.SUCCESS,
) -> Node:
    return Node(
        node_id=node_id,
        world_id=WORLD,
        parent_id=parent_id,
        creation_index=creation_index,
        round_index=round_index,
        branch_index=branch_index,
        depth=depth,
        workspace_snapshot_digest=f"ws-{node_id}",
        observation_digest=f"obs-{node_id}",
        score=score,
        evaluator_status=status,
        summary=f"node {node_id}",
    )


def _ab_nodes() -> tuple[Node, ...]:
    """Root(0) -> A(1) -> A1(2), Root -> B(3): a valid 4-node world."""
    return (
        _mk(ROOT, None, 0, 0),
        _mk("A", ROOT, 1, 1, 0.4),
        _mk("A1", "A", 2, 2, 0.5),
        _mk("B", ROOT, 3, 1, 0.7),
    )


def _tampered(node: Node, **overrides: Any) -> Node:
    """Rebuild a record bypassing ``Node.__post_init__`` (a corrupted row)."""
    clone = object.__new__(Node)
    for field_ in fields(Node):
        object.__setattr__(clone, field_.name, overrides.get(field_.name, getattr(node, field_.name)))
    return clone


# --------------------------------------------------------------------------- TR01


def test_tr01_root_children_revealed_in_creation_order() -> None:
    tree = _tree_abc()
    assert tree.children_of(ROOT) == ("A", "B", "C")

    revealed: set[str] = {ROOT}
    for expected in ("A", "B", "C"):
        assert tree.next_child(ROOT, revealed) == expected
        revealed.add(expected)
    assert tree.next_child(ROOT, revealed) is None
    assert revealed == set(tree.node_ids())

    # same answers as the reference oracle
    oracle = _oracle_world(tree.nodes())
    shown = {ROOT}
    for expected in ("A", "B", "C"):
        assert tree.next_child(ROOT, shown) == oracle.next_child(ROOT, shown) == expected
        shown.add(expected)
    assert tree.next_child(ROOT, shown) == oracle.next_child(ROOT, shown) is None


# --------------------------------------------------------------------------- TR02


def test_tr02_parent_with_a_child_is_no_longer_a_leaf() -> None:
    tree = _tree_ab()
    assert tree.children_of("A") == ("A1",)
    assert tree.is_leaf("A") is False
    assert "A" not in tree.leaves()
    assert "A" not in tree.legal_actions()

    with pytest.raises(TreeInvariantViolation):
        _add(tree, "A", "A2", score=0.9)

    # the rejected mutation left the tree untouched
    assert "A2" not in tree
    assert len(tree) == 4
    assert tree.node_ids() == (ROOT, "A", "A1", "B")
    assert tree.children_of("A") == ("A1",)


# --------------------------------------------------------------------------- TR03


def test_tr03_childless_node_has_no_invented_child() -> None:
    tree = _tree_ab()
    revealed = set(tree.node_ids())
    assert tree.next_child("B", revealed) is None
    assert _oracle_world(tree.nodes()).next_child("B", revealed) is None

    # B is still a legal parent (parent eligibility is not child existence)
    assert "B" in tree.leaves()
    assert "B" in tree.legal_actions()


# --------------------------------------------------------------------------- TR04


def test_tr04_root_appears_exactly_once_in_legal_actions() -> None:
    for tree in (_tree_ab(), _tree_abc(), _root_tree()):
        actions = tree.legal_actions()
        assert actions.count(ROOT) == 1
        assert actions[0] == ROOT
        assert actions == tuple(dict.fromkeys(actions))  # deduped, order stable


# --------------------------------------------------------------------------- TR06


def test_tr06_observation_matches_the_reference_oracle_field_by_field() -> None:
    tree = _tree_ab()
    oracle = _oracle_world(tree.nodes())
    every = set(tree.node_ids())
    subsets = (
        {ROOT},
        {ROOT, "A"},
        {ROOT, "A", "A1"},
        {ROOT, "A", "B"},
        {ROOT, "B"},
        every,
    )
    for rounds, revealed in enumerate(subsets):
        _assert_same_prefix(
            tree.observation(revealed, rounds, 2), oracle.prefix(set(revealed), rounds, 2)
        )
        # the sealed world observes exactly the same prefix
        sealed = SealedWorld.from_nodes(WORLD, tree.nodes())
        _assert_same_prefix(
            sealed.observation(revealed, rounds, 2), oracle.prefix(set(revealed), rounds, 2)
        )


def test_tr06_observation_never_leaks_hidden_nodes_or_scores() -> None:
    tree = _tree_ab()
    observed = tree.observation({ROOT}, 0, 2)
    assert [n.node_id for n in observed.nodes] == [ROOT]
    assert observed.node_ids == (ROOT,)
    assert observed.legal_actions == (ROOT,)
    assert [n.score for n in observed.nodes] == [0.0]
    assert 0.7 not in [n.score for n in observed.nodes]  # B's hidden score

    # a revealed id the world does not contain is ignored, never echoed back
    assert tree.observation({ROOT, "ghost"}, 0, 2).node_ids == (ROOT,)


# --------------------------------------------------------------------------- TR07


def test_tr07_out_of_order_batch_invents_nothing_and_mutates_nothing() -> None:
    tree = _tree_ab()
    oracle = _oracle_world(tree.nodes())
    before = tree.node_ids()
    digest_before = tree.world_digest()

    # round 1: root -> A
    revealed = frozenset({ROOT})
    assert tree.next_child(ROOT, revealed) == oracle.next_child(ROOT, set(revealed)) == "A"
    revealed = frozenset({ROOT, "A"})

    # round 2: a batch decided against the SAME pre-round prefix, in a different
    # order than creation order
    batch = ("A", ROOT)
    children = [tree.next_child(node_id, revealed) for node_id in batch]
    assert children == ["A1", "B"]
    assert children == [
        oracle.next_child(node_id, set(revealed)) for node_id in batch
    ]
    revealed = frozenset({*revealed, *children})

    # only genuine historical children were revealed; nothing was invented
    assert revealed == set(tree.node_ids())
    assert tree.node_ids() == before
    assert tree.world_digest() == digest_before

    # observations never mutate the revealed collection
    for subset in ({ROOT}, {ROOT, "A"}, set(tree.node_ids())):
        frozen = frozenset(subset)
        tree.observation(frozen, 1, 2)
        assert set(frozen) == subset


# --------------------------------------------------------------------------- TR08

DIGEST_WORLD = "w-digest"


def _digest_spec() -> list[dict[str, Any]]:
    return [
        {
            "parent_id": ROOT, "node_id": "A", "round_index": 0, "branch_index": 0,
            "workspace_snapshot_digest": "ws-A", "observation_digest": "obs-A",
            "score": 0.25, "evaluator_status": "SUCCESS", "summary": "node A",
        },
        {
            "parent_id": "A", "node_id": "A1", "round_index": 1, "branch_index": 0,
            "workspace_snapshot_digest": "ws-A1", "observation_digest": "obs-A1",
            "score": 0.5, "evaluator_status": "SUCCESS", "summary": "node A1",
        },
        {
            "parent_id": ROOT, "node_id": "B", "round_index": 0, "branch_index": 1,
            "workspace_snapshot_digest": "ws-B", "observation_digest": "obs-B",
            "score": -0.125, "evaluator_status": "FAILED", "summary": "node B",
        },
    ]


def _tree_from_spec(spec: list[dict[str, Any]], world_id: str = DIGEST_WORLD) -> DiscoveryTree:
    tree = DiscoveryTree.new_root(
        world_id=world_id,
        workspace_snapshot_digest="ws-root",
        observation_digest="obs-root",
        summary="initial workspace",
    )
    for item in spec:
        tree.add_child(
            parent_id=item["parent_id"],
            node_id=item["node_id"],
            round_index=item["round_index"],
            branch_index=item["branch_index"],
            workspace_snapshot_digest=item["workspace_snapshot_digest"],
            observation_digest=item["observation_digest"],
            score=item["score"],
            evaluator_status=EvaluatorStatus(item["evaluator_status"]),
            summary=item["summary"],
        )
    return tree


_SUBPROCESS_SCRIPT = """\
import json
import sys

sys.path.insert(0, sys.argv[1])

from dream_rsi.domain.models import EvaluatorStatus
from dream_rsi.domain.tree import DiscoveryTree

tree = DiscoveryTree.new_root(
    world_id=sys.argv[3],
    workspace_snapshot_digest="ws-root",
    observation_digest="obs-root",
    summary="initial workspace",
)
for item in json.loads(sys.argv[2]):
    tree.add_child(
        parent_id=item["parent_id"],
        node_id=item["node_id"],
        round_index=item["round_index"],
        branch_index=item["branch_index"],
        workspace_snapshot_digest=item["workspace_snapshot_digest"],
        observation_digest=item["observation_digest"],
        score=item["score"],
        evaluator_status=EvaluatorStatus(item["evaluator_status"]),
        summary=item["summary"],
    )
print(tree.world_digest())
"""


def test_tr08_world_digest_is_content_addressed() -> None:
    base = _tree_from_spec(_digest_spec())
    assert base.world_digest() == _tree_from_spec(_digest_spec()).world_digest()
    assert len(base.world_digest()) == 64
    assert set(base.world_digest()) <= set("0123456789abcdef")

    spec = _digest_spec()
    spec[0]["score"] = 0.26  # a single changed score
    assert _tree_from_spec(spec).world_digest() != base.world_digest()

    spec = _digest_spec()
    spec[0]["summary"] = "renamed"
    assert _tree_from_spec(spec).world_digest() != base.world_digest()

    spec = _digest_spec()
    spec[1]["round_index"] = 7
    assert _tree_from_spec(spec).world_digest() != base.world_digest()

    spec = _digest_spec()
    spec.append(
        {
            "parent_id": "A1", "node_id": "A2", "round_index": 2, "branch_index": 0,
            "workspace_snapshot_digest": "ws-A2", "observation_digest": "obs-A2",
            "score": 0.75, "evaluator_status": "SUCCESS", "summary": "node A2",
        }
    )
    assert _tree_from_spec(spec).world_digest() != base.world_digest()

    assert _root_tree(DIGEST_WORLD).world_digest() != base.world_digest()  # root only
    assert _tree_from_spec(_digest_spec(), world_id="w-other").world_digest() != (
        base.world_digest()
    )


def test_tr08_world_digest_is_stable_across_processes() -> None:
    digests = {
        _tree_from_spec(_digest_spec()).world_digest(),
        _tree_from_spec(_digest_spec()).world_digest(),
    }
    assert len(digests) == 1

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _SUBPROCESS_SCRIPT,
            str(SRC_DIR),
            json.dumps(_digest_spec()),
            DIGEST_WORLD,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() in digests

    # a different process with different content must not collide
    changed = _digest_spec()
    changed[2]["score"] = 0.0
    other = subprocess.run(
        [
            sys.executable,
            "-c",
            _SUBPROCESS_SCRIPT,
            str(SRC_DIR),
            json.dumps(changed),
            DIGEST_WORLD,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
    )
    assert other.returncode == 0, other.stderr
    assert other.stdout.strip() not in digests


# --------------------------------------------------------------------------- TR10


def test_tr10_fully_revealed_world_has_no_next_child() -> None:
    tree = _tree_ab()
    everything = set(tree.node_ids())
    for node_id in tree.node_ids():
        assert tree.next_child(node_id, everything) is None

    observed = tree.observation(everything, 3, 2)
    assert observed.node_ids == tree.node_ids()
    assert observed.legal_actions == (ROOT, "A1", "B")  # A is internal now
    _assert_same_prefix(observed, _oracle_world(tree.nodes()).prefix(everything, 3, 2))

    sealed = SealedWorld.from_nodes(WORLD, tree.nodes())
    for node_id in sealed.node_ids():
        assert sealed.next_child(node_id, everything) is None


# --------------------------------------------------------------------------- TR12


def test_tr12_strict_s3_has_exactly_one_root() -> None:
    tree = _tree_ab()
    roots = [n.node_id for n in tree.nodes() if n.parent_id is None]
    assert roots == [ROOT]
    assert tree.node_ids()[0] == ROOT
    assert {tree.depth_of(n.node_id) == 0 for n in tree.nodes() if n.parent_id is None} == {True}

    every = set(tree.node_ids())
    for revealed in (set(), {ROOT}, {ROOT, "A"}, {ROOT, "A1"}, every):
        observed = tree.observation(revealed, 0, 3)
        assert observed.legal_actions[0] == ROOT
        assert observed.legal_actions.count(ROOT) == 1


# ------------------------------------------------------- creation order and depth


def test_creation_index_is_contiguous_and_precedes_every_child() -> None:
    tree = _tree_abc()
    _add(tree, "A", "A1", score=0.1)
    _add(tree, "A1", "A11", score=0.2)

    records = tree.nodes()
    assert [n.creation_index for n in records] == list(range(len(tree)))
    index_of = {n.node_id: n.creation_index for n in records}
    for node in records:
        if node.parent_id is not None:
            assert index_of[node.parent_id] < node.creation_index
    assert tree.node_ids() == tuple(
        n.node_id for n in sorted(records, key=lambda n: n.creation_index)
    )


def test_depth_is_parent_depth_plus_one() -> None:
    tree = _tree_ab()
    depth = {n.node_id: n.depth for n in tree.nodes()}
    assert depth[ROOT] == 0
    for node in tree.nodes():
        if node.parent_id is not None:
            assert node.depth == depth[node.parent_id] + 1
    assert tree.depth_of("A") == 1
    assert tree.depth_of("A1") == 2
    assert tree.depth_of("B") == 1
    assert [n.depth for n in tree.observation(tree.node_ids(), 0, 2).nodes] == [0, 1, 2, 1]


def test_new_root_records_the_fixed_initial_workspace() -> None:
    tree = DiscoveryTree.new_root(
        world_id=WORLD,
        workspace_snapshot_digest="ws-root",
        observation_digest="obs-root",
        summary="initial workspace",
    )
    root = tree.node(ROOT)
    assert root.node_id == ROOT
    assert root.world_id == WORLD
    assert root.parent_id is None
    assert root.creation_index == 0
    assert root.depth == 0
    assert root.score == 0.0
    assert root.workspace_snapshot_digest == "ws-root"
    assert root.observation_digest == "obs-root"
    assert root.summary == "initial workspace"
    assert tree.world_id == WORLD and tree.root_id == ROOT
    assert len(tree) == 1

    # the bare constructor also starts from a legal root (default digests)
    plain = DiscoveryTree(WORLD)
    assert plain.node(ROOT).creation_index == 0
    assert plain.node(ROOT).score == 0.0
    assert plain.node(ROOT).parent_id is None


def test_add_child_records_every_field_and_assigns_index_and_depth() -> None:
    tree = _root_tree()
    node = tree.add_child(
        parent_id=ROOT,
        node_id="A",
        round_index=3,
        branch_index=2,
        workspace_snapshot_digest="ws-A",
        observation_digest="obs-A",
        score=1.25,
        evaluator_status=EvaluatorStatus.ERROR,
        summary="boom",
        agent_session_digest="sess-A",
        evaluator_result_digest="res-A",
        error_type="TimeoutError",
    )
    assert node.creation_index == 1
    assert node.depth == 1
    assert node.world_id == WORLD
    assert node.parent_id == ROOT
    assert node.round_index == 3 and node.branch_index == 2
    assert node.score == 1.25
    assert node.evaluator_status is EvaluatorStatus.ERROR
    assert node.error_type == "TimeoutError"
    assert node.agent_session_digest == "sess-A"
    assert node.evaluator_result_digest == "res-A"
    assert node.summary == "boom"
    assert tree.node("A") is node
    assert node in tree.nodes()

    child = _add(tree, "A", "A1")
    assert child.creation_index == 2
    assert child.depth == 2
    assert tree.leaves() == ("A1",)


def test_unknown_lookups_raise_key_error() -> None:
    tree = _tree_ab()
    assert ROOT in tree and "B" in tree
    assert "ghost" not in tree
    assert 42 not in tree
    assert tree.get("ghost") is None
    assert tree.get("A") is tree.node("A")
    with pytest.raises(KeyError):
        tree.node("ghost")
    with pytest.raises(KeyError):
        tree.children_of("ghost")
    with pytest.raises(KeyError):
        tree.next_child("ghost", {ROOT})
    with pytest.raises(KeyError):
        tree.depth_of("ghost")
    with pytest.raises(KeyError):
        tree.is_leaf("ghost")


def test_root_may_have_unlimited_children() -> None:
    tree = _root_tree()
    for branch in range(5):
        _add(tree, ROOT, f"N{branch}", branch_index=branch)
    assert tree.children_of(ROOT) == ("N0", "N1", "N2", "N3", "N4")
    assert tree.leaves() == ("N0", "N1", "N2", "N3", "N4")
    assert tree.legal_actions() == (ROOT, "N0", "N1", "N2", "N3", "N4")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"parent_id": "ghost"}, id="unknown-parent"),
        pytest.param({"parent_id": "A", "node_id": "A1"}, id="duplicate-node-id"),
        pytest.param({"parent_id": "A", "node_id": "A2"}, id="parent-no-longer-a-leaf"),
        pytest.param({"score": float("nan")}, id="nan-score"),
        pytest.param({"score": float("inf")}, id="inf-score"),
        pytest.param({"round_index": -1}, id="round-index"),
        pytest.param({"branch_index": -1}, id="branch-index"),
    ],
)
def test_add_child_rejects_illegal_mutations(kwargs: dict[str, Any]) -> None:
    tree = _tree_ab()
    full: dict[str, Any] = {
        "parent_id": ROOT,
        "node_id": "N",
        "round_index": 0,
        "branch_index": 0,
        "workspace_snapshot_digest": "ws-N",
        "observation_digest": "obs-N",
        "score": 0.0,
        "evaluator_status": EvaluatorStatus.SUCCESS,
        **kwargs,
    }
    with pytest.raises(TreeInvariantViolation):
        tree.add_child(**full)
    assert tree.node_ids() == (ROOT, "A", "A1", "B")


def test_seal_empty_and_populated_worlds() -> None:
    assert _root_tree().seal() == (0, 0.0)
    assert SealedWorld.from_nodes(WORLD, _root_tree().nodes()).seal() == (0, 0.0)
    assert SealedWorld.from_nodes(WORLD, _root_tree().nodes()).b_count == 0
    assert SealedWorld.from_nodes(WORLD, _root_tree().nodes()).s_best == 0.0

    tree = _tree_ab()
    assert tree.seal() == (3, 0.7)
    assert SealedWorld.from_nodes(WORLD, tree.nodes()).seal() == (3, 0.7)

    negative = _root_tree()
    _add(negative, ROOT, "A", score=-2.0)
    _add(negative, "A", "A1", score=-1.0)
    assert negative.seal() == (2, -1.0)  # honest: never clamped to a fake 0/first


# ------------------------------------------------------------- root-only world


def test_root_only_world_is_legal_but_empty() -> None:
    tree = _root_tree()
    assert len(tree) == 1
    assert tree.node_ids() == (ROOT,)
    assert tree.leaves() == ()
    assert tree.legal_actions() == (ROOT,)
    assert tree.children_of(ROOT) == ()
    assert tree.is_leaf(ROOT) is False
    assert tree.next_child(ROOT, {ROOT}) is None
    observed = tree.observation({ROOT}, 0, 1)
    assert observed.node_ids == (ROOT,)
    assert observed.legal_actions == (ROOT,)
    _assert_same_prefix(observed, _oracle_world(tree.nodes()).prefix({ROOT}, 0, 1))


# ------------------------------------------------------------------ sealed world


def test_sealed_world_matches_the_tree_and_normalises_order() -> None:
    tree = _tree_ab()
    sealed = SealedWorld.from_nodes(WORLD, tree.nodes())
    assert sealed.world_id == WORLD and sealed.root_id == ROOT
    assert len(sealed) == len(tree) == 4
    assert sealed.node_ids() == tree.node_ids()
    assert sealed.nodes() == tree.nodes()
    assert sealed.children_of(ROOT) == ("A", "B")
    assert sealed.children_of("A") == ("A1",)
    assert sealed.children_of("B") == ()
    assert sealed.get("A1") is not None
    with pytest.raises(KeyError):
        sealed.node("ghost")
    assert sealed.get("ghost") is None

    # records arriving in any order still normalise to creation order
    shuffled = SealedWorld.from_nodes(WORLD, tuple(reversed(tree.nodes())))
    assert shuffled.node_ids() == tree.node_ids()
    assert shuffled.world_digest() == tree.world_digest()
    assert shuffled.seal() == tree.seal()
    for revealed in ({ROOT}, {ROOT, "A"}, set(tree.node_ids())):
        assert shuffled.observation(revealed, 1, 2) == tree.observation(revealed, 1, 2)


def test_sealed_world_honours_a_custom_root_id() -> None:
    tree = _root_tree()
    tree.add_child(
        parent_id=ROOT,
        node_id="A",
        round_index=0,
        branch_index=0,
        workspace_snapshot_digest="ws-A",
        observation_digest="obs-A",
        score=0.3,
        evaluator_status=EvaluatorStatus.SUCCESS,
        summary="node A",
    )
    # renaming the root record means the old root id is simply unknown
    nodes = tuple(
        replace(node, node_id="R") if node.parent_id is None else node for node in tree.nodes()
    )
    nodes = tuple(
        replace(node, parent_id="R") if node.parent_id == ROOT else node for node in nodes
    )
    sealed = SealedWorld.from_nodes(WORLD, nodes, root_id="R")
    assert sealed.root_id == "R"
    assert sealed.node_ids() == ("R", "A")
    assert sealed.children_of("R") == ("A",)


def _invalid_world_cases() -> list[Any]:
    root, a, a1, b = _ab_nodes()
    unreachable = (
        _mk(ROOT, None, 0, 0),
        _mk("A", ROOT, 1, 1),
        # a two-node island: each record names the other as parent, so the
        # chain never reaches the root.  (With the parent-before-child rule in
        # force this is also detected earlier; the reachability pass is
        # defence in depth, exactly like the oracle's own check.)
        _mk("M", "N", 2, 2),
        _mk("N", "M", 3, 1),
    )
    return [
        pytest.param((), id="empty-world"),
        pytest.param((a, a1, b), id="missing-root"),
        pytest.param([{"node_id": ROOT}], id="not-a-node-record"),
        pytest.param((replace(root, parent_id="A"), a, a1, b), id="root-has-parent"),
        pytest.param(
            (replace(root, creation_index=1), replace(a, creation_index=0), a1, b),
            id="root-index-not-zero",
        ),
        pytest.param((replace(root, score=0.5), a, a1, b), id="root-score-not-zero"),
        pytest.param((root, a, replace(a1, node_id="A"), b), id="duplicate-node-id"),
        pytest.param((root, a, replace(a1, creation_index=1), b), id="duplicate-index"),
        pytest.param((root, a, replace(a1, creation_index=5), b), id="non-contiguous-index"),
        pytest.param((root, a, replace(a1, parent_id="ghost"), b), id="parent-absent"),
        pytest.param(
            (_mk(ROOT, None, 0, 0), _mk("B", "A", 1, 2), _mk("A", ROOT, 2, 1),
             _mk("A1", "A", 3, 2)),
            id="parent-created-after-child",
        ),
        pytest.param((root, a, a1, _mk("A2", "A", 3, 2, 0.9)), id="nonroot-two-children"),
        pytest.param((root, a, replace(a1, depth=5), b), id="wrong-depth"),
        pytest.param((_tampered(a, score=float("nan")), root, a1, b), id="nan-score"),
        pytest.param((_tampered(a, score=float("inf")), root, a1, b), id="inf-score"),
        pytest.param(unreachable, id="unreachable-component"),
    ]


@pytest.mark.parametrize("nodes", _invalid_world_cases())
def test_sealed_world_rejects_invalid_worlds(nodes: Any) -> None:
    with pytest.raises(InvalidWorld):
        SealedWorld.from_nodes(WORLD, nodes)


def test_sealed_world_rejects_a_world_without_exactly_one_root() -> None:
    # Root -> A -> A1 and Root -> B is the only accepted shape; a second
    # parentless node has no creation_index 0 and cannot be a root.
    root, a, a1, b = _ab_nodes()
    second_root = _mk(
        "B", None, a1.creation_index, 1, 0.7
    )  # duplicate index, no root parent
    with pytest.raises(InvalidWorld):
        SealedWorld.from_nodes(WORLD, (root, a, second_root, a1, b))
