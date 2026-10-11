"""Storage-layer tests: content-addressed store integrity and registry round-trips."""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from dream_rsi.domain.errors import IntegrityError
from dream_rsi.domain.models import (
    AlgorithmMode,
    Attempt,
    AttemptState,
    EvaluatorStatus,
    Event,
    Node,
    PolicyState,
    PolicyVersion,
    ReplayEpisode,
    Revision,
    Run,
    RunMode,
    RunState,
    ScoreRecord,
    StopReason,
    World,
    WorldState,
)
from dream_rsi.storage.cas import ContentAddressedStore
from dream_rsi.storage.sqlite_registry import Registry

HEX = {name: name * 64 for name in "abcdef"}


@dataclass(frozen=True)
class Sample:
    run: Run
    world: World
    other_world: World
    nodes: tuple[Node, ...]
    attempt: Attempt
    policy: PolicyVersion
    replay: ReplayEpisode
    score: ScoreRecord
    revision: Revision
    event: Event


def _sample() -> Sample:
    run = Run(
        run_id="run-1",
        mode=RunMode.OFFLINE_FIXTURE,
        algorithm_mode=AlgorithmMode.STRICT_S3,
        task_spec_digest=HEX["a"],
        provider_id="noop-for-tests",
        agent_model="fixture-model",
        fixed_evaluator_digest=HEX["b"],
        cfg_digest=HEX["c"],
    )
    world = World(
        world_id="world-1",
        outer_t=0,
        run_id="run-1",
        root_id="n0",
        policy_version_id="pol-0",
    )
    other_world = World(
        world_id="world-2",
        outer_t=1,
        run_id="run-1",
        root_id="n9",
        policy_version_id="pol-0",
    )
    nodes = (
        Node(
            node_id="n0",
            world_id="world-1",
            parent_id=None,
            creation_index=0,
            round_index=0,
            branch_index=0,
            depth=0,
            workspace_snapshot_digest=HEX["a"],
            observation_digest=HEX["b"],
            score=0.1,
            evaluator_status=EvaluatorStatus.SUCCESS,
            summary="root",
        ),
        Node(
            node_id="n1",
            world_id="world-1",
            parent_id="n0",
            creation_index=1,
            round_index=1,
            branch_index=0,
            depth=1,
            workspace_snapshot_digest=HEX["c"],
            observation_digest=HEX["d"],
            score=0.9,
            evaluator_status=EvaluatorStatus.SUCCESS,
            agent_session_digest=HEX["e"],
            evaluator_result_digest=HEX["f"],
            summary="best child",
        ),
        Node(
            node_id="n2",
            world_id="world-1",
            parent_id="n0",
            creation_index=2,
            round_index=1,
            branch_index=1,
            depth=1,
            workspace_snapshot_digest=HEX["d"],
            observation_digest=HEX["e"],
            score=0.4,
            evaluator_status=EvaluatorStatus.FAILED,
            error_type="AssertionError",
            summary="failed candidate",
        ),
    )
    attempt = Attempt(
        attempt_id="att-1",
        world_id="world-1",
        parent_id="n0",
        round_index=1,
        invocation_id="inv-1",
    )
    policy = PolicyVersion(
        version_id="pol-0",
        run_id="run-1",
        iteration_t=0,
        m=0,
        parent_version=None,
        source_sha256=HEX["a"],
        source_artifact_id="artifact-0",
        origin="incumbent",
    )
    replay = ReplayEpisode(
        episode_id="ep-1",
        world_id="world-1",
        policy_version_id="pol-0",
        rho=0.5,
        n=3,
        kstar=5,
        best_score=0.9,
        attained_quality=0.7,
        selected_batches=(("root",), ("n1", "n2")),
        revealed_order=("root", "n1", "n2"),
        stop_reason=StopReason.STOP_EMPTY_BATCH,
        replay_digest=HEX["f"],
        no_progress_count=1,
        discovery_agent_calls=2,
        evaluator_calls=3,
    )
    score = ScoreRecord(
        policy_version_id="pol-0",
        world_id="world-1",
        b_count=2,
        s_best=0.9,
        five_points=((0, 0.0), (1, 0.4), (2, 0.6), (3, 0.8), (4, 0.9)),
        frontier=((1, 0.4), (2, 0.6), (3, 0.8)),
        hv=0.54,
        parallel_penalty=0.05,
        lambda_weight=0.1,
        value=0.4631666667,
    )
    revision = Revision(
        iteration_t=0,
        m=0,
        input_policy_id="pol-0",
        output_policy_id=None,
        status="PENDING",
        feedback_digest=HEX["c"],
    )
    event = Event(run_id="run-1", kind="RUN_CREATED", payload={"note": "hello", "n": 1})
    return Sample(run, world, other_world, nodes, attempt, policy, replay, score, revision, event)


def _populate(reg: Registry, sample: Sample) -> None:
    """Insert one of every DTO, in the registry's required parent-first order."""
    reg.create_run(sample.run)
    reg.add_world(sample.world)
    reg.add_world(sample.other_world)
    reg.add_nodes(sample.nodes)
    reg.add_attempt(sample.attempt)
    reg.add_policy(sample.policy)
    reg.add_replay(sample.replay)
    reg.add_score(sample.score)
    reg.add_revision(sample.revision)
    reg.append_event(sample.event)


@pytest.fixture()
def sample() -> Sample:
    """One sample instance per test: the DTOs carry creation timestamps."""
    return _sample()


@pytest.fixture()
def registry(tmp_path: Path, sample: Sample) -> Iterator[Registry]:
    with Registry(tmp_path / "registry.sqlite3") as reg:
        _populate(reg, sample)
        yield reg


@pytest.fixture()
def cas(tmp_path: Path) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_path / "cas")


def _raw_connection(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), timeout=15.0)


def _make_symlink(source: Path, link: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(source, link, target_is_directory=directory)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows privilege dependent
        pytest.skip("this machine does not permit creating symlinks")


# --------------------------------------------------------------------- DTOs
def test_roundtrip_every_dto(registry: Registry, sample: Sample) -> None:
    assert registry.get_run("run-1") == sample.run
    assert registry.get_world("world-1") == sample.world
    assert registry.get_world("world-2") == sample.other_world
    assert registry.get_node("n1") == sample.nodes[1]
    assert registry.get_attempt("att-1") == sample.attempt
    assert registry.get_policy("pol-0") == sample.policy
    assert registry.list_replays() == [sample.replay]
    assert registry.list_scores() == [sample.score]
    assert registry.list_revisions("run-1") == [sample.revision]
    assert registry.list_events("run-1") == [sample.event]


def test_list_helpers_return_every_row(registry: Registry, sample: Sample) -> None:
    assert registry.list_runs() == [sample.run]
    assert registry.list_worlds("run-1") == [sample.world, sample.other_world]
    assert registry.list_nodes("world-1") == list(sample.nodes)
    assert registry.list_attempts("world-1") == [sample.attempt]
    assert registry.list_policies("run-1") == [sample.policy]


def test_missing_rows_read_back_as_none(registry: Registry) -> None:
    assert registry.get_run("nope") is None
    assert registry.get_world("nope") is None
    assert registry.get_node("nope") is None
    assert registry.get_attempt("nope") is None
    assert registry.get_policy("nope") is None


# ---------------------------------------------------------------- SQLite WAL
def test_journal_mode_is_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.sqlite3"
    with Registry(db_path):
        pass
    conn = _raw_connection(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_second_connection_can_read_while_open(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.sqlite3"
    sample = _sample()
    with Registry(db_path) as writer:
        writer.create_run(sample.run)
        with Registry(db_path) as reader:
            assert reader.get_run("run-1") == sample.run
        # A reader that opens while a write transaction is in flight sees the
        # pre-commit snapshot instead of blocking or exploding (WAL).
        with writer.transaction():
            writer.update_run_state("run-1", RunState.ONLINE_RUNNING)
            with Registry(db_path) as mid_flight:
                open_run = mid_flight.get_run("run-1")
                assert open_run is not None and open_run.state is RunState.CREATED
        with Registry(db_path) as after_commit:
            committed = after_commit.get_run("run-1")
            assert committed is not None and committed.state is RunState.ONLINE_RUNNING


def test_reopen_keeps_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "reopen.sqlite3"
    sample = _sample()
    with Registry(db_path) as reg:
        _populate(reg, sample)
    with Registry(db_path) as reg:
        assert reg.get_run("run-1") == sample.run
        assert reg.list_nodes("world-1") == list(sample.nodes)


# ------------------------------------------------------------- constraints
def test_duplicate_primary_key_raises(registry: Registry) -> None:
    with pytest.raises(IntegrityError):
        registry.create_run(_sample().run)
    with pytest.raises(IntegrityError):
        registry.add_world(_sample().world)
    with pytest.raises(IntegrityError):
        registry.add_node(_sample().nodes[0])
    with pytest.raises(IntegrityError):
        registry.add_policy(_sample().policy)


def test_duplicate_score_is_rejected(registry: Registry) -> None:
    with pytest.raises(IntegrityError):
        registry.add_score(_sample().score)


def test_duplicate_creation_index_is_rejected(registry: Registry) -> None:
    clash = replace(_sample().nodes[2], node_id="n3", creation_index=1)
    with pytest.raises(IntegrityError):
        registry.add_node(clash)


def test_duplicate_invocation_id_is_rejected(registry: Registry) -> None:
    clash = replace(_sample().attempt, attempt_id="att-2")
    with pytest.raises(IntegrityError):
        registry.add_attempt(clash)


def test_child_without_parent_is_rejected(registry: Registry) -> None:
    orphan = replace(_sample().nodes[0], node_id="n-orphan", world_id="world-missing")
    with pytest.raises(IntegrityError):
        registry.add_node(orphan)


def test_update_missing_row_raises(registry: Registry) -> None:
    with pytest.raises(IntegrityError):
        registry.update_run_state("nope", RunState.ONLINE_RUNNING)
    with pytest.raises(IntegrityError):
        registry.update_attempt_state("nope", AttemptState.FINISHED)
    with pytest.raises(IntegrityError):
        registry.update_policy_state("nope", PolicyState.REJECTED)


def test_operations_after_close_raise(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "closed.sqlite3")
    reg.close()
    reg.close()  # idempotent
    with pytest.raises(RuntimeError):
        reg.get_run("run-1")


# ------------------------------------------------------------- ordering
def test_list_nodes_is_ordered_by_creation_index(tmp_path: Path) -> None:
    sample = _sample()
    with Registry(tmp_path / "order.sqlite3") as reg:
        reg.create_run(sample.run)
        reg.add_world(sample.world)
        for node in (sample.nodes[2], sample.nodes[0], sample.nodes[1]):
            reg.add_node(node)
        assert [node.node_id for node in reg.list_nodes("world-1")] == ["n0", "n1", "n2"]


def test_list_worlds_sealed_only_filters(registry: Registry) -> None:
    registry.seal_world(
        "world-2",
        state=WorldState.SEALED_UNSCOREABLE,
        world_digest=HEX["a"],
        b_count=1,
        s_best=0.5,
    )
    assert len(registry.list_worlds("run-1")) == 2
    sealed = registry.list_worlds("run-1", sealed_only=True)
    assert [world.world_id for world in sealed] == ["world-2"]
    assert sealed[0].state is WorldState.SEALED_UNSCOREABLE


def test_list_events_keeps_append_order(registry: Registry) -> None:
    extra = Event(run_id="run-1", kind="WORLD_SEALED", payload={"world_id": "world-1"})
    registry.append_event(extra)
    kinds = [event.kind for event in registry.list_events("run-1")]
    assert kinds == ["RUN_CREATED", "WORLD_SEALED"]


# ---------------------------------------------------------- state updates
def test_update_run_state_records_note_and_finish(registry: Registry) -> None:
    registry.update_run_state("run-1", RunState.ONLINE_RUNNING, note="started")
    running = registry.get_run("run-1")
    assert running is not None
    assert running.state is RunState.ONLINE_RUNNING
    assert running.status_note == "started"
    assert running.started_at is not None
    assert running.finished_at is None

    registry.update_run_state("run-1", RunState.COMPLETE, note="done", finished=True)
    done = registry.get_run("run-1")
    assert done is not None
    assert done.state is RunState.COMPLETE
    assert done.status_note == "done"
    assert done.finished_at is not None


def test_update_attempt_state_records_failure(registry: Registry) -> None:
    registry.update_attempt_state(
        "att-1",
        AttemptState.ERROR_RECORDED,
        failure_reason="provider returned 500",
        workspace_snapshot_digest=HEX["c"],
    )
    attempt = registry.get_attempt("att-1")
    assert attempt is not None
    assert attempt.state is AttemptState.ERROR_RECORDED
    assert attempt.failure_reason == "provider returned 500"
    assert attempt.workspace_snapshot_digest == HEX["c"]


def test_update_policy_state_records_rejection(registry: Registry) -> None:
    registry.update_policy_state(
        "pol-0",
        PolicyState.REJECTED,
        static_validation_status="FAILED",
        sandbox_validation_status="SKIPPED",
        failure_reason="imports outside the whitelist",
    )
    policy = registry.get_policy("pol-0")
    assert policy is not None
    assert policy.state is PolicyState.REJECTED
    assert policy.static_validation_status == "FAILED"
    assert policy.sandbox_validation_status == "SKIPPED"
    assert policy.failure_reason == "imports outside the whitelist"


def test_seal_world_is_idempotent_and_final(registry: Registry) -> None:
    registry.seal_world(
        "world-1",
        state=WorldState.SEALED_SCOREABLE,
        world_digest=HEX["a"],
        b_count=2,
        s_best=0.9,
        note="batch complete",
    )
    sealed = registry.get_world("world-1")
    assert sealed is not None
    assert sealed.state is WorldState.SEALED_SCOREABLE
    assert sealed.world_digest == HEX["a"]
    assert sealed.b_count == 2
    assert sealed.s_best == 0.9
    assert sealed.sealed_at is not None
    assert sealed.seal_note == "batch complete"

    # Repeating the identical seal is a no-op (resume-safe).
    registry.seal_world(
        "world-1",
        state=WorldState.SEALED_SCOREABLE,
        world_digest=HEX["a"],
        b_count=2,
        s_best=0.9,
    )
    assert registry.get_world("world-1") == sealed

    # Sealing again with different numbers would rewrite history.
    with pytest.raises(IntegrityError):
        registry.seal_world(
            "world-1",
            state=WorldState.SEALED_SCOREABLE,
            world_digest=HEX["b"],
            b_count=3,
            s_best=1.0,
        )
    with pytest.raises(ValueError):
        registry.seal_world(
            "world-2",
            state=WorldState.BUILDING,
            world_digest=HEX["a"],
            b_count=1,
            s_best=0.1,
        )


def test_quarantine_world_keeps_reason_and_blocks_scoring(registry: Registry) -> None:
    registry.quarantine_world("world-1", "crash left a half-written batch")
    world = registry.get_world("world-1")
    assert world is not None
    assert world.state is WorldState.QUARANTINED
    assert world.seal_note is not None and "half-written" in world.seal_note
    assert registry.list_worlds("run-1", sealed_only=True) == []
    with pytest.raises(IntegrityError):
        registry.seal_world(
            "world-1",
            state=WorldState.SEALED_SCOREABLE,
            world_digest=HEX["a"],
            b_count=0,
            s_best=0.0,
        )


def test_add_nodes_is_one_transaction(registry: Registry) -> None:
    dupe = replace(_sample().nodes[2], node_id="n4", creation_index=1)
    with pytest.raises(IntegrityError):
        registry.add_nodes([replace(_sample().nodes[2], node_id="n3", creation_index=3), dupe])
    assert [node.node_id for node in registry.list_nodes("world-1")] == ["n0", "n1", "n2"]


def test_transaction_rolls_back_on_exception(registry: Registry) -> None:
    with pytest.raises(RuntimeError, match="boom"), registry.transaction():
        registry.update_run_state("run-1", RunState.ONLINE_RUNNING)
        registry.append_event(Event(run_id="run-1", kind="WILL_VANISH"))
        raise RuntimeError("boom")
    run = registry.get_run("run-1")
    assert run is not None and run.state is RunState.CREATED
    assert [event.kind for event in registry.list_events("run-1")] == ["RUN_CREATED"]


def test_nested_transactions_commit_once(registry: Registry) -> None:
    sample = _sample()
    with registry.transaction():
        with registry.transaction():
            registry.add_nodes(
                [
                    replace(node, node_id=f"{node.node_id}-x", creation_index=node.creation_index + 10)
                    for node in sample.nodes
                ]
            )
        assert len(registry.list_nodes("world-1")) == 6
    assert len(registry.list_nodes("world-1")) == 6


# ------------------------------------------------------------- tamper paths
def test_corrupt_payload_raises_integrity_error(tmp_path: Path) -> None:
    db_path = tmp_path / "corrupt.sqlite3"
    with Registry(db_path) as reg:
        _populate(reg, _sample())
    conn = _raw_connection(db_path)
    try:
        conn.execute("UPDATE runs SET payload = '{not json' WHERE run_id = 'run-1'")
        conn.commit()
    finally:
        conn.close()
    with Registry(db_path) as reg, pytest.raises(IntegrityError):
        reg.get_run("run-1")


def test_payload_column_disagreement_raises_integrity_error(tmp_path: Path) -> None:
    db_path = tmp_path / "edited.sqlite3"
    with Registry(db_path) as reg:
        _populate(reg, _sample())
    conn = _raw_connection(db_path)
    try:
        conn.execute("UPDATE runs SET state = 'COMPLETE' WHERE run_id = 'run-1'")
        conn.commit()
    finally:
        conn.close()
    with Registry(db_path) as reg, pytest.raises(IntegrityError):
        reg.get_run("run-1")


def test_unknown_schema_version_raises_integrity_error(tmp_path: Path) -> None:
    db_path = tmp_path / "future.sqlite3"
    with Registry(db_path) as reg:
        _populate(reg, _sample())
    conn = _raw_connection(db_path)
    row = conn.execute("SELECT payload FROM runs WHERE run_id = 'run-1'").fetchone()
    data = json.loads(row[0])
    data["schema_version"] = 99
    try:
        conn.execute(
            "UPDATE runs SET payload = ? WHERE run_id = 'run-1'",
            (json.dumps(data),),
        )
        conn.commit()
    finally:
        conn.close()
    with Registry(db_path) as reg, pytest.raises(IntegrityError):
        reg.get_run("run-1")


# ------------------------------------------------------------------- CAS
def test_put_get_bytes_roundtrip(cas: ContentAddressedStore) -> None:
    payload = b"workspace snapshot bytes"
    digest = cas.put_bytes(payload)
    assert len(digest) == 64
    assert cas.exists(digest)
    assert cas.verify(digest)
    assert cas.get_bytes(digest) == payload
    assert cas.path_for(digest) == cas.root / digest[:2] / digest[2:4] / digest
    assert cas.put_bytes(payload) == digest  # idempotent


def test_put_file_roundtrip(cas: ContentAddressedStore, tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"\x00\x01binary\xff")
    digest = cas.put_file(source)
    assert cas.get_bytes(digest) == b"\x00\x01binary\xff"
    assert cas.put_file(source) == digest


def test_put_json_is_canonical(cas: ContentAddressedStore) -> None:
    left = cas.put_json({"b": 1, "a": [2, 3], "c": {"y": True, "x": None}})
    right = cas.put_json({"c": {"x": None, "y": True}, "a": [2, 3], "b": 1})
    assert left == right
    assert cas.get_json(left) == {"a": [2, 3], "b": 1, "c": {"x": None, "y": True}}


def test_digest_validation_rejects_junk(cas: ContentAddressedStore) -> None:
    for bad in ("", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65):
        with pytest.raises(ValueError):
            cas.get_bytes(bad)
        with pytest.raises(ValueError):
            cas.path_for(bad)
        with pytest.raises(ValueError):
            cas.exists(bad)


def test_missing_blob_raises_integrity_error(cas: ContentAddressedStore) -> None:
    missing = "0" * 64
    assert not cas.exists(missing)
    assert not cas.verify(missing)
    with pytest.raises(IntegrityError):
        cas.get_bytes(missing)


def test_tampered_blob_is_detected(cas: ContentAddressedStore) -> None:
    digest = cas.put_bytes(b"honest content")
    cas.path_for(digest).write_bytes(b"tampered content")
    assert not cas.verify(digest)
    with pytest.raises(IntegrityError):
        cas.get_bytes(digest)
    with pytest.raises(IntegrityError):
        cas.get_json(digest)


def test_put_bytes_leaves_no_partial_blob(cas: ContentAddressedStore) -> None:
    digest = cas.put_bytes(b"x" * 10)
    blob_dir = cas.path_for(digest).parent
    leftovers = [p.name for p in blob_dir.iterdir() if p.name != digest]
    assert leftovers == []


def test_put_tree_is_deterministic(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    first = tmp_path / "tree-a"
    second = tmp_path / "tree-b"
    for root in (first, second):
        (root / "src").mkdir(parents=True)
        (root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
        (root / "README.md").write_text("docs\n", encoding="utf-8")
    digest_a = cas.put_tree(first)
    digest_b = cas.put_tree(second)
    assert digest_a == digest_b
    assert cas.exists(digest_a)  # the manifest itself is stored
    assert b"src/main.py" in cas.get_bytes(digest_a)

    (second / "README.md").write_text("changed\n", encoding="utf-8")
    assert cas.put_tree(second) != digest_a

    (second / "README.md").write_text("docs\n", encoding="utf-8")
    (second / "extra.txt").write_text("new\n", encoding="utf-8")
    assert cas.put_tree(second) != digest_a


def test_put_tree_excludes_by_name(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    tree = tmp_path / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "keep.txt").write_text("keep\n", encoding="utf-8")
    (tree / "nested" / "keep.txt").write_text("keep too\n", encoding="utf-8")
    (tree / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (tree / "nested" / "server.key").write_text("PRIVATE\n", encoding="utf-8")
    (tree / ".git").mkdir()
    (tree / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    with_secrets = cas.put_tree(tree)
    without_secrets = cas.put_tree(tree, exclude=(".env", "*.key", ".git"))
    assert with_secrets != without_secrets

    for name in (".env", "nested/server.key", ".git/config"):
        assert name.encode("utf-8") in cas.get_bytes(with_secrets)
        assert name.encode("utf-8") not in cas.get_bytes(without_secrets)

    # Removing the secrets must not change the digest of the excluded tree.
    (tree / ".env").unlink()
    (tree / "nested" / "server.key").unlink()
    (tree / ".git" / "config").unlink()
    assert cas.put_tree(tree, exclude=(".env", "*.key", ".git")) == without_secrets


def test_put_tree_refuses_link_escaping_root(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not hash me\n", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "keep.txt").write_text("keep\n", encoding="utf-8")
    _make_symlink(outside / "secret.txt", tree / "escape.txt")
    with pytest.raises(IntegrityError):
        cas.put_tree(tree)


def test_put_tree_refuses_junction_out_of_tree(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not hash me\n", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    _make_symlink(outside, tree / "escape-dir", directory=True)
    with pytest.raises(IntegrityError):
        cas.put_tree(tree)


def test_put_tree_records_internal_link_without_traversing(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas")
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "inner.txt").write_text("inner\n", encoding="utf-8")
    _make_symlink(tree / "sub", tree / "sub-alias", directory=True)
    digest = cas.put_tree(tree)
    manifest = cas.get_bytes(digest)
    assert b"sub-alias\0LINK:" in manifest
    # The alias is recorded, never walked through: no duplicated inner file.
    assert manifest.count(b"inner.txt") == 1


# ---------------------------------------------------------------- revisions
def test_revision_run_id_is_derived_from_its_policy(registry: Registry, sample: Sample) -> None:
    # input policy is registered -> the revision is visible under that run
    registered = replace(sample.revision, iteration_t=1, m=0)
    registry.add_revision(registered)
    assert registry.list_revisions("run-1") == [sample.revision, registered]
    assert registry.list_revisions("run-1", iteration_t=1) == [registered]
    assert registry.list_revisions("run-1", iteration_t=0) == [sample.revision]

    # neither policy is registered -> the row is stored but not attributed
    orphan = replace(
        sample.revision, iteration_t=2, input_policy_id="pol-missing",
        output_policy_id="pol-missing-too",
    )
    registry.add_revision(orphan)
    assert registry.list_revisions("run-1", iteration_t=2) == []


def test_duplicate_revision_is_rejected(registry: Registry) -> None:
    with pytest.raises(IntegrityError):
        registry.add_revision(_sample().revision)


# ------------------------------------------------------------------- resume
def test_resume_reopens_with_identical_values(tmp_path: Path) -> None:
    db_path = tmp_path / "resume.sqlite3"
    sample = _sample()
    with Registry(db_path) as reg:
        _populate(reg, sample)
        reg.seal_world(
            "world-1",
            state=WorldState.SEALED_SCOREABLE,
            world_digest=HEX["f"],
            b_count=2,
            s_best=0.9,
            note="nodes persisted, then sealed",
        )
        reg.update_run_state("run-1", RunState.WORLD_SEALED, note="world sealed")
        reg.append_event(Event(run_id="run-1", kind="WORLD_SEALED", payload={"world_id": "world-1"}))
        sealed = reg.get_world("world-1")
        run = reg.get_run("run-1")
        events = reg.list_events("run-1")
        assert sealed is not None and run is not None

    with Registry(db_path) as resumed:
        assert resumed.get_run("run-1") == run
        assert resumed.get_world("world-1") == sealed
        assert resumed.list_nodes("world-1") == list(sample.nodes)
        assert resumed.list_worlds("run-1") == [sealed, sample.other_world]
        assert resumed.list_worlds("run-1", sealed_only=True) == [sealed]
        assert resumed.list_attempts("world-1") == [sample.attempt]
        assert resumed.list_policies("run-1") == [sample.policy]
        assert resumed.list_replays(world_id="world-1") == [sample.replay]
        assert resumed.list_scores(policy_version_id="pol-0") == [sample.score]
        assert resumed.list_events("run-1") == events


# ------------------------------------------------------------ Windows paths
def test_paths_with_space_and_non_ascii(tmp_path: Path) -> None:
    base = tmp_path / "sp ace-ünïcode-测试"
    base.mkdir()
    sample = _sample()
    db_path = base / "reg istry.sqlite3"
    with Registry(db_path) as reg:
        _populate(reg, sample)
        assert reg.get_run("run-1") == sample.run
    with Registry(db_path) as reg:
        assert reg.list_worlds("run-1") == [sample.world, sample.other_world]

    cas = ContentAddressedStore(base / "cas store")
    digest = cas.put_bytes("ünïcode-测试".encode())
    assert cas.get_bytes(digest) == "ünïcode-测试".encode()

    tree = base / "sna pshot"
    (tree / "子目录").mkdir(parents=True)
    (tree / "子目录" / "ファイル.txt").write_text("x\n", encoding="utf-8")
    tree_digest = cas.put_tree(tree)
    assert "子目录/ファイル.txt".encode() in cas.get_bytes(tree_digest)
