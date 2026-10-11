"""End-to-end integration of the §3 loop with no model in it.

These tests drive the **real** composition root (the same wiring the CLI uses),
not a hand-built imitation: a hand-built one would pass while the shipped
program was broken. The discovery backend is the deterministic fixture, so a
green run here is ``OFFLINE_PASS`` — it is never evidence that a live model
works, and nothing in this file claims otherwise.

What is actually being pinned down:

* the loop reaches ``COMMITTED`` with every world sealed and scoreable;
* **the replay hard gate**: replay executes zero discovery-agent calls and zero
  evaluator calls;
* §3's monotonicity rule under a fixed ``H_t``: ``V(pi_{t+1}) >= V(pi_t)``;
* ``H_t`` is not degenerate — successive worlds really are different, so the
  cross-world aggregation in Eq(1) is doing work rather than averaging a value
  with itself;
* the loop is deterministic: identical inputs produce identical values.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(SRC))

from dream_rsi.cli import _build_controller  # noqa: E402
from dream_rsi.config import Config, LoopConfig  # noqa: E402
from dream_rsi.domain.models import RunState, WorldState  # noqa: E402

OUTER_ROUNDS = 3


def _config(project_root: Path, run_id: str) -> Config:
    """A Config rooted at ``project_root`` for state, but real task + policy.

    ``runtime_root``/``state_root`` derive from ``project_root``, so pointing it
    at ``tmp_path`` keeps test runs out of the developer's ``.runtime/``. The
    task and the incumbent policy keep pointing at the real repo, because those
    are inputs under test rather than scratch space.
    """
    return Config(
        project_root=project_root,
        task_dir=REPO / "tasks" / "knapsack",
        incumbent_policy_path=REPO / "policies" / "incumbent.py",
        backend="fixture",
        loop=LoopConfig(outer_rounds=OUTER_ROUNDS),
        run_id=run_id,
    )


def _run_loop(project_root: Path, run_id: str) -> Any:
    config = _config(project_root, run_id)
    config.ensure_dirs(run_id)
    controller, registry = _build_controller(config, run_id, with_developer=True)
    try:
        return controller.run()
    finally:
        registry.close()


@pytest.fixture(scope="module")
def run_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scratch project root the shared run below writes its registry into.

    Exposed separately so tests that reopen the registry read the *same* SQLite
    file the run wrote; a test that made up its own path would open an empty
    database and pass vacuously.
    """
    return tmp_path_factory.mktemp("offline")


@pytest.fixture(scope="module")
def outcome(run_root: Path) -> Any:
    """One real offline run, shared by the read-only assertions below."""
    return _run_loop(run_root, "it-offline-1")


# ---------------------------------------------------------------------------
# the loop itself
# ---------------------------------------------------------------------------


def test_loop_reaches_committed_with_sealed_worlds(outcome: Any) -> None:
    assert outcome.state is RunState.COMMITTED
    assert len(outcome.worlds) == OUTER_ROUNDS
    assert [w.outer_t for w in outcome.worlds] == list(range(OUTER_ROUNDS))
    for world in outcome.worlds:
        assert world.state is WorldState.SEALED_SCOREABLE
        # B_i = |T_i| - 1 > 0 and a positive best non-root score are what make a
        # world usable by Eq(1); a world that is not scoreable must never be
        # reported here as if it were.
        assert world.node_count > 1
        assert world.b_count == world.node_count - 1
        assert world.s_best > 0.0


def test_selection_happened_after_the_first_world(outcome: Any) -> None:
    # t=0 has an empty H_t, so pi_0 is the incumbent by construction rather than
    # by winning a contest; only t>=1 selections are real contests.
    assert [s.iteration_t for s in outcome.selections] == [1, 2]
    assert any("H_t is empty" in note for note in outcome.notes)


def test_revisions_use_the_just_scored_version_and_feedback(
    outcome: Any, run_root: Path,
) -> None:
    """§3 requires pi^(m+1) to use pi^m's code and replay measurements."""
    from dream_rsi.storage.sqlite_registry import Registry

    config = _config(run_root, outcome.run_id)
    with Registry(config.db_path) as registry:
        for t in (1, 2):
            revisions = registry.list_revisions(outcome.run_id, t)
            assert [r.m for r in revisions] == [1, 2, 3]
            incumbent = registry.get_world(f"{outcome.run_id}-w{t-1}")
            assert incumbent is not None
            predecessor = incumbent.policy_version_id
            prior_feedback: set[str] = set()
            for revision in revisions:
                assert revision.input_policy_id == predecessor
                assert revision.feedback_digest not in prior_feedback
                prior_feedback.add(revision.feedback_digest)
                proposal = registry.get_policy(revision.output_policy_id or "")
                assert proposal is not None
                source = Path(proposal.source_artifact_id).read_text(encoding="utf-8")
                previous = registry.get_policy(predecessor)
                assert previous is not None
                previous_path = (
                    config.policy_root(outcome.run_id) / f"pi_{t-1}.py"
                    if revision.m == 1 else Path(previous.source_artifact_id)
                )
                digest = hashlib.sha256(previous_path.read_bytes()).hexdigest()
                assert f"incumbent sha256={digest}" in source
                predecessor = proposal.version_id


def test_resume_quarantines_an_interrupted_world_and_keeps_prior_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dream_rsi.runtime import controller as controller_module
    from dream_rsi.storage.sqlite_registry import Registry

    run_id = "resume-interrupted"
    original = controller_module.run_online_rollout
    calls = 0

    def interrupt_second_world(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interruption before a batch committed")
        return original(**kwargs)

    monkeypatch.setattr(controller_module, "run_online_rollout", interrupt_second_world)
    with pytest.raises(KeyboardInterrupt):
        _run_loop(tmp_path, run_id)
    monkeypatch.setattr(controller_module, "run_online_rollout", original)

    config = dataclasses.replace(_config(tmp_path, run_id), resume=True)
    controller, registry = _build_controller(config, run_id, with_developer=True)
    try:
        resumed = controller.run()
    finally:
        registry.close()
    assert resumed.state is RunState.COMMITTED
    assert [w.outer_t for w in resumed.worlds] == [0, 1, 2]
    with Registry(config.db_path) as registry:
        worlds = registry.list_worlds(run_id)
        assert [w.state for w in worlds if w.outer_t == 1] == [
            WorldState.QUARANTINED, WorldState.SEALED_SCOREABLE,
        ]
        assert len(registry.list_nodes(f"{run_id}-w0")) == resumed.worlds[0].node_count
        assert len(registry.list_revisions(run_id, 1)) == config.loop.M - 1


def test_sealed_history_detects_node_payload_tampering(tmp_path: Path) -> None:
    from dream_rsi.domain.errors import IntegrityError

    run_id = "tamper-history"
    result = _run_loop(tmp_path, run_id)
    db_path = _config(tmp_path, run_id).db_path
    node_id = f"{result.worlds[0].world_id}-root"
    with sqlite3.connect(db_path) as connection:
        payload = connection.execute(
            "SELECT payload FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert payload is not None
        changed = json.loads(payload[0])
        changed["summary"] = "tampered after sealing"
        connection.execute(
            "UPDATE nodes SET payload = ? WHERE node_id = ?",
            (json.dumps(changed), node_id),
        )

    config = dataclasses.replace(_config(tmp_path, run_id), resume=True)
    controller, registry = _build_controller(config, run_id, with_developer=True)
    try:
        with pytest.raises(IntegrityError, match="changed"):
            controller._settle_history(run_id, 1)
    finally:
        registry.close()


def test_all_failed_model_attempts_cannot_commit_as_live(tmp_path: Path) -> None:
    """A fixed evaluator may score an unchanged seed; that is not a model run."""
    from dream_rsi.adapters.base import GeneratedAttempt
    from dream_rsi.domain.errors import BlockedModelConfiguration, IntegrityError
    from dream_rsi.storage.sqlite_registry import Registry

    run_id = "failed-live-gate"
    config = _config(tmp_path, run_id)
    controller, registry = _build_controller(config, run_id, with_developer=True)
    backend = controller.deps.backend
    backend.is_live = True
    assert controller.deps.developer is not None
    controller.deps.developer.is_live = True

    def fail_generation(**kwargs: Any) -> GeneratedAttempt:
        return GeneratedAttempt(
            invocation_id=kwargs["invocation_id"], backend_id="simulated-pi",
            model="simulated", workspace_path=kwargs["output_workspace"],
            entry_file_digest=None, succeeded=False, duration_s=0.0,
            error_type="PI_PROMPT_REJECTED",
        )

    backend.generate = fail_generation
    try:
        with pytest.raises(BlockedModelConfiguration, match="all Pi discovery attempts failed"):
            controller.run()
    finally:
        registry.close()
    with Registry(config.db_path) as persisted:
        run = persisted.get_run(run_id)
        assert run is not None and run.state is RunState.BLOCKED_EXTERNAL
    resumed = dataclasses.replace(config, resume=True)
    controller, registry = _build_controller(resumed, run_id, with_developer=True)
    try:
        with pytest.raises(IntegrityError, match="externally blocked"):
            controller.run()
    finally:
        registry.close()


def test_policy_developer_transport_failure_blocks_live_run(tmp_path: Path) -> None:
    from dream_rsi.domain.errors import BlockedModelConfiguration
    from dream_rsi.storage.sqlite_registry import Registry

    run_id = "failed-developer-gate"
    config = _config(tmp_path, run_id)
    controller, registry = _build_controller(config, run_id, with_developer=True)
    controller.deps.backend.is_live = True
    developer = controller.deps.developer
    assert developer is not None
    developer.is_live = True

    def fail_development(**kwargs: Any) -> None:
        raise RuntimeError("m=1: call failed (PI_PROMPT_REJECTED)")

    developer.propose = fail_development
    try:
        with pytest.raises(BlockedModelConfiguration, match="call failed"):
            controller.run()
    finally:
        registry.close()
    with Registry(config.db_path) as persisted:
        run = persisted.get_run(run_id)
        assert run is not None and run.state is RunState.BLOCKED_EXTERNAL
        assert len(persisted.list_worlds(run_id)) == 1


def test_resume_of_committed_run_does_not_append_new_events(tmp_path: Path) -> None:
    from dream_rsi.domain.errors import IntegrityError
    from dream_rsi.storage.sqlite_registry import Registry

    run_id = "committed-resume"
    _run_loop(tmp_path, run_id)
    config = dataclasses.replace(_config(tmp_path, run_id), resume=True)
    with Registry(config.db_path) as persisted:
        events_before = len(persisted.list_events(run_id))
    controller, registry = _build_controller(config, run_id, with_developer=True)
    try:
        outcome = controller.run()
    finally:
        registry.close()
    assert outcome.state is RunState.COMMITTED
    assert len(outcome.worlds) == OUTER_ROUNDS
    with Registry(config.db_path) as persisted:
        assert len(persisted.list_events(run_id)) == events_before

    deployed = config.policy_root(run_id) / f"pi_{OUTER_ROUNDS-1}.py"
    deployed.write_text("# tampered deployment\n", encoding="utf-8")
    controller, registry = _build_controller(config, run_id, with_developer=True)
    try:
        with pytest.raises(IntegrityError, match="changed its deployed policy"):
            controller.run()
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# the replay hard gate
# ---------------------------------------------------------------------------


def test_replay_made_no_external_calls(outcome: Any, run_root: Path) -> None:
    """§3 replay must not re-run an agent or the evaluator — it re-reads."""
    from dream_rsi.storage.sqlite_registry import Registry

    config = _config(run_root, "it-offline-1")
    registry = Registry(config.db_path)
    try:
        policy_ids = {p.version_id for p in registry.list_policies(outcome.run_id)}
        episodes = [e for e in registry.list_replays() if e.policy_version_id in policy_ids]
    finally:
        registry.close()

    assert episodes, "the run recorded no replay episodes, so the gate proves nothing"
    assert sum(e.discovery_agent_calls for e in episodes) == 0
    assert sum(e.evaluator_calls for e in episodes) == 0
    # The online side is the complement: it is where the calls are allowed.
    assert outcome.online_agent_calls > 0 or outcome.online_evaluator_calls > 0


def test_replay_module_cannot_reach_a_model_at_all() -> None:
    """The counters above are backed by structure, not by discipline.

    ``replay.py`` takes a sealed world and a policy function and nothing else —
    it is never handed a backend or an evaluator. This asserts the stronger
    property that it does not import one either, so a future edit that wanted to
    call a model would have to add an import that this test would catch.
    """
    tree = ast.parse((SRC / "dream_rsi" / "runtime" / "replay.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden_roots = ("adapters", "evaluator", "subprocess", "socket", "urllib", "http")
    offenders = sorted(
        name
        for name in imported
        if any(part in forbidden_roots for part in name.split("."))
    )
    assert not offenders, f"replay.py imported something that could make a live call: {offenders}"


# ---------------------------------------------------------------------------
# the event log
# ---------------------------------------------------------------------------


def test_the_event_log_is_mirrored_to_events_jsonl(outcome: Any, run_root: Path) -> None:
    """``contracts/SCHEMAS.md`` names the event log as ``events.jsonl`` **and**
    SQLite WAL. The table is authoritative and is asserted against elsewhere;
    this pins the other half, and pins that the two agree.

    Agreement is the whole point. A mirror that quietly drifted — a dropped
    line, a reordered pair, a rolled-back event that got written anyway — would
    look like an audit trail while describing a run that did not happen, which
    is worse than having no mirror at all.
    """
    from dream_rsi.storage.sqlite_registry import Registry

    config = _config(run_root, "it-offline-1")
    registry = Registry(config.db_path)
    try:
        authoritative = registry.list_events(outcome.run_id)
    finally:
        registry.close()

    assert authoritative, "the run recorded no events, so this test proves nothing"

    path = config.run_root(outcome.run_id) / "events.jsonl"
    assert path.is_file(), f"the run wrote no events.jsonl at {path}"

    raw = path.read_bytes()
    # LF is load-bearing across this project (the static policy gate rejects
    # CRLF), and a file opened without an explicit newline would be CRLF here.
    assert b"\r\n" not in raw, "the mirror must be written with LF, not CRLF"

    mirrored = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert [e["kind"] for e in mirrored] == [e.kind for e in authoritative], (
        "the mirror and the registry disagree on which events happened, or on their order"
    )
    assert [e["payload"] for e in mirrored] == [dict(e.payload) for e in authoritative]
    assert [e["at"] for e in mirrored] == [e.at for e in authoritative]


# ---------------------------------------------------------------------------
# §3 semantics
# ---------------------------------------------------------------------------


def test_value_is_monotone_under_a_fixed_history(outcome: Any) -> None:
    """V(pi_{t+1}) >= V(pi_t), measured on the same H_t."""
    for sel in outcome.selections:
        values = dict(sel.mean_values)
        assert sel.selected_policy_id in values
        assert sel.deployed_policy_id in values
        assert values[sel.selected_policy_id] >= values[sel.deployed_policy_id] - 1e-12, (
            f"t={sel.iteration_t} deployed {sel.deployed_policy_id} at "
            f"{values[sel.deployed_policy_id]!r} but selected "
            f"{sel.selected_policy_id} at {values[sel.selected_policy_id]!r}"
        )
        # deployed_again is exactly "the contest did not change the deployment".
        assert sel.deployed_again == (sel.selected_policy_id == sel.deployed_policy_id)


def test_history_growth_is_not_degenerate(outcome: Any) -> None:
    """Successive worlds must genuinely differ.

    Every world starts from the same fixed seed workspace (ACCEPTANCE §2), so if
    the backend ignored ``history`` the worlds would be byte-identical and every
    cross-world mean would be an average of a number with itself. That would make
    the aggregation in Eq(1) untestable while still looking green.
    """
    digests = [w.world_digest for w in outcome.worlds]
    assert all(digests), "worlds must be sealed with a digest"
    assert len(set(digests)) == len(digests), f"worlds are not distinct: {digests}"

    # And the difference must show up where it matters: the same policy scored
    # over a growing H_t moves.
    first, second = outcome.selections[0], outcome.selections[1]
    before = dict(first.mean_values)[second.deployed_policy_id]
    after = dict(second.mean_values)[second.deployed_policy_id]
    assert before != after, (
        "the incumbent scored identically over H_1 and H_2, which means world 1 "
        "added no information"
    )


def test_loop_is_deterministic(tmp_path: Path) -> None:
    """Same inputs, same values — the fixture has no hidden entropy."""
    first = _run_loop(tmp_path / "a", "det-a")
    second = _run_loop(tmp_path / "b", "det-b")

    def shape(run: Any) -> list[tuple[int, int, int, float]]:
        return [(w.outer_t, w.node_count, w.b_count, round(w.s_best, 12)) for w in run.worlds]

    assert shape(first) == shape(second)
    assert [s.deployed_again for s in first.selections] == [
        s.deployed_again for s in second.selections
    ]
    # Policy ids embed the run id, so compare the contest's *values*, which must
    # match exactly if the two runs really explored the same worlds.
    for a, b in zip(first.selections, second.selections, strict=True):
        assert sorted(v for _, v in a.mean_values) == sorted(v for _, v in b.mean_values)


# ---------------------------------------------------------------------------
# the run record is honest about what it is
# ---------------------------------------------------------------------------


def test_offline_run_is_labelled_offline(outcome: Any) -> None:
    """A fixture run must never be reported as a live one.

    The mode is what keeps `is_live=False` attempts out of live statistics; if
    this ever reads LIVE, every downstream claim about model quality is void.
    """
    from dream_rsi.domain.models import RunMode

    assert outcome.mode is RunMode.OFFLINE_FIXTURE
    assert not outcome.live


def test_run_record_has_no_workspace_paths(outcome: Any, run_root: Path) -> None:
    """Nodes carry a content digest, never a path.

    Paths would make a world unreplayable anywhere but the machine that built it.
    """
    from dream_rsi.storage.sqlite_registry import Registry

    config = _config(run_root, "it-offline-1")
    registry = Registry(config.db_path)
    try:
        nodes = registry.list_nodes(outcome.worlds[0].world_id)
    finally:
        registry.close()

    assert nodes
    blob = json.dumps([dataclasses.asdict(n) for n in nodes], default=str)
    assert str(run_root) not in blob
    assert ".runtime" not in blob
    for node in nodes:
        assert node.workspace_snapshot_digest
        assert not Path(node.workspace_snapshot_digest).is_absolute()
