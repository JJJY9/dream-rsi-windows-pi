"""Focused tests for ``tools/verify_run.py``.

The point of these is narrow, and worth stating plainly. A checker whose PASS is
quoted as acceptance evidence has to be shown to *fail* when the thing it claims
to check is absent. Three checks repaired in this review are exercised here:

* ``run.completed`` -- an unfinished run must not pass;
* ``replay.every_version_on_every_history_world`` -- a required policy/world
  pair that was never scored must not pass;
* ``policy_development.real_pi_calls`` -- a revision accepted on a Pi reply that
  is failed or carries no policy must not pass.

Each test builds a complete synthetic run in a temporary project, asserts the
baseline passes, and then mutates exactly one thing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dream_rsi.config import load_config  # noqa: E402
from dream_rsi.domain.models import (  # noqa: E402
    AlgorithmMode,
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
from dream_rsi.domain.tree import SealedWorld  # noqa: E402
from dream_rsi.score.equation1 import equation1  # noqa: E402
from dream_rsi.storage.sqlite_registry import Registry  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verify_run_tool", REPO / "tools" / "verify_run.py"
)
assert _spec is not None and _spec.loader is not None
verify_run = importlib.util.module_from_spec(_spec)
sys.modules["verify_run_tool"] = verify_run
_spec.loader.exec_module(verify_run)

#: The parameters the synthetic run is built and declared with. They are *not*
#: the tool's defaults, so the provenance check has to do real work.
OUTER_ROUNDS = 2
K1 = 1
M = 2

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-01T00:01:00+00:00"
T2 = "2026-01-01T00:02:00+00:00"
T3 = "2026-01-01T00:03:00+00:00"


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _node(world_id: str, node_id: str, parent: str | None, index: int, score: float) -> Node:
    return Node(
        node_id=node_id,
        world_id=world_id,
        parent_id=parent,
        creation_index=index,
        round_index=0,
        branch_index=0,
        depth=index,
        workspace_snapshot_digest=_hex(f"{node_id}-ws"),
        observation_digest=_hex(f"{node_id}-obs"),
        score=score,
        evaluator_status=EvaluatorStatus.SUCCESS,
    )


@dataclass
class Fixture:
    """A built run plus the handles a test needs to mutate it."""

    cfg: Any
    registry: Registry
    run_id: str
    policy_root: Path
    session_dir: Path
    versions: dict[str, str]

    def result_file(self) -> Path:
        return self.session_dir / "pi_request.json.result.json"


def _build(
    tmp_path: Path, *, finished: bool = True, reply_ok: bool = True,
    score_revision: bool = True, rollout_cap: int = K1,
) -> Fixture:
    project = tmp_path / "proj"
    (project / ".runtime").mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        project_root=project,
        backend="pi",
        developer="llm",
        provider="cc-switch-deep-seek",
        model="deepseek-v4-pro",
        overrides={"outer_rounds": OUTER_ROUNDS, "K1": rollout_cap, "M": M},
    )
    run_id = "run-fixture"
    registry = Registry(cfg.db_path)
    policy_root = cfg.policy_root(run_id)
    policy_root.mkdir(parents=True, exist_ok=True)

    registry.create_run(Run(
        run_id=run_id,
        mode=RunMode.LIVE,
        algorithm_mode=AlgorithmMode.STRICT_S3,
        task_spec_digest=_hex("task"),
        provider_id=str(cfg.provider),
        agent_model=str(cfg.model),
        fixed_evaluator_digest=_hex("evaluator"),
        cfg_digest=cfg.cfg_digest,
        state=RunState.ONLINE_RUNNING,
        created_at=T0,
    ))
    registry.update_run_state(
        run_id, RunState.COMMITTED if finished else RunState.ONLINE_RUNNING,
        finished=finished,
    )

    # ---------------------------------------------------------------- worlds
    worlds: list[World] = []
    for t in range(OUTER_ROUNDS):
        world_id = f"{run_id}-w{t}"
        root_id = f"{world_id}-root"
        nodes = [
            _node(world_id, root_id, None, 0, 0.0),
            _node(world_id, f"{world_id}-n1", root_id, 1, 1.0),
        ]
        # The world row comes first: nodes carry a foreign key to it.
        world = World(
            world_id=world_id, outer_t=t, run_id=run_id, root_id=root_id,
            policy_version_id=f"{run_id}-pi0", state=WorldState.BUILDING, created_at=T0,
        )
        registry.add_world(world)
        registry.add_nodes(nodes)
        sealed = SealedWorld.from_nodes(world_id, nodes, root_id=root_id)
        registry.seal_world(
            world_id, state=WorldState.SEALED_SCOREABLE,
            world_digest=sealed.world_digest(),
            b_count=sealed.b_count, s_best=sealed.s_best,
        )
        worlds.append(world)

    # -------------------------------------------------------------- policies
    incumbent_source = "def select(observation, rho):\n    return []\n"
    (policy_root / "pi_0.py").write_text(incumbent_source, encoding="utf-8", newline="\n")

    revision_body = "def select(observation, rho):\n    return [observation.node_ids[-1]]\n"
    header = "# revision m=1 at iteration t=1; fixture\n"
    proposal_dir = policy_root / "proposals_t1"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposal_dir / "llm_t1_m1_exploit.py"
    proposal_path.write_text(header + revision_body, encoding="utf-8", newline="\n")
    (policy_root / "pi_1.py").write_text(header + revision_body, encoding="utf-8", newline="\n")

    versions = {"incumbent": f"{run_id}-pi0", "revision": f"{run_id}-t1-m1"}
    registry.add_policy(PolicyVersion(
        version_id=versions["incumbent"], run_id=run_id, iteration_t=0, m=0,
        parent_version=None, source_sha256=hashlib.sha256(
            incumbent_source.encode()).hexdigest(),
        source_artifact_id=str(policy_root / "pi_0.py"), origin="incumbent",
        state=PolicyState.REPLAYED, created_at=T0,
    ))
    registry.add_policy(PolicyVersion(
        version_id=versions["revision"], run_id=run_id, iteration_t=1, m=1,
        parent_version=versions["incumbent"],
        source_sha256=hashlib.sha256((header + revision_body).encode()).hexdigest(),
        source_artifact_id=str(proposal_path), origin="revision",
        state=PolicyState.REPLAYED, created_at=T2,
    ))

    # --------------------------------------------------------------- replays
    # The revision attains quality 1.0 where the incumbent attains 0.5, so the
    # contest has a winner that is not the incumbent.
    quality = {versions["incumbent"]: 0.5, versions["revision"]: 1.0}
    world0 = worlds[0].world_id
    scored_versions = [versions["incumbent"]]
    if score_revision:
        scored_versions.append(versions["revision"])
    scores: dict[tuple[str, str], float] = {}
    for vid in scored_versions:
        episodes = [
            ReplayEpisode(
                episode_id=f"{vid}-{world0}-{rho}", world_id=world0,
                policy_version_id=vid, rho=rho, n=0, kstar=0,
                best_score=quality[vid], attained_quality=quality[vid],
                selected_batches=(), revealed_order=(),
                stop_reason=StopReason.MAX_ROUNDS,
                replay_digest=_hex(f"{vid}-{world0}-{rho}"),
                created_at=T1,
            )
            for rho in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        for episode in episodes:
            registry.add_replay(episode)
        result = equation1(
            episodes, b_count=1, s_best=1.0, lambda_weight=0.1,
            expected_rhos=(0.0, 0.25, 0.5, 0.75, 1.0),
        )
        scores[(vid, world0)] = result.value
        registry.add_score(ScoreRecord(
            policy_version_id=vid, world_id=world0, b_count=1, s_best=1.0,
            five_points=result.points, frontier=result.frontier, hv=result.hv,
            parallel_penalty=result.penalty, lambda_weight=0.1, value=result.value,
            created_at=T1,
        ))
    assert (versions["revision"], world0) not in scores or (
        scores[(versions["revision"], world0)] > scores[(versions["incumbent"], world0)]
    ), "the fixture must have a revision that beats the incumbent"

    # ------------------------------------------------------------- revision
    registry.add_revision(Revision(
        iteration_t=1, m=1, input_policy_id=versions["incumbent"],
        output_policy_id=versions["revision"], status="ELIGIBLE",
        feedback_digest=_hex("feedback"), proposal_digest=_hex("proposal"),
        created_at=T2,
    ))

    # -------------------------------------------------- the developer's call
    session_dir = cfg.run_root(run_id) / "policy_agent" / "t1_m1_a1"
    session_dir.mkdir(parents=True, exist_ok=True)
    mean_incumbent = scores[(versions["incumbent"], world0)]
    (session_dir / "pi_request.json").write_text(json.dumps({
        "prompt": (
            f"policy id {versions['incumbent']}\n"
            f"world {world0}\n"
            f"mean V {mean_incumbent:.12f}\n"
        ),
    }), encoding="utf-8", newline="\n")
    reply = (
        "Here is the revision.\n=== POLICY BEGIN ===\n" + revision_body
        + "=== POLICY END ===\n"
    )
    (session_dir / "pi_request.json.result.json").write_text(json.dumps({
        "ok": reply_ok, "final_response": reply if reply_ok else "no policy here",
        "error_type": None if reply_ok else "AGENT_FAILED",
        "session_id": "sess-1", "event_types": ["agent_start"],
    }), encoding="utf-8", newline="\n")

    # ----------------------------------------------------- discovery attempts
    for world in worlds:
        (cfg.run_root(run_id) / "workspaces" / world.world_id / "attempt-1").mkdir(
            parents=True, exist_ok=True)
    attempt_dir = cfg.run_root(run_id) / "agent" / "attempt-1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "pi.result.json").write_text(json.dumps({
        "ok": True, "error_type": None, "event_types": ["agent_start", "agent_settled"],
    }), encoding="utf-8", newline="\n")

    # --------------------------------------------------------------- events
    events = [
        Event(run_id=run_id, kind="run.start", payload={}, at=T0),
        Event(run_id=run_id, kind="run.components",
              payload={"backend_is_live": True, "developer_is_live": True}, at=T0),
        Event(run_id=run_id, kind="policy.no_contest", payload={"iteration_t": 0}, at=T0),
        Event(run_id=run_id, kind="world.sealed", payload={"world_id": world0}, at=T1),
        Event(run_id=run_id, kind="policy.selected",
              payload={"iteration_t": 1, "selected": versions["revision"]}, at=T2),
    ]
    if finished:
        events.append(Event(run_id=run_id, kind="run.finish",
                            payload={"state": "COMMITTED"}, at=T3))
    for event in events:
        registry.append_event(event)
    (cfg.run_root(run_id) / "events.jsonl").write_text(
        "\n".join(json.dumps(e.to_dict()) for e in events) + "\n",
        encoding="utf-8", newline="\n",
    )
    return Fixture(cfg=cfg, registry=registry, run_id=run_id, policy_root=policy_root,
                   session_dir=session_dir, versions=versions)


def _expectations(*, rollout_cap: int = K1) -> Any:
    return verify_run.Expectations(
        backend="pi", developer="llm", provider="cc-switch-deep-seek",
        model="deepseek-v4-pro", outer_rounds=OUTER_ROUNDS, K1=rollout_cap,
        W=4, M=M,
    )


def _failed(fixture: Fixture) -> set[str]:
    checks = verify_run.verify(
        fixture.registry, fixture.cfg, fixture.run_id,
        _expectations(rollout_cap=fixture.cfg.loop.K1),
    )
    return {row["check"] for row in checks.failed}


def test_complete_run_passes_every_check(tmp_path: Path) -> None:
    """The baseline: if this fails, the negative tests below prove nothing."""
    fixture = _build(tmp_path)
    try:
        assert _failed(fixture) == set()
    finally:
        fixture.registry.close()


def test_unfinished_run_cannot_pass(tmp_path: Path) -> None:
    """No COMMITTED state, no finish event and no finish timestamp."""
    fixture = _build(tmp_path, finished=False)
    try:
        failed = _failed(fixture)
    finally:
        fixture.registry.close()
    assert "run.completed" in failed


def test_completed_run_may_stop_before_the_online_round_cap(tmp_path: Path) -> None:
    """K1=2 permits a policy to stop after one nonempty round."""
    fixture = _build(tmp_path, rollout_cap=2)
    try:
        assert _failed(fixture) == set()
    finally:
        fixture.registry.close()


def test_event_mirror_with_same_count_but_changed_contents_cannot_pass(tmp_path: Path) -> None:
    fixture = _build(tmp_path)
    try:
        assert _failed(fixture) == set()
        mirror = fixture.cfg.run_root(fixture.run_id) / "events.jsonl"
        events = [json.loads(line) for line in mirror.read_text(encoding="utf-8").splitlines()]
        events[0]["payload"] = {"changed_without_changing_line_count": True}
        mirror.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
        )
        assert "events.mirror_matches_registry" in _failed(fixture)
    finally:
        fixture.registry.close()


def test_missing_required_score_pair_cannot_pass(tmp_path: Path) -> None:
    """The revision was never scored on H_1, so a required pair has no score."""
    fixture = _build(tmp_path, score_revision=False)
    try:
        failed = _failed(fixture)
    finally:
        fixture.registry.close()
    assert "replay.every_version_on_every_history_world" in failed


def test_accepted_revision_with_unusable_reply_cannot_pass(tmp_path: Path) -> None:
    """A revision recorded ELIGIBLE whose stored Pi reply failed and holds no policy."""
    fixture = _build(tmp_path, reply_ok=False)
    try:
        failed = _failed(fixture)
    finally:
        fixture.registry.close()
    assert "policy_development.real_pi_calls" in failed


def test_declared_parameters_are_proven_against_the_run(tmp_path: Path) -> None:
    """Wrong declarations must fail loudly rather than check invented counts."""
    fixture = _build(tmp_path)
    try:
        wrong = verify_run.Expectations(
            backend="pi", developer="llm", provider="cc-switch-deep-seek",
            model="deepseek-v4-pro", outer_rounds=3, K1=2, W=4, M=3,
        )
        checks = verify_run.verify(fixture.registry, fixture.cfg, fixture.run_id, wrong)
        failed = {row["check"] for row in checks.failed}
    finally:
        fixture.registry.close()
    assert "parameters.cfg_digest_matches" in failed
    assert "run.completed" in failed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
