#!/usr/bin/env python3
"""Cross-check a finished run against the §3 ordering it claims to implement.

This reads the run's registry rows and its on-disk artifacts and reports what it
finds. It is a checker, not a proof: every check states the evidence it used, so
a reader can disagree with a check rather than having to trust it. Nothing here
writes to the registry, and a FAIL is reported as a FAIL.

The ordering under test is the one §3 specifies and that an earlier version of
this project got wrong: score the current policy version, hand *that* version's
replay trajectories and scores to the development agent, score the version it
returns, and only then ask for the next one.

Three things this checker deliberately does **not** take on trust:

* **The run's parameters.** The count expectations on the command line are not
  read back out of the run they are checking. They are re-derived into a
  :class:`~dream_rsi.config.Config` and the resulting ``cfg_digest`` is compared
  with ``run.cfg_digest``, which the controller wrote at creation. A declared
  parameter set that does not reproduce the recorded digest is reported as a
  FAIL, so the completion counts that follow are never checked against invented
  numbers.
* **Which policy/world pairs were required.** The set is rebuilt from the
  incumbent, the round's eligible candidates and ``H_t`` rather than read from
  the score table, so a pair that was never scored at all shows up as missing
  instead of being invisible.
* **Policy-development replies.** Accepted revisions carry
  ``status == "ELIGIBLE"``, not ``"PENDING"``; validation keys off the accepted
  status, so a revision cannot be accepted on a reply that never arrived.

Usage::

    python tools/verify_run.py --run-id <RUN_ID> [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dream_rsi.config import KNOWN_TASKS, load_config  # noqa: E402
from dream_rsi.domain.models import PolicyState, RunState, WorldState  # noqa: E402
from dream_rsi.domain.tree import SealedWorld  # noqa: E402
from dream_rsi.runtime.llm_policy_developer import extract_policy  # noqa: E402
from dream_rsi.score.equation1 import equation1  # noqa: E402
from dream_rsi.storage.sqlite_registry import Registry  # noqa: E402

#: The status a revision carries once its proposal survived screening and was
#: accepted into the contest. Rejected proposals carry a different status.
ACCEPTED_REVISION = "ELIGIBLE"


@dataclass(frozen=True)
class Expectations:
    """What the run is *claimed* to have been created with.

    These are declarations, not observations. ``parameters.cfg_digest_matches``
    proves or disproves them against the digest the controller wrote, and the
    completion counts below are only meaningful once that proof holds.
    """

    backend: str
    developer: str
    provider: str | None
    model: str | None
    outer_rounds: int
    K1: int
    W: int
    M: int
    agent_timeout_s: float | None = None
    task: str | None = None


class Checks:
    """Collects check results; a run is only OK if every check passed."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, check_id: str, ok: bool, detail: str, **evidence: Any) -> None:
        self.rows.append({"check": check_id, "ok": bool(ok), "detail": detail, **evidence})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if not r["ok"]]

    def print(self) -> None:
        width = max(len(r["check"]) for r in self.rows)
        for row in self.rows:
            mark = "PASS" if row["ok"] else "FAIL"
            print(f"[{mark}] {row['check']:<{width}}  {row['detail']}")
            for key, value in row.items():
                if key in ("check", "ok", "detail"):
                    continue
                print(f"          {key}: {json.dumps(value, sort_keys=True)}")


def _scores_by_version(registry: Registry, version_ids: list[str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for vid in version_ids:
        out[vid] = registry.list_scores(policy_version_id=vid)
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def verify(
    registry: Registry,
    config: Any,
    run_id: str,
    expect: Expectations,
    *,
    expectation_source: str = "command line",
) -> Checks:
    checks = Checks()

    run = registry.get_run(run_id)
    if run is None:
        checks.add("run.recorded", False, f"run {run_id!r} is not in the registry")
        return checks
    checks.add(
        "run.recorded", True,
        f"run {run_id!r} is in the registry (existence gate only; completion is "
        "checked by run.completed)",
        state=run.state.value, mode=run.mode.value, algorithm_mode=run.algorithm_mode.value,
        cfg_digest=run.cfg_digest, provider_id=run.provider_id, agent_model=run.agent_model,
        task_spec_digest=run.task_spec_digest,
        fixed_evaluator_digest=run.fixed_evaluator_digest,
    )

    events = registry.list_events(run_id)
    components = [e for e in events if e.kind == "run.components"]
    if not components:
        checks.add("run.live_components", False, "no run.components event")
    else:
        payload = components[0].payload
        ok = bool(payload.get("backend_is_live")) and bool(payload.get("developer_is_live"))
        checks.add(
            "run.live_components", ok and run.mode.value == "live",
            "both sides report live" if ok else "a side is not live",
            payload=payload, run_mode=run.mode.value,
        )

    # ------------------------------------------------------------ parameters
    # The expectations are re-derived into a Config and hashed. cfg_digest
    # excludes absolute paths, so a verifier on any machine can reproduce it
    # from the declared parameters alone. This is what makes the completion
    # counts below evidence rather than assumption.
    provenance_ok = False
    provenance_detail: dict[str, Any] = dict(asdict(expect), source=expectation_source)
    try:
        rebuilt = load_config(
            project_root=config.project_root,
            backend=expect.backend,
            developer=expect.developer,
            provider=expect.provider,
            model=expect.model,
            overrides={"outer_rounds": expect.outer_rounds, "K1": expect.K1,
                       "W": expect.W, "M": expect.M},
            agent_timeout_s=expect.agent_timeout_s,
            task=expect.task,
        )
        provenance_detail["rebuilt_cfg_digest"] = rebuilt.cfg_digest
        provenance_detail["recorded_cfg_digest"] = run.cfg_digest
        provenance_detail["rebuilt_loop"] = asdict(rebuilt.loop)
        provenance_detail["provider_matches"] = rebuilt.provider == run.provider_id
        provenance_detail["model_matches"] = rebuilt.model == run.agent_model
        provenance_ok = (
            rebuilt.cfg_digest == run.cfg_digest
            and provenance_detail["provider_matches"]
            and provenance_detail["model_matches"]
        )
        detail = (
            f"declared parameters reproduce run.cfg_digest {run.cfg_digest[:16]}…"
            if provenance_ok else
            f"declared parameters hash to {rebuilt.cfg_digest[:16]}… but the run recorded "
            f"{run.cfg_digest[:16]}…; the counts below are not checked against proven values"
        )
    except Exception as exc:  # a bad declaration must not crash the checker
        provenance_detail["error"] = f"{type(exc).__name__}: {exc}"
        detail = f"could not rebuild the declared parameters: {exc}"
    checks.add("parameters.cfg_digest_matches", provenance_ok, detail, **provenance_detail)

    # ------------------------------------------------------------------ worlds
    worlds = registry.list_worlds(run_id)
    tampered: list[str] = []
    unscoreable: list[str] = []
    for world in worlds:
        nodes = registry.list_nodes(world.world_id)
        try:
            sealed = SealedWorld.from_nodes(world.world_id, nodes, root_id=world.root_id)
        except Exception as exc:
            tampered.append(f"{world.world_id}: rebuild failed: {exc}")
            continue
        if world.state is not WorldState.SEALED_SCOREABLE:
            unscoreable.append(f"{world.world_id}={world.state.value}")
        if sealed.world_digest() != world.world_digest:
            tampered.append(f"{world.world_id}: digest drifted")
        if (sealed.b_count, sealed.s_best) != (world.b_count, world.s_best):
            tampered.append(f"{world.world_id}: B/S drifted")
        if sealed.b_count <= 0 or sealed.s_best <= 0:
            unscoreable.append(f"{world.world_id}: B={sealed.b_count} S={sealed.s_best}")
    checks.add(
        "worlds.sealed_and_intact", not tampered and not unscoreable,
        "every world rebuilds to its stored digest and (B, S)"
        if not tampered and not unscoreable else "world records do not match",
        worlds=[
            {"world_id": w.world_id, "outer_t": w.outer_t, "B": w.b_count, "S": w.s_best,
             "collector": w.policy_version_id, "digest": w.world_digest}
            for w in worlds
        ],
        tampered=tampered, unscoreable=unscoreable,
    )

    # ------------------------------------------------------- real discovery
    # Pi writes ``attempt-*/pi_request.json.result.json`` when its worker
    # finishes; the Claude Code adapter writes ``attempt-*/claude_result.json``
    # and the Codex adapter ``attempt-*/codex_result.json``. A result counts as
    # a real discovery call only when the transport reports success AND carries
    # the agent's own session evidence (Pi: an ``agent_start`` event after
    # settle; Claude: the CLI's session id; Codex: the thread id, recorded as
    # ``session_id``) — an acknowledgement alone is not a completed call.
    agent_dir = config.run_root(run_id) / "agent"
    discovery: list[dict[str, Any]] = []
    per_world_ok: dict[str, int] = {w.world_id: 0 for w in worlds}
    result_files = (
        sorted(agent_dir.glob("attempt-*/*.result.json"))
        + sorted(agent_dir.glob("attempt-*/claude_result.json"))
        + sorted(agent_dir.glob("attempt-*/codex_result.json"))
    )
    for result_file in result_files:
        attempt_id = result_file.parent.name.removeprefix("attempt-")
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        seen = list(payload.get("event_types") or [])
        session_id = payload.get("session_id")
        ok = bool(payload.get("ok")) and ("agent_start" in seen or bool(session_id))
        record = {
            "attempt": attempt_id, "ok": ok, "error_type": payload.get("error_type"),
            "agent_start": "agent_start" in seen, "session_recorded": bool(session_id),
            "events_kept": len(seen),
        }
        discovery.append(record)
        if ok:
            for world in worlds:
                workspace = config.run_root(run_id) / "workspaces" / world.world_id
                if (workspace / f"attempt-{attempt_id}").is_dir():
                    per_world_ok[world.world_id] += 1
    failed_discovery = [d for d in discovery if not d["ok"]]
    starved = [w for w, n in per_world_ok.items() if n == 0]
    checks.add(
        "worlds.real_discovery", not failed_discovery and not starved and bool(discovery),
        f"{len(discovery)} discovery calls, {len(discovery) - len(failed_discovery)} succeeded, "
        f"every world has at least one"
        if not failed_discovery and not starved and discovery
        else "a discovery call failed or a world has no successful generation",
        per_world_successful=per_world_ok, failures=failed_discovery,
        note="ok=true is only written after the worker observed agent_settled; "
             "event_types is the first 80 events, so agent_settled is normally truncated away",
    )

    # ------------------------------------------------------------------ policies
    policies = {p.version_id: p for p in registry.list_policies(run_id)}
    revisions = registry.list_revisions(run_id)
    scores = _scores_by_version(registry, list(policies))

    # §3 ordering, part 1: each revision is built on the version that was just
    # scored. Within a round that means m-1's *accepted* output; for m=1 it means
    # whoever was deployed when the round opened, which is the previous round's
    # winner and may itself be a late revision. The chain is therefore walked
    # forward with a cursor rather than assumed to be m-1.
    def deployed_version_at(iteration_t: int) -> str | None:
        """Version deployed when round ``t`` opened: the bytes of ``pi_{t-1}.py``."""
        if iteration_t < 1:
            return None
        path = config.policy_root(run_id) / f"pi_{iteration_t - 1}.py"
        if not path.is_file():
            return None
        blob = path.read_bytes()
        for policy in policies.values():
            source = Path(policy.source_artifact_id)
            if source.is_file() and source.read_bytes() == blob:
                return policy.version_id
        return None

    chain_problems: list[str] = []
    chain: list[dict[str, Any]] = []
    cursor: dict[int, str | None] = {}
    for rev in sorted(revisions, key=lambda r: (r.iteration_t, r.m)):
        t = rev.iteration_t
        if t not in cursor:
            cursor[t] = deployed_version_at(t)
        expected_input = cursor[t]
        if expected_input is None:
            chain_problems.append(
                f"t={t} m={rev.m}: cannot tell which policy was deployed when the round opened"
            )
            continue
        if rev.input_policy_id != expected_input:
            chain_problems.append(
                f"t={t} m={rev.m}: revision was built on {rev.input_policy_id} "
                f"but the policy in place was {expected_input}"
            )
        output = policies.get(rev.output_policy_id or "")
        if output is None:
            if rev.status not in ("DEVELOPER_ERROR", "STATIC_REJECTED"):
                chain_problems.append(
                    f"t={t} m={rev.m}: status {rev.status} with no output policy"
                )
        else:
            if output.parent_version != rev.input_policy_id:
                chain_problems.append(
                    f"t={t} m={rev.m}: {output.version_id} records parent "
                    f"{output.parent_version}, not its revision input {rev.input_policy_id}"
                )
            if output.state is not PolicyState.REJECTED:
                # Only a version that survived screening and replay becomes the
                # base for the next request.
                cursor[t] = output.version_id
        chain.append({
            "t": t, "m": rev.m, "status": rev.status, "expected_input": expected_input,
            "input": rev.input_policy_id, "output": rev.output_policy_id,
        })
    checks.add(
        "revisions.sequential_chain", not chain_problems and bool(chain),
        f"{len(chain)} revision records each build on the version just scored"
        if not chain_problems and chain else "the revision chain is broken",
        chain=chain, problems=chain_problems,
    )

    # §3 ordering, part 2: the model call happened *after* the scores it was
    # shown. `Revision.created_at` is stamped once the developer call returns,
    # and a score row is written the moment its replay completes, so
    # created_at(revision m) > created_at(every score of revision m's input)
    # is the ordering evidence. Timestamps come from the same clock.
    order_problems: list[str] = []
    order_rows: list[dict[str, Any]] = []
    for rev in sorted(revisions, key=lambda r: (r.iteration_t, r.m)):
        input_scores = scores.get(rev.input_policy_id, [])
        if not input_scores:
            order_problems.append(
                f"t={rev.iteration_t} m={rev.m}: input {rev.input_policy_id} has no scores"
            )
            continue
        # Only scores that existed when the revision was requested count: the
        # incumbent is re-scored on the growing H_t every selection round, and
        # those later rows must not retroactively invalidate an ordering that
        # held at request time. What must hold is that the input had already
        # been scored on every world of H_t.
        prior = [s for s in input_scores if s.created_at < rev.created_at]
        required_worlds = {w.world_id for w in worlds if w.outer_t < rev.iteration_t}
        missing = sorted(required_worlds - {s.world_id for s in prior})
        last_score = max((s.created_at for s in prior), default=None)
        if missing:
            order_problems.append(
                f"t={rev.iteration_t} m={rev.m}: input {rev.input_policy_id} was not "
                f"scored on {missing} before the revision was written at {rev.created_at}"
            )
        order_rows.append({
            "t": rev.iteration_t, "m": rev.m, "input": rev.input_policy_id,
            "input_scored_worlds_before_request": len({s.world_id for s in prior}),
            "required_worlds": sorted(required_worlds),
            "last_prior_input_score_at": last_score, "revision_written_at": rev.created_at,
        })
    checks.add(
        "revisions.scored_before_requested", not order_problems and bool(order_rows),
        "every revision was requested after its input had been scored on all of H_t"
        if not order_problems and order_rows else "a revision predates its input's scores",
        rows=order_rows, problems=order_problems,
    )

    # §3 ordering, part 3: what the developer was actually shown. The prompt is
    # on disk, so this reads the model's own input rather than trusting that
    # the controller built it correctly.
    feedback_problems: list[str] = []
    feedback_rows: list[dict[str, Any]] = []
    for rev in sorted(revisions, key=lambda r: (r.iteration_t, r.m)):
        session = config.run_root(run_id) / "policy_agent" / f"t{rev.iteration_t}_m{rev.m}_a1"
        # Either transport's stored request is acceptable evidence: Pi writes
        # ``pi_request.json``, the Claude Code adapter writes
        # ``claude_request.json`` and the Codex adapter ``codex_request.json``
        # (all carry the prompt under ``prompt``).
        request = next(
            (p for p in (session / "pi_request.json",
                         session / "claude_request.json",
                         session / "codex_request.json")
             if p.is_file()),
            None,
        )
        if request is None:
            feedback_problems.append(f"t={rev.iteration_t} m={rev.m}: no stored request")
            continue
        prompt = str(json.loads(request.read_text(encoding="utf-8")).get("prompt", ""))
        input_scores = scores.get(rev.input_policy_id, [])
        mean = _mean([s.value for s in input_scores]) if input_scores else float("nan")
        carries_mean = f"{mean:.12f}" in prompt
        carries_id = rev.input_policy_id in prompt
        carries_worlds = all(s.world_id in prompt for s in input_scores)
        if not (carries_mean and carries_id and carries_worlds):
            feedback_problems.append(
                f"t={rev.iteration_t} m={rev.m}: prompt missing "
                f"mean={not carries_mean} id={not carries_id} worlds={not carries_worlds}"
            )
        feedback_rows.append({
            "t": rev.iteration_t, "m": rev.m, "input": rev.input_policy_id,
            "prompt_chars": len(prompt), "mean_V_shown": mean,
            "carries_mean": carries_mean, "carries_policy_id": carries_id,
            "carries_every_world_id": carries_worlds,
        })
    checks.add(
        "revisions.prompt_carries_measured_feedback", not feedback_problems and bool(feedback_rows),
        "each stored prompt contains its input policy's id, its measured mean V and every world id"
        if not feedback_problems and feedback_rows else "a prompt did not carry its input's scores",
        rows=feedback_rows, problems=feedback_problems,
    )

    # ------------------------------------------------------------------ replay
    # The required set is *constructed*: for every contested round t, the
    # incumbent plus every candidate that survived screening must carry a score
    # on every world of H_t. Reading the score table would only confirm what is
    # already there; building the set is what makes an absent pair visible.
    # A pair required at two rounds is one pair, and reuse of a cached pair
    # therefore still yields exactly one score row.
    required: dict[tuple[str, str], list[str]] = {}
    for t in sorted({w.outer_t for w in worlds}):
        if t < 1:
            continue  # t=0 contests nothing: §3 has no history to select on
        h_t = [w for w in worlds if w.outer_t < t]
        if not h_t:
            continue
        incumbent_id = deployed_version_at(t)
        if incumbent_id is not None:
            for w in h_t:
                required.setdefault((incumbent_id, w.world_id), []).append(f"t={t} incumbent")
        for rev in revisions:
            if rev.iteration_t != t:
                continue
            proposal = policies.get(rev.output_policy_id or "")
            if proposal is None or proposal.state is PolicyState.REJECTED:
                continue  # a rejected proposal never reaches the contest
            for w in h_t:
                required.setdefault((proposal.version_id, w.world_id), []).append(
                    f"t={t} m={rev.m} candidate"
                )

    # A pair with no score is a failure unless the controller recorded *why* it
    # has none: a policy can be rejected at replay time, and that is a recorded
    # outcome, not a hole in the record.
    recorded_failures = {
        (str(e.payload.get("version_id")), str(e.payload.get("world_id"))): str(e.payload.get("reason", ""))
        for e in events if e.kind == "policy.replay_failed"
    }
    missing_pairs: list[dict[str, Any]] = []
    explained_pairs: list[dict[str, Any]] = []
    for (vid, wid), reasons in sorted(required.items()):
        if any(row.world_id == wid for row in scores.get(vid, [])):
            continue
        entry = {"version_id": vid, "world_id": wid, "required_by": sorted(set(reasons))}
        if (vid, wid) in recorded_failures:
            entry["recorded_failure"] = recorded_failures[(vid, wid)]
            explained_pairs.append(entry)
        else:
            missing_pairs.append(entry)

    replay_problems: list[str] = []
    recompute_problems: list[str] = []
    calls: list[str] = []
    episode_total = 0
    for vid in sorted({v for v, rows in scores.items() if rows}):
        rows = scores[vid]
        by_world: dict[str, list[Any]] = {}
        for row in rows:
            by_world.setdefault(row.world_id, []).append(row)
        for world_id, world_rows in sorted(by_world.items()):
            episodes = registry.list_replays(policy_version_id=vid, world_id=world_id)
            episode_total += len(episodes)
            rhos = sorted(round(e.rho, 6) for e in episodes)
            if rhos != sorted(round(r, 6) for r in config.loop.rho_grid):
                replay_problems.append(
                    f"{vid} on {world_id}: rhos {rhos}, expected {sorted(config.loop.rho_grid)}"
                )
            for episode in episodes:
                if episode.discovery_agent_calls or episode.evaluator_calls:
                    calls.append(
                        f"{vid} on {world_id} rho={episode.rho}: "
                        f"agent={episode.discovery_agent_calls} "
                        f"evaluator={episode.evaluator_calls}"
                    )
            if not episodes:
                continue
            record = world_rows[0]
            result = equation1(
                episodes, b_count=record.b_count, s_best=record.s_best,
                lambda_weight=config.loop.lambda_weight, expected_rhos=config.loop.rho_grid,
            )
            for field, got, stored in (
                ("hv", result.hv, record.hv),
                ("penalty", result.penalty, record.parallel_penalty),
                ("value", result.value, record.value),
            ):
                if abs(got - stored) > 1e-12:
                    recompute_problems.append(
                        f"{vid} on {world_id}: {field} stored {stored} but replays give {got}"
                    )
    satisfied = len(required) - len(missing_pairs) - len(explained_pairs)
    checks.add(
        "replay.every_version_on_every_history_world",
        not missing_pairs and not replay_problems,
        f"{len(required)} required policy/world pairs: {satisfied} scored, "
        f"{len(explained_pairs)} with a recorded replay failure, {len(missing_pairs)} missing; "
        f"{episode_total} episodes each covering five rho values"
        if not missing_pairs and not replay_problems
        else "a required policy/world pair has no score or an incomplete rho grid",
        required_pairs=[
            {"version_id": v, "world_id": w, "required_by": sorted(set(r))}
            for (v, w), r in sorted(required.items())
        ],
        missing=missing_pairs, explained=explained_pairs, rho_problems=replay_problems,
        episodes=episode_total,
    )
    checks.add(
        "replay.zero_agent_and_evaluator_calls", not calls,
        "no replay episode reported an agent or evaluator call"
        if not calls else "a replay re-ran an agent or the evaluator",
        offending=calls,
    )
    checks.add(
        "score.recomputes_from_stored_replays", not recompute_problems,
        "recomputing Eq(1) from the stored episodes reproduces every stored score"
        if not recompute_problems else "a stored score does not follow from its replays",
        problems=recompute_problems,
    )

    # --------------------------------------------------------------- selection
    selection_rows: list[dict[str, Any]] = []
    selection_problems: list[str] = []
    history = [w for w in worlds if w.state is WorldState.SEALED_SCOREABLE]
    for t in sorted({p.iteration_t for p in policies.values()}):
        h_t = [w for w in history if w.outer_t < t]
        if not h_t:
            continue  # t=0 has an empty history: no contest, nothing to check
        world_ids = {w.world_id for w in h_t}
        incumbent_id = deployed_version_at(t)

        # The pool `select()` sees: the incumbent (always revision 0 there, even
        # when its own version id ends in a later m) plus everything proposed
        # this round that was scored on the whole of H_t.
        pool: list[dict[str, Any]] = []
        for policy in policies.values():
            per_world = {
                row.world_id: row.value for row in scores.get(policy.version_id, [])
                if row.world_id in world_ids
            }
            if len(per_world) != len(h_t):
                continue
            is_incumbent = policy.version_id == incumbent_id
            if not is_incumbent and policy.iteration_t != t:
                continue
            pool.append({
                "version_id": policy.version_id, "m": 0 if is_incumbent else policy.m,
                "mean_V": _mean(list(per_world.values())), "per_world": per_world,
            })
        if not pool:
            selection_problems.append(f"t={t}: nothing was scored on H_t")
            continue
        incumbent = next((c for c in pool if c["m"] == 0), None)
        if incumbent is None:
            selection_problems.append(
                f"t={t}: the incumbent {incumbent_id!r} has no scores on the whole of H_t"
            )
            continue
        best = max(c["mean_V"] for c in pool)
        tied = [c for c in pool if best - c["mean_V"] <= 1e-12]
        expected = min(tied, key=lambda c: c["m"])
        if expected["mean_V"] < incumbent["mean_V"] - 1e-12:
            selection_problems.append(f"t={t}: the winner scores below the incumbent")

        deployed_file = config.policy_root(run_id) / f"pi_{t}.py"
        selected_version = None
        if deployed_file.is_file():
            blob = deployed_file.read_bytes()
            for policy in policies.values():
                source = Path(policy.source_artifact_id)
                if source.is_file() and source.read_bytes() == blob:
                    selected_version = policy.version_id
        if selected_version is None:
            selection_problems.append(f"t={t}: {deployed_file.name} matches no recorded version")
        elif selected_version != expected["version_id"]:
            selection_problems.append(
                f"t={t}: deployed {selected_version} but the rule picks {expected['version_id']}"
            )
        selection_rows.append({
            "t": t, "H_t": sorted(world_ids),
            "incumbent": incumbent["version_id"], "incumbent_mean_V": incumbent["mean_V"],
            "pool": sorted(
                ({"version_id": c["version_id"], "m": c["m"], "mean_V": c["mean_V"]}
                 for c in pool), key=lambda c: (-c["mean_V"], c["m"]),
            ),
            "rule_picks": expected["version_id"], "rule_picks_mean_V": expected["mean_V"],
            "deployed": selected_version, "deployed_file": deployed_file.name,
            "monotone": incumbent["mean_V"] <= expected["mean_V"] + 1e-12,
        })
    checks.add(
        "selection.argmax_and_monotone", not selection_problems and bool(selection_rows),
        "each round deployed the argmax of mean V on its own H_t, never below the incumbent"
        if not selection_problems and selection_rows else "selection does not follow the rule",
        rows=selection_rows, problems=selection_problems,
    )

    # -------------------------------------------------------------- deployment
    deploy_problems: list[str] = []
    deploy_rows: list[dict[str, Any]] = []
    for path in sorted(config.policy_root(run_id).glob("pi_*.py")):
        blob = path.read_bytes()
        matches = [p.version_id for p in policies.values()
                   if Path(p.source_artifact_id).is_file()
                   and Path(p.source_artifact_id).read_bytes() == blob]
        deploy_rows.append({
            "file": path.name, "sha256_prefix": hashlib.sha256(blob).hexdigest()[:16],
            "bytes": len(blob), "matches_versions": matches,
        })
        if not matches:
            deploy_problems.append(f"{path.name} is byte-identical to no recorded policy version")
    checks.add(
        "deployment.files_are_recorded_policies", not deploy_problems and bool(deploy_rows),
        "every deployed file is byte-identical to a version in the registry"
        if not deploy_problems and deploy_rows else "a deployed file matches no recorded version",
        rows=deploy_rows, problems=deploy_problems,
    )

    # -------------------------------------------------------- policy-dev calls
    # Accepted revisions carry status "ELIGIBLE" -- not "PENDING" -- so the
    # reply check keys off acceptance. Every attempt is read, because a repair
    # is a second call and it is the *last* one that produced the file. The
    # reply is parsed with the developer's own extractor, and the source it
    # yields must be the tail of the proposal file the run actually used: that
    # ties the stored conversation to the bytes that entered the contest.
    dev_problems: list[str] = []
    dev_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    session_root = config.run_root(run_id) / "policy_agent"
    proposal_root = config.policy_root(run_id)
    for rev in sorted(revisions, key=lambda r: (r.iteration_t, r.m)):
        attempts: list[dict[str, Any]] = []
        for session in sorted(session_root.glob(f"t{rev.iteration_t}_m{rev.m}_a*")):
            # Pi stores ``pi_request.json.result.json``; the Claude Code
            # adapter stores ``claude_result.json`` and the Codex adapter
            # ``codex_result.json`` (both with the reply text under
            # ``final_response`` and ``final_text``). Any one backs the reply.
            result_file = next(
                (p for p in (session / "pi_request.json.result.json",
                             session / "claude_result.json",
                             session / "codex_result.json")
                 if p.is_file()),
                None,
            )
            if result_file is None:
                attempts.append({"session": session.name, "ok": None,
                                 "note": "no result written; the call did not complete"})
                continue
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            text = str(payload.get("final_response") or payload.get("final_text") or "")
            extracted, problem = extract_policy(text)
            attempts.append({
                "session": session.name, "ok": bool(payload.get("ok")),
                "error_type": payload.get("error_type"),
                "extracts_policy": extracted is not None, "extract_problem": problem,
                "reply_chars": len(text),
            })
            # Only the accepted attempt's source can be compared with the file:
            # a rejected attempt may have been overwritten by the repair.
            attempts[-1]["_source"] = extracted
        accepted = rev.status == ACCEPTED_REVISION
        last = attempts[-1] if attempts else None
        row = {
            "t": rev.iteration_t, "m": rev.m, "status": rev.status,
            "accepted": accepted,
            "attempts": [{k: v for k, v in a.items() if k != "_source"} for a in attempts],
            "has_stored_request": any(
                p.is_file() for p in (
                    session_root / f"t{rev.iteration_t}_m{rev.m}_a1" / "pi_request.json",
                    session_root / f"t{rev.iteration_t}_m{rev.m}_a1" / "claude_request.json",
                    session_root / f"t{rev.iteration_t}_m{rev.m}_a1" / "codex_request.json",
                )
            ),
        }
        if accepted:
            proposals = sorted(proposal_root.glob(
                f"proposals_t{rev.iteration_t}/llm_t{rev.iteration_t}_m{rev.m}_*.py"))
            row["proposal_files"] = [p.name for p in proposals]
            reply_source = (last or {}).get("_source")
            if len(proposals) == 1 and reply_source is not None:
                # `_propose_one` writes a two-line provenance header followed by
                # exactly the extracted source, so the file must end with it.
                row["proposal_ends_with_extracted_source"] = (
                    proposals[0].read_text(encoding="utf-8").endswith(reply_source)
                )
                row["proposal_sha256"] = hashlib.sha256(
                    proposals[0].read_bytes()).hexdigest()
        dev_rows.append(row)
        if accepted:
            if last is None:
                dev_problems.append(
                    f"t={rev.iteration_t} m={rev.m}: accepted with no stored Pi session"
                )
            elif not (last.get("ok") and last.get("extracts_policy")):
                dev_problems.append(
                    f"t={rev.iteration_t} m={rev.m}: accepted as a proposal but the reply that "
                    f"produced it is ok={last.get('ok')} with extract problem "
                    f"{last.get('extract_problem')!r}"
                )
            elif not row.get("proposal_ends_with_extracted_source"):
                dev_problems.append(
                    f"t={rev.iteration_t} m={rev.m}: accepted, but the proposal file on disk "
                    f"is not the source found in its own Pi reply"
                )
        else:
            # Not a success, and reported as what it is rather than omitted.
            rejected.append({
                "t": rev.iteration_t, "m": rev.m, "status": rev.status,
                "failure_reason": rev.failure_reason,
                "attempts": [{k: v for k, v in a.items() if k != "_source"} for a in attempts],
            })
    checks.add(
        "policy_development.real_pi_calls",
        not dev_problems and bool(dev_rows) and any(r["accepted"] for r in dev_rows),
        f"{sum(1 for r in dev_rows if r['accepted'])} accepted revisions each backed by a "
        f"successful Pi reply carrying a policy file; "
        f"{len(rejected)} revision(s) not accepted, listed with their status"
        if not dev_problems and dev_rows and any(r["accepted"] for r in dev_rows)
        else "a revision was accepted on a reply that is missing, failed, or carries no policy",
        rows=dev_rows, rejected=rejected, problems=dev_problems,
        note="ok=true means the worker saw agent_settled; a prompt acknowledgement is not enough",
    )

    # ------------------------------------------------------- run completion
    finish_events = [e for e in events if e.kind == "run.finish"]
    selection_events = [e for e in events if e.kind == "policy.selected"]
    rounds_present = sorted({w.outer_t for w in worlds})
    nodes_by_world = {w.world_id: registry.list_nodes(w.world_id) for w in worlds}
    all_nodes = [n for nodes in nodes_by_world.values() for n in nodes]
    used_online_rounds = max(
        (n.round_index for n in all_nodes if n.parent_id is not None), default=-1
    ) + 1
    derived_m = max((r.m for r in revisions), default=0) + 1
    expected_selections = expect.outer_rounds - 1
    completion_problems: list[str] = []
    if run.state is not RunState.COMMITTED:
        completion_problems.append(f"state is {run.state.value}, not COMMITTED")
    if not run.finished_at:
        completion_problems.append("finished_at is empty")
    if not finish_events:
        completion_problems.append("no run.finish event")
    elif finish_events[-1] is not events[-1]:
        completion_problems.append("run.finish is not the last event")
    if not provenance_ok:
        completion_problems.append("declared parameters were not proven against run.cfg_digest")
    if rounds_present != list(range(expect.outer_rounds)):
        completion_problems.append(
            f"worlds cover outer rounds {rounds_present}, expected "
            f"{list(range(expect.outer_rounds))}"
        )
    if len(selection_events) != expected_selections:
        completion_problems.append(
            f"{len(selection_events)} policy.selected events, expected {expected_selections}"
        )
    # K1 is an upper bound. The policy may return an empty batch before it is
    # exhausted; the configuration digest proves the cap, not the tree depth.
    if used_online_rounds > expect.K1:
        completion_problems.append(
            f"used {used_online_rounds} online rounds, exceeding K1={expect.K1}"
        )
    if derived_m != expect.M:
        completion_problems.append(f"revisions give M={derived_m}, declared {expect.M}")
    checks.add(
        "run.completed", not completion_problems,
        f"COMMITTED with a finish event, {len(worlds)} outer rounds, "
        f"{len(selection_events)} recorded selections, K1 cap={expect.K1} "
        f"(used at most {used_online_rounds}), M={derived_m}, "
        "parameters proven against cfg_digest"
        if not completion_problems else "the run is not complete under its declared parameters",
        state=run.state.value, finished_at=run.finished_at,
        finish_events=len(finish_events), outer_rounds_present=rounds_present,
        selections_recorded=len(selection_events), selections_expected=expected_selections,
        world_count=len(worlds), expected_world_count=expect.outer_rounds,
        max_online_rounds_used=used_online_rounds, M_derived=derived_m,
        declared={"outer_rounds": expect.outer_rounds, "K1": expect.K1, "M": expect.M},
        problems=completion_problems,
    )

    # ------------------------------------------------------------------ events
    kinds: dict[str, int] = {}
    for event in events:
        kinds[event.kind] = kinds.get(event.kind, 0) + 1
    mirror = config.run_root(run_id) / "events.jsonl"
    mirror_rows: list[Any] = []
    mirror_problem: str | None = None
    try:
        mirror_rows = [json.loads(line) for line in mirror.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        mirror_problem = f"{type(exc).__name__}: {exc}"
    mirror_matches = mirror_problem is None and mirror_rows == [e.to_dict() for e in events]
    checks.add(
        "events.mirror_matches_registry", mirror_matches,
        f"{len(events)} events in the registry, {len(mirror_rows)} in the mirror; "
        + ("ordered contents match" if mirror_matches else "contents differ or mirror is unreadable"),
        kinds=kinds, parse_error=mirror_problem,
    )
    return checks


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--json", default="", help="also write the result here")
    # Declared parameters. The defaults name the task-1 verification run; they
    # are *declarations* and are proven against run.cfg_digest before any count
    # below them is trusted.
    ap.add_argument("--expect-backend", default="pi")
    ap.add_argument("--expect-developer", default="llm")
    ap.add_argument("--expect-provider", default="cc-switch-deep-seek")
    ap.add_argument("--expect-model", default="deepseek-v4-pro")
    ap.add_argument("--expect-outer-rounds", type=int, default=3)
    ap.add_argument("--expect-k1", type=int, default=2)
    ap.add_argument("--expect-workers", type=int, default=4)
    ap.add_argument("--expect-versions", type=int, default=3)
    ap.add_argument("--expect-agent-timeout-s", type=float, default=None)
    ap.add_argument("--expect-task", default=None, choices=KNOWN_TASKS)
    args = ap.parse_args(argv[1:])

    root = Path(args.project_root).resolve()
    config = load_config(project_root=root)
    expect = Expectations(
        backend=args.expect_backend, developer=args.expect_developer,
        provider=args.expect_provider, model=args.expect_model,
        outer_rounds=args.expect_outer_rounds, K1=args.expect_k1,
        W=args.expect_workers, M=args.expect_versions,
        agent_timeout_s=args.expect_agent_timeout_s,
        task=args.expect_task,
    )
    supplied = [a for a in argv[1:] if a.startswith("--expect-")]
    source = ("command line" if supplied
              else "tool defaults for the task-1 run, unproven until cfg_digest matches")
    registry = Registry(config.db_path)
    try:
        checks = verify(registry, config, args.run_id, expect, expectation_source=source)
    finally:
        registry.close()

    checks.print()
    failed = checks.failed
    print()
    print(f"{len(checks.rows) - len(failed)}/{len(checks.rows)} checks passed "
          f"for run {args.run_id!r}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"run_id": args.run_id, "checks": checks.rows,
                        "expectations": asdict(expect),
                        "expectation_source": source,
                        "failed": [r["check"] for r in failed]},
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
