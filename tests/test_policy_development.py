"""How policy revisions are produced, and how that gets labelled.

**Nothing in this file is live evidence.** ``LlmPolicyDeveloper``'s transport is
stubbed via ``monkeypatch`` wherever a call would otherwise happen, so these
tests exercise the *plumbing* around a model call — prompt construction,
response extraction, the repair round, failure routing, provenance — and say
nothing about whether any model was ever reached. Whether a run really used a
model is decided by ``LlmPolicyDeveloper.sessions`` on a real run, and that
number is what the report prints as ``policy_agent_calls``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dream_rsi.adapters._pi_session import PiSessionResult
from dream_rsi.domain.models import RunMode
from dream_rsi.runtime import llm_policy_developer as lpd
from dream_rsi.runtime.controller import Controller
from dream_rsi.runtime.llm_policy_developer import (
    BEGIN,
    END,
    ROLES,
    LlmPolicyDeveloper,
    extract_policy,
)
from dream_rsi.runtime.policy_development import (
    ALLOWED_IMPORTS,
    FORBIDDEN_CALLS,
    ScriptedVariantDeveloper,
    validate_policy_source,
)

#: Used to assert that no Pi child is ever started inside the repository.
#: `dream_rsi` itself comes from the editable install, so nothing here needs to
#: touch `sys.path`.
REPO = Path(__file__).resolve().parents[1]

INCUMBENT = """\
from typing import Any


def select(observation: Any, rho: float) -> list[str]:
    legal = list(observation.legal_actions)
    return legal[: max(1, int(observation.max_parallelism))]
"""

#: A response shaped exactly like a compliant model reply.
GOOD_BODY = """\
def select(observation: Any, rho: float) -> list[str]:
    legal = list(observation.legal_actions)
    width = 1 + round(max(0.0, min(1.0, rho)) * (int(observation.max_parallelism) - 1))
    return legal[: max(1, width)]
"""


def _reply(source: str = GOOD_BODY, **over: Any) -> PiSessionResult:
    payload = {"ok": True, "final_response": f"{BEGIN}\n{source}{END}\n"}
    payload.update(over)
    return PiSessionResult(
        ok=payload["ok"], final_text=str(payload["final_response"]),
        session_id="sess-test", event_types=("agent_settled",),
    )


def _developer(tmp_path: Path) -> LlmPolicyDeveloper:
    return LlmPolicyDeveloper(
        workdir_root=tmp_path / "policy_agent",
        provider="test-provider", model="test-model",
    )


@pytest.fixture()
def incumbent(tmp_path: Path) -> Path:
    path = tmp_path / "incumbent.py"
    path.write_text(INCUMBENT, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------- labelling


class _StubBackend:
    """Just enough backend to answer the one question the mode depends on."""

    def __init__(self, live: bool) -> None:
        self.is_live = live
        self.backend_id = "stub"


def _mode_for(*, backend_live: bool, developer: Any) -> RunMode:
    config = SimpleNamespace(loop=SimpleNamespace(M=4))
    controller = Controller(config, SimpleNamespace(  # type: ignore[arg-type]
        backend=_StubBackend(backend_live), developer=developer,
    ))
    return controller._run_mode()


def test_a_live_backend_with_the_scripted_developer_is_not_labelled_live() -> None:
    """The defect this encodes: a run's label used to come from the backend alone.

    Every policy revision in such a run is a bundled constant, so calling the
    whole run ``live`` would put offline artifacts into live statistics. The
    label takes the conservative value; the run record names the developer so
    the mix stays visible.
    """
    assert _mode_for(
        backend_live=True, developer=ScriptedVariantDeveloper()
    ) is RunMode.OFFLINE_FIXTURE
    assert _mode_for(
        backend_live=True, developer=LlmPolicyDeveloper(
            workdir_root=Path("."), provider="p", model="m")
    ) is RunMode.LIVE
    # A fixture backend is never live, whatever the developer claims.
    assert _mode_for(
        backend_live=False, developer=LlmPolicyDeveloper(
            workdir_root=Path("."), provider="p", model="m")
    ) is RunMode.OFFLINE_FIXTURE
    # M=1 means no revisions at all, so there is no policy side to be offline.
    assert _mode_for(backend_live=True, developer=None) is RunMode.LIVE


# ---------------------------------------------------------------- prompt


def test_prompt_states_the_signature_the_hard_rules_and_the_stance() -> None:
    prompt = lpd.build_developer_prompt(
        incumbent_source=INCUMBENT, incumbent_digest="a" * 64,
        feedback=["world w0: |T|-1=3, best non-root score=0.500000"],
        role=ROLES[1], iteration_t=1,
    )
    assert "def select(observation: Any, rho: float) -> list[str]:" in prompt
    assert "observation.max_parallelism" in prompt
    # The rules the static gate actually enforces must be stated, or the model
    # is being failed for breaking a rule it was never told. Derived from the
    # gate's own constants rather than restated here, so that changing the gate
    # without changing the prompt fails this test instead of silently costing
    # the model a wasted call on a rule it could not have known.
    for module in sorted(ALLOWED_IMPORTS):
        assert module in prompt, f"the prompt never names the allowed module {module!r}"
    for call in sorted(FORBIDDEN_CALLS):
        assert call in prompt, f"the prompt never names the forbidden call {call!r}"
    for rule in ("LF", "must not be decorated"):
        assert rule in prompt, rule
    assert ROLES[1].instruction.split(".")[0] in prompt
    assert "0.500000" in prompt, "measured feedback must reach the prompt"
    assert "`B` is the number of non-root nodes in the sealed world" in prompt
    assert "`n` is the number of nodes this policy revealed" in prompt
    assert "does not prove better future" in prompt
    assert INCUMBENT.splitlines()[-1] in prompt, "incumbent source must reach the prompt"
    assert BEGIN in prompt and END in prompt


def test_prompt_says_so_when_there_is_no_history_and_when_repairing() -> None:
    no_history = lpd.build_developer_prompt(
        incumbent_source=INCUMBENT, incumbent_digest="b" * 64,
        feedback=(), role=ROLES[0], iteration_t=0,
    )
    assert "(no history yet)" in no_history
    assert "REJECTED" not in no_history

    repair = lpd.build_developer_prompt(
        incumbent_source=INCUMBENT, incumbent_digest="b" * 64, feedback=(),
        role=ROLES[0], iteration_t=0,
        repair_reason="forbidden import 'os'",
    )
    assert "REJECTED" in repair and "forbidden import 'os'" in repair


def test_every_role_asks_for_a_different_stance() -> None:
    """Three identical "improve it" prompts converge on one corner of (n, a)."""
    assert len({r.key for r in ROLES}) == len(ROLES)
    assert len({r.instruction for r in ROLES}) == len(ROLES)


# ---------------------------------------------------------------- extraction


def test_extract_round_trips_a_policy_that_clears_the_real_gate(tmp_path: Path) -> None:
    source, problem = extract_policy(f"prose\n{BEGIN}\n{GOOD_BODY}{END}\ntrailing prose")
    assert problem is None
    written = tmp_path / "candidate.py"
    written.write_text(source, encoding="utf-8", newline="\n")
    check = validate_policy_source(written)
    assert check.ok, check.reason


def test_extract_normalises_crlf_because_the_gate_rejects_it() -> None:
    source, problem = extract_policy(
        f"{BEGIN}\r\n" + GOOD_BODY.replace("\n", "\r\n") + f"{END}"
    )
    assert problem is None
    assert "\r" not in source


def test_extract_refuses_a_truncated_response_instead_of_returning_a_fragment() -> None:
    """The transport keeps the *tail* of a long reply, so a cut response loses
    its opening. Returning what survived would deploy a different program than
    the one the model wrote, silently."""
    source, problem = extract_policy(f"{BEGIN}\n{GOOD_BODY[:40]}")
    assert source is None
    assert "truncated" in problem


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("", "no text"),
        ("I could not do this task.", f"no {BEGIN}"),
        (f"{BEGIN}\n\n\n{END}", "empty"),
        (f"{BEGIN}\n```python\n{GOOD_BODY}{END}", "unterminated"),
    ],
)
def test_extract_reports_every_way_a_response_can_be_unusable(
    text: str, needle: str
) -> None:
    source, problem = extract_policy(text)
    assert source is None, source
    assert needle in problem


def test_extract_tolerates_a_single_wrapping_fence() -> None:
    source, problem = extract_policy(f"{BEGIN}\n```python\n{GOOD_BODY}```\n{END}")
    assert problem is None
    assert source.startswith("def select")


# ---------------------------------------------------------------- proposing


def test_proposals_are_lf_and_carry_the_model_that_produced_them(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def fake(request: dict[str, Any], **_: Any) -> PiSessionResult:
        prompts.append(str(request["prompt"]))
        return _reply()

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    proposals = developer.propose(
        incumbent_path=incumbent, feedback=["world w0: |T|-1=3"],
        output_dir=tmp_path / "out", count=3, iteration_t=1,
    )

    assert [p.m for p in proposals] == [1, 2, 3]
    assert all(p.developer == "llm-pi" and p.is_live for p in proposals)
    for proposal in proposals:
        raw = proposal.path.read_bytes()
        assert b"\r\n" not in raw, "the static gate rejects CRLF"
        text = raw.decode("utf-8")
        assert "provider=test-provider" in text and "model=test-model" in text
        assert "sess-test" in proposal.summary
        assert validate_policy_source(proposal.path).ok
    # Three revisions must be three different questions, not one asked thrice.
    assert len(set(prompts)) == 3
    assert len(developer.sessions) == 3
    assert all(s["ok"] for s in developer.sessions)


def test_one_sequential_revision_uses_its_slot_and_measured_feedback(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def fake(request: dict[str, Any], **_: Any) -> PiSessionResult:
        prompts.append(str(request["prompt"]))
        return _reply()

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    proposals = developer.propose(
        incumbent_path=incumbent,
        feedback=["policy pi1 mean V=0.42", "rho=0.5: n=1, k=1, batches=(('root',),)"],
        output_dir=tmp_path / "out", count=1, iteration_t=2, start_m=2,
    )
    assert [p.m for p in proposals] == [2]
    assert "role=explore" in proposals[0].path.read_text(encoding="utf-8")
    assert len(prompts) == 1
    assert "policy pi1 mean V=0.42" in prompts[0]
    assert "rho=0.5: n=1, k=1" in prompts[0]


def test_a_rejected_candidate_is_repaired_once_with_the_real_reason(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live call is expensive; a mechanical rejection should not waste one."""
    prompts: list[str] = []
    bad = "import os\n\n\ndef select(observation, rho):\n    return []\n"

    def fake(request: dict[str, Any], **_: Any) -> PiSessionResult:
        prompts.append(str(request["prompt"]))
        return _reply(bad) if len(prompts) == 1 else _reply()

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    proposals = developer.propose(
        incumbent_path=incumbent, feedback=[],
        output_dir=tmp_path / "out", count=1, iteration_t=2,
    )

    assert len(proposals) == 1
    assert len(prompts) == 2, "exactly one repair round, not a retry loop"
    # Assert the *actual* reason rather than a remembered phrase. The point of
    # the repair prompt is that the model is told what really went wrong, so the
    # test asks the gate what that is instead of restating it -- which would
    # otherwise silently pass on a prompt carrying a stale or wrong reason.
    probe = tmp_path / "probe.py"
    probe.write_text(bad, encoding="utf-8", newline="\n")
    reason = validate_policy_source(probe).reason
    assert reason, "the gate accepted `import os`, so this test proves nothing"
    assert reason in prompts[1], f"the repair prompt did not carry the gate's reason {reason!r}"
    assert validate_policy_source(proposals[0].path).ok
    assert len(developer.sessions) == 2


def test_a_response_with_no_policy_is_not_retried(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is nothing to repair, so a second call would just be a reroll."""
    calls = 0

    def fake(request: dict[str, Any], **_: Any) -> PiSessionResult:
        nonlocal calls
        calls += 1
        # No sentinels at all: the model answered in prose and produced no file.
        return PiSessionResult(
            ok=True, final_text="I'm not able to help with that.",
            session_id="sess-test", event_types=("agent_settled",),
        )

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    with pytest.raises(RuntimeError, match="every policy-development call failed"):
        developer.propose(
            incumbent_path=incumbent, feedback=[],
            output_dir=tmp_path / "out", count=2, iteration_t=1,
        )
    assert calls == 2, "one call per revision, none retried"


def test_every_call_failing_raises_so_the_controller_records_the_real_reason(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently returning [] would re-select the incumbent and look like a
    successful round in which the model simply had nothing to add."""

    def fake(request: dict[str, Any], **_: Any) -> PiSessionResult:
        return PiSessionResult(ok=False, error_type="AUTHENTICATION_FAILED",
                               error="invalid x-api-key")

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        developer.propose(
            incumbent_path=incumbent, feedback=[],
            output_dir=tmp_path / "out", count=2, iteration_t=1,
        )
    assert "AUTHENTICATION_FAILED" in str(excinfo.value)


def test_a_partial_failure_still_yields_the_revisions_that_worked(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing one revision is a weaker round, not a broken one — so it does not
    take the whole selection down, and the survivors are still screened."""
    seen = {"n": 0}

    def fake(request: dict[str, Any], **_: Any) -> PiSessionResult:
        seen["n"] += 1
        if seen["n"] == 2:
            return PiSessionResult(ok=False, error_type="TIMEOUT", error="900s")
        return _reply()

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    proposals = developer.propose(
        incumbent_path=incumbent, feedback=[],
        output_dir=tmp_path / "out", count=3, iteration_t=1,
    )
    assert [p.m for p in proposals] == [1, 3]
    assert all(validate_policy_source(p.path).ok for p in proposals)


def test_developer_refuses_to_exist_without_a_named_model(tmp_path: Path) -> None:
    """A developer that cannot say which model it used is indistinguishable
    from one that used no model, so it does not get to exist."""
    for provider, model in (("", "m"), ("p", ""), ("", "")):
        with pytest.raises(ValueError, match="explicit provider and model"):
            LlmPolicyDeveloper(workdir_root=tmp_path, provider=provider, model=model)


def test_pi_runs_in_an_empty_directory_never_the_repository(
    tmp_path: Path, incumbent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incumbent is pasted into the prompt, so the model needs no file
    access; a cwd inside the project would let an agentic detour edit it."""
    cwds: list[Path] = []

    def fake(request: dict[str, Any], run_dir: Path, **_: Any) -> PiSessionResult:
        cwds.append(Path(str(request["cwd"])))
        return _reply()

    monkeypatch.setattr(lpd, "run_pi_prompt", fake)
    developer = _developer(tmp_path)
    developer.propose(
        incumbent_path=incumbent, feedback=[],
        output_dir=tmp_path / "out", count=2, iteration_t=1,
    )
    for cwd in cwds:
        assert REPO not in cwd.parents and cwd != REPO
        assert tmp_path in cwd.parents
