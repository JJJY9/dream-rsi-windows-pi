"""Claude Code transport contract and backend semantics.

No model and no CLI are needed: ``subprocess.Popen`` is replaced with a stub,
so these tests pin the *parent-side* contract — argv shape, prompt delivery via
stdin, stdout/jSON parsing, timeout reaping, evidence persistence — and the
backend-level semantics the loop relies on (a byte-identical artifact is
``NO_ARTIFACT_CHANGE``, an errored call is recorded rather than raised).

The developer tests stub the transport one level higher and pin the pieces
that make model-written policies trustworthy: the sentinel extraction, the
static gate, the single repair round that carries the rejection reason, and
the rule that a run of failed calls raises instead of silently proposing
nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(SRC))

from dream_rsi.adapters import claude_code as cc  # noqa: E402
from dream_rsi.adapters.base import TaskSpec  # noqa: E402
from dream_rsi.adapters.claude_code import (  # noqa: E402
    ClaudeCodeBackend,
    find_claude_executable,
    run_claude_prompt,
)
from dream_rsi.domain.errors import BlockedModelConfiguration  # noqa: E402
from dream_rsi.runtime import llm_policy_developer as devmod  # noqa: E402
from dream_rsi.runtime.llm_policy_developer import (  # noqa: E402
    BEGIN,
    END,
    ClaudePolicyDeveloper,
)


def _ok_stdout(text: str = "done", cost: float = 0.12) -> bytes:
    return json.dumps(
        {
            "is_error": False,
            "subtype": "success",
            "session_id": "sess-xyz",
            "result": text,
            "total_cost_usd": cost,
        }
    ).encode("utf-8")


class _RecordingProc:
    """Stands in for the claude CLI process; records how it was launched."""

    last: _RecordingProc | None = None
    stdout_bytes: bytes = b""
    stderr_bytes: bytes = b""
    timeout_first: bool = False
    on_spawn: Any = None

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = argv
        self.cwd = kwargs.get("cwd")
        self.env = kwargs.get("env") or {}
        self.stdin_input: bytes | None = None
        self.returncode = 0
        _RecordingProc.last = self
        if type(self).on_spawn is not None:
            type(self).on_spawn(argv, kwargs)

    def communicate(self, input: bytes | None = None, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.stdin_input = input
        if type(self).timeout_first:
            type(self).timeout_first = False
            raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout or 0)
        return type(self).stdout_bytes, type(self).stderr_bytes


class _NullJob:
    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> _NullJob:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def assign(self, proc: Any) -> None:
        pass


@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingProc.stdout_bytes = b""
    _RecordingProc.stderr_bytes = b""
    _RecordingProc.timeout_first = False
    _RecordingProc.on_spawn = None
    monkeypatch.setattr(cc, "WindowsJob", _NullJob)
    monkeypatch.setattr(cc.subprocess, "Popen", _RecordingProc)


class TestStdoutParsing:
    def test_clean_json(self) -> None:
        assert cc._parse_cli_stdout('{"a": 1}') == {"a": 1}

    def test_surrounding_whitespace(self) -> None:
        assert cc._parse_cli_stdout('  \n{"a": 1}\n  ') == {"a": 1}

    def test_banner_then_json_on_the_last_line(self) -> None:
        assert cc._parse_cli_stdout('warning: something\n{"a": 2}\n') == {"a": 2}

    def test_garbage_is_none(self) -> None:
        assert cc._parse_cli_stdout("not json at all") is None

    def test_non_dict_json_is_none(self) -> None:
        assert cc._parse_cli_stdout("[1, 2]") is None

    def test_empty_is_none(self) -> None:
        assert cc._parse_cli_stdout("   ") is None


class TestExecutableResolution:
    def test_env_override_wins_when_the_file_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake = tmp_path / "claude.exe"
        fake.write_bytes(b"")
        monkeypatch.setenv(cc.CLAUDE_BIN_ENV, str(fake))
        assert find_claude_executable() == str(fake)

    def test_env_override_pointing_nowhere_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(cc.CLAUDE_BIN_ENV, "Z:/definitely/not/here/claude.exe")
        assert find_claude_executable() is None

    def test_path_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(cc.CLAUDE_BIN_ENV, raising=False)
        monkeypatch.setattr(cc.shutil, "which", lambda name: f"/found/{name}")
        assert find_claude_executable() == "/found/claude"

    def test_nothing_anywhere_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(cc.CLAUDE_BIN_ENV, raising=False)
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        assert find_claude_executable() is None


class TestRunClaudePrompt:
    def test_argv_prompt_and_environment(self, stubbed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _RecordingProc.stdout_bytes = _ok_stdout("hello")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid/anthropic")

        result = run_claude_prompt(
            prompt="PROMPT TEXT",
            cwd=tmp_path / "cwd",
            model="claude-sonnet-5",
            timeout_s=10.0,
            executable="C:/fake/claude.exe",
        )

        proc = _RecordingProc.last
        assert proc is not None
        assert proc.argv == [
            "C:/fake/claude.exe",
            "-p",
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--model", "claude-sonnet-5",
        ]
        assert proc.stdin_input == b"PROMPT TEXT"
        assert proc.env["PYTHONIOENCODING"] == "utf-8"
        # The user's own ANTHROPIC_* configuration is forwarded on purpose.
        assert proc.env["ANTHROPIC_BASE_URL"] == "https://example.invalid/anthropic"
        assert result.ok and result.final_text == "hello"
        assert result.session_id == "sess-xyz"
        assert result.total_cost_usd == pytest.approx(0.12)

    def test_is_error_maps_the_terminal_reason(self, stubbed: None, tmp_path: Path) -> None:
        _RecordingProc.stdout_bytes = json.dumps(
            {
                "is_error": True,
                "subtype": "success",  # observed: stays "success" even on api errors
                "terminal_reason": "api_error",
                "result": "Not logged in",
            }
        ).encode("utf-8")
        result = run_claude_prompt(
            prompt="x", cwd=tmp_path, model=None, timeout_s=10.0, executable="C:/fake/claude.exe"
        )
        assert not result.ok
        assert result.error_type == "api_error"
        assert result.error  # carries the human-readable cause

    def test_unreadable_stdout_names_the_problem(self, stubbed: None, tmp_path: Path) -> None:
        _RecordingProc.stdout_bytes = b"definitely not json"
        result = run_claude_prompt(
            prompt="x", cwd=tmp_path, model=None, timeout_s=10.0, executable="C:/fake/claude.exe"
        )
        assert not result.ok
        assert result.error_type == "UNREADABLE_CLI_OUTPUT"

    def test_timeout_reaps_the_tree_and_is_reported(
        self, stubbed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _RecordingProc.timeout_first = True
        reaped: list[str] = []
        monkeypatch.setattr(cc, "terminate_tree", lambda proc, job: reaped.append("killed"))
        result = run_claude_prompt(
            prompt="x", cwd=tmp_path, model=None, timeout_s=0.1, executable="C:/fake/claude.exe"
        )
        assert result.timed_out and result.error_type == "TIMEOUT"
        assert reaped == ["killed"]

    def test_missing_executable_blocks_before_spawning(
        self, stubbed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cc, "find_claude_executable", lambda: None)
        with pytest.raises(BlockedModelConfiguration):
            run_claude_prompt(prompt="x", cwd=tmp_path, model=None, timeout_s=10.0)

    def test_evidence_files_land_in_the_evidence_dir_not_the_cwd(
        self, stubbed: None, tmp_path: Path
    ) -> None:
        _RecordingProc.stdout_bytes = _ok_stdout("evidence check")
        cwd = tmp_path / "work"
        evidence = tmp_path / "evidence"
        run_claude_prompt(
            prompt="THE PROMPT",
            cwd=cwd,
            model="m",
            timeout_s=10.0,
            executable="C:/fake/claude.exe",
            evidence_dir=evidence,
        )
        request = json.loads((evidence / "claude_request.json").read_text(encoding="utf-8"))
        result = json.loads((evidence / "claude_result.json").read_text(encoding="utf-8"))
        assert request["prompt"] == "THE PROMPT"
        assert result["ok"] is True
        # `verify_run` reads either key; the adapter writes both on purpose.
        assert result["final_response"] == "evidence check"
        assert result["final_text"] == "evidence check"
        assert not (cwd / "claude_request.json").exists()
        assert not (cwd / "claude_result.json").exists()


class TestGenerate:
    def _task(self, seed: Path) -> TaskSpec:
        return TaskSpec(
            task_id="packing",
            entry_filename="candidate.py",
            instructions="does not matter here",
            seed_workspace=seed,
        )

    def _backend(self, tmp_path: Path) -> ClaudeCodeBackend:
        return ClaudeCodeBackend(
            workdir_root=tmp_path / "agent",
            provider="p",
            model="claude-sonnet-5",
            timeout_s=5.0,
            executable="C:/fake/claude.exe",
        )

    def _parent(self, tmp_path: Path) -> Path:
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
        return parent

    def test_a_real_edit_is_a_success_with_clean_workspace_evidence(
        self, stubbed: None, tmp_path: Path
    ) -> None:
        parent = self._parent(tmp_path)
        _RecordingProc.stdout_bytes = _ok_stdout("I edited the file")
        _RecordingProc.on_spawn = lambda argv, kwargs: (
            Path(kwargs["cwd"]) / "candidate.py"
        ).write_text("VALUE = 2\n", encoding="utf-8")
        output = tmp_path / "output"

        attempt = self._backend(tmp_path).generate(
            task=self._task(parent),
            parent_workspace=parent,
            history=(),
            output_workspace=output,
            invocation_id="attempt-1",
        )

        assert attempt.succeeded and attempt.error_type is None
        assert attempt.extra["entry_changed"] is True
        assert (output / "candidate.py").read_text(encoding="utf-8") == "VALUE = 2\n"
        assert attempt.summary.startswith("I edited the file")
        # Audit evidence goes to the per-attempt run dir, never into the
        # candidate workspace (it would otherwise enter workspace digests).
        run_dir = tmp_path / "agent" / "attempt-1"
        assert (run_dir / "claude_request.json").is_file()
        assert (run_dir / "claude_result.json").is_file()
        assert not (output / "claude_request.json").exists()

    def test_a_settled_call_that_changed_nothing_is_not_a_success(
        self, stubbed: None, tmp_path: Path
    ) -> None:
        parent = self._parent(tmp_path)
        _RecordingProc.stdout_bytes = _ok_stdout("all done!")
        attempt = self._backend(tmp_path).generate(
            task=self._task(parent),
            parent_workspace=parent,
            history=(),
            output_workspace=tmp_path / "output",
            invocation_id="attempt-2",
        )
        assert not attempt.succeeded
        assert attempt.error_type == "NO_ARTIFACT_CHANGE"

    def test_an_agent_error_is_recorded_not_raised(self, stubbed: None, tmp_path: Path) -> None:
        parent = self._parent(tmp_path)
        _RecordingProc.stdout_bytes = json.dumps(
            {"is_error": True, "terminal_reason": "api_error", "result": "Not logged in"}
        ).encode("utf-8")
        attempt = self._backend(tmp_path).generate(
            task=self._task(parent),
            parent_workspace=parent,
            history=(),
            output_workspace=tmp_path / "output",
            invocation_id="attempt-3",
        )
        assert not attempt.succeeded
        assert attempt.error_type == "api_error"


# ---------------------------------------------------------------------------
# developer
# ---------------------------------------------------------------------------

VALID_POLICY = '''from __future__ import annotations


def select(observation, rho):
    """Return at most one legal action, preferring the newest node."""
    legal = list(observation.legal_actions)
    if not legal:
        return []
    return legal[:1]
'''

INVALID_POLICY = '''from __future__ import annotations


def select(observation, rho):
    handle = open("sneaky.txt")
    return [handle.name]
'''


def _reply(policy: str) -> str:
    return f"Here is the revision.\n{BEGIN}\n{policy}\n{END}\n"


def _fake_result(text: str, *, ok: bool = True, error_type: str | None = None) -> Any:
    return SimpleNamespace(
        ok=ok,
        failed=not ok,
        final_text=text,
        session_id="sess-dev",
        error_type=error_type,
        error=None if ok else "transport died",
        duration_s=0.5,
        total_cost_usd=0.01,
    )


class TestClaudePolicyDeveloper:
    def _developer(self, tmp_path: Path) -> ClaudePolicyDeveloper:
        incumbent = tmp_path / "incumbent.py"
        incumbent.write_text(VALID_POLICY, encoding="utf-8")
        return ClaudePolicyDeveloper(
            workdir_root=tmp_path / "policy_agent",
            provider="cc-switch-deepseek",
            model="claude-sonnet-5",
            timeout_s=5.0,
            executable="C:/fake/claude.exe",
        )

    def _propose(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        replies: list[Any],
    ) -> tuple[list[Any], list[str]]:
        prompts: list[str] = []

        def fake_run_claude_prompt(*, prompt: str, **_: Any) -> Any:
            prompts.append(prompt)
            return replies[len(prompts) - 1]

        monkeypatch.setattr(devmod, "run_claude_prompt", fake_run_claude_prompt)
        developer = self._developer(tmp_path)
        proposals = developer.propose(
            incumbent_path=tmp_path / "incumbent.py",
            feedback=("world w0: V=0.2",),
            output_dir=tmp_path / "proposals",
            count=1,
            iteration_t=1,
        )
        return list(proposals), prompts

    def test_a_valid_reply_becomes_a_provenanced_proposal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        proposals, prompts = self._propose(
            monkeypatch, tmp_path, [_fake_result(_reply(VALID_POLICY))]
        )
        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.developer == "llm-claude"
        assert proposal.is_live
        # m=1 rotates to the exploit stance; its name lands in the file name.
        assert "exploit" in proposal.path.name
        text = proposal.path.read_text(encoding="utf-8")
        assert "developer=llm-claude" in text
        assert "model=claude-sonnet-5" in text
        assert "attempt=1" in text
        assert text.endswith(VALID_POLICY)
        # The developer's measured feedback reached the prompt.
        assert "V=0.2" in prompts[0]

    def test_a_static_rejection_is_repaired_with_the_exact_reason(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        proposals, prompts = self._propose(
            monkeypatch,
            tmp_path,
            [_fake_result(_reply(INVALID_POLICY)), _fake_result(_reply(VALID_POLICY))],
        )
        assert len(proposals) == 1
        assert len(prompts) == 2, "a rejection must be followed by exactly one repair call"
        assert "REJECTED by the static gate" in prompts[1]
        assert "forbidden call 'open'()" in prompts[1]
        assert "attempt=2" in proposals[0].path.read_text(encoding="utf-8")

    def test_a_reply_without_a_policy_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="every policy-development call failed"):
            self._propose(monkeypatch, tmp_path, [_fake_result("no markers anywhere")])
        # exactly one call: a missing marker is not a repairable shape error

    def test_transport_failures_raise_instead_of_proposing_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="every policy-development call failed"):
            self._propose(
                monkeypatch,
                tmp_path,
                [_fake_result("", ok=False, error_type="TIMEOUT")],
            )
