"""Codex transport contract and backend semantics.

No model and no CLI are needed: ``subprocess.Popen`` is replaced with a stub,
so these tests pin the *parent-side* contract — argv shape, prompt delivery via
stdin, JSONL event folding, timeout reaping, evidence persistence — and the
backend-level semantics the loop relies on (a byte-identical artifact is
``NO_ARTIFACT_CHANGE``, an errored call is recorded rather than raised).

The event schema is the observed one (codex-cli 0.162): ``thread.started`` /
``item.completed`` (``agent_message`` carries the final text, ``error`` items
are non-fatal warnings) / ``turn.completed`` / ``turn.failed``.
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

from dream_rsi.adapters import codex as cx  # noqa: E402
from dream_rsi.adapters.base import TaskSpec  # noqa: E402
from dream_rsi.adapters.codex import (  # noqa: E402
    CodexBackend,
    find_codex_executable,
    parse_codex_events,
    run_codex_prompt,
)
from dream_rsi.domain.errors import BlockedModelConfiguration  # noqa: E402
from dream_rsi.runtime import llm_policy_developer as devmod  # noqa: E402
from dream_rsi.runtime.llm_policy_developer import (  # noqa: E402
    BEGIN,
    END,
    CodexPolicyDeveloper,
)


def _stream(*events: dict[str, Any]) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")


def _ok_stream(text: str = "done") -> bytes:
    return _stream(
        {"type": "thread.started", "thread_id": "thread-xyz"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "error",
                                            "message": "config warning (non-fatal)"}},
        {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                                            "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 7}},
    )


class _RecordingProc:
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

    def communicate(
        self, input: bytes | None = None, timeout: float | None = None
    ) -> tuple[bytes, bytes]:
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
    monkeypatch.setattr(cx, "WindowsJob", _NullJob)
    monkeypatch.setattr(cx.subprocess, "Popen", _RecordingProc)


class TestEventFolding:
    def test_a_completed_turn_with_warnings_still_succeeds(self) -> None:
        summary = parse_codex_events(_ok_stream("hello").decode("utf-8"))
        assert summary["completed"] and not summary["failed"]
        assert summary["final_text"] == "hello"
        assert summary["thread_id"] == "thread-xyz"
        assert summary["usage"] == {"input_tokens": 100, "output_tokens": 7}

    def test_the_last_agent_message_wins(self) -> None:
        stream = _stream(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "second"}},
            {"type": "turn.completed", "usage": {}},
        )
        summary = parse_codex_events(stream.decode("utf-8"))
        assert summary["final_text"] == "second"

    def test_turn_failed_is_a_failure_with_its_message(self) -> None:
        message = json.dumps({"error": {"message": "bad model", "code": "invalid_request_error"}})
        stream = _stream({"type": "turn.failed", "error": {"message": message}})
        summary = parse_codex_events(stream.decode("utf-8"))
        assert summary["failed"] and not summary["completed"]
        assert cx._nested_error_code(summary["failure_message"]) == "invalid_request_error"

    def test_a_top_level_error_without_completion_is_not_success(self) -> None:
        stream = _stream({"type": "error", "message": "boom"})
        summary = parse_codex_events(stream.decode("utf-8"))
        assert not summary["completed"] and not summary["failed"]
        assert summary["top_error"] == "boom"

    def test_stray_lines_are_counted_not_fatal(self) -> None:
        text = "not json\n" + _ok_stream("x").decode("utf-8")
        summary = parse_codex_events(text)
        assert summary["completed"] and summary["unparsed_lines"] == 1


class TestExecutableResolution:
    def test_env_override_wins_when_the_file_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake = tmp_path / "codex.exe"
        fake.write_bytes(b"")
        monkeypatch.setenv(cx.CODEX_BIN_ENV, str(fake))
        assert find_codex_executable() == str(fake)

    def test_env_override_pointing_nowhere_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(cx.CODEX_BIN_ENV, "Z:/definitely/not/here/codex.exe")
        assert find_codex_executable() is None

    def test_path_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(cx.CODEX_BIN_ENV, raising=False)
        monkeypatch.setattr(cx.shutil, "which", lambda name: f"/found/{name}")
        assert find_codex_executable() == "/found/codex"


class TestRunCodexPrompt:
    def test_argv_prompt_and_environment(self, stubbed: None, tmp_path: Path) -> None:
        _RecordingProc.stdout_bytes = _ok_stream("hello")
        cwd = tmp_path / "work"

        result = run_codex_prompt(
            prompt="PROMPT TEXT",
            cwd=cwd,
            model="deepseek-v4-pro",
            timeout_s=10.0,
            executable="C:/fake/codex.exe",
        )

        proc = _RecordingProc.last
        assert proc is not None
        assert proc.argv == [
            "C:/fake/codex.exe", "exec",
            "--skip-git-repo-check",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", str(cwd.resolve()),
            "-m", "deepseek-v4-pro",
            "-",
        ]
        assert proc.stdin_input == b"PROMPT TEXT"
        assert proc.env["PYTHONIOENCODING"] == "utf-8"
        assert result.ok and result.final_text == "hello"
        assert result.session_id == "thread-xyz"
        assert result.usage["output_tokens"] == 7

    def test_a_failed_turn_maps_the_nested_error_code(self, stubbed: None, tmp_path: Path) -> None:
        message = json.dumps({"error": {"message": "bad model", "code": "invalid_request_error"}})
        _RecordingProc.stdout_bytes = _stream(
            {"type": "thread.started", "thread_id": "t"},
            {"type": "turn.failed", "error": {"message": message}},
        )
        result = run_codex_prompt(
            prompt="x", cwd=tmp_path, model="m", timeout_s=10.0,
            executable="C:/fake/codex.exe",
        )
        assert not result.ok
        assert result.error_type == "invalid_request_error"

    def test_timeout_reaps_the_tree_and_is_reported(
        self, stubbed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _RecordingProc.timeout_first = True
        reaped: list[str] = []
        monkeypatch.setattr(cx, "terminate_tree", lambda proc, job: reaped.append("killed"))
        result = run_codex_prompt(
            prompt="x", cwd=tmp_path, model="m", timeout_s=0.1,
            executable="C:/fake/codex.exe",
        )
        assert result.timed_out and result.error_type == "TIMEOUT"
        assert reaped == ["killed"]

    def test_missing_executable_blocks_before_spawning(
        self, stubbed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cx, "find_codex_executable", lambda: None)
        with pytest.raises(BlockedModelConfiguration):
            run_codex_prompt(prompt="x", cwd=tmp_path, model=None, timeout_s=10.0)

    def test_evidence_files_land_in_the_evidence_dir_not_the_cwd(
        self, stubbed: None, tmp_path: Path
    ) -> None:
        _RecordingProc.stdout_bytes = _ok_stream("evidence check")
        cwd = tmp_path / "work"
        evidence = tmp_path / "evidence"
        run_codex_prompt(
            prompt="THE PROMPT",
            cwd=cwd,
            model="m",
            timeout_s=10.0,
            executable="C:/fake/codex.exe",
            evidence_dir=evidence,
        )
        request = json.loads((evidence / "codex_request.json").read_text(encoding="utf-8"))
        result = json.loads((evidence / "codex_result.json").read_text(encoding="utf-8"))
        assert request["prompt"] == "THE PROMPT"
        assert result["ok"] is True
        assert result["final_response"] == "evidence check"
        assert result["final_text"] == "evidence check"
        assert not (cwd / "codex_request.json").exists()
        assert not (cwd / "codex_result.json").exists()


class TestGenerate:
    def _task(self, seed: Path) -> TaskSpec:
        return TaskSpec(
            task_id="packing",
            entry_filename="candidate.py",
            instructions="does not matter here",
            seed_workspace=seed,
        )

    def _backend(self, tmp_path: Path) -> CodexBackend:
        return CodexBackend(
            workdir_root=tmp_path / "agent",
            provider="p",
            model="deepseek-v4-pro",
            timeout_s=5.0,
            executable="C:/fake/codex.exe",
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
        _RecordingProc.stdout_bytes = _ok_stream("I edited the file")
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
        run_dir = tmp_path / "agent" / "attempt-1"
        assert (run_dir / "codex_request.json").is_file()
        assert (run_dir / "codex_result.json").is_file()
        assert not (output / "codex_request.json").exists()

    def test_a_settled_call_that_changed_nothing_is_not_a_success(
        self, stubbed: None, tmp_path: Path
    ) -> None:
        parent = self._parent(tmp_path)
        _RecordingProc.stdout_bytes = _ok_stream("all done!")
        attempt = self._backend(tmp_path).generate(
            task=self._task(parent),
            parent_workspace=parent,
            history=(),
            output_workspace=tmp_path / "output",
            invocation_id="attempt-2",
        )
        assert not attempt.succeeded
        assert attempt.error_type == "NO_ARTIFACT_CHANGE"

    def test_a_failed_turn_is_recorded_not_raised(self, stubbed: None, tmp_path: Path) -> None:
        parent = self._parent(tmp_path)
        message = json.dumps({"error": {"message": "bad model", "code": "invalid_request_error"}})
        _RecordingProc.stdout_bytes = _stream(
            {"type": "turn.failed", "error": {"message": message}},
        )
        attempt = self._backend(tmp_path).generate(
            task=self._task(parent),
            parent_workspace=parent,
            history=(),
            output_workspace=tmp_path / "output",
            invocation_id="attempt-3",
        )
        assert not attempt.succeeded
        assert attempt.error_type == "invalid_request_error"


VALID_POLICY = '''from __future__ import annotations


def select(observation, rho):
    """Return at most one legal action."""
    legal = list(observation.legal_actions)
    return legal[:1]
'''


def _reply(policy: str) -> str:
    return f"Here is the revision.\n{BEGIN}\n{policy}\n{END}\n"


def _fake_result(text: str, *, ok: bool = True, error_type: str | None = None) -> Any:
    return SimpleNamespace(
        ok=ok,
        failed=not ok,
        final_text=text,
        session_id="thread-dev",
        error_type=error_type,
        error=None if ok else "transport died",
        duration_s=0.5,
        usage={"input_tokens": 10, "output_tokens": 2},
    )


class TestCodexPolicyDeveloper:
    def test_a_valid_reply_becomes_a_provenanced_proposal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        incumbent = tmp_path / "incumbent.py"
        incumbent.write_text(VALID_POLICY, encoding="utf-8")
        prompts: list[str] = []

        def fake_run_codex_prompt(*, prompt: str, **_: Any) -> Any:
            prompts.append(prompt)
            return _fake_result(_reply(VALID_POLICY))

        monkeypatch.setattr(devmod, "run_codex_prompt", fake_run_codex_prompt)
        developer = CodexPolicyDeveloper(
            workdir_root=tmp_path / "policy_agent",
            provider="cc-switch-deepseek",
            model="deepseek-v4-pro",
            timeout_s=5.0,
            executable="C:/fake/codex.exe",
        )
        proposals = list(developer.propose(
            incumbent_path=incumbent,
            feedback=("world w0: V=0.2",),
            output_dir=tmp_path / "proposals",
            count=1,
            iteration_t=1,
        ))
        assert len(proposals) == 1
        assert proposals[0].developer == "llm-codex"
        text = proposals[0].path.read_text(encoding="utf-8")
        assert "developer=llm-codex" in text
        assert "model=deepseek-v4-pro" in text
        assert text.endswith(VALID_POLICY)

    def test_transport_failures_raise_instead_of_proposing_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        incumbent = tmp_path / "incumbent.py"
        incumbent.write_text(VALID_POLICY, encoding="utf-8")
        monkeypatch.setattr(
            devmod, "run_codex_prompt",
            lambda **_: _fake_result("", ok=False, error_type="TIMEOUT"),
        )
        developer = CodexPolicyDeveloper(
            workdir_root=tmp_path / "policy_agent",
            provider="p",
            model="m",
            timeout_s=5.0,
            executable="C:/fake/codex.exe",
        )
        with pytest.raises(RuntimeError, match="every policy-development call failed"):
            developer.propose(
                incumbent_path=incumbent,
                feedback=(),
                output_dir=tmp_path / "proposals",
                count=1,
                iteration_t=1,
            )
