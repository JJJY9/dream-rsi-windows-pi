"""Live discovery backend backed by the Claude Code CLI (headless ``claude -p``).

This adapter drives an installed Claude Code executable in non-interactive
print mode. One ``generate`` call is one full agent session in the attempt's
own workspace: the agent is handed the task prompt, edits the entry file in
place, and the caller reads back what actually changed on disk.

Facts that shape the adapter:

* The executable is resolved from ``DREAM_RSI_CLAUDE_BIN`` first (an explicit
  path to a bundled or versioned binary), then from ``PATH``. Nothing here ever
  reads Claude Code's own configuration files; authentication is whatever the
  CLI already has (its own login or the ``ANTHROPIC_*`` environment), exactly
  like the Pi adapter forwards the user's own Pi credentials and nothing else.
* ``--dangerously-skip-permissions`` is required for headless operation: there
  is no human to answer a permission prompt, and an edit that is silently
  denied would be recorded as ``NO_ARTIFACT_CHANGE`` — the most misleading
  possible reading of "the agent never got to act". Confinement is the
  attempt workspace plus the OS process container, not a permission dialog
  (see ``docs/SECURITY.md``; that document's posture applies unchanged).
* A settled session that left the entry file byte-identical is reported as
  ``NO_ARTIFACT_CHANGE``, not as a success: an untouched artifact is "no work",
  however fluently the agent's closing sentence is written.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.errors import BlockedModelConfiguration
from ..runtime.winjob import WindowsJob, terminate_tree
from .base import GeneratedAttempt, HistoryEntry, TaskSpec, digest_file, digest_text
from .dsh import build_prompt

#: Explicit executable override. Point this at a versioned/bundled binary when
#: ``claude`` is not (or should not be) on ``PATH``.
CLAUDE_BIN_ENV = "DREAM_RSI_CLAUDE_BIN"


def find_claude_executable() -> str | None:
    """The Claude Code executable to use, or ``None`` when none is available."""
    override = os.environ.get(CLAUDE_BIN_ENV)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("claude")


@dataclass(frozen=True)
class ClaudeSessionResult:
    """What one headless ``claude -p`` round trip produced."""

    ok: bool
    final_text: str = ""
    session_id: str | None = None
    error_type: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    total_cost_usd: float | None = None
    stderr_tail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return not self.ok


def _parse_cli_stdout(raw: str) -> dict[str, Any] | None:
    """Parse the single JSON document ``--output-format json`` emits.

    Tolerates surrounding blank lines; falls back to the last non-empty line,
    which is where the result object lands if the CLI ever prints a banner.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def run_claude_prompt(
    *,
    prompt: str,
    cwd: Path,
    model: str | None,
    timeout_s: float,
    executable: str | None = None,
    memory_limit_mb: int | None = 4096,
    evidence_dir: Path | None = None,
) -> ClaudeSessionResult:
    """Run one prompt through the Claude Code CLI and read back the result.

    ``cwd`` is the working directory the agent runs in. ``evidence_dir`` is
    where the request and result are persisted for later auditing; it defaults
    to ``cwd`` but discovery attempts pass their per-attempt run directory, so
    audit files never end up inside the candidate workspace.

    Never raises for a model-side failure: a refused or errored call comes back
    as ``ok=False`` with an ``error_type``, because a failed attempt still has
    to become a recorded node rather than an exception that loses the attempt.
    """
    started = time.monotonic()
    cwd = Path(cwd).resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    evidence = Path(evidence_dir).resolve() if evidence_dir is not None else cwd
    evidence.mkdir(parents=True, exist_ok=True)

    exe = executable or find_claude_executable()
    if not exe:
        raise BlockedModelConfiguration(
            "the Claude Code executable was not found. Install Claude Code and "
            f"make `claude` available on PATH, or set {CLAUDE_BIN_ENV} to the "
            "full path of the binary."
        )

    argv = [exe, "-p", "--output-format", "json", "--dangerously-skip-permissions"]
    if model:
        argv += ["--model", model]

    # Persist the exact request and reply for later auditing, the same way the
    # Pi transport stores its RPC request: ``tools/verify_run.py`` audits the
    # developer's input and the accepted reply from these files, and evidence
    # that only ever existed in memory cannot be audited after the fact.
    request_path = evidence / "claude_request.json"
    request_path.write_text(
        json.dumps(
            {"prompt": prompt, "model": model, "cwd": str(cwd), "argv": argv},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result_path = evidence / "claude_result.json"

    # The parent environment is forwarded as-is: authentication for the CLI is
    # whatever the user set up for it (its own stored login, or ANTHROPIC_*).
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    stdout = b""
    stderr = b""
    timed_out = False
    payload: dict[str, Any] = {}
    error_type: str | None = None

    with WindowsJob(memory_limit_mb=memory_limit_mb, active_process_limit=8) as job:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        job.assign(proc)
        try:
            stdout, stderr = proc.communicate(
                input=prompt.encode("utf-8"), timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_tree(proc, job)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                stdout, stderr = b"", b""

    duration = time.monotonic() - started
    if timed_out:
        error_type = "TIMEOUT"
    else:
        parsed = _parse_cli_stdout(stdout.decode("utf-8", "replace"))
        if parsed is None:
            error_type = "UNREADABLE_CLI_OUTPUT"
        else:
            payload = parsed
            if payload.get("is_error"):
                # ``subtype`` stays "success" even for an API-side failure in
                # observed builds, so the is_error flag and terminal_reason are
                # the fields that actually carry the outcome.
                error_type = str(
                    payload.get("terminal_reason")
                    or payload.get("api_error_status")
                    or "AGENT_ERROR"
                )

    final_text = payload.get("result")
    if not isinstance(final_text, str):
        final_text = ""
    error_message = None
    if error_type is not None:
        tail = stderr[-800:].decode("utf-8", "replace").strip()
        error_message = tail or str(payload.get("result") or "")[:400] or error_type

    total_cost = payload.get("total_cost_usd")
    result = ClaudeSessionResult(
        ok=error_type is None,
        final_text=final_text,
        session_id=payload.get("session_id"),
        error_type=error_type,
        error=error_message,
        duration_s=duration,
        timed_out=timed_out,
        total_cost_usd=float(total_cost) if isinstance(total_cost, (int, float)) else None,
        stderr_tail=stderr[-800:].decode("utf-8", "replace"),
        payload=payload,
    )
    result_path.write_text(
        json.dumps(
            {
                "ok": result.ok,
                "error_type": result.error_type,
                "error": result.error,
                "session_id": result.session_id,
                "final_text": result.final_text,
                "final_response": result.final_text,
                "duration_s": round(result.duration_s, 3),
                "timed_out": result.timed_out,
                "total_cost_usd": result.total_cost_usd,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


class ClaudeCodeBackend:
    """Real model calls through headless Claude Code. ``is_live`` is True."""

    backend_id = "claude"
    is_live = True

    def __init__(
        self,
        *,
        workdir_root: Path,
        provider: str | None = None,
        model: str | None = None,
        timeout_s: float = 900.0,
        executable: str | None = None,
        python_executable: str | None = None,
        memory_limit_mb: int | None = 4096,
    ) -> None:
        self.workdir_root = Path(workdir_root)
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        self.executable = executable
        self.python_executable = python_executable or sys.executable
        self.memory_limit_mb = memory_limit_mb

    def preflight(self) -> None:
        if find_claude_executable() is None and not (
            self.executable and Path(self.executable).is_file()
        ):
            raise BlockedModelConfiguration(
                "the Claude Code executable was not found. Install Claude Code "
                f"and make `claude` available on PATH, or set {CLAUDE_BIN_ENV} "
                "to the full path of the binary."
            )

    def generate(
        self,
        *,
        task: TaskSpec,
        parent_workspace: Path,
        history: Sequence[HistoryEntry],
        output_workspace: Path,
        invocation_id: str,
    ) -> GeneratedAttempt:
        started = time.monotonic()
        parent_workspace = Path(parent_workspace).resolve()
        output_workspace = Path(output_workspace).resolve()
        output_workspace.mkdir(parents=True, exist_ok=True)
        if parent_workspace.is_dir() and parent_workspace != output_workspace:
            shutil.copytree(parent_workspace, output_workspace, dirs_exist_ok=True)

        entry_file = output_workspace / task.entry_filename
        before = digest_file(entry_file)

        run_dir = (self.workdir_root / invocation_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_claude_prompt(
            prompt=build_prompt(task, history, time_limit_s=self.timeout_s),
            cwd=output_workspace,
            model=self.model,
            timeout_s=self.timeout_s,
            executable=self.executable,
            memory_limit_mb=self.memory_limit_mb,
            evidence_dir=run_dir,
        )

        duration = time.monotonic() - started
        error_type = result.error_type
        after = digest_file(entry_file)
        # A settled agent that touched nothing has not done the job. Without
        # this the attempt would be recorded as a success with an unchanged
        # artifact, which is the most flattering possible reading of "no work".
        if error_type is None and before == after:
            error_type = "NO_ARTIFACT_CHANGE"

        return GeneratedAttempt(
            invocation_id=invocation_id,
            backend_id=self.backend_id,
            model=self.model or "claude-default",
            workspace_path=output_workspace,
            entry_file_digest=after,
            succeeded=error_type is None,
            duration_s=duration,
            error_type=error_type,
            summary=result.final_text.strip()[:500],
            session_id=result.session_id,
            raw_response_digest=digest_text(result.final_text) or None,
            extra={
                "is_live": True,
                "entry_changed": before != after,
                "total_cost_usd": result.total_cost_usd,
                "stderr_tail": result.stderr_tail,
                "worker_error": result.error,
            },
        )


__all__ = [
    "CLAUDE_BIN_ENV",
    "ClaudeCodeBackend",
    "ClaudeSessionResult",
    "find_claude_executable",
    "run_claude_prompt",
]
