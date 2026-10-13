"""Live discovery backend backed by the OpenAI Codex CLI (headless ``codex exec``).

This adapter drives an installed Codex executable in non-interactive mode. One
``generate`` call is one full agent session in the attempt's own workspace: the
agent is handed the task prompt on stdin, edits the entry file in place, and
the caller reads back what actually changed on disk.

Facts that shape the adapter (observed on codex-cli 0.162):

* The executable is resolved from ``DREAM_RSI_CODEX_BIN`` first (an explicit
  path to a bundled or versioned binary), then from ``PATH``. Auth and provider
  configuration are the CLI's own (``~/.codex/config.toml`` / ``auth.json``);
  nothing here reads or rewrites them.
* The model is always passed explicitly with ``-m``: a config that names a
  model from a different provider is a real observed failure mode, and a run
  must only ever be attributed to the model it actually invoked.
* ``exec``'s non-interactive default is a **read-only** sandbox under which
  every write is denied by the approval policy (observed: ``patch rejected:
  writing is blocked by read-only sandbox``), so a default invocation could
  never edit the candidate. The adapter passes
  ``--dangerously-bypass-approvals-and-sandbox`` instead; confinement is the
  attempt workspace plus the OS process container, exactly the posture
  ``docs/SECURITY.md`` describes for every live backend here — not the flag.
* ``--skip-git-repo-check``: attempt workspaces are not repositories.
* Output is JSONL on stdout. The final agent text is the last
  ``item.completed`` of type ``agent_message``; failure is ``turn.failed``.
  Non-fatal ``item.completed`` events of type ``error`` (config warnings) must
  NOT be mistaken for failures — a warning followed by a completed turn is a
  successful call.
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
#: ``codex`` is not (or should not be) on ``PATH``.
CODEX_BIN_ENV = "DREAM_RSI_CODEX_BIN"


def find_codex_executable() -> str | None:
    """The Codex executable to use, or ``None`` when none is available."""
    override = os.environ.get(CODEX_BIN_ENV)
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("codex")


@dataclass(frozen=True)
class CodexSessionResult:
    """What one headless ``codex exec`` round trip produced."""

    ok: bool
    final_text: str = ""
    session_id: str | None = None
    error_type: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    stderr_tail: str = ""

    @property
    def failed(self) -> bool:
        return not self.ok


def _nested_error_code(message: str) -> str | None:
    """Pull ``error.code`` out of the API error JSON Codex embeds in messages.

    ``turn.failed`` carries the provider's error body as a JSON *string*
    (observed with the model-name rejection); the code inside it is the most
    precise error_type available. Anything unparseable yields ``None`` and the
    caller falls back to a generic type.
    """
    try:
        parsed = json.loads(message)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return str(error["code"])
    return None


def parse_codex_events(stdout_text: str) -> dict[str, Any]:
    """Fold a JSONL event stream into one outcome summary.

    Tolerant by design: unparseable lines (a stray log line on stdout) are
    counted and skipped rather than failing the whole call.
    """
    summary: dict[str, Any] = {
        "thread_id": None,
        "final_text": "",
        "completed": False,
        "failed": False,
        "failure_message": None,
        "top_error": None,
        "usage": {},
        "unparsed_lines": 0,
    }
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            summary["unparsed_lines"] += 1
            continue
        if not isinstance(event, dict):
            summary["unparsed_lines"] += 1
            continue
        kind = event.get("type")
        if kind == "thread.started" and isinstance(event.get("thread_id"), str):
            summary["thread_id"] = event["thread_id"]
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                summary["final_text"] = item["text"]
        elif kind == "turn.failed":
            summary["failed"] = True
            error = event.get("error") or {}
            summary["failure_message"] = str(error.get("message") or "turn failed")
        elif kind == "turn.completed":
            summary["completed"] = True
            usage = event.get("usage")
            if isinstance(usage, dict):
                summary["usage"] = {
                    str(k): int(v) for k, v in usage.items() if isinstance(v, int)
                }
        elif kind == "error":
            summary["top_error"] = str(event.get("message") or "error event")
    return summary


def run_codex_prompt(
    *,
    prompt: str,
    cwd: Path,
    model: str | None,
    timeout_s: float,
    executable: str | None = None,
    memory_limit_mb: int | None = 4096,
    evidence_dir: Path | None = None,
) -> CodexSessionResult:
    """Run one prompt through the Codex CLI and read back the result.

    ``cwd`` is the working directory (and workspace root) the agent runs in.
    ``evidence_dir`` is where the request and result are persisted; it defaults
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

    exe = executable or find_codex_executable()
    if not exe:
        raise BlockedModelConfiguration(
            "the Codex executable was not found. Install the Codex CLI and "
            f"make `codex` available on PATH, or set {CODEX_BIN_ENV} to the "
            "full path of the binary."
        )

    argv = [
        exe, "exec",
        "--skip-git-repo-check",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", str(cwd),
    ]
    if model:
        argv += ["-m", model]
    argv += ["-"]  # read the prompt from stdin

    request_path = evidence / "codex_request.json"
    request_path.write_text(
        json.dumps(
            {"prompt": prompt, "model": model, "cwd": str(cwd), "argv": argv},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result_path = evidence / "codex_result.json"

    # The parent environment is forwarded as-is: authentication for the CLI is
    # whatever the user set up for it (~/.codex auth or OPENAI_API_KEY).
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    stdout = b""
    stderr = b""
    timed_out = False
    error_type: str | None = None
    error_message: str | None = None
    final_text = ""
    session_id: str | None = None
    usage: dict[str, int] = {}

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
        error_message = f"codex exceeded {timeout_s}s"
    else:
        summary = parse_codex_events(stdout.decode("utf-8", "replace"))
        final_text = str(summary["final_text"])
        session_id = summary["thread_id"]
        usage = dict(summary["usage"])
        if summary["failed"]:
            message = str(summary["failure_message"])
            error_type = _nested_error_code(message) or "TURN_FAILED"
            error_message = message[:400]
        elif not summary["completed"]:
            top = summary["top_error"]
            error_type = (_nested_error_code(str(top)) if top else None) or "NO_TURN_COMPLETION"
            error_message = str(top or stderr[-800:].decode("utf-8", "replace"))[:400]

    result = CodexSessionResult(
        ok=error_type is None,
        final_text=final_text,
        session_id=session_id,
        error_type=error_type,
        error=error_message,
        duration_s=duration,
        timed_out=timed_out,
        usage=usage,
        stderr_tail=stderr[-800:].decode("utf-8", "replace"),
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
                "usage": result.usage,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


class CodexBackend:
    """Real model calls through headless Codex CLI. ``is_live`` is True."""

    backend_id = "codex"
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
        if find_codex_executable() is None and not (
            self.executable and Path(self.executable).is_file()
        ):
            raise BlockedModelConfiguration(
                "the Codex executable was not found. Install the Codex CLI and "
                f"make `codex` available on PATH, or set {CODEX_BIN_ENV} to "
                "the full path of the binary."
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
        result = run_codex_prompt(
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
            model=self.model or "codex-default",
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
                "usage": dict(result.usage),
                "stderr_tail": result.stderr_tail,
                "worker_error": result.error,
            },
        )


__all__ = [
    "CODEX_BIN_ENV",
    "CodexBackend",
    "CodexSessionResult",
    "find_codex_executable",
    "parse_codex_events",
    "run_codex_prompt",
]
