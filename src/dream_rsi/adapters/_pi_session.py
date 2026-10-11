"""One Pi RPC round trip, shared by everything that needs a live model call.

The discovery backend and the policy developer need exactly the same thing:
hand a prompt to ``pi --mode rpc``, wait for the agent to *settle* rather than
merely acknowledge, and read back the final assistant text. Keeping a single
copy matters because the parts being shared are the parts that are easy to get
subtly wrong — Job Object containment, the timeout-then-reap path, and the
"a prompt ack is not a completion" rule.

The worker itself (``_pi_worker``) already exists and is what actually speaks
the protocol; this module is the parent-side half that starts it and interprets
what it wrote.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.winjob import WindowsJob, terminate_tree

#: The minimum environment a child needs to start and find its tools. Anything
#: not listed here is dropped rather than inherited.
_BASE_ENV = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "APPDATA", "LOCALAPPDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "USERNAME",
)

#: Credential variables belonging to *other* agents. Matched against the
#: variable NAME. This is what stops a Pi child from silently picking up Claude
#: Code's own key and reporting success that was never Pi's to claim.
_FORBIDDEN_NAME_SUBSTRINGS = ("ANTHROPIC", "DEEPSEEK", "OPENAI", "CLAUDE")


def pi_child_env() -> dict[str, str]:
    """Environment for a Pi child: the OS minimum plus the user's own ``PI_*``.

    Everything else — most importantly any ``*_API_KEY`` belonging to another
    tool — is dropped rather than inherited.
    """
    env: dict[str, str] = {}
    for name in _BASE_ENV:
        value = os.environ.get(name)
        if value:
            env[name] = value
    for name, value in os.environ.items():
        if not name.upper().startswith("PI_"):
            continue
        if any(bad in name.upper() for bad in _FORBIDDEN_NAME_SUBSTRINGS):
            continue
        env[name] = value
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@dataclass(frozen=True)
class PiSessionResult:
    """What one ``pi --mode rpc`` round trip produced."""

    ok: bool
    final_text: str = ""
    session_id: str | None = None
    event_types: tuple[str, ...] = ()
    error_type: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    stderr_tail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return not self.ok


def run_pi_prompt(
    *,
    request: dict[str, Any],
    run_dir: Path,
    timeout_s: float,
    python_executable: str | None = None,
    memory_limit_mb: int | None = 4096,
) -> PiSessionResult:
    """Run one prompt through Pi and return its final assistant text.

    Never raises for a model-side failure: a refused or errored call comes back
    as ``ok=False`` with an ``error_type``, because a failed attempt still has
    to become a recorded node rather than an exception that loses the attempt.
    """
    started = time.monotonic()
    python_executable = python_executable or sys.executable
    # Resolved, not merely wrapped: the child runs with `cwd=run_dir`, so a
    # relative request path handed to it would be resolved against the *child's*
    # directory and the child would read a file that is not there. It still
    # exits 0 after writing its error payload — to yet another wrong path — so
    # the parent sees only "no result file" and loses the real reason.
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    request_path = run_dir / "pi_request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    result_path = Path(str(request_path) + ".result.json")

    argv = [
        python_executable, "-m", "dream_rsi.adapters._pi_worker", str(request_path),
    ]
    payload: dict[str, Any] = {}
    timed_out = False
    stdout = b""
    stderr = b""
    error_type: str | None = None

    with WindowsJob(memory_limit_mb=memory_limit_mb, active_process_limit=16) as job:
        proc = subprocess.Popen(
            argv, cwd=str(run_dir), env=pi_child_env(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False,
        )
        job.assign(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_tree(proc, job)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                stdout, stderr = b"", b""

    # The worker reports through its result file, never through stdout. The
    # pipes are still drained above, because a child that writes more than the
    # pipe buffer holds would otherwise block forever waiting for a reader.
    del stdout

    duration = time.monotonic() - started
    if timed_out:
        error_type = "TIMEOUT"
    elif result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            error_type = "UNREADABLE_WORKER_RESULT"
        else:
            if not payload.get("ok"):
                error_type = str(payload.get("error_type") or "AGENT_FAILED")
    else:
        error_type = f"WORKER_NO_RESULT_EXIT_{proc.returncode}"

    response = payload.get("final_response")
    text = response if isinstance(response, str) else ""
    return PiSessionResult(
        ok=error_type is None,
        final_text=text,
        session_id=payload.get("session_id"),
        event_types=tuple(payload.get("event_types") or ()),
        error_type=error_type,
        error=str(payload.get("error"))[:400] if payload.get("error") else None,
        duration_s=duration,
        timed_out=timed_out,
        stderr_tail=stderr[-800:].decode("utf-8", "replace"),
        payload=payload,
    )


__all__ = ["PiSessionResult", "pi_child_env", "run_pi_prompt"]
