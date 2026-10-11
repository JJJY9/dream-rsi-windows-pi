"""Fixed-command evaluator: run a pinned grading script against a candidate.

Design notes
------------
* The grading script is loaded from a project-owned, agent-inaccessible path —
  never from the candidate workspace. A discovery agent that edits its own copy
  of the grader changes nothing.
* The child runs with ``-I`` (isolated) and a **scrubbed environment**: only a
  small allowlist survives, and anything whose name looks like a credential is
  dropped even if it would otherwise be allowed.
* A hang is a hard failure, not a hang: the child gets a deadline and its whole
  process tree is reaped via a Windows Job Object.
* Failure to grade yields ``EvaluatorStatus.ERROR`` with a task-protocol score
  mapping (default 0.0). A genuine, measured zero yields ``FAILED``. The two are
  never conflated.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..domain.models import EvaluatorStatus
from ..runtime.winjob import WindowsJob, terminate_tree
from .protocol import EvaluationResult

# Kept deliberately small. Everything else is dropped before the child starts.
_ENV_ALLOWLIST = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
    "LANG", "LC_ALL", "PYTHONIOENCODING",
)
# Belt and braces: never forward these even if someone adds them to the allowlist.
_SECRET_MARKERS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
    "SESSION", "COOKIE", "ANTHROPIC", "DEEPSEEK", "OPENAI", "PI_", "DSH_",
)


def scrubbed_env() -> dict[str, str]:
    """A minimal child environment with every credential-shaped var removed."""
    env: dict[str, str] = {}
    for name in _ENV_ALLOWLIST:
        upper = name.upper()
        if any(marker in upper for marker in _SECRET_MARKERS):
            continue
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    # Keep candidate workspaces (and therefore their digests) clean.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FixedCommandEvaluator:
    """Runs ``harness_path`` as a subprocess and reads back a JSON result file."""

    def __init__(
        self,
        *,
        harness_path: Path,
        workdir_root: Path,
        timeout_s: float = 120.0,
        python_executable: str | None = None,
        memory_limit_mb: int | None = 2048,
    ) -> None:
        self.harness_path = Path(harness_path).resolve()
        self.workdir_root = Path(workdir_root).resolve()
        self.timeout_s = float(timeout_s)
        self.python_executable = python_executable or sys.executable
        self.memory_limit_mb = memory_limit_mb
        if not self.harness_path.is_file():
            raise FileNotFoundError(f"evaluator harness not found: {self.harness_path}")

    @property
    def spec_digest(self) -> str:
        """Pins the exact grader: its bytes plus the interpreter and deadline."""
        material = hashlib.sha256(self.harness_path.read_bytes()).hexdigest()
        return _digest({
            "harness_sha256": material,
            "harness_name": self.harness_path.name,
            "timeout_s": self.timeout_s,
            "kind": "fixed_command",
        })

    def evaluate(self, candidate_dir: Path, *, invocation_id: str) -> EvaluationResult:
        candidate_dir = Path(candidate_dir).resolve()
        if not candidate_dir.is_dir():
            raise FileNotFoundError(f"candidate directory not found: {candidate_dir}")

        workdir = self.workdir_root / invocation_id
        workdir.mkdir(parents=True, exist_ok=True)
        result_path = workdir / "evaluator_result.json"
        if result_path.exists():
            result_path.unlink()

        argv: Sequence[str] = [
            self.python_executable, "-I", str(self.harness_path),
            str(candidate_dir), str(result_path),
        ]
        started = time.monotonic()
        timed_out = False
        proc: subprocess.Popen[bytes] | None = None

        with WindowsJob(memory_limit_mb=self.memory_limit_mb, active_process_limit=8) as job:
            proc = subprocess.Popen(
                list(argv),
                cwd=str(workdir),
                env=scrubbed_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            job.assign(proc)
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_tree(proc, job)
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except (subprocess.TimeoutExpired, ValueError, OSError):
                    stdout, stderr = b"", b""
            duration = time.monotonic() - started

        returncode = proc.returncode if proc is not None else None
        base_detail: dict[str, Any] = {
            "returncode": returncode,
            "timed_out": timed_out,
            "duration_s": round(duration, 4),
            "harness": self.harness_path.name,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_tail": stdout[-2000:].decode("utf-8", "replace"),
            "stderr_tail": stderr[-2000:].decode("utf-8", "replace"),
        }

        if timed_out:
            payload = {"status": "ERROR", "error_type": "TIMEOUT",
                       "error": f"evaluator exceeded {self.timeout_s}s", "score": 0.0}
            return EvaluationResult(
                status=EvaluatorStatus.ERROR, score=0.0, duration_s=duration,
                result_digest=_digest(payload), detail={**base_detail, **payload},
                error_type="TIMEOUT",
            )

        if not result_path.is_file():
            payload = {
                "status": "ERROR", "error_type": "NO_RESULT_FILE",
                "error": f"harness exited {returncode} without writing a result",
                "score": 0.0,
            }
            return EvaluationResult(
                status=EvaluatorStatus.ERROR, score=0.0, duration_s=duration,
                result_digest=_digest(payload), detail={**base_detail, **payload},
                error_type="NO_RESULT_FILE",
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            bad = {"status": "ERROR", "error_type": "UNREADABLE_RESULT",
                   "error": str(exc)[:300], "score": 0.0}
            return EvaluationResult(
                status=EvaluatorStatus.ERROR, score=0.0, duration_s=duration,
                result_digest=_digest(bad), detail={**base_detail, **bad},
                error_type="UNREADABLE_RESULT",
            )

        raw_score = payload.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        if score != score or score in (float("inf"), float("-inf")):  # NaN / inf guard
            score = 0.0
        score = max(0.0, min(1.0, score))

        harness_status = str(payload.get("status", "OK")).upper()
        if harness_status == "ERROR":
            status = EvaluatorStatus.ERROR
            error_type = str(payload.get("error_type") or "EVALUATOR_ERROR")
        elif score > 0.0:
            status = EvaluatorStatus.SUCCESS
            error_type = None
        else:
            # The grader ran and genuinely measured zero. That is FAILED, not ERROR.
            status = EvaluatorStatus.FAILED
            error_type = str(payload.get("error_type") or "ZERO_SCORE")

        return EvaluationResult(
            status=status,
            score=score,
            duration_s=duration,
            result_digest=_digest(payload),
            detail={**base_detail, "harness_payload": payload},
            error_type=error_type,
        )


__all__ = ["FixedCommandEvaluator", "scrubbed_env"]
