"""Policy execution.

Two runtimes, deliberately unequal:

``InProcessPolicySession``
    For **trusted, project-owned** policies only — the incumbent shipped in
    ``policies/`` and test doubles. It imports the module into the controller
    process, so it must never be pointed at agent-generated code.

``SubprocessPolicySession``
    The real runtime for every agent-generated policy. One worker process per
    replay episode, short-lived, with an explicit ``cwd``, a scrubbed
    environment, a per-call deadline, and a Job Object that reaps the tree.

Neither is a security sandbox. The subprocess still runs as the current user
and can read files and reach the network; ``SECURITY_AND_RECOVERY.md`` requires
that be stated rather than glossed over.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from ..domain.errors import PolicyRejected
from ..runtime.winjob import WindowsJob, terminate_tree
from .protocol import PrefixObservation

# A policy needs no credentials, no project metadata and no user profile.
_ENV_ALLOWLIST = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)


def policy_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class PolicySession(Protocol):
    def select(self, observation: PrefixObservation, rho: float) -> Any: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Trusted in-process execution
# --------------------------------------------------------------------------


def load_policy_module(path: Path) -> Any:
    """Import a policy module into THIS process.

    Only ever call this with project-owned source. Agent-generated policy code
    must go through :class:`SubprocessPolicyRunner`.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"dream_rsi_trusted_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PolicyRejected(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "select", None)):
        raise PolicyRejected(f"{path.name} does not define select(observation, rho)")
    return module


class InProcessPolicySession:
    """Trusted policies only. No isolation of any kind is claimed."""

    isolated = False

    def __init__(self, module: Any) -> None:
        self._module = module
        self._select = module.select
        self.calls = 0

    @classmethod
    def from_path(cls, path: Path) -> InProcessPolicySession:
        return cls(load_policy_module(path))

    def select(self, observation: PrefixObservation, rho: float) -> Any:
        self.calls += 1
        return self._select(observation, rho)

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# Contained subprocess execution (the real path for generated policies)
# --------------------------------------------------------------------------


class SubprocessPolicySession:
    """One worker process, one replay episode, K2 request/response round trips."""

    isolated = True

    def __init__(
        self,
        *,
        python_executable: str,
        policy_path: Path,
        cwd: Path,
        timeout_s: float,
        memory_limit_mb: int | None,
        stderr_path: Path,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.cwd = Path(cwd)
        self.timeout_s = float(timeout_s)
        self.cwd.mkdir(parents=True, exist_ok=True)
        self._job = WindowsJob(memory_limit_mb=memory_limit_mb, active_process_limit=8)
        self._stderr_file = open(stderr_path, "wb")  # noqa: SIM115 - closed in close()
        self.calls = 0
        self._request_id = 0
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._dead = False

        self._proc = subprocess.Popen(
            [python_executable, "-m", "dream_rsi.policy._policy_worker", str(self.policy_path)],
            cwd=str(self.cwd),
            env=policy_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            shell=False,
        )
        self._job.assign(self._proc)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for raw in iter(self._proc.stdout.readline, b""):
            try:
                self._responses.put(json.loads(raw.decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError):
                self._responses.put({"ok": False, "error_type": "ProtocolError",
                                     "error": "worker emitted a non-JSON line"})

    def _fail(self, error_type: str, message: str) -> PolicyRejected:
        self._kill()
        return PolicyRejected(f"{error_type}: {message}")

    def _kill(self) -> None:
        if self._dead:
            return
        self._dead = True
        terminate_tree(self._proc, self._job)

    def select(self, observation: PrefixObservation, rho: float) -> Any:
        if self._dead:
            raise PolicyRejected("policy worker is no longer running")
        self.calls += 1
        self._request_id += 1
        request_id = self._request_id
        payload = {
            "id": request_id,
            "observation": observation.to_dict(),
            "rho": float(rho),
        }
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise self._fail("PolicyWorkerGone", str(exc)) from exc

        end = time.monotonic() + self.timeout_s
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise self._fail(
                    "PolicyTimeout", f"select() exceeded {self.timeout_s}s"
                )
            try:
                response = self._responses.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise self._fail(
                        "PolicyWorkerExited",
                        f"worker process exited with code {self._proc.returncode}",
                    ) from None
                continue
            break

        if response.get("id") not in (request_id, None):
            raise self._fail("ProtocolError", "worker replied with a mismatched request id")
        if not response.get("ok"):
            raise PolicyRejected(
                f"{response.get('error_type', 'PolicyError')}: "
                f"{str(response.get('error'))[:400]}"
            )
        actions = response.get("actions")
        if response.get("was_set"):
            # Preserve the "a set is not a legal batch" rule across the boundary
            # (ADR-0003) instead of letting JSON quietly make it a list.
            return frozenset(actions or [])
        return actions

    def close(self) -> None:
        if not self._dead:
            try:
                if self._proc.poll() is None and self._proc.stdin is not None:
                    self._proc.stdin.write(b'{"op":"close"}\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=2)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._kill()
        self._job.close()
        with contextlib.suppress(OSError):
            self._stderr_file.close()

    def __enter__(self) -> SubprocessPolicySession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()



class SubprocessPolicyRunner:
    """Factory for contained policy sessions."""

    def __init__(
        self,
        *,
        workdir_root: Path,
        python_executable: str | None = None,
        timeout_s: float = 5.0,
        memory_limit_mb: int | None = 256,
    ) -> None:
        self.workdir_root = Path(workdir_root)
        self.python_executable = python_executable or sys.executable
        self.timeout_s = float(timeout_s)
        self.memory_limit_mb = memory_limit_mb

    def open(self, *, policy_path: Path, policy_version_id: str, session_key: str) -> SubprocessPolicySession:
        safe_key = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_key)
        cwd = self.workdir_root / policy_version_id / f"{safe_key}-{uuid.uuid4().hex[:8]}"
        cwd.mkdir(parents=True, exist_ok=True)
        return SubprocessPolicySession(
            python_executable=self.python_executable,
            policy_path=Path(policy_path).resolve(),
            cwd=cwd,
            timeout_s=self.timeout_s,
            memory_limit_mb=self.memory_limit_mb,
            stderr_path=cwd / "policy_stderr.log",
        )


__all__ = [
    "InProcessPolicySession",
    "PolicySession",
    "SubprocessPolicyRunner",
    "SubprocessPolicySession",
    "load_policy_module",
    "policy_env",
]
