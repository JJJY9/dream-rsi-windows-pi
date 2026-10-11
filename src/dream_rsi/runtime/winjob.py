"""Best-effort Windows Job Object containment.

**This is not a security boundary.** A Job Object can cap memory, cap the number
of active processes, and guarantee that the whole descendant tree dies with the
job — it cannot stop a process from reading arbitrary files or opening a
network connection. `docs/SECURITY.md` requires that this
residual risk be stated plainly rather than dressed up as a sandbox.

On non-Windows platforms every method degrades to a no-op so the module stays
importable for static analysis; the project targets native Windows only.
"""
from __future__ import annotations

import contextlib
import ctypes
import subprocess
import sys
from typing import Any

_IS_WINDOWS = sys.platform == "win32"

# --- Win32 constants -------------------------------------------------------
_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

if _IS_WINDOWS:  # pragma: no cover - exercised on Windows only
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),  # ULONG_PTR
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class WindowsJob:
    """Owns a Job Object handle and, optionally, an attached child process."""

    def __init__(
        self,
        *,
        memory_limit_mb: int | None = None,
        active_process_limit: int | None = None,
        kill_on_close: bool = True,
    ) -> None:
        self._handle: Any = None
        self._available = False
        self._last_error: str | None = None
        if not _IS_WINDOWS:
            self._last_error = "not windows"
            return

        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            self._last_error = f"CreateJobObjectW failed ({ctypes.get_last_error()})"
            return

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = 0
        if kill_on_close:
            flags |= _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_mb is not None:
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = int(memory_limit_mb) * 1024 * 1024
        if active_process_limit is not None:
            flags |= _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = int(active_process_limit)
        info.BasicLimitInformation.LimitFlags = flags

        ok = _kernel32.SetInformationJobObject(
            handle,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            _kernel32.CloseHandle(handle)
            self._last_error = f"SetInformationJobObject failed ({err})"
            return

        self._handle = handle
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def assign(self, process: subprocess.Popen[Any]) -> bool:
        """Attach a started process so its whole tree lives inside the job."""
        if not self._available:
            return False
        handle = getattr(process, "_handle", None)
        if handle is None:  # pragma: no cover - defensive
            return False
        return bool(_kernel32.AssignProcessToJobObject(self._handle, int(handle)))

    def terminate(self) -> None:
        """Kill every process in the job. Safe to call more than once."""
        if not self._available:
            return
        _kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        if self._available and self._handle:
            _kernel32.CloseHandle(self._handle)
        self._handle = None
        self._available = False

    def __enter__(self) -> WindowsJob:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def terminate_tree(process: subprocess.Popen[Any], job: WindowsJob | None = None) -> None:
    """Terminate ``process`` and, on Windows, its descendants.

    Prefers the Job Object (which reliably reaps grandchildren such as a shell
    the child spawned). Falls back to ``taskkill /T /F`` and finally ``kill``.
    """
    if job is not None and job.available:
        job.terminate()
    with contextlib.suppress(OSError):
        if process.poll() is None:
            process.kill()
    if _IS_WINDOWS:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )


def job_support_report() -> dict[str, Any]:
    """Non-secret summary for the environment/acceptance reports."""
    job = WindowsJob()
    report = {
        "platform": sys.platform,
        "available": job.available,
        "detail": job.last_error,
        "note": (
            "Job Object constrains process count/memory and reaps the process tree; "
            "it does NOT restrict filesystem or network access."
        ),
    }
    job.close()
    return report


__all__ = ["WindowsJob", "job_support_report", "terminate_tree"]
