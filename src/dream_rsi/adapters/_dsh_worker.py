#!/usr/bin/env python3
"""Child process that performs ONE DeepSeekHarness run and reports JSON.

Why a separate process: a live model call can hang, and the parent must be able
to enforce a real deadline and reap the whole process tree. Running the SDK
in-process would make the timeout advisory only.

Invoked as::

    python -m dream_rsi.adapters._dsh_worker <request.json>

The request file holds the prompt and paths; credentials are **not** put in the
file — the child inherits only `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` from its
parent's scrubbed environment. The result is written to `result_path` as JSON.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    # Imported here so the module can be imported for static analysis even when
    # the optional SDK extra is not installed.
    from deepseek_harness import DeepSeekHarness

    prompt = str(request["prompt"])
    session_id = str(request["session_id"])
    with DeepSeekHarness(
        profile=str(request.get("profile", "sdk-minimal")),
        provider=str(request["provider"]),
        model=str(request["model"]),
        cwd=str(request["cwd"]),
        dsh_home=str(request["dsh_home"]),
        max_tokens=int(request.get("max_tokens", 8192)),
    ) as harness:
        result = harness.run(prompt, session_id=session_id)

    final_response = getattr(result, "final_response", None)
    session = getattr(result, "session_id", None)
    return {
        "ok": isinstance(final_response, str) and bool(final_response.strip()),
        "final_response": final_response if isinstance(final_response, str) else None,
        "session_id": session if isinstance(session, str) else session_id,
        "error_type": None if isinstance(final_response, str) and final_response.strip()
        else "EMPTY_FINAL_RESPONSE",
        "error": None,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _dsh_worker.py <request.json>", file=sys.stderr)
        return 2
    request_path = Path(argv[1]).resolve()
    result_path = Path(str(request_path) + ".result.json")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = run_request(request)
    except Exception as exc:
        payload = {
            "ok": False,
            "final_response": None,
            "session_id": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
            "traceback": traceback.format_exc(limit=4)[-1500:],
        }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
