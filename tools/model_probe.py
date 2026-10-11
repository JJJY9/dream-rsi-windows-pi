#!/usr/bin/env python3
"""Real, optional Windows DSH/Pi model smoke. Does not verify Dream-RSI implementation.

Only prints fixed status tokens. Caller injects model config via process env.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path


def dsh_probe(root: Path) -> None:
    from deepseek_harness import DeepSeekHarness

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("missing DEEPSEEK_API_KEY")
    model = os.environ.get("DSH_MODEL", "deepseek-v4-flash")
    provider = os.environ.get("DSH_PROVIDER", "deepseek-official")
    folder = root / ".runtime" / "model-check" / ("dsh-" + uuid.uuid4().hex)
    work, home = folder / "work", folder / "home"
    work.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    with DeepSeekHarness(
        provider=provider,
        model=model,
        profile="sdk-minimal",
        cwd=str(work),
        dsh_home=str(home),
        max_tokens=1024,
    ) as harness:
        result = harness.run(
            "Reply with the exact token DREAM_RSI_CONNECT_OK and no other text.",
            session_id="model-check-" + uuid.uuid4().hex,
        )
    response = result.final_response
    if not isinstance(response, str) or "DREAM_RSI_CONNECT_OK" not in response:
        raise RuntimeError("DSH returned an empty/unexpected final response")
    print("LIVE_MODEL_OK DSH nonempty final_response")


def pi_probe(root: Path) -> None:
    pi = shutil.which("pi")
    if not pi:
        raise RuntimeError("pi executable unavailable")
    folder = root / ".runtime" / "model-check" / ("pi-" + uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=True)
    args = [pi, "--mode", "rpc", "--no-session"]
    if os.environ.get("PI_PROVIDER"):
        args += ["--provider", os.environ["PI_PROVIDER"]]
    if os.environ.get("PI_MODEL"):
        args += ["--model", os.environ["PI_MODEL"]]
    proc = subprocess.Popen(
        args,
        cwd=folder,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        shell=False,
    )
    output: queue.Queue[dict] = queue.Queue()
    errors: list[str] = []
    completed_responses: list[str] = []
    seen_agent_error = False
    settled_aborted = False
    # Provider-supplied error text. It is carried into the exception so that
    # main() can classify it (401 -> AUTHENTICATION_FAILED and so on); main()
    # prints only the derived category, so nothing here reaches the terminal.
    provider_errors: list[str] = []

    def read_stdout() -> None:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.readline()
            if not raw:
                break
            try:
                output.put(json.loads(raw.decode("utf-8").rstrip("\r\n")))
            except (UnicodeError, json.JSONDecodeError):
                errors.append("invalid JSONL record")

    def drain_stderr() -> None:
        assert proc.stderr is not None
        # Discard stderr to prevent leaking provider messages and to avoid pipe deadlock.
        while proc.stderr.read(4096):
            pass

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=drain_stderr, daemon=True).start()

    def send(command: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()

    def note_provider_error(record: dict) -> None:
        """Keep the provider's own wording so main() can classify it."""
        for source in (record, record.get("message") or {}):
            for key in ("errorMessage", "error", "message"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    provider_errors.append(value.strip())
                    return

    def inspect_event(record: dict) -> None:
        nonlocal seen_agent_error, settled_aborted
        if record.get("type") in ("agent_error", "extension_error", "error"):
            seen_agent_error = True
            note_provider_error(record)
        if record.get("type") == "message_end":
            m = record.get("message") or {}
            if m.get("role") == "assistant":
                if m.get("stopReason") in ("error", "aborted"):
                    seen_agent_error = True
                    note_provider_error(record)
                content = m.get("content", [])
                if isinstance(content, str):
                    completed_responses.append(content)
                elif isinstance(content, list):
                    completed_responses.extend(
                        str(x.get("text", ""))
                        for x in content
                        if isinstance(x, dict) and x.get("type") == "text"
                    )
        if record.get("type") == "agent_settled" and record.get("aborted"):
            settled_aborted = True

    def recv_until(test, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                record = output.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if proc.poll() is not None:
                    # The queue being empty is the normal wait path, not the
                    # cause; `from None` keeps the traceback about the real fault.
                    raise RuntimeError("Pi process ended before completion") from None
                continue
            inspect_event(record)
            if test(record):
                return record
        raise RuntimeError("Pi RPC wait timed out")

    try:
        send({"id": "state-1", "type": "get_state"})
        handshake = recv_until(
            lambda m: m.get("type") == "response" and m.get("id") == "state-1", 15
        )
        if not handshake.get("success"):
            raise RuntimeError("Pi RPC get_state failed")
        send({"id": "prompt-1", "type": "prompt", "message": "Reply only DREAM_RSI_CONNECT_OK."})
        ack = recv_until(lambda m: m.get("type") == "response" and m.get("id") == "prompt-1", 20)
        if not ack.get("success"):
            raise RuntimeError("Pi prompt rejected")
        if ack.get("data", {}).get("disposition") == "handled":
            raise RuntimeError("Pi prompt handled without model run")
        # Waiting for agent_settled *is* the assertion: a prompt ack means the
        # request was accepted, not that the model finished. The returned event
        # carries nothing further that this probe checks.
        recv_until(lambda m: m.get("type") == "agent_settled", 105)
        if errors or seen_agent_error or settled_aborted:
            # Including the provider's text lets main() report e.g.
            # AUTHENTICATION_FAILED instead of a generic provider error.
            detail = provider_errors[0] if provider_errors else ""
            raise RuntimeError(
                "Pi RPC event stream reported a provider or agent error"
                + (f": {detail}" if detail else "")
            )
        if not any("DREAM_RSI_CONNECT_OK" in x for x in completed_responses):
            raise RuntimeError("Pi did not return expected assistant text")
        send({"id": "state-2", "type": "get_state"})
        after = recv_until(lambda m: m.get("type") == "response" and m.get("id") == "state-2", 10)
        if not after.get("success"):
            raise RuntimeError("Pi final state query failed")
        print(
            "LIVE_MODEL_OK Pi RPC prompt completed, nonempty expected assistant text, agent_settled"
        )
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("dsh", "pi"), required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    try:
        (dsh_probe if args.backend == "dsh" else pi_probe)(args.root.resolve())
        return 0
    except Exception as exc:
        # Print only a coarse category; provider errors may contain secret URLs or tokens.
        message = str(exc).lower()
        category = "UNKNOWN_API_OR_RUNTIME_ERROR"
        for needle, code in (
            ("401", "AUTHENTICATION_FAILED"),
            ("403", "AUTHORIZATION_FAILED"),
            ("unauthorized", "AUTHENTICATION_FAILED"),
            ("api key", "AUTHENTICATION_FAILED"),
            ("not found", "MODEL_OR_ENDPOINT_NOT_FOUND"),
            ("404", "MODEL_OR_ENDPOINT_NOT_FOUND"),
            ("timeout", "TIMEOUT"),
            ("timed out", "TIMEOUT"),
            ("connect", "CONNECTION_FAILED"),
            ("429", "RATE_LIMITED"),
            ("unexpected", "UNEXPECTED_RESPONSE"),
            ("empty", "EMPTY_RESPONSE"),
            ("missing", "MISSING_CONFIGURATION"),
            ("error", "PROVIDER_OR_AGENT_ERROR"),
        ):
            if needle in message:
                category = code
                break
        print("BLOCKED_MODEL_CONFIGURATION: " + category)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
