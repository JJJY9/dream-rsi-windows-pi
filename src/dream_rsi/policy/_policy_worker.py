#!/usr/bin/env python3
"""Isolated policy worker: loads ONE policy file and answers ``select`` calls.

Protocol — LF-delimited JSON, one object per line.

    ->  {"id": 1, "observation": {...}, "rho": 0.5}
    <-  {"id": 1, "ok": true,  "actions": ["root"], "was_set": false}
    <-  {"id": 1, "ok": false, "error_type": "PolicyRaised", "error": "..."}
    ->  {"id": 2, "op": "close"}

Process hygiene
---------------
* The policy module is loaded from an explicit path; it is **never** imported
  into the controller process.
* File descriptor 1 is duplicated for the protocol and then re-pointed at
  stderr, so a policy that calls ``print()`` cannot corrupt the message stream.
* The hidden world, database, and credentials are not passed in the worker's
  protocol input. The Windows process still has the user's filesystem rights.

This is a containment measure, not a sandbox. The worker still runs with the
current user's rights and can read files and open sockets; see
``docs/SECURITY.md``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from typing import Any

# Reserve the real stdout for the protocol; everything else goes to stderr.
_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
_PROTOCOL = os.fdopen(_PROTOCOL_FD, "w", encoding="utf-8", newline="\n")


def _send(payload: dict[str, Any]) -> None:
    _PROTOCOL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def _load_policy(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("dream_rsi_candidate_policy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dream_rsi_candidate_policy"] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "select", None)):
        raise AttributeError("policy module must define a callable `select(observation, rho)`")
    return module


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        _send({"id": None, "ok": False, "error_type": "USAGE",
               "error": "usage: _policy_worker.py <policy_path>"})
        return 2

    policy_path = os.path.abspath(argv[1])
    directory = os.path.dirname(policy_path)
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)

    try:
        module = _load_policy(policy_path)
    except Exception as exc:
        _send({
            "id": None,
            "ok": False,
            "error_type": f"PolicyImportFailed:{type(exc).__name__}",
            "error": str(exc)[:800],
            "traceback": traceback.format_exc(limit=4)[-1200:],
        })
        return 3

    from dream_rsi.policy.protocol import PrefixObservation

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _send({"id": None, "ok": False, "error_type": "BadRequest",
                   "error": str(exc)[:300]})
            continue
        if request.get("op") == "close":
            break

        request_id = request.get("id")
        try:
            observation = PrefixObservation.from_dict(request["observation"])
            result = module.select(observation, float(request["rho"]))
        except Exception as exc:
            _send({
                "id": request_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:800],
                "traceback": traceback.format_exc(limit=4)[-1200:],
            })
            continue

        was_set = isinstance(result, (set, frozenset))
        if was_set:
            items: Any = sorted(str(x) for x in result)
        elif isinstance(result, (list, tuple)):
            items = [x if isinstance(x, str) else repr(x) for x in result]
        elif result is None:
            _send({"id": request_id, "ok": False, "error_type": "ReturnedNone",
                   "error": "select() returned None; return a list of node ids"})
            continue
        else:
            _send({"id": request_id, "ok": False,
                   "error_type": f"BadReturnType:{type(result).__name__}",
                   "error": "select() must return a list/tuple of node ids"})
            continue
        _send({"id": request_id, "ok": True, "actions": items, "was_set": was_set})

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
