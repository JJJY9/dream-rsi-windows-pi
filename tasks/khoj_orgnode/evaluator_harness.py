#!/usr/bin/env python3
"""FIXED evaluator harness for the khoj orgnode speed task. Stdlib only.

    *** This grades a real third-party project (khoj-ai/khoj, AGPL-3.0). ***

Domain: speed up the org-mode parser in
``processor/content/org_mode/orgnode.py`` while reproducing the shipped
parser's node output exactly on the frozen gate corpus.

Gate and score:

* **Integrity** - every file that shipped in the initial workspace (except the
  entry file) must match its frozen sha256. New scratch files are allowed;
  editing the framework scores 0.
* **Behavior** - 15 frozen jobs (14 documents plus a re-parse of doc 3 after
  all the others, so a candidate that wrongly memoises across calls fails) must
  reproduce ``reactor/expected.json`` exactly, compared through the node's
  public API (level/heading/body/tags/todo/priority/scheduled/deadline/closed/
  properties/logbook/ancestors).
* **Speed** - a ~65k-line workload is parsed under one shared protocol
  (``corpus.time_makelist``: >=4 repetitions, >=0.4s total, best kept). Both
  sides are timed by ``reactor/timer_runner.py`` in fresh subprocesses of
  their own - the frozen reference (``reactor/orgnode_ref.py``) and the
  candidate entry file - so neither a bloated caller process nor shared module
  state can tilt the comparison. Score = ``min(1, speedup / 2.5)``: the fixture
  variants measure ~1.6x (precompiled patterns), ~2.0x (+ substring gates) and
  ~2.6-3.0x (routing plain lines past the drawer chain, hoisted filename
  normalisation, clock memoisation), so 1.0 means a genuinely restructured
  hot path.

Usage:
    python -I evaluator_harness.py <workspace_dir> <result_json_path>
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REACTOR = HERE / "reactor"
MANIFEST_PATH = REACTOR / "manifest.json"
EXPECTED_PATH = REACTOR / "expected.json"
REF_MODULE = REACTOR / "orgnode_ref.py"
TIMER_RUNNER = REACTOR / "timer_runner.py"

ENTRY_REL = "processor/content/org_mode/orgnode.py"
TARGET_SPEEDUP = 2.5

#: Candidate modules are loaded under a fresh name per grade: harnesses may
#: grade several workspaces in one process (the test suite does), and a cached
#: module would make the second grade re-score the first tree.
_GRADE_SEQ = itertools.count()


def _load_corpus() -> Any:
    # Loaded by path under a task-specific name rather than ``import corpus``:
    # in one process (the test suite grades several tasks' workspaces in a
    # single interpreter) sys.modules is shared, and a bare import would
    # collide with another task's module of the same name.
    spec = importlib.util.spec_from_file_location("khoj_orgnode_corpus", REACTOR / "corpus.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_integrity(workspace: Path) -> str | None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for rel, expected in manifest.items():
        if rel == ENTRY_REL:
            continue
        target = workspace / rel
        if not target.is_file():
            return f"required file {rel!r} is missing from the workspace"
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            return f"file {rel!r} changed: only {ENTRY_REL} may be edited"
    return None


def _load_candidate(workspace: Path) -> Any:
    name = f"orgnode_candidate_{next(_GRADE_SEQ)}_{abs(hash(str(workspace))) % 100000}"
    spec = importlib.util.spec_from_file_location(name, workspace / ENTRY_REL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_difference(got: list[dict], want: list[dict]) -> str:
    if len(got) != len(want):
        return f"node count {len(got)} != reference {len(want)}"
    for index, (g, w) in enumerate(zip(got, want)):
        if g != w:
            for field in w:
                if g.get(field) != w[field]:
                    return (
                        f"node {index} field {field!r}: "
                        f"{str(g.get(field))[:60]!r} != reference {str(w[field])[:60]!r}"
                    )
    return "contents differ"


def _behavior_gate(mod: Any, corpus: Any) -> str | None:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    for key, doc in corpus.gate_jobs():
        got = corpus.dump(mod.makelist(doc, corpus.GATE_FILENAME))
        if got != expected[key]:
            return f"job {key}: {_first_difference(got, expected[key])}"
    return None


def _seconds_for(module_path: Path) -> float:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, str(TIMER_RUNNER), str(module_path)],
        capture_output=True,
        timeout=600,
        env=env,
    )
    line = proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1]
    return float(json.loads(line)["seconds"])


def run(workspace: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "domain": "khoj-orgnode-speed-0.1",
        "is_paper_benchmark": False,
        "upstream": "khoj-ai/khoj@ae229ca (AGPL-3.0)",
        "entry": ENTRY_REL,
        "target_speedup": TARGET_SPEEDUP,
    }
    entry = workspace / ENTRY_REL
    if entry.is_file():
        report["candidate_sha256"] = hashlib.sha256(entry.read_bytes()).hexdigest()

    problem = _check_integrity(workspace)
    if problem is not None:
        return {**report, "status": "OK", "error_type": "CONSTRAINT_VIOLATION",
                "error": problem, "score": 0.0, "speedup": 0.0}

    corpus = _load_corpus()
    try:
        mod = _load_candidate(workspace)
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": str(exc)[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}

    try:
        mismatch = _behavior_gate(mod, corpus)
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": f"behavior gate crashed: {exc}"[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}
    if mismatch is not None:
        return {**report, "status": "OK", "error_type": "CONSTRAINT_VIOLATION",
                "error": f"behavior differs from the reference: {mismatch}"[:500],
                "score": 0.0, "speedup": 0.0}

    try:
        t_ref = _seconds_for(REF_MODULE)
        t_cand = _seconds_for(workspace / ENTRY_REL)
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": f"timing crashed: {exc}"[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}

    speedup = t_ref / t_cand if t_cand > 0 else float("inf")
    score = min(1.0, speedup / TARGET_SPEEDUP)
    return {
        **report,
        "status": "OK",
        "error_type": None,
        "error": None,
        "score": round(score, 9),
        "speedup": round(speedup, 6),
        "reference_seconds": round(t_ref, 6),
        "candidate_seconds": round(t_cand, 6),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: evaluator_harness.py <workspace_dir> <result_json_path>", file=sys.stderr)
        return 2
    workspace = Path(argv[1]).resolve()
    result_path = Path(argv[2]).resolve()
    try:
        payload = run(workspace)
    except Exception as exc:
        payload = {"status": "ERROR", "error_type": type(exc).__name__,
                   "error": str(exc)[:500], "score": 0.0, "is_paper_benchmark": False}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
