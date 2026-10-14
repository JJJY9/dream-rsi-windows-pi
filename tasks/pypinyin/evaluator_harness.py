#!/usr/bin/env python3
"""FIXED evaluator harness for the pypinyin speed task. Stdlib only.

    *** This grades a real third-party project (mozillazg/pypinyin, MIT). ***

Domain: speed up the per-character conversion pipeline in
``pypinyin/converter.py`` while keeping the library's behavior exactly
(``pinyin()`` / ``lazy_pinyin()`` outputs across 7 styles must equal the
frozen reference on the gate corpus).

Gate and score:

* **Integrity** — every file that shipped in the initial workspace (except
  ``pypinyin/converter.py``) must match its frozen sha256. New scratch files
  are allowed; editing the framework scores 0.
* **Behavior** — 84 frozen jobs (12 texts x 7 styles, incl. multi-pronunciation
  phrases, mixed-script and empty inputs) must reproduce
  ``reactor/expected.json`` exactly.
* **Speed** — a ~100k-character workload (``pinyin()`` + ``lazy_pinyin()``) is
  timed; the reference time comes from a subprocess running the frozen
  package (``reactor/ref_runner.py``). Score = ``min(1, speedup / 2.0)``:
  the fixture variants measure ~1.4x (copy-tax fix) and ~2.3x (style
  memoisation), so 1.0 means a genuinely restructured hot path.

Usage:
    python -I evaluator_harness.py <workspace_dir> <result_json_path>
"""
from __future__ import annotations

import hashlib
import importlib
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
REF_RUNNER = REACTOR / "ref_runner.py"

ENTRY_REL = "pypinyin/converter.py"
TARGET_SPEEDUP = 2.0


def _load_corpus() -> Any:
    sys.path.insert(0, str(REACTOR))
    import corpus

    return corpus


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


def _behavior_gate(workspace: Path, pkg: Any, corpus: Any) -> str | None:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    for ti, text in enumerate(corpus.GATE_TEXTS):
        for style in corpus.GATE_STYLES:
            key = f"{ti}:{style}"
            if style == "default":
                got = pkg.pinyin(text)
            else:
                got = pkg.pinyin(text, style=getattr(pkg.Style, style))
            if got != expected[key]:
                return (
                    f"job {key} (text {text[:14]!r}) returned {got!r}, "
                    f"reference says {expected[key]!r}"
                )
    return None


def _reference_seconds() -> float:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, str(REF_RUNNER)],
        capture_output=True,
        timeout=300,
        env=env,
    )
    line = proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1]
    return float(json.loads(line)["ref_seconds"])


def _time_candidate(pkg: Any, corpus: Any) -> float:
    texts = corpus.workload_texts()
    best = float("inf")
    total = 0.0
    reps = 0
    while reps < 40:
        started = time.perf_counter()
        pkg.pinyin(texts[0])
        pkg.lazy_pinyin(texts[1])
        elapsed = time.perf_counter() - started
        best = min(best, elapsed)
        total += elapsed
        reps += 1
        if total >= 0.5 and reps >= 5:
            break
    return best


def run(workspace: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "domain": "pypinyin-speed-0.1",
        "is_paper_benchmark": False,
        "upstream": "mozillazg/pypinyin@0.55.0 (sdist, pinned)",
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
    # A fresh process per evaluation in production, but test harnesses may
    # grade several workspaces in one process — a cached package would grade
    # the wrong tree. Purge before importing the candidate.
    for name in list(sys.modules):
        if name == "pypinyin" or name.startswith("pypinyin."):
            del sys.modules[name]
    importlib.invalidate_caches()
    sys.path.insert(0, str(workspace))
    try:
        import pypinyin as pkg
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": str(exc)[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}

    try:
        mismatch = _behavior_gate(workspace, pkg, corpus)
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": f"behavior gate crashed: {exc}"[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}
    if mismatch is not None:
        return {**report, "status": "OK", "error_type": "CONSTRAINT_VIOLATION",
                "error": f"behavior differs from the reference: {mismatch}"[:500],
                "score": 0.0, "speedup": 0.0}

    try:
        t_ref = _reference_seconds()
        t_cand = _time_candidate(pkg, corpus)
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
