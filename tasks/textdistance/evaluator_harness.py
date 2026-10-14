#!/usr/bin/env python3
"""FIXED evaluator harness for the textdistance speed task. Stdlib only.

    *** This grades a real third-party project (life4/textdistance, MIT). ***

Domain: speed up the `Levenshtein` implementation in
``textdistance/algorithms/edit_based.py`` while keeping the module's behavior
exactly. The workspace is a full copy of the upstream repository at a pinned
commit; the candidate may edit **only** the entry file.

Gate and score:

* **Integrity** — every file that shipped in the initial workspace (except the
  entry file itself) must still match its frozen sha256; new scratch files are
  allowed. A changed framework file scores 0 (a one-file optimization task).
* **Behavior** — the entry module's classes must agree with the frozen
  reference implementation (a copy of the same pinned commit) on a
  deterministic corpus: exact equality for integer distances, 1e-12 for
  similarity floats. Any mismatch scores 0.
* **Speed** — a fixed workload (240 deterministic string pairs, lengths 40-64)
  is timed against the frozen reference in the same process. Score is
  ``min(1, speedup / TARGET_SPEEDUP)`` with TARGET_SPEEDUP = 20: the known
  bit-parallel (Myers) rewrite measures ~26x on this machine, so 1.0 means a
  genuinely algorithmic improvement; the shipped code scores ~0.05.

Usage:
    python -I evaluator_harness.py <workspace_dir> <result_json_path>
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reactor"
MANIFEST_PATH = REFERENCE_DIR / "manifest.json"

ENTRY_REL = "textdistance/algorithms/edit_based.py"
TARGET_SPEEDUP = 20.0


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


def _corpus(n: int, lo: int, hi: int, seed: int) -> list[tuple[str, str]]:
    g = _lcg(seed)
    letters = "abcd"
    pairs = []
    for _ in range(n):
        l1 = lo + next(g) % (hi - lo + 1)
        l2 = lo + next(g) % (hi - lo + 1)
        a = "".join(letters[next(g) % 4] for _ in range(l1))
        b = "".join(letters[next(g) % 4] for _ in range(l2))
        pairs.append((a, b))
    return pairs


GATE_PAIRS = _corpus(320, 0, 96, seed=20261014)
EDGE_PAIRS = [
    ("", ""), ("", "abc"), ("abc", ""), ("a" * 70, "a" * 70),
    ("ab" * 40, "ba" * 40), ("kitten", "sitting"),
    ("中文测试字符串", "中文字符串测试"), ("🙂🙂🙂", "🙂🙂"),
]
LIGHT_PAIRS = _corpus(24, 0, 18, seed=777)
WORKLOAD = _corpus(240, 40, 64, seed=5150)


class CandidateBroken(Exception):
    """The candidate cannot be loaded or crashed; this is an evaluator ERROR."""


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
            return (
                f"file {rel!r} changed: only {ENTRY_REL} may be edited"
            )
    # Interpreter caches are not part of the contract: a workspace that
    # regenerated __pycache__ while the agent tested its work is fine, and the
    # manifest never lists such files.
    return None


def _load_modules(workspace: Path) -> tuple[Any, Any]:
    """Load the workspace package and the frozen reference as distinct packages.

    Both packages are purged from ``sys.modules`` first: production runs one
    evaluation per fresh interpreter, but test harnesses may grade several
    workspaces in one process, and a cached module would silently grade the
    wrong tree.
    """
    for name in list(sys.modules):
        if name.split(".")[0] in ("textdistance", "textdistance_ref"):
            del sys.modules[name]
    importlib.invalidate_caches()
    sys.path.insert(0, str(REFERENCE_DIR))
    sys.path.insert(0, str(workspace))
    ref = importlib.import_module("textdistance_ref")
    cand = importlib.import_module("textdistance")
    return ref, cand


def _behavior_gate(ref: Any, cand: Any) -> str | None:
    checks = [
        ("Levenshtein", GATE_PAIRS + EDGE_PAIRS, True),
        ("Hamming", LIGHT_PAIRS, True),
        ("DamerauLevenshtein", LIGHT_PAIRS, True),
        ("Jaro", LIGHT_PAIRS, False),
        ("JaroWinkler", LIGHT_PAIRS, False),
        ("StrCmp95", LIGHT_PAIRS, False),
    ]
    for name, pairs, exact in checks:
        ref_algo = getattr(ref, name)(external=False)
        cand_algo = getattr(cand, name)(external=False)
        for a, b in pairs:
            want = ref_algo(a, b)
            got = cand_algo(a, b)
            if exact:
                ok = got == want
            else:
                ok = isinstance(got, (int, float)) and math.isclose(got, want, abs_tol=1e-12)
            if not ok:
                return f"{name}({a[:20]!r}, {b[:20]!r}) = {got!r}, reference says {want!r}"
    return None


def _time_candidate(cand: Any) -> tuple[float, float]:
    ref = importlib.import_module("textdistance_ref")
    ref_algo = ref.Levenshtein(external=False)
    cand_algo = cand.Levenshtein(external=False)

    started = time.perf_counter()
    for a, b in WORKLOAD:
        ref_algo(a, b)
    t_ref = time.perf_counter() - started

    best = float("inf")
    reps = 0
    total = 0.0
    while reps < 50:
        started = time.perf_counter()
        for a, b in WORKLOAD:
            cand_algo(a, b)
        elapsed = time.perf_counter() - started
        best = min(best, elapsed)
        total += elapsed
        reps += 1
        if total >= 0.25 and reps >= 5:
            break
    return t_ref, best


def run(workspace: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "domain": "textdistance-speed-0.1",
        "is_paper_benchmark": False,
        "upstream": "life4/textdistance@d6a68d61088a40eef5c88191ccf79323dbf34850",
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

    try:
        ref, cand = _load_modules(workspace)
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": str(exc)[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}

    try:
        mismatch = _behavior_gate(ref, cand)
    except Exception as exc:
        return {**report, "status": "ERROR", "error_type": type(exc).__name__,
                "error": f"behavior gate crashed: {exc}"[:500], "score": 0.0,
                "traceback": traceback.format_exc(limit=3)[-1500:]}
    if mismatch is not None:
        return {**report, "status": "OK", "error_type": "CONSTRAINT_VIOLATION",
                "error": f"behavior differs from the reference: {mismatch}"[:500],
                "score": 0.0, "speedup": 0.0}

    try:
        t_ref, t_cand = _time_candidate(cand)
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
