#!/usr/bin/env python3
"""FIXED evaluator harness for this project's *test domain*. Stdlib only.

    *** THIS IS NOT THE PAPER'S §4 BENCHMARK. ***

The paper's §4 task suite is explicitly out of scope. This is a small,
deterministic, self-contained grading script that gives the Dream-RSI loop a
graded optimisation signal to climb: a discovery agent may rewrite
``candidate.py``, and a score is computed by *measuring the result*, never by
trusting anything the agent says about itself.

Domain: spread exactly ``N_POINTS`` points in the unit square so that the
**minimum pairwise distance** is as large as possible. The candidate must
expose

    construct() -> list[tuple[float, float]]

Score is the minimum pairwise euclidean distance over all point pairs (a pure
geometric measurement of the returned points), in ``[0, ~0.3]``. A uniform
6x5 lattice scores exactly 0.2. Constraint violations score 0 for that
candidate.

The optimum is not needed and not computed: the score is a direct, honest
reading of the artifact, and any improvement on it is an improvement in fact.

Usage:
    python -I evaluator_harness.py <candidate_dir> <result_json_path>

Writes a JSON object to ``<result_json_path>`` and prints nothing meaningful on
stdout (candidate code may print whatever it likes; the result file is the only
channel that is read).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

# --- fixed problem definition ----------------------------------------------

N_POINTS = 30


# --- candidate loading and grading -----------------------------------------


def load_candidate(candidate_dir: Path) -> Any:
    module_path = candidate_dir / "candidate.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"candidate.py not found in {candidate_dir}")
    spec = importlib.util.spec_from_file_location("dream_rsi_candidate", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not build an import spec for candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "construct", None)):
        raise AttributeError("candidate.py must define a callable `construct()`")
    return module


def _coerce_points(raw: Any) -> tuple[list[tuple[float, float]] | None, str | None]:
    """Validate the raw return value; never raises. Returns (points, error)."""
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        return None, "construct() must return an iterable of (x, y) pairs"
    points: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (str, bytes)) or not hasattr(item, "__iter__"):
            return None, f"point {item!r} is not an (x, y) pair"
        pair = list(item)
        if len(pair) != 2:
            return None, f"point {item!r} does not have exactly two coordinates"
        values: list[float] = []
        for value in pair:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"coordinate {value!r} is not a real number"
            number = float(value)
            if not math.isfinite(number):
                return None, f"coordinate {value!r} is not finite"
            if not 0.0 <= number <= 1.0:
                return None, f"coordinate {value!r} is outside the unit square [0, 1]"
            values.append(number)
        points.append((values[0], values[1]))
    if len(points) != N_POINTS:
        return None, f"expected exactly {N_POINTS} points, got {len(points)}"
    return points, None


def min_pairwise_distance(points: list[tuple[float, float]]) -> float:
    best = math.inf
    for i in range(len(points)):
        xi, yi = points[i]
        for j in range(i + 1, len(points)):
            xj, yj = points[j]
            distance = math.hypot(xi - xj, yi - yj)
            if distance < best:
                best = distance
    return best


def run(candidate_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "domain": "packing-0.1-fixed",
        "is_paper_benchmark": False,
        "n_points": N_POINTS,
    }

    candidate_file = candidate_dir / "candidate.py"
    if candidate_file.is_file():
        report["candidate_sha256"] = hashlib.sha256(candidate_file.read_bytes()).hexdigest()

    try:
        module = load_candidate(candidate_dir)
    except Exception as exc:
        return {
            **report,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "score": 0.0,
            "traceback": traceback.format_exc(limit=3)[-1500:],
        }

    try:
        raw = module.construct()
    except Exception as exc:
        return {
            **report,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "score": 0.0,
            "traceback": traceback.format_exc(limit=3)[-1500:],
        }

    points, problem = _coerce_points(raw)
    if points is None:
        # A violated a constraint is a measured zero, not a grader error: the
        # candidate ran and its output was graded, and it scored 0.
        return {
            **report,
            "status": "OK",
            "error_type": "CONSTRAINT_VIOLATION",
            "error": problem,
            "score": 0.0,
            "min_distance": 0.0,
            "points_digest": None,
        }

    score = min_pairwise_distance(points)
    return {
        **report,
        "status": "OK",
        "error_type": None,
        "error": None,
        "score": score,
        "min_distance": score,
        "points_digest": hashlib.sha256(
            json.dumps(sorted(points), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: evaluator_harness.py <candidate_dir> <result_json_path>", file=sys.stderr)
        return 2
    candidate_dir = Path(argv[1]).resolve()
    result_path = Path(argv[2]).resolve()
    try:
        payload = run(candidate_dir)
    except Exception as exc:
        payload = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "score": 0.0,
            "is_paper_benchmark": False,
        }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
