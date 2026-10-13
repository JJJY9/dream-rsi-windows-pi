#!/usr/bin/env python3
"""FIXED evaluator harness for the circle-packing task. Stdlib only.

    *** Follows the AlphaEvolve/OpenEvolve circle-packing benchmark metric;
        it is NOT a benchmark result claimed by this project. ***

Domain: place exactly ``N_CIRCLES`` circles in the unit square so that their
radii sum is as large as possible without overlaps. The candidate exposes

    construct() -> list[tuple[float, float, float]]

The reported ``score`` is the sum of the 26 radii (after full validation,
mirroring the OpenEvolve evaluator's tolerances: 1e-6 slack on the square
boundary and on pairwise distances) **divided by the published target
2.635983** — the run protocol scores in [0, 1], so the task normalises against
the benchmark's best known result (1.0 would match AlphaEvolve's n=26 SOTA).
The raw sum is kept in ``sum_radii``. Constraint violations score 0.

Usage:
    python -I evaluator_harness.py <candidate_dir> <result_json_path>

Writes a JSON object to ``<result_json_path>``; the result file is the only
channel that is read.
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

N_CIRCLES = 26
TOL = 1e-6  # matches the OpenEvolve benchmark tolerances
TARGET_VALUE = 2.635983  # published AlphaEvolve result for n=26


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


def _coerce_circles(raw: Any) -> tuple[list[tuple[float, float, float]] | None, str | None]:
    """Validate the raw return value; never raises. Returns (circles, error)."""
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        return None, "construct() must return an iterable of (x, y, r) triples"
    circles: list[tuple[float, float, float]] = []
    for item in raw:
        if isinstance(item, (str, bytes)) or not hasattr(item, "__iter__"):
            return None, f"circle {item!r} is not an (x, y, r) triple"
        triple = list(item)
        if len(triple) != 3:
            return None, f"circle {item!r} does not have exactly three values"
        values: list[float] = []
        for value in triple:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"value {value!r} is not a real number"
            number = float(value)
            if not math.isfinite(number):
                return None, f"value {value!r} is not finite"
            values.append(number)
        circles.append((values[0], values[1], values[2]))
    if len(circles) != N_CIRCLES:
        return None, f"expected exactly {N_CIRCLES} circles, got {len(circles)}"
    return circles, None


def validate_circles(circles: list[tuple[float, float, float]]) -> str | None:
    """Return a violation description, or None when the packing is valid."""
    for index, (x, y, r) in enumerate(circles):
        if r < 0:
            return f"circle {index} has negative radius {r}"
        if x - r < -TOL or x + r > 1.0 + TOL or y - r < -TOL or y + r > 1.0 + TOL:
            return (
                f"circle {index} at ({x}, {y}) with radius {r} is outside the unit square"
            )
    for i in range(len(circles)):
        xi, yi, ri = circles[i]
        for j in range(i + 1, len(circles)):
            xj, yj, rj = circles[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist < ri + rj - TOL:
                return (
                    f"circles {i} and {j} overlap: distance {dist} < r_i + r_j "
                    f"{ri + rj}"
                )
    return None


def run(candidate_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "domain": "circle-packing-26-0.1",
        "is_paper_benchmark": False,
        "n_circles": N_CIRCLES,
        "target_value": 2.635983,
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

    circles, problem = _coerce_circles(raw)
    if circles is None:
        return {
            **report,
            "status": "OK",
            "error_type": "CONSTRAINT_VIOLATION",
            "error": problem,
            "score": 0.0,
            "sum_radii": 0.0,
            "circles_digest": None,
        }

    violation = validate_circles(circles)
    if violation is not None:
        return {
            **report,
            "status": "OK",
            "error_type": "CONSTRAINT_VIOLATION",
            "error": violation,
            "score": 0.0,
            "sum_radii": 0.0,
            "circles_digest": None,
        }

    score = sum(r for _, _, r in circles)
    return {
        **report,
        "status": "OK",
        "error_type": None,
        "error": None,
        "score": score / TARGET_VALUE,
        "sum_radii": score,
        "circles_digest": hashlib.sha256(
            json.dumps(sorted(circles), separators=(",", ":")).encode("utf-8")
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
