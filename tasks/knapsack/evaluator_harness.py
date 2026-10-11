#!/usr/bin/env python3
"""FIXED evaluator harness for this project's *test domain*. Stdlib only.

    *** THIS IS NOT THE PAPER'S §4 BENCHMARK. ***

The paper's §4 task suite is explicitly out of scope. This is a small,
deterministic, self-contained grading script that gives the Dream-RSI loop a
real optimisation signal to climb: a discovery agent may rewrite
``candidate.py``, and a score is computed by *measuring the result*, never by
trusting anything the agent says about itself.

Domain: 0/1 knapsack. The candidate must expose

    solve(weights: list[int], values: list[int], capacity: int) -> list[int]

returning the 0-based indices of the chosen items.

Score is the mean ratio of achieved value to the exact optimum (computed here
by dynamic programming), in ``[0, 1]``. Infeasible answers score 0 for that
instance. The instance family is generated from a fixed seed that is **not**
exposed to the agent's workspace, so hard-coding answers is not a viable
strategy.

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
import sys
import traceback
from pathlib import Path
from typing import Any

# --- fixed instance family -------------------------------------------------

SEED = 20261010
N_INSTANCES = 12
MAX_CALL_SECONDS = 5.0  # advisory; the caller enforces the hard timeout


class _Deterministic:
    """A tiny LCG. Deliberately version-independent, unlike `random`."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0x7FFFFFFF

    def next_below(self, bound: int) -> int:
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return self._state % bound


def build_instances() -> list[tuple[list[int], list[int], int]]:
    rng = _Deterministic(SEED)
    instances: list[tuple[list[int], list[int], int]] = []
    for index in range(N_INSTANCES):
        size = 16 + (index % 7)  # 16..22 items
        weights = [rng.next_below(50) + 1 for _ in range(size)]
        values = [rng.next_below(60) + 1 for _ in range(size)]
        capacity = int(sum(weights) * (0.35 + 0.03 * (index % 5)))
        instances.append((weights, values, capacity))
    return instances


def optimal_value(weights: list[int], values: list[int], capacity: int) -> int:
    """Exact 0/1 knapsack optimum by DP."""
    best = [0] * (capacity + 1)
    for weight, value in zip(weights, values, strict=True):
        for cap in range(capacity, weight - 1, -1):
            candidate = best[cap - weight] + value
            if candidate > best[cap]:
                best[cap] = candidate
    return best[capacity]


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
    if not callable(getattr(module, "solve", None)):
        raise AttributeError("candidate.py must define a callable `solve(...)`")
    return module


def grade_solution(
    chosen: Any, weights: list[int], values: list[int], capacity: int
) -> tuple[bool, int, str | None]:
    """Return (feasible, achieved_value, error). Never raises."""
    try:
        if isinstance(chosen, (str, bytes)) or not hasattr(chosen, "__iter__"):
            return False, 0, "solve() must return an iterable of item indices"
        indices: list[int] = []
        for item in chosen:
            if isinstance(item, bool) or not isinstance(item, int):
                return False, 0, f"non-integer index {item!r} in the returned solution"
            indices.append(item)
        if len(set(indices)) != len(indices):
            return False, 0, "duplicate item indices in the returned solution"
        total_weight = 0
        total_value = 0
        for idx in indices:
            if idx < 0 or idx >= len(weights):
                return False, 0, f"index {idx} out of range"
            total_weight += weights[idx]
            total_value += values[idx]
        if total_weight > capacity:
            return False, 0, f"over capacity: {total_weight} > {capacity}"
        return True, total_value, None
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"


def run(candidate_dir: Path) -> dict[str, Any]:
    instances = build_instances()
    optima = [optimal_value(w, v, c) for w, v, c in instances]
    report: dict[str, Any] = {
        "domain": "knapsack-0.1-fixed",
        "is_paper_benchmark": False,
        "n_instances": len(instances),
        "instances_digest": hashlib.sha256(
            json.dumps(instances, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
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
            "per_instance": [],
            "traceback": traceback.format_exc(limit=3)[-1500:],
        }

    per_instance: list[dict[str, Any]] = []
    total_ratio = 0.0
    solved = 0
    for index, ((weights, values, capacity), optimum) in enumerate(
        zip(instances, optima, strict=True)
    ):
        try:
            chosen = module.solve(list(weights), list(values), capacity)
            feasible, achieved, error = grade_solution(chosen, weights, values, capacity)
        except Exception as exc:
            feasible, achieved, error = False, 0, f"{type(exc).__name__}: {exc}"
        ratio = (achieved / optimum) if (feasible and optimum > 0) else 0.0
        total_ratio += ratio
        solved += 1 if feasible else 0
        per_instance.append({
            "index": index,
            "n_items": len(weights),
            "capacity": capacity,
            "optimum": optimum,
            "achieved": achieved,
            "feasible": feasible,
            "ratio": ratio,
            "error": error,
        })

    score = total_ratio / len(instances) if instances else 0.0
    return {
        **report,
        "status": "OK",
        "error_type": None,
        "error": None,
        "score": score,
        "optimal_count": sum(1 for row in per_instance if row["ratio"] >= 1.0),
        "feasible_count": solved,
        "per_instance": per_instance,
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
