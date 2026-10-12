# Task: spread 30 points in the unit square (`candidate.py`)

**This is this project's own small test domain. It is NOT the paper's §4 benchmark.**

## Interface (must keep working)

```python
def construct() -> list[tuple[float, float]]
```

Return exactly **30** points `(x, y)`, each with `0.0 <= x <= 1.0` and
`0.0 <= y <= 1.0`.

## Rules the grader enforces

- Exactly 30 points; each must be a pair of finite numbers inside the unit
  square. Any violation scores **0** for that candidate.
- `construct()` runs inside the grader's interpreter. Standard library only
  (no third-party packages), and it must finish quickly — the whole evaluation
  of one candidate is wall-clock capped, so keep `construct()` well under a
  few seconds of computation.
- Prefer deterministic output. If you use randomness, seed it with a fixed
  constant so the same candidate always measures the same.
- Helper functions and extra module attributes are fine. Keep the module
  importable and `construct` importable from it.

## How you are scored

The grader is a **fixed external script** that you cannot see and cannot
modify. It calls `construct()` once, checks the constraints, and computes the
**minimum pairwise distance** between the 30 returned points:

```
score = min over all point pairs (i, j) of the euclidean distance between them
```

Higher is better. A uniform 6×5 lattice scores `0.2` exactly; random points
score far below that. The score is a pure geometric measurement of what you
return — a candidate may embed any construction you have worked out, because
the points themselves *are* the answer.

## Notes

- This is a classic packing problem, and good answers are not uniform grids:
  boundary points can spread further than interior intuition suggests, and
  hexagonal-style staggering beats square rows.
- Local improvement (jitter points, keep changes that raise the minimum pair
  distance, repeat) run *inside* `construct()` is a legitimate and effective
  approach — budget your iterations so the whole call stays fast.
- Doing nothing scores the lattice value; small geometric insights move the
  score meaningfully.
