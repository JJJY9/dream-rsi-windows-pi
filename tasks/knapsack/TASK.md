# Task: 0/1 knapsack solver (`candidate.py`)

**This is this project's own small test domain. It is NOT the paper's §4 benchmark.**

## Interface (must keep working)

```python
def solve(weights: list[int], values: list[int], capacity: int) -> list[int]
```

Return the **0-based indices** of the items to pack into the knapsack.

## Rules the grader enforces

- Every returned index must be an `int` in `range(len(weights))`.
- No duplicate indices.
- The sum of `weights[i]` over the returned indices must be `<= capacity`.
- Any violation scores **0** for that instance (it is not an exception, just a zero).
- Returning extra attributes, helper functions, or new imports is fine. Keep the
  module importable and keep `solve` importable from it.

## How you are scored

The grader is a **fixed external script** that you cannot see and cannot modify.
It generates a family of instances from a seed that is not present in your
workspace, computes each instance's exact optimum with dynamic programming, and
scores you as

```
score = mean over instances of (your_total_value / optimal_total_value)
```

so the range is `0.0 .. 1.0`, and `1.0` means you matched the exact optimum on
every instance. Hard-coding answers is not viable: you do not have the instances.

## Notes

- Instances are small (tens of items, capacity a few hundred), so an exact
  algorithm is entirely affordable. Runtime per instance should stay well under
  a few seconds.
- Doing nothing about values will score poorly. Think about what actually
  maximises total value under a weight budget.
- Standard library only. Do not add third-party dependencies.
