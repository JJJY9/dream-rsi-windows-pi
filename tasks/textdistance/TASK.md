# Task: speed up `textdistance`'s Levenshtein — a real third-party project

**The workspace is a full copy of [life4/textdistance](https://github.com/life4/textdistance)
at commit `d6a68d6108` (MIT license, vendored under `tasks/textdistance/initial/`).**
This task asks for a genuine performance improvement to a real community
library — the kind of contribution a maintainer would merge.

## What you may edit

**Exactly one file:**

```
textdistance/algorithms/edit_based.py
```

Everything else in the repository (the rest of the package, the tests, the
metadata) is read-only reference. The grader verifies the hashes of every other
file; touching any of them scores 0. New scratch files you create for your own
experiments are allowed and ignored.

## What must stay true

- The module's **public behavior, exactly**: every class in the file
  (`Hamming`, `Levenshtein`, `DamerauLevenshtein`, `Jaro`, `JaroWinkler`,
  `StrCmp95`, …) must return identical results to the shipped implementation on
  a large deterministic corpus (exact equality for distances, 1e-12 for
  similarity floats). A behavior mismatch scores 0.
- Keep the constructor signatures (the grader calls `Levenshtein(external=False)`).
- **Standard library only** — no new dependencies in the module. (The upstream
  file optionally imports numpy; the grader's environment may or may not have
  it, and its current use in the hot loop is not helping. Pure-Python wins here.)
- The module must import cleanly when the grader loads the package.

## How you are scored

The grader times a fixed workload — 240 deterministic string pairs of length
40–64 — against a **frozen copy of the original implementation**, in the same
process:

```
speedup = time(original) / time(yours)
score   = min(1.0, speedup / 20)
```

The full test suite of the repository (`tests/`) is available in the workspace
as reference for expected behavior, but the grader uses its own frozen corpus
— editing tests cannot change anything.

Points of reference measured on the authoring machine: the shipped
implementation scores ~0.05; a tight row-wise DP rewrite ~1.5x (0.07); the
**Myers bit-parallel algorithm ~26x (1.0)**. The cell `textdistance/algorithms/edit_based.py`
docstring itself links the classic reference for this.

## Notes

- `Levenshtein.__call__` trims common prefix/suffix and short-circuits equal
  strings before entering `_cycled`; every remaining case goes through the
  quadratic inner loop — that loop is the hot path.
- The workload strings are short (≤ 64 symbols): bit-parallel techniques fit in
  one machine word, but behavior on longer inputs must still match the
  reference (the gate includes pairs up to length 96).
- **Keep the file importable at every save** — the call can end at any moment,
  and whatever the file contains then is what gets graded.
