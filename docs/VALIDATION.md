# Validation scope

The independent local audit completed on 2026-10-11 accepted the **Pi-only core §3 mechanism** on native Windows. Its checked run had three sealed scoreable worlds, two contested policy selections, seven successful Pi discovery calls, and four successful model calls writing policy revisions. Recorded replay episodes made zero discovery-agent and evaluator calls. An independent arithmetic path recomputed recorded and held-out Eq. (1) values to within `1e-12`.

These facts support the implementation claim: online discovery, fixed evaluation, sequential policy revision, historical replay, Eq. (1) scoring, selection, and deployment all executed. The demonstration task was small and saturated. The audit therefore did **not** establish useful self-improvement across a broad task distribution or reproduce the paper's §4 benchmark results. DSH live operation and numerical coverage targets were not verified.

The raw live-run archive is deliberately absent from this public snapshot because it contains local paths, model transcripts, and development records. The local audit statement above is therefore not independently reproducible from the files on GitHub alone. The offline tests and fixture run below are reproducible from this source.

To check the public source without a model account, create the virtual environment as shown in the README, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . offline-test --outer-rounds 3
```

The offline fixture is deterministic and never constitutes a live-model claim. A fresh live result requires a working Pi installation, explicit provider and model selection, and inspection of the recorded run state and event log. A run in which every Pi attempt in a world fails is blocked even if an interrupted attempt left a file that the evaluator can score.

The earlier MOF screening comparison is not included in this public snapshot. It was exploratory, used a small worker setting, and ended early in the Dream-RSI arm. A revised MOF study will need a separately frozen protocol and new results before any efficacy claim.
