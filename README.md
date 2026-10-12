# Dream-RSI §3: an independent Windows/Pi reproduction

**Language:** English | [简体中文](README.zh-CN.md)

This repository independently implements the core loop in [Dream-RSI v2 §3](https://arxiv.org/abs/2609.14858v2) on native Windows. It is not the authors' official code. The earlier revision was exercised with real Pi discovery and model-written policies. The v2 context and consistency changes have separate [acceptance evidence](docs/V2_ACCEPTANCE.md); that earlier live audit does not validate new code. The DSH adapter is included but has not passed a live acceptance run. The paper's §4 benchmarks are not reproduced here.

The repository currently publishes the **algorithm and a small knapsack demonstration task**. A separate MOF screening experiment is being redesigned with a larger worker setting. Its previous runs are exploratory and are not evidence of improved MOF discovery, so no MOF result is claimed in this release.

## How the loop works

1. A discovery agent edits a candidate program. An independent evaluator scores it and adds a node to a discovery tree.
2. The current policy and model-written revisions are replayed over sealed historical worlds. Replay reuses recorded nodes and scores; it makes **zero** discovery-agent and evaluator calls.
3. Each policy is evaluated at the paper's five `ρ` settings using Eq. (1). The policy with the highest mean historical value is deployed for the next online world.

The Eq. (1) value measures the historical trade-off between discovered quality and search cost. It is an **internal policy-selection score**, not the candidate's task score and not a guarantee of better performance on unseen worlds. See [Algorithm](docs/ALGORITHM.md) for the definitions and paper-to-code mapping.

## Quick start

Supported development environment: Windows 11 x64, native PowerShell, Python 3.11 or newer. Offline tests need no model account.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . offline-test --outer-rounds 3
.\.venv\Scripts\python.exe -m pytest -q
```

`offline-test` exercises the loop with a deterministic fixture and reports `OFFLINE_PASS` when successful. It is not a live-model result.

For a live run, install and authenticate [Pi](https://github.com/earendil-works/pi) and configure a provider and model available to your account. These commands make real model requests and can incur costs:

```powershell
.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . model-check pi `
    --provider "YOUR_PROVIDER" --model "YOUR_MODEL"

.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . run `
    --backend pi --developer llm `
    --provider "YOUR_PROVIDER" --model "YOUR_MODEL" `
    --outer-rounds 3
```

The bundled knapsack task is selected by default. `--developer llm` makes policy revisions with a model; `--developer scripted` is a diagnostic fixture and must not be reported as a fully live run. Run artifacts are written under `.runtime/`, which is excluded from Git.

## Evidence and limits

The [validation summary](docs/VALIDATION.md) describes what was independently checked on the Pi path and what remains unverified. The implementation preserves the §3 root/leaf action set, prefix-only replay, sequential policy revision, and Eq. (1). Its existing live demonstration used a small, saturated task; it does not establish useful self-improvement on a broad task distribution.

Model-generated programs run with the user's Windows permissions. Process limits and workspace conventions do not provide a filesystem or network security boundary. See [Security and execution](docs/SECURITY.md) before running an untrusted model. Data and software sources are listed in [References](docs/REFERENCES.md).

## Repository contents

| Path | Purpose |
|---|---|
| `src/dream_rsi/` | §3 controller, tree, replay, Eq. (1), adapters, and CLI |
| `tasks/knapsack/` | Small fixed-evaluator demonstration task |
| `policies/` and `reference/` | Starting policy and independent mathematical oracle |
| `tests/` | Offline and regression tests |
| `docs/` | Public algorithm, validation, security, and source notes |

This public snapshot starts with a clean Git history. Internal development instructions, superseded acceptance reports, raw model transcripts, local runtime data, and the unfinished MOF experiment are not part of the public release.
