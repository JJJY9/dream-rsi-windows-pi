# Dream-RSI v2 §3 implementation acceptance

Date: 2026-10-12. Scope: the mechanisms in arXiv:2609.14858v2 §3,
using the project's fixed knapsack example as a diagnostic task. The user's
scenario experiment and the paper's §4 benchmarks are outside this acceptance.

## Changes and evidence

| Gap in the initial audit | Implementation | Regression evidence |
| --- | --- | --- |
| G1: inherited branch observations | Root-to-parent public observations, generation records and restored workspaces accompany each attempt. | Diagnostic-only repair; sibling isolation; incomplete branch rejection. |
| G2: truncated historical context | Every completed world is exported with all public nodes and artifact references. | Seven-world history includes and restores the earliest world's artifact. |
| G3: different policy state semantics | Online and replay open a fresh contained policy process for every decision. | Stateful policy resets across decisions and rho ordering; replay external-call spies. |
| G4: incomplete persistence | CAS stores actual file bytes, TREE2 manifests, complete semantic Pi events, public observations and private evaluator records separately. | Restore after source deletion; corrupt blob and unsafe path rejection; long Pi tool-result preservation. |
| G5: missing terminal offline update | The last sealed world is followed by the final contest and deployment of `pi_N.py`. | Integration selection counts; interrupted final seal resumes once; committed resume checks prior deployed policies. |
| G6: unreadable replay feedback | Each decision records its visible prefix, selected batch and revealed public results; feedback includes the trace. | Trace reconstruction without policy/model/evaluator execution; tampered counters and missing decisions rejected. |

The existing v2 Eq. (1), fixed five-point rho grid, incumbent-inclusive M
versions and sequential M−1 revisions are retained. The arithmetic oracle and
existing action/replay regression tests remain part of acceptance.

## Local validation

Environment: native Windows 11 x64, Python 3.14.6, an editable installation in
`.venv`. Tests use temporary directories outside the repository, both to keep
Windows paths short and to exercise the existing isolated-working-directory
contract. There are no paid model calls in the regression suite.

Commands:

```powershell
python -m ruff check .
python -m mypy src/dream_rsi --no-incremental
.\.venv\Scripts\python.exe -m pytest -q --basetemp "$env:TEMP/drsi-v2-complete"
```

Ruff passes. Mypy passes for 39 source files. The final full suite completed:
**248 passed, 3 skipped in 550.73 seconds**. All three skips are the existing
symlink tests on this machine without link-creation privileges. A focused
50-test suite also passed after the policy worker lifecycle change.

Earlier focused checks passed: 229 tests with 3 skips outside the integration
module, and 40 tests covering v2 context, policy development and Pi tracing.
An integration failure exposed a missing check for tampered historical
deployments after final selection; the fix and the targeted regression passed.

## Local Pi acceptance

The user separately authorized the installed Pi agent to use its configured
`cc-switch-deep-seek / deepseek-v4-pro` provider/model. Pi runs in JSONL RPC mode;
the payload consists of the public knapsack example, generated code, public
measurements and policy replay feedback. Calls are real and consume model
account quota. They are separate from the offline regressions above.

The first run, `v2-pi-20261012`, was configured for two online rounds and two
outer iterations, W=1, K2=2, M=2 and a 240-second per-call ceiling. Two discovery calls succeeded
and produced a sealed scoreable world (B=2, S=1). Policy development was still
emitting thinking deltas when the parent timeout expired. The run correctly
became `BLOCKED_EXTERNAL`; it is **not** a passing closed-loop acceptance.

The follow-up run, `v2-pi-20261012-final`, uses one outer iteration and the
default 900-second ceiling, keeping the other parameters above. Its budget is
three discovery/development calls. It completed as **COMMITTED / live**:

- Two successful discovery calls and two fixed evaluator calls; B=2, S=1.
- One successful Pi policy-development call; its revision passed screening.
- Ten replay episodes, each reporting zero discovery-agent/evaluator calls.
- The terminal contest deployed `pi_1.py` and retained the incumbent:
  mean V=0.36 versus the revision's V=0.30. No improvement is claimed.
- Six discovery/development invocations across both runs in total; no syntax
  repair or further model call was needed.
- The separate read-only verifier passed **18/18 checks**, including actual
  context/artifact integrity, trace reconstruction, Eq. (1) recomputation,
  sequential feedback, terminal selection, deployment and event consistency.

Implementation commit: `f19a94247cbf68b3c51354ba773af468574f8705`.
The final world's digest is
`9cb6479ca6b6eed2b8e9088b857edfc2dc3ab857de4eff3417e61ad912da2396`;
its declared configuration digest is
`703af7417ec783ed60ba1c41b6505988126bdb082200aeffe3381a7cabb89c9f`.

```powershell
$env:PI_PROVIDER = "cc-switch-deep-seek"
$env:PI_MODEL = "deepseek-v4-pro"
.\.venv\Scripts\python.exe -m dream_rsi.cli --project-root . run `
  --backend pi --developer llm `
  --provider cc-switch-deep-seek --model deepseek-v4-pro `
  --outer-rounds 1 --k1 2 --k2 2 --workers 1 --versions 2 `
  --max-model-calls 3 --agent-timeout-s 900 --run-id v2-pi-20261012-final

.\.venv\Scripts\python.exe tools/verify_run.py --project-root . `
  --run-id v2-pi-20261012-final --expect-backend pi --expect-developer llm `
  --expect-provider cc-switch-deep-seek --expect-model deepseek-v4-pro `
  --expect-outer-rounds 1 --expect-k1 2 --expect-k2 2 --expect-workers 1 `
  --expect-versions 2 --expect-agent-timeout-s 900 `
  --json .runtime/v2-pi-verification.json
```

The provider/model environment is supplied only to this invocation to satisfy
the existing configuration-presence check. Credentials remain in Pi's own
local configuration and are not copied to the repository. Raw local runtime
records are excluded from the public PR; its live summary is therefore not
independently reproducible without a new model-account run. The seven-world
history and sibling-isolation assertions are reproducible local tests, not
claims inferred from this small live example.

## Boundaries

- Windows link-creation tests may be skipped when the OS denies symlink
  privileges. Ordinary files and empty-directory restoration are exercised.
- Legacy hash-only snapshots and incomplete observations are readable as old
  records, but cannot silently become complete v2 context or resume under the
  new configuration digest.
- Context files are a read-only convention, not an OS filesystem sandbox.
  Private evaluator results stay outside exported context; the framework
  validates its own authoritative CAS copies.
- DSH live operation and its full tool transcript remain unverified.
- Successful mechanism acceptance would not establish useful self-improvement
  or reproduce the paper's benchmark results. Raw local transcripts and model
  configuration are excluded from Git.
