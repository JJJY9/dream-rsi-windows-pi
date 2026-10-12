# Algorithm and notation

This document describes the implemented core of [Dream-RSI v2 §3](https://arxiv.org/abs/2609.14858v2). It also separates the paper's mechanism from this repository's task and platform choices. The independent arithmetic oracle is in `reference/oracle.py`; production scoring is in `src/dream_rsi/score/equation1.py`.

## Objects and parameters

| Symbol | Meaning |
|---|---|
| `t` | Outer online-discovery and policy-improvement iteration. |
| `T_i` | A sealed discovery tree, or historical world, produced online at iteration `i`. |
| `H_t` | The sealed historical worlds available when selecting the next policy. |
| `π_t` | The policy deployed at iteration `t`. A policy selects a batch of parent nodes. |
| `ρ` | Budget-preference input to a policy. Replay uses the fixed grid `{0, 0.25, 0.5, 0.75, 1}`. |
| `W` | Maximum number of parent nodes selected in one batch. |
| `K₁`, `K₂` | Maximum sequential rounds in online discovery and historical replay, respectively. |
| `M` | Number of policy versions considered at a selection step, including the incumbent. |
| `λ` | Weight of the parallelism penalty in Eq. (1). |

The values of `W`, `K₁`, `K₂`, `M`, `λ`, and the live `ρ` are experiment configuration. They are not universal constants supplied by the paper.

## Online discovery

The policy sees a **prefix** of the tree: the root and nodes already revealed by earlier completed rounds. Its legal actions are the root and the leaves of that prefix. It may return at most `W` distinct legal parent IDs, or an empty list to stop. For each selected parent, the discovery agent works from that parent's candidate workspace. The independent evaluator executes and scores the resulting candidate. The new node, score, error state, and workspace identity are recorded before the next policy decision.

Online discovery creates new evidence. A world is sealed before it is used for replay. The task's fixed evaluator supplies scores; model-written text is never accepted as a score.

Each attempt receives the complete root-to-parent observation chain and an index of **all** previously completed worlds. The index exposes their recorded public observations, generation records, and candidate workspaces for tool-based reading. These are discovery inputs, separate from the tree under construction. Pi can start a new session because the framework reconstructs the inherited context explicitly. The bounded event log is a diagnostic aid; a separate semantic-message log preserves complete finalized messages and tool results.

Evaluator `detail` is archived privately. Only explicitly declared `public_detail`, measured score/status, and error type enter discovery and policy observations. Task evaluators should populate `public_detail` with useful diagnostic feedback without revealing private test inputs. The context directory is a read-only input convention, not a filesystem sandbox; the framework retains its authoritative hash-verified copy in CAS.

After the last world is sealed, a final offline selection is performed and stored as `pi_N.py` and `final_policy_id`. Thus N normally completed, scoreable iterations have N selections. The last online world records the policy that actually generated it, which can differ from the final policy. No scoreable history produces an explicit skipped-final-selection event; a call-budget cut remains recorded even when an incumbent-only contest can complete.

## Historical replay

Replay begins with only the root of a sealed world visible. At each round, the policy receives the current visible prefix and selects legal parent nodes. The replay controller reveals the corresponding next children from the immutable world. The policy never receives the hidden full tree. Selecting several parents in one batch uses the same pre-round prefix for every selection.

Replay stops on an empty batch, `K₂` completed rounds, or a fully revealed world. It reads recorded scores and performs **no discovery-agent or evaluator calls**. A nonempty batch counts as a round even if it reveals no new child.

Online and replay both load a fresh isolated policy process for every decision, so module state cannot persist across rounds or rho values. This costs process launches but preserves identical decision semantics. Replay records each visible observation, selected batch, and newly revealed public node results, including empty stopping decisions. The verifier reconstructs these traces without calling a model, evaluator, or policy. Policy-development feedback includes these readable trajectories from the just-evaluated version.

## Artifact versions and resume

The SQLite DTO envelope remains version 1 with backward-readable additions. New nodes use `observation_format=2`, new replay episodes use `trajectory_format=2`, and new worlds use the v2 digest header. CAS snapshots start with `TREE2`, store all referenced file bytes, preserve directories, normalize internal links, reject special files, and restore only into new/empty targets. Restoration validates all file blobs and paths first; creating links on Windows requires the usual OS permissions.

Old world digests remain readable under the original header. Legacy observation records and hash-only workspace manifests cannot be promoted to complete context. The configuration digest now pins the v2 context implementation, so old runs cannot silently resume under changed semantics. `verify_run.py` reports legacy/incomplete contexts and decision traces as failures. A crash after the final seal can resume the final contest; completed contests are reused. Interrupted model revisions with uncertain outcomes fail closed rather than repeat potentially paid calls.

## Eq. (1): score on one world

For a sealed world `T_i`, let `B_i = |T_i| − 1` be its number of non-root nodes and `S_i` the best non-root score. The equation requires `B_i > 0` and `S_i > 0`; an unscoreable world is reported rather than silently repaired.

For one replay at `ρ`, let `n_ρ` be the number of non-root nodes revealed, `s_ρ` the best score among them (zero if none), and `k*_ρ` the number of completed nonempty replay rounds. Its normalized quality is

```text
a_ρ = clip(s_ρ / S_i, 0, 1).
```

Reduce the five points `(n_ρ, a_ρ)` to distinct nondominated points `(n̄_j, ā_j)`, ordered by increasing `n̄_j`, with `ā_0 = 0`. Lower `n` and higher `a` are preferred. The hypervolume and penalty are

```text
HV_i = (1 / B_i) Σ_j (ā_j − ā_{j−1})(B_i − n̄_j)
P_i  = (1 / |R|) Σ_{ρ∈R, n_ρ>0} (k*_ρ / n_ρ)
V_i  = HV_i − λ P_i,
```

where `R` is the five-element `ρ` grid. A zero-reveal replay contributes zero to the penalty, but the denominator remains `|R| = 5`. The point `(0, 0)` participates in dominance and contributes no positive hypervolume area.

For candidate policy version `m`, the selection value is the arithmetic mean of `V_i^m` over the worlds in `H_t`. The incumbent is evaluated first. Each model-written revision is produced from the **just-evaluated** preceding version and its replay traces and scores, then evaluated before the next revision is requested. The highest mean value is selected; a tie retains the incumbent.

The selection guarantee is only relative to this fixed historical set and this internal equation. It does not imply improved candidate quality in a new online world. On a small or saturated world, Eq. (1) can reward stopping early; that behavior must be reported rather than hidden by changing the equation after observing a result.

## Repository-specific choices

Python, SQLite, Windows Job Objects, Pi RPC, the bundled knapsack evaluator, time limits, and provider configuration are implementation choices. The DSH adapter has not been validated with a live provider in the current acceptance scope. No claim is made about the paper's §4 benchmarks.
