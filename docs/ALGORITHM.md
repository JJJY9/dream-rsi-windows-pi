# Algorithm and notation

This document describes the implemented core of [Dream-RSI §3](https://arxiv.org/abs/2609.14858). It also separates the paper's mechanism from this repository's task and platform choices. The independent arithmetic oracle is in `reference/oracle.py`; production scoring is in `src/dream_rsi/score/equation1.py`.

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

## Historical replay

Replay begins with only the root of a sealed world visible. At each round, the policy receives the current visible prefix and selects legal parent nodes. The replay controller reveals the corresponding next children from the immutable world. The policy never receives the hidden full tree. Selecting several parents in one batch uses the same pre-round prefix for every selection.

Replay stops on an empty batch, `K₂` completed rounds, or a fully revealed world. It reads recorded scores and performs **no discovery-agent or evaluator calls**. A nonempty batch counts as a round even if it reveals no new child.

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
