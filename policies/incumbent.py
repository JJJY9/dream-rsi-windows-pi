"""Incumbent policy pi_1 for the strict-§3 loop.

This file is the **starting point** of the policy search and the incumbent that
every revision must beat. It is deliberately small, readable, and deterministic,
because the Policy Developer agent revises *this source* and must be able to
understand it.

Contract (the only interface a policy may rely on)
--------------------------------------------------
``select(observation, rho) -> list[str]``

``observation`` is a ``PrefixObservation`` carrying:
  * ``nodes``            - the already-revealed nodes, as ``PublicNode``
  * ``legal_actions``    - root plus the leaves of the revealed subtree
  * ``completed_rounds`` - rounds finished in this rollout
  * ``max_parallelism``  - the worker cap W
  * ``node_ids``         - ids of the revealed nodes, in creation order

``rho`` in ``[0, 1]`` is the budget-preference knob: small means "few nodes,
deep", large means "many nodes, wide".

Hard rules
----------
* Return only ids from ``observation.legal_actions``; returning an id the policy
  has not observed, a duplicate, or more than ``max_parallelism`` entries is
  rejected by the controller and fails the whole episode closed.
* Return ``[]`` to stop.
* The policy can never see hidden nodes, hidden scores, or the full world. Do
  not attempt to read them; they are not passed and the runtime is a separate
  process.
"""
from __future__ import annotations

from typing import Any


def select(observation: Any, rho: float) -> list[str]:
    legal = list(observation.legal_actions)
    if not legal:
        return []

    workers = max(1, int(observation.max_parallelism))
    rho_clamped = max(0.0, min(1.0, float(rho)))

    by_id = {node.node_id: node for node in observation.nodes}
    root_id = observation.node_ids[0] if observation.node_ids else None

    # Budget knob -> how many parents to expand this round.
    width = 1 + round(rho_clamped * (workers - 1))
    width = max(1, min(width, workers, len(legal)))

    def rank(node_id: str) -> tuple[float, int, int]:
        node = by_id.get(node_id)
        if node is None:
            return (0.0, 0, -1)
        # Exploit: highest observed score first. Tie-break on shallower depth
        # (more room left below) and then on newer creation index.
        return (float(node.score), -int(node.depth), int(node.creation_index))

    explored = [nid for nid in legal if nid != root_id]
    explored.sort(key=rank, reverse=True)
    chosen = explored[:width]

    # Root is a pure exploration slot: expanding it reveals the earliest
    # unrevealed root child, which is the only way the tree gets a new branch.
    # Spend a slot on it when the budget is wide, or when there is nothing else.
    wants_root = root_id is not None and root_id in legal and (rho_clamped >= 0.5 or not chosen)
    if wants_root and root_id not in chosen:
        if len(chosen) >= width:
            chosen = chosen[: width - 1]
        chosen.append(root_id)

    # Stable order so the replay digest is reproducible.
    chosen.sort(key=lambda nid: by_id[nid].creation_index if nid in by_id else -1)
    return chosen
