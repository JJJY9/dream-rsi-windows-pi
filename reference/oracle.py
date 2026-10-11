"""Small, auditable strict-§3 REFERENCE ORACLE, not a production runtime.

Pure Python, standard library only. Its job is to pin down:
(1) root+leaf action sets and historical replay revelation; (2) Eq(1) math.
No real agent, no evaluator, no persistence, no OS sandbox. NEVER call this a
complete Dream-RSI implementation. Agent-generated policies are NOT safe to run
in this process. Only trusted test policies may be passed to replay().
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable

RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Node:
    id: str
    parent: str | None
    created: int
    score: float
    # Sensitive unseen fields should NEVER be included in PrefixNode.
    observation: str = ""


@dataclass(frozen=True)
class PrefixNode:
    id: str
    parent: str | None
    created: int
    score: float
    observation: str


@dataclass(frozen=True)
class Prefix:
    nodes: tuple[PrefixNode, ...]
    legal: tuple[str, ...]
    rounds: int
    workers: int


@dataclass(frozen=True)
class ReplayResult:
    rho: float
    revealed: tuple[str, ...]
    actions: tuple[tuple[str, ...], ...]
    kstar: int
    n: int
    best_score: float
    stop_reason: str


class InvalidWorld(ValueError):
    pass


class InvalidAction(ValueError):
    pass


class FullWorld:
    def __init__(self, nodes: Iterable[Node], root_id: str = "root") -> None:
        raw_nodes = list(nodes)
        self.nodes = {n.id: n for n in raw_nodes}
        if not self.nodes or root_id not in self.nodes:
            raise InvalidWorld("missing root")
        if len(self.nodes) != len(raw_nodes):
            raise InvalidWorld("duplicate id")
        self.root_id = root_id
        root = self.nodes[root_id]
        if root.parent is not None or root.created != 0 or root.score != 0:
            raise InvalidWorld("root invariants violated")
        if len({n.created for n in self.nodes.values()}) != len(self.nodes):
            raise InvalidWorld("duplicate creation index")
        if any(not isfinite(n.score) for n in self.nodes.values()):
            raise InvalidWorld("nonfinite score")
        if any(n.created < 0 for n in self.nodes.values()):
            raise InvalidWorld("negative creation index")
        self.children: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for n in self.nodes.values():
            if n.id == root_id:
                continue
            if n.parent not in self.nodes:
                raise InvalidWorld("parent not in world")
            assert n.parent is not None
            if self.nodes[n.parent].created >= n.created:
                raise InvalidWorld("parent not created before child")
            self.children[n.parent].append(n.id)
        for kids in self.children.values():
            kids.sort(key=lambda nid: self.nodes[nid].created)
        for parent, kids in self.children.items():
            if parent != root_id and len(kids) > 1:
                raise InvalidWorld("nonroot has multiple children")
        self.ordered = tuple(n.id for n in sorted(self.nodes.values(), key=lambda n: n.created))
        if tuple(self.nodes[nid].created for nid in self.ordered) != tuple(range(len(self.nodes))):
            raise InvalidWorld("creation indices must be contiguous from zero")
        seen = {root_id}
        todo = [root_id]
        while todo:
            p = todo.pop()
            for kid in self.children[p]:
                if kid in seen:
                    raise InvalidWorld("duplicate/cyclic reachability")
                seen.add(kid)
                todo.append(kid)
        if len(seen) != len(self.nodes):
            raise InvalidWorld("disconnected world")

    def prefix(self, shown: set[str], rounds: int, W: int) -> Prefix:
        # No hidden child count or hidden node metadata is leaked.
        selected = tuple(nid for nid in self.ordered if nid in shown)
        leaves = [nid for nid in selected if not any(c in shown for c in self.children[nid])]
        legal = tuple(dict.fromkeys((self.root_id, *leaves)))
        public = tuple(PrefixNode(
            self.nodes[nid].id, self.nodes[nid].parent,
            self.nodes[nid].created, self.nodes[nid].score,
            self.nodes[nid].observation,
        ) for nid in selected)
        return Prefix(public, legal, rounds, W)

    def next_child(self, selected_node: str, shown: set[str]) -> str | None:
        for child in self.children[selected_node]:
            if child not in shown:
                return child
        return None


def replay(
    world: FullWorld,
    policy: Callable[[Prefix, float], list[str] | tuple[str, ...]],
    rho: float,
    W: int,
    K2: int,
) -> ReplayResult:
    if W < 1 or K2 < 0:
        raise ValueError("W>=1, K2>=0 required")
    shown = {world.root_id}
    action_trace: list[tuple[str, ...]] = []
    stop_reason = "K2"
    rounds = 0
    while rounds < K2:
        if len(shown) == len(world.nodes):
            stop_reason = "FULLY_REVEALED"
            break
        prefix = world.prefix(shown, rounds, W)
        selection = policy(prefix, rho)
        if not isinstance(selection, (list, tuple)) or any(not isinstance(x, str) for x in selection):
            raise InvalidAction("policy must return a list/tuple of node IDs")
        batch = tuple(selection)
        if not batch:
            stop_reason = "STOP_EMPTY_BATCH"
            break
        if len(batch) != len(set(batch)):
            raise InvalidAction("duplicate node in a batch")
        if len(batch) > W:
            raise InvalidAction("exceeds max workers")
        if any(nid not in prefix.legal for nid in batch):
            raise InvalidAction("illegal or hidden node")
        # Simultaneous batch: decide each child against the SAME pre-round prefix.
        children = [world.next_child(nid, shown) for nid in batch]
        action_trace.append(batch)
        rounds += 1  # per paper, each NONEMPTY batch counts, including no progress
        shown.update(child for child in children if child is not None)
    else:
        if len(shown) == len(world.nodes):
            stop_reason = "FULLY_REVEALED"
    revealed = tuple(nid for nid in world.ordered if nid in shown)
    nonroot = revealed[1:]
    best = max((world.nodes[nid].score for nid in nonroot), default=0.0)
    return ReplayResult(rho, revealed, tuple(action_trace), rounds, len(nonroot), best, stop_reason)


def nondominated(points: Iterable[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    """Minimize n; maximize a; equality is not strict dominance."""
    data = sorted(set(points))
    for n, a in data:
        if not isinstance(n, int) or n < 0 or not isfinite(a):
            raise ValueError("invalid quality-cost point")
        if not 0.0 <= a <= 1.0:
            raise ValueError("quality must be clipped into [0,1]")
    return tuple((n, a) for n, a in data if not any(
        (nn <= n and aa >= a) and (nn < n or aa > a)
        for nn, aa in data
    ))


@dataclass(frozen=True)
class Eq1:
    hv: float
    penalty: float
    value: float
    frontier: tuple[tuple[int, float], ...]


def equation1(
    results: Iterable[ReplayResult],
    B: int,
    S: float,
    lambda_weight: float,
    expected_rhos: tuple[float, ...] = RHO_GRID,
) -> Eq1:
    samples = tuple(results)
    if B <= 0 or not isfinite(S) or S <= 0:
        raise ValueError("paper's Eq(1) requires B>0, S>0")
    if not isfinite(lambda_weight) or lambda_weight < 0:
        raise ValueError("invalid lambda")
    if len(samples) != len(expected_rhos) or sorted(r.rho for r in samples) != sorted(expected_rhos):
        raise ValueError("must provide exactly one sample per configured rho")
    for r in samples:
        if r.n < 0 or r.n > B or r.kstar < 0 or not isfinite(r.best_score):
            raise ValueError("invalid replay sample")
    points = ((r.n, max(0.0, min(1.0, r.best_score / S))) for r in samples)
    frontier = nondominated(points)
    hv = 0.0
    previous_a = 0.0
    for n, a in frontier:
        if a < previous_a:
            raise AssertionError("ordered Pareto quality must increase")
        hv += (a - previous_a) * (B - n) / B
        previous_a = a
    penalty = sum(r.kstar / r.n for r in samples if r.n > 0) / len(expected_rhos)
    return Eq1(hv, penalty, hv - lambda_weight * penalty, frontier)
