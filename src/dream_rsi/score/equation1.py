"""Equation (1): the replay objective of Dream-RSI §3.4.

For one replay world ``i`` with ``B_i = |T_i| - 1 > 0`` nodes and best
non-root score ``S_i > 0``, and the fixed budget grid ``R`` (default
``RHO_GRID``):

* ``n_rho = |R_final| - 1``, ``a_rho = clip(s_best / S_i, 0, 1)``;
* the five ``(n_rho, a_rho)`` outcomes are reduced to their distinct
  non-dominated frontier (ascending ``n``, ``abar_0 = 0``):

  ``HV_i = (1 / B_i) * sum_j (abar_j - abar_{j-1}) * (B_i - nbar_j)``

* ``P_i = (1 / |R|) * sum_{rho in R, n_rho > 0} (kstar_rho / n_rho)`` — the
  denominator is the **whole** grid size, not the number of positive-``n``
  samples, so zero-call rhos contribute ``0`` while still counting;
* ``V_i = HV_i - lambda * P_i`` and ``V^m = (1/t) * sum_i V_i^m``.

Paper fidelity rules enforced here (never worked around):
``B > 0`` and finite ``S > 0`` are required — a world with ``S <= 0`` is not
scoreable and must be recorded and reported as such, never silently repaired
with ``max(1, S)``; ``n = 0`` contributes no positive area and no penalty but
still participates in Pareto dominance; quality is clipped only through
:func:`clip_quality`.

This is the production twin of ``reference/oracle.equation1`` and must stay
mathematically identical to it (verified by ``tests/test_score.py``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from dream_rsi.score.frontier import nondominated

RHO_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
SCORE_FORMULA_VERSION: str = "eq1-v2"

__all__ = [
    "RHO_GRID",
    "SCORE_FORMULA_VERSION",
    "Eq1Result",
    "ReplaySample",
    "clip_quality",
    "equation1",
    "mean_value",
]


@dataclass(frozen=True)
class Eq1Result:
    """Outcome of one ``equation1`` call.

    ``frontier`` is the distinct non-dominated frontier (ascending ``n``);
    ``points`` is the five ``(n, a)`` outcomes in ``expected_rhos`` order, so a
    persisted record stays comparable across calls regardless of input order.
    """

    hv: float
    penalty: float
    value: float
    frontier: tuple[tuple[int, float], ...]
    points: tuple[tuple[int, float], ...]


class ReplaySample(Protocol):
    """Structural type for one ``(world, policy, rho)`` replay outcome.

    Duck-typed: any object exposing these four attributes works, in particular
    the frozen ``dream_rsi.domain.models.ReplayEpisode``.

    The members are declared read-only on purpose. With plain variable
    annotations mypy requires *settable* attributes, which a frozen dataclass
    does not have ("Protocol member ... expected settable variable, got
    read-only attribute"), so ``list[ReplayEpisode]`` would not type-check as
    ``Sequence[ReplaySample]``. Read-only members accept both frozen and
    mutable implementations and add no requirement to callers.
    """

    @property
    def rho(self) -> float: ...

    @property
    def n(self) -> int: ...

    @property
    def kstar(self) -> int: ...

    @property
    def best_score(self) -> float: ...


def clip_quality(best_score: float, s_best: float) -> float:
    """``a = clip(best_score / s_best, 0, 1)`` (§3.4).

    Raises:
        ValueError: ``s_best`` is not finite and ``> 0`` (the paper's Eq(1)
            cannot be evaluated; never substitute ``max(1, S)``), or
            ``best_score`` is not finite.
    """
    if not isfinite(s_best) or s_best <= 0:
        raise ValueError("paper's Eq(1) requires S>0: world is not scoreable")
    if not isfinite(best_score):
        raise ValueError("best_score must be finite")
    return max(0.0, min(1.0, best_score / s_best))


def equation1(
    samples: Sequence[ReplaySample],
    *,
    b_count: int,
    s_best: float,
    lambda_weight: float,
    expected_rhos: Sequence[float] = RHO_GRID,
) -> Eq1Result:
    """Evaluate Eq(1) for one replay world from its five rho outcomes.

    Args:
        samples: exactly one replay outcome per entry of ``expected_rhos``.
        b_count: ``B = |T_i| - 1``, the number of non-root world nodes, ``> 0``.
        s_best: ``S = max`` non-root score, finite and ``> 0``.
        lambda_weight: parallel-penalty weight, finite and ``>= 0``.
        expected_rhos: the configured budget grid (default :data:`RHO_GRID`).

    Returns:
        :class:`Eq1Result` with ``hv``, ``penalty``, ``value``, the ``frontier``
        and the five ``(n, a)`` ``points`` in ``expected_rhos`` order.

    Raises:
        ValueError: ``B <= 0`` or ``S`` non-finite / ``<= 0``; invalid
            ``lambda_weight``; the rho multiset does not match
            ``expected_rhos`` exactly; or a sample violates
            ``0 <= n <= B``, ``kstar >= 0``, finite ``best_score``.
    """
    if b_count <= 0 or not isfinite(s_best) or s_best <= 0:
        raise ValueError("paper's Eq(1) requires B>0, S>0")
    if not isfinite(lambda_weight) or lambda_weight < 0:
        raise ValueError("lambda_weight must be finite and >= 0")
    if not expected_rhos:
        raise ValueError("expected_rhos must be non-empty: Eq(1) divides the penalty by |R|")
    if len(samples) != len(expected_rhos) or sorted(
        sample.rho for sample in samples
    ) != sorted(expected_rhos):
        raise ValueError("must provide exactly one sample per configured rho")
    for sample in samples:
        if sample.n < 0 or sample.n > b_count or sample.kstar < 0 or not isfinite(sample.best_score):
            raise ValueError("invalid replay sample")

    ordered = _in_rho_grid_order(samples, expected_rhos)
    points = tuple((sample.n, clip_quality(sample.best_score, s_best)) for sample in ordered)

    frontier = nondominated(points)
    hv = 0.0
    previous_a = 0.0
    for n, a in frontier:
        # Unreachable for a genuine frontier (distinct costs imply strictly
        # increasing quality); kept as a hard guard against a broken frontier.
        if a < previous_a:
            raise AssertionError("ordered Pareto quality must increase")
        hv += (a - previous_a) * (b_count - n) / b_count
        previous_a = a

    # n == 0 contributes 0 but the denominator stays the full |R|.
    penalty = sum(sample.kstar / sample.n for sample in samples if sample.n > 0) / len(
        expected_rhos
    )
    return Eq1Result(
        hv=hv,
        penalty=penalty,
        value=hv - lambda_weight * penalty,
        frontier=frontier,
        points=points,
    )


def mean_value(values: Sequence[float]) -> float:
    """``V^m = (1/t) * sum_i V_i^m`` — equal weight over worlds.

    Raises:
        ValueError: ``values`` is empty (an empty mean has no defined value).
    """
    if not values:
        raise ValueError("V^m requires at least one world score; refusing an empty mean")
    return sum(values) / len(values)


def _in_rho_grid_order(
    samples: Sequence[ReplaySample], expected_rhos: Sequence[float]
) -> tuple[ReplaySample, ...]:
    """Order samples to match ``expected_rhos`` positionally, not call order.

    The rho multiset has already been validated by the caller; this only fixes
    the order of ``points`` (a duplicated rho in ``expected_rhos`` consumes its
    slots in sample order).
    """
    pending: dict[float, list[int]] = {}
    for slot, rho in enumerate(expected_rhos):
        pending.setdefault(rho, []).append(slot)
    assigned: list[tuple[int, ReplaySample]] = []
    for sample in samples:
        slots = pending.get(sample.rho)
        if not slots:
            raise ValueError("must provide exactly one sample per configured rho")
        assigned.append((slots.pop(0), sample))
    assigned.sort(key=lambda item: item[0])
    return tuple(sample for _, sample in assigned)
