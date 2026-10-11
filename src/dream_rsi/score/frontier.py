"""Pareto frontier over ``(n, a)`` replay outcomes for Equation (1) (§3.4).

Cost ``n`` is minimised, quality ``a`` is maximised, and equality is **not**
strict dominance: ``p`` dominates ``q`` iff ``n_p <= n_q`` and ``a_p >= a_q``
and at least one of the two is strict. Points are deduplicated and returned in
ascending ``n`` order, one quality per cost.

This is the production twin of ``reference/oracle.nondominated``. Both must
return identical tuples for every valid input; ``tests/test_score.py`` checks
that on the golden fixture and on seeded random point sets.
"""
from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

__all__ = ["nondominated"]


def nondominated(points: Iterable[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    """Return the distinct non-dominated ``(n, a)`` points, ascending by ``n``.

    Args:
        points: candidate ``(cost, quality)`` outcomes. ``n`` must be a
            non-negative ``int``; ``a`` must be finite and inside ``[0, 1]``.

    Returns:
        The frontier as a tuple, sorted by ascending ``n``.

    Raises:
        ValueError: a point has a non-``int`` or negative ``n``, or a
            non-finite / out-of-range ``a``. Out-of-contract points are never
            silently clipped: ``clip_quality`` is the only place clipping is
            allowed, and it happens before this function is called.
    """
    seen: set[tuple[int, float]] = set()
    distinct: list[tuple[int, float]] = []
    for n, a in points:
        if not isinstance(n, int) or n < 0:
            raise ValueError(
                f"invalid quality-cost point: n={n!r} must be a non-negative int"
            )
        if not isfinite(a) or not 0.0 <= a <= 1.0:
            raise ValueError(
                f"invalid quality-cost point: a={a!r} must be finite within [0, 1]"
            )
        point = (n, a)
        if point not in seen:
            seen.add(point)
            distinct.append(point)

    # Descending quality inside each cost: with the sweep below, a point is
    # non-dominated iff its quality is strictly greater than every quality
    # already seen (smaller-or-equal cost). Equal quality is domination by the
    # cheaper point, so it is dropped — exactly the oracle's rule.
    ordered = sorted(distinct, key=lambda point: (point[0], -point[1]))
    frontier: list[tuple[int, float]] = []
    best_quality = float("-inf")
    for n, a in ordered:
        if a > best_quality:
            frontier.append((n, a))
            best_quality = a
    return tuple(frontier)
