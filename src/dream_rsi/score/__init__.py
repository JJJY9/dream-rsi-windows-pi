"""Equation (1) scoring layer: Pareto hypervolume minus the parallel penalty.

Public API (frozen contract — the controller imports these exact names):

* :func:`dream_rsi.score.frontier.nondominated` — distinct non-dominated
  ``(n, a)`` points, ascending ``n``.
* :data:`dream_rsi.score.equation1.RHO_GRID` — the five-rho budget sweep.
* :data:`dream_rsi.score.equation1.SCORE_FORMULA_VERSION` — ``"eq1-v2"``.
* :class:`dream_rsi.score.equation1.Eq1Result` — hv / penalty / value /
  frontier / the five points in rho-grid order.
* :class:`dream_rsi.score.equation1.ReplaySample` — structural sample type.
* :func:`dream_rsi.score.equation1.clip_quality` — ``clip(s/S, 0, 1)``.
* :func:`dream_rsi.score.equation1.equation1` — the §3.4 objective.
* :func:`dream_rsi.score.equation1.mean_value` — ``V^m`` equal-weight mean.

The submodule ``equation1`` is deliberately *not* re-exported as a package
attribute here: doing so would shadow the submodule of the same name and break
``import dream_rsi.score.equation1`` for callers. Import from the submodules.
The math is pinned against ``reference/oracle.py`` by ``tests/test_score.py``.
"""
