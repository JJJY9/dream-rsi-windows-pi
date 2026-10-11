"""EQ01-EQ10 for the Equation (1) scoring module, plus oracle cross-checks.

The golden fixture (B=10, S=100, five points ``(2,.4),(4,.4),(5,.8),(8,.9),
(9,.9)`` with sequential rounds ``kstar=[2,3,4,5,6]`` over ``rho=[0,.25,.5,
.75,1]``) is a **hand-built math fixture** explained in ``docs/ALGORITHM.md`` -- it
is NOT paper experimental data.

``reference/oracle.py`` is the authoritative math reference and is imported
read-only to prove the production module is numerically identical.
"""
from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

# tests/ has no package __init__; make the repo root (reference/) importable and
# prepend src/ so the working-tree dream_rsi wins over any installed copy.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from reference.oracle import ReplayResult  # noqa: E402
from reference.oracle import equation1 as oracle_equation1  # noqa: E402
from reference.oracle import nondominated as oracle_nondominated  # noqa: E402

from dream_rsi.score.equation1 import (  # noqa: E402
    RHO_GRID,
    SCORE_FORMULA_VERSION,
    clip_quality,
    equation1,
    mean_value,
)
from dream_rsi.score.frontier import nondominated  # noqa: E402

GOLDEN_B = 10
GOLDEN_S = 100.0
GOLDEN_POINTS: tuple[tuple[int, float], ...] = ((2, 0.4), (4, 0.4), (5, 0.8), (8, 0.9), (9, 0.9))
GOLDEN_KSTARS: tuple[int, ...] = (2, 3, 4, 5, 6)
GOLDEN_LAMBDA = 0.1


@dataclass(frozen=True)
class Sample:
    """Structural stand-in for a ReplayEpisode (duck-typed ReplaySample)."""

    rho: float
    n: int
    kstar: int
    best_score: float


def golden_samples() -> list[Sample]:
    """The §4 fixture; ``best_score = a * S`` so ``clip_quality`` reproduces ``a``."""
    return [
        Sample(rho=rho, n=n, kstar=kstar, best_score=a * GOLDEN_S)
        for rho, (n, a), kstar in zip(RHO_GRID, GOLDEN_POINTS, GOLDEN_KSTARS, strict=True)
    ]


def oracle_view(samples: list[Sample]) -> list[ReplayResult]:
    """Same numbers through the oracle's ReplayResult carrier."""
    return [
        ReplayResult(
            rho=s.rho,
            revealed=("root",),
            actions=(),
            kstar=s.kstar,
            n=s.n,
            best_score=s.best_score,
            stop_reason="synthetic",
        )
        for s in samples
    ]


# --- EQ01-EQ04: the golden fixture -----------------------------------------


def test_eq01_golden_frontier() -> None:
    assert nondominated(GOLDEN_POINTS) == ((2, 0.4), (5, 0.8), (8, 0.9))
    result = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert result.frontier == ((2, 0.4), (5, 0.8), (8, 0.9))


def test_eq02_golden_hv() -> None:
    result = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert result.hv == pytest.approx(0.54, abs=1e-12)
    # hand-check: .4*8/10 + (.8-.4)*5/10 + (.9-.8)*2/10
    assert result.hv == pytest.approx(0.32 + 0.2 + 0.02, abs=1e-12)


def test_eq03_golden_penalty() -> None:
    result = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert result.penalty == pytest.approx((1 + 0.75 + 0.8 + 0.625 + 6 / 9) / 5, abs=1e-12)


def test_eq04_golden_value() -> None:
    result = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert result.value == pytest.approx(0.46316666666666667, abs=1e-12)
    assert result.value == pytest.approx(result.hv - GOLDEN_LAMBDA * result.penalty, abs=1e-12)


def test_golden_points_are_reported_in_rho_grid_order() -> None:
    samples = golden_samples()
    shuffled = [samples[i] for i in (3, 0, 4, 2, 1)]
    result = equation1(shuffled, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert result.points == GOLDEN_POINTS
    assert len(result.points) == len(RHO_GRID)


def test_sample_order_does_not_change_hv_penalty_value() -> None:
    samples = golden_samples()
    shuffled = [samples[i] for i in (4, 3, 2, 1, 0)]
    a = equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    b = equation1(shuffled, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert (a.hv, a.penalty, a.value, a.frontier) == (b.hv, b.penalty, b.value, b.frontier)


# --- EQ05: zero-call rho ----------------------------------------------------


def test_eq05_zero_call_rho_contributes_zero_but_denominator_stays_five() -> None:
    samples = golden_samples()
    samples[0] = replace(samples[0], n=0, kstar=0, best_score=0.0)
    result = equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    positive_terms = 0.75 + 0.8 + 0.625 + 6 / 9
    assert result.penalty == pytest.approx((0 + positive_terms) / 5, abs=1e-12)
    # not the 4-sample mean that a wrong denominator would give
    assert result.penalty != pytest.approx(positive_terms / 4, abs=1e-12)
    # (0, 0) is nondominated -- no point is cheaper -- so it stays on the
    # frontier while contributing exactly zero area (abar_1 - abar_0 == 0).
    assert result.frontier == ((0, 0.0), (4, 0.4), (5, 0.8), (8, 0.9))
    # zero area from (0, 0): (0-0)*(B-0)/B contributes nothing, so HV is the
    # (.4-0)*(10-4)/10 + (.8-.4)*5/10 + (.9-.8)*2/10 sweep that follows.
    assert result.hv == pytest.approx(0.4 * 6 / 10 + 0.4 * 5 / 10 + 0.1 * 2 / 10, abs=1e-12)


def test_all_rhos_stopped_immediately_is_valid() -> None:
    samples = [Sample(rho=rho, n=0, kstar=0, best_score=0.0) for rho in RHO_GRID]
    result = equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    assert (result.hv, result.penalty, result.value) == (0.0, 0.0, 0.0)
    assert result.frontier == ((0, 0.0),)  # (0,0) is nondominated when alone


# --- EQ06: unscoreable worlds ----------------------------------------------


@pytest.mark.parametrize(
    ("b_count", "s_best"),
    [
        (0, 100.0),
        (-1, 100.0),
        (10, 0.0),
        (10, -1.0),
        (10, float("nan")),
        (10, float("inf")),
        (10, float("-inf")),
    ],
)
def test_eq06_unscoreable_world_raises_never_substitutes(b_count: int, s_best: float) -> None:
    with pytest.raises(ValueError, match=re.escape("paper's Eq(1) requires B>0, S>0")):
        equation1(golden_samples(), b_count=b_count, s_best=s_best, lambda_weight=GOLDEN_LAMBDA)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_clip_quality_rejects_nonfinite_best_score(bad: float) -> None:
    with pytest.raises(ValueError):
        clip_quality(bad, GOLDEN_S)


def test_clip_quality_clips_but_rejects_bad_s() -> None:
    assert clip_quality(150.0, GOLDEN_S) == 1.0
    assert clip_quality(50.0, GOLDEN_S) == 0.5
    assert clip_quality(0.0, GOLDEN_S) == 0.0
    assert clip_quality(-5.0, GOLDEN_S) == 0.0  # negative raw score clips to 0, not silently
    for bad_s in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            clip_quality(1.0, bad_s)


def test_negative_best_score_clips_to_zero_not_an_exception() -> None:
    # A revealed max below zero is legitimate while S>0 (the world holds a
    # positive unrevealed score); Eq(1) maps it to a=0. The world-level
    # "negative scores => not_scoreable" decision is the §4 S>0 gate above.
    samples = golden_samples()
    samples[0] = replace(samples[0], best_score=-10.0)
    result = equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.0)
    assert result.points[0] == (2, 0.0)


@pytest.mark.parametrize("bad_lambda", [float("nan"), float("inf"), float("-inf"), -0.1, -1.0])
def test_invalid_lambda_raises(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="lambda"):
        equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=bad_lambda)


def test_lambda_zero_is_valid() -> None:
    result = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.0)
    assert result.value == result.hv


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", -1),
        ("n", 11),  # n > B
        ("kstar", -1),
        ("best_score", float("nan")),
        ("best_score", float("inf")),
    ],
)
def test_invalid_sample_raises(field: str, value: float) -> None:
    samples = golden_samples()
    samples[2] = replace(samples[2], **{field: value})
    with pytest.raises(ValueError, match="invalid replay sample"):
        equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)


def test_rho_multiset_must_match_the_configured_grid() -> None:
    samples = golden_samples()
    with pytest.raises(ValueError, match="exactly one sample per configured rho"):
        equation1(samples[:4], b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    wrong_rho = [replace(s, rho=0.6) if s.rho == 0.5 else s for s in samples]
    with pytest.raises(ValueError, match="exactly one sample per configured rho"):
        equation1(wrong_rho, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    duplicated = [samples[0], *samples[1:4], samples[1]]
    with pytest.raises(ValueError, match="exactly one sample per configured rho"):
        equation1(duplicated, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    with pytest.raises(ValueError):
        equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA,
                  expected_rhos=())


# --- EQ07: same cost, different quality -------------------------------------


def test_eq07_same_n_keeps_only_the_highest_quality_point() -> None:
    samples = [
        Sample(rho=0.0, n=4, kstar=1, best_score=20.0),   # a=.2, dominated at n=4
        Sample(rho=0.25, n=4, kstar=1, best_score=70.0),  # a=.7, kept
        Sample(rho=0.5, n=4, kstar=1, best_score=50.0),   # a=.5, dominated at n=4
        Sample(rho=0.75, n=9, kstar=1, best_score=90.0),  # a=.9, kept
        Sample(rho=1.0, n=4, kstar=1, best_score=70.0),   # duplicate of (4,.7)
    ]
    result = equation1(samples, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.0)
    assert result.frontier == ((4, 0.7), (9, 0.9))
    assert nondominated([(4, 0.2), (4, 0.7), (4, 0.5), (9, 0.9), (4, 0.7)]) == ((4, 0.7), (9, 0.9))


# --- EQ08: dominated / duplicate points -------------------------------------


def test_eq08_only_distinct_nondominated_points_survive() -> None:
    points = [(5, 0.9), (5, 0.9), (6, 0.5), (7, 0.6), (8, 0.9), (9, 0.1)]
    assert nondominated(points) == ((5, 0.9),)
    assert nondominated(points) == oracle_nondominated(points)
    assert nondominated([]) == ()
    assert nondominated([(3, 0.5), (3, 0.5), (3, 0.5)]) == ((3, 0.5),)


@pytest.mark.parametrize(
    "points",
    [
        ((-1, 0.5),),
        ((2.0, 0.5),),
        (("2", 0.5),),
        ((None, 0.5),),
        ((2, float("nan")),),
        ((2, float("inf")),),
        ((2, 1.5),),
        ((2, -0.1),),
    ],
)
def test_nondominated_rejects_out_of_contract_points(points: tuple[tuple[int, float], ...]) -> None:
    with pytest.raises(ValueError):
        nondominated(points)


# --- EQ09: monotone in quality at fixed cost --------------------------------


def test_eq09_higher_quality_at_fixed_cost_never_decreases_hv() -> None:
    base = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.0)
    raised = [
        replace(s, best_score=95.0) if s.rho == 0.5 else s  # (5,.8) -> (5,.95)
        for s in golden_samples()
    ]
    improved = equation1(raised, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.0)
    assert improved.hv >= base.hv
    assert improved.hv == pytest.approx(0.32 + 0.55 * 5 / 10, abs=1e-12)


def test_eq09_hv_is_monotone_in_a_single_frontier_quality() -> None:
    samples = golden_samples()
    previous = -1.0
    for best_score in (0.0, 20.0, 60.0, 80.0):
        tuned = [replace(samples[1], best_score=best_score), *samples[:1], *samples[2:]]
        result = equation1(tuned, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.0)
        assert result.hv >= previous
        previous = result.hv


# --- EQ10: differentiated vs degenerate rho sweep ---------------------------


def test_eq10_rho_sweep_differentiated_and_degenerate() -> None:
    differentiated = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S,
                               lambda_weight=GOLDEN_LAMBDA)
    assert len(set(differentiated.points)) > 1
    assert len(differentiated.points) == len(RHO_GRID)
    assert differentiated.frontier != differentiated.points  # (4,.4) and (9,.9) are dominated

    # Degenerate sweep: every rho reaches the same (n, a). Accepted, no crash --
    # but flagged here as degenerate, per TEST_MATRIX EQ10.
    degenerate = [Sample(rho=rho, n=3, kstar=1, best_score=50.0) for rho in RHO_GRID]
    result = equation1(degenerate, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    assert result.points == ((3, 0.5),) * 5
    assert result.frontier == ((3, 0.5),)
    assert result.hv == pytest.approx(0.5 * (GOLDEN_B - 3) / GOLDEN_B, abs=1e-12)
    assert result.penalty == pytest.approx(1 / 3, abs=1e-12)


# --- frozen constants, result shape, V^m ------------------------------------


def test_frozen_constants_and_result_shape() -> None:
    assert RHO_GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert SCORE_FORMULA_VERSION == "eq1-v2"
    result = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=0.1)
    assert isinstance(result.frontier, tuple) and isinstance(result.points, tuple)
    assert all(isinstance(n, int) and isinstance(a, float) for n, a in result.points)


def test_domain_replay_episode_dto_satisfies_replay_sample() -> None:
    """Duck typing: the real frozen domain DTO is a valid ReplaySample carrier."""
    from dream_rsi.domain.models import ReplayEpisode, StopReason

    episodes = [
        ReplayEpisode(
            episode_id=f"ep-{i}",
            world_id="world-1",
            policy_version_id="policy-1",
            rho=sample.rho,
            n=sample.n,
            kstar=sample.kstar,
            best_score=sample.best_score,
            attained_quality=0.0,
            selected_batches=(),
            revealed_order=("root",),
            stop_reason=StopReason.MAX_ROUNDS,
            replay_digest="digest",
        )
        for i, sample in enumerate(golden_samples())
    ]
    result = equation1(episodes, b_count=GOLDEN_B, s_best=GOLDEN_S, lambda_weight=GOLDEN_LAMBDA)
    assert result.hv == pytest.approx(0.54, abs=1e-12)
    assert result.points == GOLDEN_POINTS


def test_mean_value() -> None:
    assert mean_value([1.0]) == 1.0
    assert mean_value([0.5, 1.5]) == pytest.approx(1.0, abs=1e-12)
    assert mean_value([-1.0, 1.0, 3.0]) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        mean_value([])


# --- cross-checks against reference/oracle.py -------------------------------


def test_equation1_agrees_with_oracle_on_golden_data() -> None:
    mine = equation1(golden_samples(), b_count=GOLDEN_B, s_best=GOLDEN_S,
                     lambda_weight=GOLDEN_LAMBDA)
    theirs = oracle_equation1(oracle_view(golden_samples()), GOLDEN_B, GOLDEN_S, GOLDEN_LAMBDA)
    assert mine.frontier == theirs.frontier
    assert mine.hv == theirs.hv
    assert mine.penalty == theirs.penalty
    assert mine.value == theirs.value


def test_equation1_agrees_with_oracle_on_seeded_random_worlds() -> None:
    rng = random.Random(20241010)
    qualities = (0.0, 0.2, 0.4, 0.5, 0.5, 0.6, 0.75, 0.9, 1.0)
    for _ in range(300):
        b_count = rng.randint(1, 12)
        s_best = rng.choice((0.5, 1.0, 2.5, 10.0, 100.0))
        lam = rng.choice((0.0, 0.1, 0.5, 2.0))
        samples = [
            Sample(
                rho=rho,
                n=rng.choice((0, rng.randint(0, b_count), b_count)),
                kstar=rng.randint(0, 8),
                best_score=rng.choice(qualities) * s_best,
            )
            for rho in RHO_GRID
        ]
        mine = equation1(samples, b_count=b_count, s_best=s_best, lambda_weight=lam)
        theirs = oracle_equation1(oracle_view(samples), b_count, s_best, lam)
        assert mine.frontier == theirs.frontier
        assert mine.hv == pytest.approx(theirs.hv, abs=1e-12)
        assert mine.penalty == pytest.approx(theirs.penalty, abs=1e-12)
        assert mine.value == pytest.approx(theirs.value, abs=1e-12)


def test_nondominated_agrees_with_oracle_on_seeded_random_point_sets() -> None:
    rng = random.Random(42)
    qualities = (0.0, 0.2, 0.4, 0.5, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0)
    for _ in range(400):
        points = [
            (rng.choice((0, 1, 2, 3, 4, 4, 5, 8)), rng.choice(qualities))
            for _ in range(rng.randint(0, 9))
        ]
        mine = nondominated(points)
        theirs = oracle_nondominated(points)
        assert mine == theirs
        assert list(mine) == sorted(mine)  # ascending n, as the oracle guarantees
