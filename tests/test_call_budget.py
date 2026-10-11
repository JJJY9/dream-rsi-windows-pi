"""Tests for the hard ceiling on model calls.

A budget that only *counts* is not a budget. The properties that matter are
that it never grants more than the limit, that it holds under the parallel
batches the rollout actually uses, and that running out stops the run without
turning into a failure -- a run that stopped early is a result, not an error,
and must stay readable as one.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(SRC))

from dream_rsi.runtime.online import CallBudget  # noqa: E402

# ---------------------------------------------------------------------------
# the primitive
# ---------------------------------------------------------------------------


def test_reserve_grants_up_to_the_limit() -> None:
    budget = CallBudget(3)
    assert budget.reserve(2) == 2
    assert budget.spent == 2
    assert budget.reserve(2) == 1, "a partial grant must be reported as partial"
    assert budget.exhausted
    assert budget.reserve(5) == 0


def test_refusals_are_counted_not_hidden() -> None:
    """A cut has to be visible, or a short run looks like a short policy."""
    budget = CallBudget(1)
    budget.reserve(1)
    assert budget.refusals == 0
    budget.reserve(4)
    assert budget.refusals == 4
    assert budget.to_dict()["refusals"] == 4


def test_no_limit_means_no_limit() -> None:
    budget = CallBudget(None)
    assert budget.reserve(10_000) == 10_000
    assert budget.remaining() is None
    assert not budget.exhausted
    assert budget.refusals == 0
    assert budget.to_dict()["limit"] is None


def test_a_zero_budget_permits_nothing() -> None:
    """Zero is a real setting: 'make no model calls at all'."""
    budget = CallBudget(0)
    assert budget.exhausted
    assert budget.reserve(1) == 0


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        CallBudget(-1)


def test_reserving_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        CallBudget(5).reserve(-1)


def test_remaining_never_goes_negative() -> None:
    budget = CallBudget(2)
    budget.reserve(2)
    assert budget.remaining() == 0


def test_concurrent_reservations_cannot_overshoot() -> None:
    """The rollout dispatches batches in parallel, so this is the real case.

    Every worker claims under the same lock; without that, two threads could
    both see one slot free and both take it.
    """
    limit = 50
    budget = CallBudget(limit)
    granted: list[int] = []
    granted_lock = threading.Lock()

    def claim() -> None:
        got = budget.reserve(1)
        with granted_lock:
            granted.append(got)

    threads = [threading.Thread(target=claim) for _ in range(400)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(granted) == limit
    assert sum(1 for g in granted if g == 1) == limit
    assert budget.spent == limit
    assert budget.refusals == 400 - limit


def test_a_batch_is_granted_atomically_relative_to_other_batches() -> None:
    budget = CallBudget(5)
    results: list[int] = []
    lock = threading.Lock()

    def claim_batch() -> None:
        got = budget.reserve(2)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=claim_batch) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 5 batches of 2 against a limit of 5: 2+2+1 and then nothing, in some order.
    assert sorted(results, reverse=True) == [2, 2, 1, 0, 0]
    assert budget.spent == 5


def test_the_ledger_states_its_own_unit() -> None:
    """The limit is in invocations, not model turns, and has to say so.

    A Pi discovery call is a whole agent session that makes many model turns
    internally. Quoting a call limit without that caveat would read as a token
    bound it does not provide.
    """
    unit = CallBudget(10).to_dict()["unit"]
    assert "invocation" in unit
    assert "not model turns" in unit
