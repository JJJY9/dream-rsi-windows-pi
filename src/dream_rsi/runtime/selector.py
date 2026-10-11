"""Policy selection (§3.3, ACCEPTANCE §A.6).

Every candidate policy — the incumbent included — is replayed on the *same*
frozen world set ``H_t`` across the same five-rho grid. The candidate with the
highest mean ``V`` is deployed.

Two properties this module is responsible for, and which the controller asserts
rather than assumes:

*   **Monotonicity on a fixed `H_t`.** The incumbent is always a candidate, so
    the winner's mean ``V`` can never be below the incumbent's. ``V(π_{t+1}) >=
    V(π_t)`` therefore holds *as measured on `H_t`* — it is a property of the
    selection rule, not a claim about future real-world performance, and it is
    never reported as one.
*   **Stability under ties.** Floating-point equality is meaningless here, so
    "tie" means "within ``TIE_EPSILON``". Among tied candidates the incumbent
    wins; otherwise the lowest revision index wins. A policy is only ever
    replaced by one that is *actually better*, never by a coin flip.

An invalid proposal never reaches this module: it is recorded as a rejected
revision and the incumbent is carried forward.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..domain.errors import PolicyRejected

#: Two mean values closer than this are the same policy as far as deployment is
#: concerned. Chosen well above float64 accumulation noise for a handful of
#: five-point sums, and far below any difference a real policy change produces.
TIE_EPSILON = 1e-12


@dataclass(frozen=True)
class PolicyScore:
    """One policy version's measured performance on a fixed ``H_t``."""

    policy_version_id: str
    m: int
    module_digest: str
    per_world_values: tuple[float, ...]
    per_world_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.per_world_values:
            raise PolicyRejected(
                f"policy {self.policy_version_id!r} has no world results; "
                "a policy must be replayed on the whole of H_t before it can be scored"
            )
        if self.per_world_digests and len(self.per_world_digests) != len(self.per_world_values):
            raise PolicyRejected(
                f"policy {self.policy_version_id!r} has {len(self.per_world_values)} values "
                f"but {len(self.per_world_digests)} digests"
            )

    @property
    def world_count(self) -> int:
        return len(self.per_world_values)

    @property
    def mean_value(self) -> float:
        return sum(self.per_world_values) / len(self.per_world_values)

    def is_deployed(self) -> bool:
        """The incumbent is always revision 0."""
        return self.m == 0


@dataclass(frozen=True)
class Selection:
    selected: PolicyScore
    deployed: PolicyScore
    ranking: tuple[PolicyScore, ...]
    deployed_again: bool
    reason: str

    @property
    def improvement(self) -> float:
        return self.selected.mean_value - self.deployed.mean_value


def rank(candidates: Sequence[PolicyScore]) -> tuple[PolicyScore, ...]:
    """Best first: mean ``V`` descending, then revision index ascending.

    The secondary key is what makes the order total and reproducible — without
    it, two tied policies would rank by input order, which is not a property of
    the policies themselves.
    """
    return tuple(sorted(candidates, key=lambda c: (-c.mean_value, c.m)))


def select(
    *,
    deployed: PolicyScore,
    candidates: Sequence[PolicyScore],
) -> Selection:
    """Pick the next policy to deploy on the frozen ``H_t``.

    ``candidates`` are the *revisions proposed this round* (m >= 1). The
    incumbent is added here rather than trusted to be present, so monotonicity
    is structural: the winner is chosen from a set that contains ``deployed``.
    """
    if not deployed.is_deployed():
        raise PolicyRejected(
            f"the deployed policy must be revision 0, got m={deployed.m}"
        )

    pool = (deployed, *candidates)
    for candidate in candidates:
        if candidate.world_count != deployed.world_count:
            raise PolicyRejected(
                f"policy {candidate.policy_version_id!r} was scored on "
                f"{candidate.world_count} worlds but the incumbent on "
                f"{deployed.world_count}; every candidate must see the same H_t"
            )
        if candidate.policy_version_id == deployed.policy_version_id:
            raise PolicyRejected(
                f"revision {candidate.m} reuses the incumbent's id "
                f"{deployed.policy_version_id!r}"
            )

    ordered = rank(pool)
    best = ordered[0]
    best_mean = best.mean_value

    # Stable tie-break: among everything statistically tied with the best, the
    # lowest revision index wins, and the incumbent is index 0.
    tied = [c for c in pool if best_mean - c.mean_value <= TIE_EPSILON]
    winner = min(tied, key=lambda c: c.m)

    deployed_again = winner.policy_version_id == deployed.policy_version_id
    if deployed_again:
        others = [c for c in pool if c is not deployed]
        if not others:
            reason = "no revisions were proposed; the incumbent stands"
        elif best_mean - deployed.mean_value <= TIE_EPSILON and best.policy_version_id != deployed.policy_version_id:
            reason = (
                f"best revision {best.policy_version_id!r} ties the incumbent at "
                f"V={best_mean:.12f}; the incumbent stands"
            )
        else:
            reason = (
                f"no revision beat the incumbent (best V={best_mean:.12f} vs "
                f"incumbent V={deployed.mean_value:.12f})"
            )
    else:
        reason = (
            f"{winner.policy_version_id!r} improves mean V from "
            f"{deployed.mean_value:.12f} to {winner.mean_value:.12f}"
        )

    return Selection(
        selected=winner,
        deployed=deployed,
        ranking=ordered,
        deployed_again=deployed_again,
        reason=reason,
    )


def assert_monotonic(selection: Selection) -> None:
    """``V(π_(t+1)) >= V(π_t)`` on the fixed ``H_t``, or the run is invalid.

    This can only fire if :func:`select` is broken or a caller bypassed it —
    which is exactly when the run's results stop meaning anything, so it is a
    hard failure rather than a warning.
    """
    if selection.selected.mean_value < selection.deployed.mean_value - TIE_EPSILON:
        raise PolicyRejected(
            "policy selection violated V(pi_(t+1)) >= V(pi_t) on a fixed H_t: "
            f"selected {selection.selected.policy_version_id!r} "
            f"V={selection.selected.mean_value:.12f} < incumbent "
            f"V={selection.deployed.mean_value:.12f}"
        )


__all__ = ["TIE_EPSILON", "PolicyScore", "Selection", "assert_monotonic", "rank", "select"]
