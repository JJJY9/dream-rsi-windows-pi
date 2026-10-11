"""Exception hierarchy for the strict-§3 Dream-RSI runtime.

Kept deliberately narrow: the runtime distinguishes *invalid input* (a bug or a
hostile policy), *integrity failure* (tampered artifact), and *external block*
(missing model configuration / unavailable OS facility). External blocks must
never be reported as an algorithm success or failure.
"""
from __future__ import annotations


class DreamRsiError(Exception):
    """Base class for every error raised by this package."""


class InvalidWorld(DreamRsiError, ValueError):
    """A world/tree violates the structural invariants of §3."""


class InvalidAction(DreamRsiError, ValueError):
    """A policy returned a batch that is not legal for the given prefix."""


class IllegalBatch(InvalidAction):
    """Batch is not a subset of {root} ∪ leaves, has duplicates, or exceeds W."""


class HiddenNodeAccess(InvalidAction):
    """Policy referenced a node it has not observed yet."""


class TreeInvariantViolation(DreamRsiError, ValueError):
    """Online tree mutation would break root/leaf/creation-order invariants."""


class IntegrityError(DreamRsiError):
    """A content-addressed artifact failed its hash check or was tampered with."""


class EvaluationError(DreamRsiError):
    """The fixed evaluator could not produce a trustworthy score."""


class PolicyRejected(DreamRsiError):
    """A candidate policy failed static/sandbox validation and cannot be used."""


class BlockedError(DreamRsiError):
    """Base class for genuine external blockers (never a normal completion)."""


class BlockedModelConfiguration(BlockedError):
    """Model credentials/provider/model id are missing or unusable.

    Status token: ``BLOCKED_MODEL_CONFIGURATION``.
    """


class BlockedEnvironment(BlockedError):
    """A required OS facility is unavailable (status: ``BLOCKED_ENVIRONMENT``)."""


__all__ = [
    "BlockedEnvironment",
    "BlockedError",
    "BlockedModelConfiguration",
    "DreamRsiError",
    "EvaluationError",
    "HiddenNodeAccess",
    "IllegalBatch",
    "IntegrityError",
    "InvalidAction",
    "InvalidWorld",
    "PolicyRejected",
    "TreeInvariantViolation",
]
