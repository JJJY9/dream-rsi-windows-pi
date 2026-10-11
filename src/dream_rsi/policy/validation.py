"""Server-side batch validation. A policy is never trusted to police itself.

Frozen contract: worker modules must not edit this file.

§3: a batch is ``C ⊆ {root} ∪ leaves(R)`` with distinct members and ``|C| ≤ W``.
The controller re-validates every action a policy proposes, in both online and
offline mode, because policies are LLM-generated source and may be wrong or
adversarial.

Ordering note (project ADR-0003): the paper models a batch as a *set*. We accept
only ordered ``list``/``tuple`` and reject ``set``/``frozenset``, because the
replay digest must be byte-reproducible and Python set iteration order is not a
contract. Rejecting is safer than silently sorting, which would mask a policy
that is incidentally order-dependent.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..domain.errors import HiddenNodeAccess, IllegalBatch


def _coerce_sequence(batch: Any) -> Sequence[Any]:
    if isinstance(batch, (str, bytes, bytearray)):
        raise IllegalBatch("policy returned a bare string, not a batch of node ids")
    if isinstance(batch, (set, frozenset)):
        raise IllegalBatch(
            "policy returned a set; batches must be ordered list/tuple so the replay "
            "digest is reproducible (ADR-0003)"
        )
    if not isinstance(batch, Sequence):
        raise IllegalBatch(f"policy must return list/tuple of node ids, got {type(batch).__name__}")
    return batch


def validate_batch(
    batch: Any,
    *,
    legal_actions: Sequence[str],
    observed_ids: Sequence[str],
    max_parallelism: int,
) -> tuple[str, ...]:
    """Return the batch as a canonical tuple, or raise ``InvalidAction``.

    ``[]`` is legal and means "stop"; it is returned as an empty tuple.
    """
    if not isinstance(max_parallelism, int) or max_parallelism < 1:
        raise ValueError("max_parallelism (W) must be an integer >= 1")

    items = _coerce_sequence(batch)
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise IllegalBatch(f"batch entries must be node id strings, got {type(item).__name__}")
        normalized.append(item)

    if len(normalized) > max_parallelism:
        raise IllegalBatch(
            f"batch size {len(normalized)} exceeds max_parallelism W={max_parallelism}"
        )

    seen: set[str] = set()
    for node_id in normalized:
        if node_id in seen:
            raise IllegalBatch(f"duplicate node id in one batch: {node_id!r}")
        seen.add(node_id)

    legal = set(legal_actions)
    observed = set(observed_ids)
    for node_id in normalized:
        if node_id not in legal:
            if node_id in observed:
                raise IllegalBatch(
                    f"{node_id!r} is observed but not a legal action "
                    "(not root and not a current leaf)"
                )
            raise HiddenNodeAccess(
                f"{node_id!r} is not in the observed prefix; a policy may only act on "
                "root or a revealed leaf"
            )
    return tuple(normalized)


__all__ = ["validate_batch"]
