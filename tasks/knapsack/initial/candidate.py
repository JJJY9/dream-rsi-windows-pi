"""Knapsack candidate — ROOT workspace (deliberately weak starting point).

A discovery agent may rewrite this file freely. The only hard requirement is the
public interface, which the fixed evaluator calls:

    solve(weights: list[int], values: list[int], capacity: int) -> list[int]

Return the 0-based indices of the items to pack. The grader rejects duplicate
indices, out-of-range indices, and any selection whose total weight exceeds
`capacity`.

This starting point is a first-fit scan in input order: it never exceeds the
capacity, but it ignores value density entirely, so it leaves a lot of value on
the table.
"""
from __future__ import annotations


def solve(weights: list[int], values: list[int], capacity: int) -> list[int]:
    chosen: list[int] = []
    remaining = capacity
    for index, weight in enumerate(weights):
        if weight <= remaining:
            chosen.append(index)
            remaining -= weight
    return chosen
