"""Packing candidate — ROOT workspace (deliberately simple starting point).

A discovery agent may rewrite this file freely. The only hard requirement is
the public interface, which the fixed evaluator calls:

    construct() -> list[tuple[float, float]]

Return exactly 30 points in the unit square. The grader scores the minimum
pairwise distance between them (higher is better).

This starting point is a plain 6x5 lattice spanning the full square: evenly
spaced, every neighbour at distance 0.2, and it leaves real room on the table
for staggered/hexagonal arrangements and local refinement.
"""
from __future__ import annotations


def construct() -> list[tuple[float, float]]:
    columns = 6
    rows = 5
    points: list[tuple[float, float]] = []
    for row in range(rows):
        for col in range(columns):
            points.append((col / (columns - 1), row / (rows - 1)))
    return points
