"""Circle-packing candidate — ROOT workspace (deliberately simple starting point).

A discovery agent may rewrite this file freely. The only hard requirement is
the public interface, which the fixed evaluator calls:

    construct() -> list[tuple[float, float, float]]

Return exactly 26 circles ``(x, y, r)`` inside the unit square with no
overlaps. The grader scores the sum of the radii (higher is better; the
published target for n=26 is 2.635983).

This starting point is the classic ring arrangement (one circle in the middle,
rings of 8 and 16 around it), with the radii shrunk proportionally until every
pair is disjoint and all circles fit the square. It is valid but far from
optimal: the proportional shrink throws away radius everywhere, and the outer
ring wastes the square's corners.
"""
from __future__ import annotations

import math

N = 26


def construct() -> list[tuple[float, float, float]]:
    centers: list[tuple[float, float]] = [(0.5, 0.5)]
    for i in range(9):
        angle = 2.0 * math.pi * i / 9.0
        centers.append((0.5 + 0.3 * math.cos(angle), 0.5 + 0.3 * math.sin(angle)))
    for i in range(16):
        angle = 2.0 * math.pi * i / 16.0
        centers.append((0.5 + 0.7 * math.cos(angle), 0.5 + 0.7 * math.sin(angle)))
    centers = [(min(0.99, max(0.01, x)), min(0.99, max(0.01, y))) for x, y in centers]

    # Radius for each circle: distance to the square border, then shrunk
    # pairwise so every pair is disjoint.
    radii = [min(x, y, 1.0 - x, 1.0 - y) for x, y in centers]
    for _ in range(3):
        for i in range(N):
            xi, yi = centers[i]
            for j in range(i + 1, N):
                xj, yj = centers[j]
                dist = math.hypot(xi - xj, yi - yj)
                if radii[i] + radii[j] > dist > 0:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale

    return [(centers[i][0], centers[i][1], radii[i]) for i in range(N)]
