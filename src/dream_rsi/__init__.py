"""Dream-RSI strict §3 reproduction (Windows native).

Scope: Discovery Tree, Root/Leaf online rollout, deterministic historical
replay, five-rho sweep, Equation (1) Pareto hypervolume minus a parallel
penalty, and M-version policy revision/selection. The paper's §4 benchmark is
explicitly out of scope.

Nothing in this package claims to be the paper's original implementation. See
``docs/ALGORITHM.md`` for the paper-rule / engineering-choice split.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
