from __future__ import annotations

import numpy as np


def is_dominated(candidate: np.ndarray, others: np.ndarray) -> bool:
    """Return True if candidate is dominated by any row in others (minimization)."""
    le = np.all(others <= candidate, axis=1)
    lt = np.any(others < candidate, axis=1)
    return bool(np.any(le & lt))


def pareto_front_indices(objectives_min: np.ndarray) -> np.ndarray:
    """Fast-ish O(N^2) Pareto front for minimization objectives."""
    costs = np.asarray(objectives_min, dtype=float)
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue
        c = costs[i]
        active_idx = np.where(is_efficient)[0]
        active = costs[active_idx]
        dominated_by_i = np.all(c <= active, axis=1) & np.any(c < active, axis=1)
        for j in active_idx[dominated_by_i]:
            if j != i:
                is_efficient[j] = False

    return np.where(is_efficient)[0]


def pareto_ranks(objectives_min: np.ndarray) -> np.ndarray:
    """Non-dominated sorting rank: 0 = Pareto front, 1 = second front, ..."""
    costs = np.asarray(objectives_min, dtype=float)
    n = costs.shape[0]
    remaining = np.arange(n)
    ranks = np.full(n, fill_value=-1, dtype=int)
    rank = 0

    while remaining.size > 0:
        front_local = pareto_front_indices(costs[remaining])
        front_global = remaining[front_local]
        ranks[front_global] = rank

        mask = np.ones(remaining.size, dtype=bool)
        mask[front_local] = False
        remaining = remaining[mask]
        rank += 1

    return ranks
