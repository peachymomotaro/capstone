from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DedupResult:
    keep_indices: list[int]
    dropped_duplicate_internal: list[int]
    dropped_duplicate_observed: list[int]


def quantize_scalar_floor(x: float, decimals: int = 6, lower: float = 0.0, upper: float = 0.999999) -> float:
    x = min(max(float(x), lower), upper)
    scale = 10**decimals
    return math.floor(x * scale) / scale


def quantize_array_floor(
    arr: np.ndarray,
    decimals: int = 6,
    lower: float = 0.0,
    upper: float = 0.999999,
) -> np.ndarray:
    scale = 10**decimals
    clipped = np.clip(np.asarray(arr, dtype=float), lower, upper)
    return np.floor(clipped * scale) / scale


def format_portal_string(x_row: np.ndarray, decimals: int = 6, lower: float = 0.0, upper: float = 0.999999) -> str:
    q = quantize_array_floor(np.asarray(x_row, dtype=float), decimals=decimals, lower=lower, upper=upper)
    return "-".join(f"{float(v):.{decimals}f}" for v in q.tolist())


def dedup_after_rounding(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    scores: np.ndarray | None = None,
    decimals: int = 6,
    lower: float = 0.0,
    upper: float = 0.999999,
) -> DedupResult:
    cand_np = candidates.detach().cpu().numpy()
    obs_np = observed.detach().cpu().numpy()

    cand_q = quantize_array_floor(cand_np, decimals=decimals, lower=lower, upper=upper)
    obs_q = quantize_array_floor(obs_np, decimals=decimals, lower=lower, upper=upper)

    obs_set = {tuple(row.tolist()) for row in obs_q}

    dropped_obs: list[int] = []
    dropped_internal: list[int] = []

    if scores is None:
        scores = np.zeros(cand_q.shape[0], dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)

    best_for_key: dict[tuple[float, ...], tuple[int, float]] = {}
    for i, row in enumerate(cand_q):
        key = tuple(row.tolist())
        if key in obs_set:
            dropped_obs.append(i)
            continue

        prev = best_for_key.get(key)
        cur_score = float(scores[i])
        if prev is None:
            best_for_key[key] = (i, cur_score)
        else:
            prev_i, prev_score = prev
            if cur_score > prev_score:
                dropped_internal.append(prev_i)
                best_for_key[key] = (i, cur_score)
            else:
                dropped_internal.append(i)

    keep = sorted(i for i, _ in best_for_key.values())

    return DedupResult(
        keep_indices=keep,
        dropped_duplicate_internal=sorted(set(dropped_internal)),
        dropped_duplicate_observed=sorted(set(dropped_obs)),
    )
