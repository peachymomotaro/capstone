from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from config import RebuildConfig
from stagnation import robust_y_scale, scale_aware_tolerance


@dataclass
class TrustRegionState:
    center: np.ndarray
    length: float
    success_count: int
    failure_count: int
    n_restarts: int


def reconstruct_tr_state(X_hist: np.ndarray, y_hist: np.ndarray, cfg: RebuildConfig) -> TrustRegionState:
    X_hist = np.asarray(X_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float).reshape(-1)
    if X_hist.shape[0] == 0:
        return TrustRegionState(np.array([], dtype=float), float(cfg.tr_init_length), 0, 0, 0)

    best = float(y_hist[0])
    center = X_hist[int(np.argmax(y_hist))].copy()
    length = float(cfg.tr_init_length)
    success = 0
    failure = 0
    n_restarts = 0
    y_scale = robust_y_scale(y_hist)
    improvement_tol = scale_aware_tolerance(
        y_scale,
        rel_tol=float(cfg.tr_stagnation_rel_tol),
        abs_floor=float(cfg.tr_stagnation_abs_floor),
    )

    for i in range(1, X_hist.shape[0]):
        yi = float(y_hist[i])
        improved = yi > best + improvement_tol
        if improved:
            best = yi
            center = X_hist[i].copy()
            success += 1
            failure = 0
            if success >= int(cfg.tr_success_tolerance):
                length = min(float(cfg.tr_max_length), length * 1.5)
                success = 0
        else:
            failure += 1
            success = 0
            if failure >= int(cfg.tr_failure_tolerance):
                length = max(float(cfg.tr_min_length), length / 2.0)
                failure = 0
        if length <= float(cfg.tr_min_length):
            n_restarts += 1
            length = float(cfg.tr_init_length)
            center = X_hist[int(np.argmax(y_hist[: i + 1]))].copy()
            success = 0
            failure = 0

    return TrustRegionState(center, float(length), int(success), int(failure), int(n_restarts))


def sample_in_tr(
    state: TrustRegionState,
    n: int,
    d: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    lower: float,
    upper: float,
) -> torch.Tensor:
    if n <= 0 or d <= 0 or state.center.size == 0:
        return torch.empty((0, d), device=device, dtype=dtype)

    sobol = SobolEngine(dimension=d, scramble=True, seed=int(seed))
    unit = sobol.draw(int(n)).detach().cpu().numpy()
    centered = (unit - 0.5) * float(state.length)
    pts = np.clip(state.center[None, :] + centered, lower, upper)
    return torch.tensor(pts, device=device, dtype=dtype)
