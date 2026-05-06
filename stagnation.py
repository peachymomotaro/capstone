from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RebuildConfig


@dataclass(frozen=True)
class StagnationStats:
    flag: bool
    gain: float
    threshold: float
    scale: float
    recent_best_start: float
    recent_best_end: float
    window: int


def _as_1d_float_array(y_raw) -> np.ndarray:
    if hasattr(y_raw, "detach"):
        y_raw = y_raw.detach().cpu().numpy()
    return np.asarray(y_raw, dtype=float).reshape(-1)


def robust_y_scale(y_raw) -> float:
    y = _as_1d_float_array(y_raw)
    if y.size < 2:
        return 0.0
    q25, q75 = np.quantile(y, [0.25, 0.75])
    return float(max(0.0, q75 - q25))


def scale_aware_tolerance(
    scale: float,
    *,
    rel_tol: float,
    abs_floor: float,
) -> float:
    return float(max(float(abs_floor), float(rel_tol) * max(0.0, float(scale))))


def compute_stagnation_stats(y_raw, cfg: RebuildConfig) -> StagnationStats:
    y = _as_1d_float_array(y_raw)
    window = max(2, int(cfg.tr_stagnation_window))
    scale = robust_y_scale(y)
    threshold = scale_aware_tolerance(
        scale,
        rel_tol=float(cfg.tr_stagnation_rel_tol),
        abs_floor=float(cfg.tr_stagnation_abs_floor),
    )

    if y.size <= window:
        return StagnationStats(
            flag=False,
            gain=float("nan"),
            threshold=threshold,
            scale=scale,
            recent_best_start=float("nan"),
            recent_best_end=float("nan"),
            window=window,
        )

    best_so_far = np.maximum.accumulate(y)
    recent = best_so_far[-window:]
    recent_best_start = float(recent[0])
    recent_best_end = float(recent[-1])
    gain = float(recent_best_end - recent_best_start)

    return StagnationStats(
        flag=bool(gain <= threshold),
        gain=gain,
        threshold=threshold,
        scale=scale,
        recent_best_start=recent_best_start,
        recent_best_end=recent_best_end,
        window=window,
    )
