from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping


PROJECT_DIR = Path(__file__).resolve().parent


def _default_per_function_overrides() -> Mapping[str, Mapping[str, object]]:
    return {
        "function_1": {
            "use_output_power_transform": False,
            "portfolio_primary": "global_explore",
            "portfolio_when_stagnant": "global_explore",
        }
    }


@dataclass(frozen=True)
class RebuildConfig:
    data_dir: str = str(PROJECT_DIR / "data")
    reports_dir: str = str(PROJECT_DIR / "reports")
    report_date: str = date.today().isoformat()
    overwrite: bool = False

    seed: int = 0
    dtype: str = "float64"

    # Feature flags
    use_output_power_transform: bool = True
    output_transform_mode: str = "power"  # power | copula
    use_input_warp: bool = True
    kernel_mode: str = "lin_matern32_add"  # baseline | lin_matern32_add
    acq_mode: str = "portfolio_mixed_pool"
    selection_mode: str = "robust_consensus"  # robust_consensus | single_seed
    consensus_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    consensus_min_successes: int = 3
    consensus_tie_break: str = "highest_mean_submission_score"

    # Portfolio / ensemble
    portfolio_primary: str = "primary_pareto_sigma"
    portfolio_when_stagnant: str = "trust_region_ts"
    enable_model_ensemble: bool = False
    ensemble_kernel_modes: tuple[str, ...] = ("lin_matern32_add", "baseline")
    enable_robust_acq_noise: bool = False
    robust_acq_noise_frac: float = 0.02

    # Candidate generation
    n_sobol: int = 4096
    n_sobol_global: int = 3072
    n_sobol_tr: int = 1024
    n_sobol_elite: int = 512
    topk_per_acq: int = 32
    ucb_betas: tuple[float, ...] = (
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
    )
    ts_n_draws: int = 1
    use_trust_region: bool = True
    elite_frac: float = 0.2
    elite_margin: float = 0.05
    tr_init_length: float = 0.35
    tr_min_length: float = 0.05
    tr_max_length: float = 0.80
    tr_success_tolerance: int = 3
    tr_failure_tolerance: int = 3
    tr_stagnation_window: int = 6
    tr_stagnation_rel_tol: float = 0.02
    tr_stagnation_abs_floor: float = 1e-15
    stagnation_hybrid_ucb_rel_tol: float = 0.05

    # Portal / rounding
    decimals: int = 6
    lower: float = 0.0
    upper: float = 0.999999

    # Fit robustness
    fit_retry_max_attempts: int = 7

    # Boundary-hugging control for higher-dimensional functions.
    boundary_penalty_weight: float = 0.10
    boundary_penalty_min_dim: int = 4
    boundary_margin: float = 0.02

    # Optional overrides
    per_function_overrides: Mapping[str, Mapping[str, object]] = field(default_factory=_default_per_function_overrides)


def sigma_quantile_schedule(n_points: int) -> float:
    if n_points < 20:
        return 0.50
    if n_points < 35:
        return 0.35
    return 0.20
