from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from acquisition.scores import compute_acq_scores
from candidates.pool import build_candidate_pool
from candidates.selectors import select_portfolio
from candidates.trust_region import reconstruct_tr_state
from config import RebuildConfig
from data_io import list_function_csvs, load_function_dataset
from models.warped_gp import fit_warped_gp
from reporting.export_csv import write_report_csvs
from rounding import dedup_after_rounding, format_portal_string, quantize_array_floor
from stagnation import compute_stagnation_stats


@dataclass
class FunctionRunContext:
    function: str
    d: int
    n: int
    report_dir: Path


@dataclass
class FunctionSeedResult:
    success: bool
    function: str
    seed: int
    error_type: str | None
    error_message: str | None
    candidate_rows: list[dict]
    recommendation_rows: list[dict]
    portal_row: dict | None
    diagnostics: dict | None
    submission_portal_string: str | None
    submission_score: float | None
    submission_mu_t: float | None
    warning_counts: dict[str, int]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


def _nn_stats(X: np.ndarray) -> tuple[float, float]:
    n = X.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    dists = []
    for i in range(n):
        di = np.linalg.norm(X - X[i], axis=1)
        di[i] = np.inf
        dists.append(di.min())
    dists = np.asarray(dists, dtype=float)
    return float(np.min(dists)), float(np.median(dists))


def _boundary_prox(X: np.ndarray) -> np.ndarray:
    return np.min(np.minimum(X, 1.0 - X), axis=1)


def _boundary_penalty(
    X: np.ndarray,
    *,
    d: int,
    weight: float,
    min_dim: int,
    margin: float,
) -> np.ndarray:
    if weight <= 0.0 or d < int(min_dim):
        return np.zeros(X.shape[0], dtype=float)
    m = float(max(0.0, margin))
    if m <= 0.0:
        return np.zeros(X.shape[0], dtype=float)
    prox = _boundary_prox(X)
    # Linear ramp: full penalty at boundary, zero by `margin`.
    return float(weight) * np.clip((m - prox) / m, 0.0, 1.0)


def _nearest_distance(candidates: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if observed.shape[0] == 0:
        return np.full(candidates.shape[0], fill_value=np.nan)
    out = np.empty(candidates.shape[0], dtype=float)
    for i, x in enumerate(candidates):
        out[i] = float(np.min(np.linalg.norm(observed - x, axis=1)))
    return out


def _make_random_nonduplicate(
    *,
    observed: torch.Tensor,
    d: int,
    cfg: RebuildConfig,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    obs_q = quantize_array_floor(
        observed.detach().cpu().numpy(),
        decimals=cfg.decimals,
        lower=cfg.lower,
        upper=cfg.upper,
    )
    obs_set = {tuple(r.tolist()) for r in obs_q}

    for _ in range(20000):
        x = rng.random(d)
        q = quantize_array_floor(x, decimals=cfg.decimals, lower=cfg.lower, upper=cfg.upper)
        if tuple(q.tolist()) not in obs_set:
            return torch.tensor(x, dtype=dtype, device=device).unsqueeze(0)

    # Last-resort deterministic fallback.
    x = np.full(d, 0.5, dtype=float)
    return torch.tensor(x, dtype=dtype, device=device).unsqueeze(0)


def _apply_function_overrides(cfg: RebuildConfig, function: str) -> RebuildConfig:
    overrides = cfg.per_function_overrides.get(function, {})
    if not overrides:
        return cfg

    valid = {f.name for f in dataclasses.fields(RebuildConfig)}
    clean = {k: v for k, v in overrides.items() if k in valid}
    return dataclasses.replace(cfg, **clean)


def _count_warning_types(records: list[warnings.WarningMessage]) -> dict[str, int]:
    out = {
        "optimization_warning": 0,
        "negative_variance_warning": 0,
        "other_warning": 0,
    }
    for r in records:
        cat = getattr(r.category, "__name__", str(r.category))
        msg = str(r.message)
        if "OptimizationWarning" in cat or "OptimizationWarning" in msg:
            out["optimization_warning"] += 1
        elif "Negative variance values detected" in msg:
            out["negative_variance_warning"] += 1
        else:
            out["other_warning"] += 1
    return out


def _stagnation_flag(y_raw: torch.Tensor, cfg: RebuildConfig) -> bool:
    return bool(compute_stagnation_stats(y_raw, cfg).flag)


def _run_one_function(
    *,
    ds,
    local_cfg: RebuildConfig,
    func_seed: int,
    report_dir: Path,
) -> FunctionSeedResult:
    np.random.seed(func_seed)
    torch.manual_seed(func_seed)

    ctx = FunctionRunContext(
        function=ds.function_id,
        d=ds.d,
        n=ds.n,
        report_dir=report_dir / ds.function_id,
    )

    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")

        fitted = fit_warped_gp(ds.X, ds.y, local_cfg)
        model = fitted.model
        stagnation = compute_stagnation_stats(ds.y, local_cfg)
        stagnation_flag = bool(stagnation.flag)
        tr_state = reconstruct_tr_state(
            ds.X.detach().cpu().numpy(),
            ds.y.detach().cpu().numpy(),
            local_cfg,
        )

        pool = build_candidate_pool(
            model=model,
            y_t=fitted.y_t,
            X_obs=ds.X,
            y_obs=ds.y,
            d=ctx.d,
            cfg=local_cfg,
            seed=func_seed + 7,
            device=ds.X.device,
            dtype=ds.X.dtype,
            tr_state=tr_state,
        )

        X_pool = pool.X_pool
        scores = pool.scores
        candidate_origin_pool = pool.candidate_origin

        # Submission score for shadow comparison (old-style): mu + 0.5*sigma
        shadow_score = scores.mu_t + 0.5 * scores.sigma_t

        dedup = dedup_after_rounding(
            candidates=X_pool,
            observed=ds.X,
            scores=shadow_score,
            decimals=local_cfg.decimals,
            lower=local_cfg.lower,
            upper=local_cfg.upper,
        )

        keep_idx = np.asarray(dedup.keep_indices, dtype=int)

        if keep_idx.size == 0:
            X_keep = _make_random_nonduplicate(
                observed=ds.X,
                d=ctx.d,
                cfg=local_cfg,
                seed=func_seed + 99,
                device=ds.X.device,
                dtype=ds.X.dtype,
            )
            rescored = compute_acq_scores(
                model=model,
                y_t=fitted.y_t,
                X_pool=X_keep,
                ucb_betas=local_cfg.ucb_betas,
                seed=func_seed + 11,
                n_ts_draws=int(local_cfg.ts_n_draws),
                robust_noise_frac=float(local_cfg.robust_acq_noise_frac) if bool(local_cfg.enable_robust_acq_noise) else 0.0,
            )

            mu_t = rescored.mu_t
            sigma_t = rescored.sigma_t
            ei = rescored.ei
            log_ei = rescored.log_ei
            pi = rescored.pi
            ucb = rescored.ucb_max
            ucb_beta = rescored.ucb_max_beta
            ts = rescored.ts_max
            X_keep_np = X_keep.detach().cpu().numpy()
            candidate_origin = np.array(["fallback"], dtype=object)
            boundary_penalty = _boundary_penalty(
                X_keep_np,
                d=ctx.d,
                weight=float(local_cfg.boundary_penalty_weight),
                min_dim=int(local_cfg.boundary_penalty_min_dim),
                margin=float(local_cfg.boundary_margin),
            )
            mu_t_adj = mu_t - boundary_penalty
            nn_dist = _nearest_distance(X_keep_np, ds.X.detach().cpu().numpy())
            nn_min, nn_med = _nn_stats(ds.X.detach().cpu().numpy())
            novelty_ratio = nn_dist / nn_med if np.isfinite(nn_med) and nn_med > 0 else np.full_like(nn_dist, np.nan)
            novelty_ratio_safe = np.nan_to_num(novelty_ratio, nan=0.0, posinf=0.0, neginf=0.0)
            exploration_score = sigma_t * novelty_ratio_safe
            pareto_sel, strategy_indices = select_portfolio(
                cfg=local_cfg,
                n_points=ctx.n,
                mu_t=mu_t,
                sigma_t=sigma_t,
                ei=ei,
                pi=pi,
                ucb=ucb,
                ts=ts,
                exploration_score=exploration_score,
                stagnation_flag=stagnation_flag,
                candidate_origin=candidate_origin,
                mu_t_tiebreak=mu_t_adj,
                stagnation_scale=float(stagnation.scale),
            )
            selected_local = int(pareto_sel.selected_index)
            shadow_local = int(np.argmax(mu_t + 0.5 * sigma_t))
            source_pool_idx = np.array([-1], dtype=int)
        else:
            X_keep = X_pool[keep_idx]
            mu_t = scores.mu_t[keep_idx]
            sigma_t = scores.sigma_t[keep_idx]
            ei = scores.ei[keep_idx]
            log_ei = scores.log_ei[keep_idx]
            pi = scores.pi[keep_idx]
            ucb = scores.ucb_max[keep_idx]
            ucb_beta = scores.ucb_max_beta[keep_idx]
            ts = scores.ts_max[keep_idx]
            source_pool_idx = keep_idx.copy()
            X_keep_np = X_keep.detach().cpu().numpy()
            candidate_origin = candidate_origin_pool[keep_idx]
            boundary_penalty = _boundary_penalty(
                X_keep_np,
                d=ctx.d,
                weight=float(local_cfg.boundary_penalty_weight),
                min_dim=int(local_cfg.boundary_penalty_min_dim),
                margin=float(local_cfg.boundary_margin),
            )
            mu_t_adj = mu_t - boundary_penalty
            nn_dist = _nearest_distance(X_keep_np, ds.X.detach().cpu().numpy())
            nn_min, nn_med = _nn_stats(ds.X.detach().cpu().numpy())
            novelty_ratio = nn_dist / nn_med if np.isfinite(nn_med) and nn_med > 0 else np.full_like(nn_dist, np.nan)
            novelty_ratio_safe = np.nan_to_num(novelty_ratio, nan=0.0, posinf=0.0, neginf=0.0)
            exploration_score = sigma_t * novelty_ratio_safe
            pareto_sel, strategy_indices = select_portfolio(
                cfg=local_cfg,
                n_points=ctx.n,
                mu_t=mu_t,
                sigma_t=sigma_t,
                ei=ei,
                pi=pi,
                ucb=ucb,
                ts=ts,
                exploration_score=exploration_score,
                stagnation_flag=stagnation_flag,
                candidate_origin=candidate_origin,
                mu_t_tiebreak=mu_t_adj,
                stagnation_scale=float(stagnation.scale),
            )
            selected_local = int(pareto_sel.selected_index)
            shadow_local = int(np.argmax(mu_t + 0.5 * sigma_t))

        X_keep_q = quantize_array_floor(X_keep_np, decimals=local_cfg.decimals, lower=local_cfg.lower, upper=local_cfg.upper)

        mu_raw_est = fitted.y_transform.inverse_transform(mu_t)
        nn_dist = _nearest_distance(X_keep_np, ds.X.detach().cpu().numpy())
        nn_min, nn_med = _nn_stats(ds.X.detach().cpu().numpy())
        novelty_ratio = nn_dist / nn_med if np.isfinite(nn_med) and nn_med > 0 else np.full_like(nn_dist, np.nan)
        novelty_ratio_safe = np.nan_to_num(novelty_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        exploration_score = sigma_t * novelty_ratio_safe
        bprox = _boundary_prox(X_keep_np)

        pareto_set = set(int(i_) for i_ in pareto_sel.pareto_indices.tolist())

        def _portal(i_local: int) -> str:
            return format_portal_string(
                X_keep_np[i_local],
                decimals=local_cfg.decimals,
                lower=local_cfg.lower,
                upper=local_cfg.upper,
            )

        submission_idx = selected_local
        submission_portal = _portal(submission_idx)

        portal_row = {
            "function": ctx.function,
            "candidate": "chosen_submission",
            "portal_string": submission_portal,
            "seed": int(func_seed),
        }

        # Recommendations table includes strategy anchors.
        strategy_indices = {
            "chosen_submission": submission_idx,
            "primary_pareto_sigma": int(strategy_indices.get("primary_pareto_sigma", submission_idx)),
            "trust_region_ts": int(strategy_indices.get("trust_region_ts", submission_idx)),
            "global_explore": int(strategy_indices.get("global_explore", submission_idx)),
            "best_mu": int(np.argmax(mu_t)),
            "best_sigma": int(np.argmax(sigma_t)),
            "best_logEI": int(np.argmax(log_ei)),
            "best_PI": int(np.argmax(pi)),
            "best_UCB": int(np.argmax(ucb)),
            "best_exploration": int(np.argmax(exploration_score)),
            "shadow_oldstyle": shadow_local,
        }

        recommendation_rows: list[dict] = []
        for strategy, idx_local in strategy_indices.items():
            recommendation_rows.append(
                {
                    "function": ctx.function,
                    "recommendation_type": strategy,
                    "candidate_label": f"pool_{idx_local:06d}",
                    "portal_string": _portal(idx_local),
                    "mu_t": float(mu_t[idx_local]),
                    "sigma_t": float(sigma_t[idx_local]),
                    "ei": float(ei[idx_local]),
                    "log_ei": float(log_ei[idx_local]),
                    "pi": float(pi[idx_local]),
                    "ucb": float(ucb[idx_local]),
                    "ucb_beta": float(ucb_beta[idx_local]),
                    "ts": float(ts[idx_local]),
                    "exploration_score": float(exploration_score[idx_local]),
                    "is_submission": bool(strategy == "chosen_submission"),
                    "seed": int(func_seed),
                    "candidate_origin": str(candidate_origin[idx_local]),
                    "stagnation_flag": bool(stagnation_flag),
                    "stagnation_gain": float(stagnation.gain),
                    "stagnation_threshold": float(stagnation.threshold),
                    "stagnation_scale": float(stagnation.scale),
                    "why": (
                        f"mu_t={mu_t[idx_local]:.6g}, sigma_t={sigma_t[idx_local]:.6g}, "
                        f"ei={ei[idx_local]:.6g}, pi={pi[idx_local]:.6g}, "
                        f"ucb={ucb[idx_local]:.6g}, beta={ucb_beta[idx_local]:.2f}, ts={ts[idx_local]:.6g}, "
                        f"explore={exploration_score[idx_local]:.6g}, "
                        f"stagnation_gain={stagnation.gain:.6g}, threshold={stagnation.threshold:.6g}"
                    ),
                }
            )

        submission_score = mu_t + 0.5 * sigma_t

        candidate_rows: list[dict] = []
        for j in range(X_keep_np.shape[0]):
            row = {
                "function": ctx.function,
                "candidate_label": f"pool_{j:06d}",
                "source_pool_index": int(source_pool_idx[j]) if j < len(source_pool_idx) else -1,
                "d": ctx.d,
                "n": ctx.n,
                "portal_string": _portal(j),
                "mu_t": float(mu_t[j]),
                "mu_t_tiebreak": float(mu_t_adj[j]),
                "sigma_t": float(sigma_t[j]),
                "mu_raw_est": float(mu_raw_est[j]),
                "ei": float(ei[j]),
                "log_ei": float(log_ei[j]),
                "pi": float(pi[j]),
                "ucb": float(ucb[j]),
                "ucb_best_beta": float(ucb_beta[j]),
                "ts_score": float(ts[j]),
                "is_pareto": bool(j in pareto_set),
                "pareto_rank": int(pareto_sel.pareto_ranks[j]),
                "novelty_nn_dist": float(nn_dist[j]),
                "novelty_ratio": float(novelty_ratio[j]) if np.isfinite(novelty_ratio[j]) else np.nan,
                "exploration_score": float(exploration_score[j]),
                "boundary_prox": float(bprox[j]),
                "boundary_penalty": float(boundary_penalty[j]),
                "flag_boundary": bool(bprox[j] < 0.01),
                "duplicate_after_rounding": False,
                "is_submission": bool(j == submission_idx),
                "is_shadow_oldstyle": bool(j == shadow_local),
                "submission_score": float(submission_score[j]),
                "sigma_quantile_used": float(pareto_sel.sigma_quantile_used),
                "sigma_threshold": float(pareto_sel.sigma_threshold),
                "pareto_size": int(len(pareto_sel.pareto_indices)),
                "eligible_count": int(X_keep_np.shape[0]),
                "candidate_origin": str(candidate_origin[j]),
                "in_trust_region": bool(candidate_origin[j] == "tr"),
                "tr_length": float(tr_state.length) if tr_state is not None else np.nan,
                "stagnation_flag": bool(stagnation_flag),
                "stagnation_gain": float(stagnation.gain),
                "stagnation_threshold": float(stagnation.threshold),
                "stagnation_scale": float(stagnation.scale),
                "seed": int(func_seed),
            }
            for k in range(ctx.d):
                row[f"x{k}"] = float(X_keep_np[j, k])
                row[f"x{k}_q"] = float(X_keep_q[j, k])

            candidate_rows.append(row)

        d_payload = fitted.diagnostics()
        d_payload.update(
            {
                "function": ctx.function,
                "d": ctx.d,
                "n": ctx.n,
                "kernel_mode": local_cfg.kernel_mode,
                "acq_mode": local_cfg.acq_mode,
                "ucb_betas": list(local_cfg.ucb_betas),
                "n_sobol": int(local_cfg.n_sobol),
                "topk_per_acq": int(local_cfg.topk_per_acq),
                "pool_size": int(X_pool.shape[0]),
                "eligible_count": int(X_keep_np.shape[0]),
                "dropped_duplicate_internal": int(len(dedup.dropped_duplicate_internal)),
                "dropped_duplicate_observed": int(len(dedup.dropped_duplicate_observed)),
                "pareto_size": int(len(pareto_sel.pareto_indices)),
                "selected_local_index": int(submission_idx),
                "selected_source_pool_index": int(source_pool_idx[submission_idx]) if submission_idx < len(source_pool_idx) else -1,
                "shadow_oldstyle_local_index": int(shadow_local),
                "sigma_quantile_used": float(pareto_sel.sigma_quantile_used),
                "sigma_threshold": float(pareto_sel.sigma_threshold),
                "boundary_penalty_weight": float(local_cfg.boundary_penalty_weight),
                "boundary_penalty_min_dim": int(local_cfg.boundary_penalty_min_dim),
                "boundary_margin": float(local_cfg.boundary_margin),
                "selected_boundary_penalty": float(boundary_penalty[submission_idx]),
                "selected_exploration_score": float(exploration_score[submission_idx]),
                "selected_strategy_name": str(pareto_sel.strategy_name),
                "selected_candidate_origin": str(candidate_origin[submission_idx]),
                "stagnation_flag": bool(stagnation_flag),
                "stagnation_gain": float(stagnation.gain),
                "stagnation_threshold": float(stagnation.threshold),
                "stagnation_scale": float(stagnation.scale),
                "stagnation_window": int(stagnation.window),
                "trust_region_length": float(tr_state.length) if tr_state is not None else np.nan,
                "trust_region_restarts": int(tr_state.n_restarts) if tr_state is not None else 0,
                "best_y_raw": float(ds.y.max().detach().cpu().item()),
                "best_y_transformed": float(fitted.y_t.max().detach().cpu().item()),
                "nn_min_observed": float(nn_min) if np.isfinite(nn_min) else np.nan,
                "nn_median_observed": float(nn_med) if np.isfinite(nn_med) else np.nan,
                "seed": int(func_seed),
            }
        )

        warn_counts = _count_warning_types(wlog)

        return FunctionSeedResult(
            success=True,
            function=ctx.function,
            seed=int(func_seed),
            error_type=None,
            error_message=None,
            candidate_rows=candidate_rows,
            recommendation_rows=recommendation_rows,
            portal_row=portal_row,
            diagnostics=d_payload,
            submission_portal_string=submission_portal,
            submission_score=float(submission_score[submission_idx]),
            submission_mu_t=float(mu_t[submission_idx]),
            warning_counts=warn_counts,
        )


def _run_single_seed(cfg: RebuildConfig, seed_value: int, report_dir: Path) -> dict[str, FunctionSeedResult]:
    dtype = _torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    out: dict[str, FunctionSeedResult] = {}
    csvs = list_function_csvs(Path(cfg.data_dir))

    for i, csv_path in enumerate(csvs):
        ds = load_function_dataset(csv_path, dtype=dtype)
        local_cfg = _apply_function_overrides(cfg, ds.function_id)
        local_cfg = dataclasses.replace(local_cfg, seed=int(seed_value))
        func_seed = int(local_cfg.seed + 1000 * i)

        try:
            out[ds.function_id] = _run_one_function(
                ds=ds,
                local_cfg=local_cfg,
                func_seed=func_seed,
                report_dir=report_dir,
            )
        except Exception as exc:
            out[ds.function_id] = FunctionSeedResult(
                success=False,
                function=ds.function_id,
                seed=int(func_seed),
                error_type=type(exc).__name__,
                error_message=str(exc),
                candidate_rows=[],
                recommendation_rows=[],
                portal_row=None,
                diagnostics={
                    "function": ds.function_id,
                    "fit_failed": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "seed": int(func_seed),
                },
                submission_portal_string=None,
                submission_score=None,
                submission_mu_t=None,
                warning_counts={
                    "optimization_warning": 0,
                    "negative_variance_warning": 0,
                    "other_warning": 0,
                },
            )

    return out


def _consensus_pick(rows: list[dict], tie_break: str) -> tuple[str, int, float, float]:
    # rows fields: portal_string, submission_score, mu_t
    by_portal: dict[str, list[dict]] = {}
    for r in rows:
        by_portal.setdefault(str(r["portal_string"]), []).append(r)

    ranked = []
    for portal, grp in by_portal.items():
        freq = len(grp)
        mean_score = float(np.mean([g["submission_score"] for g in grp]))
        mean_mu = float(np.mean([g["mu_t"] for g in grp]))
        ranked.append((portal, freq, mean_score, mean_mu))

    # Frequency first.
    ranked.sort(key=lambda x: (-x[1], x[0]))
    top_freq = ranked[0][1]
    tied = [r for r in ranked if r[1] == top_freq]

    if len(tied) == 1:
        return tied[0]

    if tie_break == "highest_mean_submission_score":
        tied.sort(key=lambda x: (-x[2], -x[3], x[0]))
    else:
        tied.sort(key=lambda x: (-x[3], -x[2], x[0]))
    return tied[0]


def run_weekly(
    cfg: RebuildConfig,
    write_outputs: bool = True,
    write_dashboard: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    data_dir = Path(cfg.data_dir)
    report_dir = Path(cfg.reports_dir) / cfg.report_date

    if report_dir.exists() and not cfg.overwrite and write_outputs:
        raise SystemExit(
            f"Report folder already exists: {report_dir}\n"
            "Use a new --report_date or pass --overwrite."
        )

    all_functions = [p.stem for p in list_function_csvs(data_dir)]

    if cfg.selection_mode == "single_seed":
        seed_runs = {int(cfg.seed): _run_single_seed(cfg=cfg, seed_value=int(cfg.seed), report_dir=report_dir)}
        consensus_seeds = [int(cfg.seed)]
        min_success = 1
    else:
        consensus_seeds = [int(s) for s in cfg.consensus_seeds]
        seed_runs = {
            s: _run_single_seed(cfg=cfg, seed_value=s, report_dir=report_dir)
            for s in consensus_seeds
        }
        min_success = max(1, int(cfg.consensus_min_successes))

    candidate_rows: list[dict] = []
    recommendation_rows: list[dict] = []
    portal_rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    seed_suggestions_rows: list[dict] = []
    stability_rows: list[dict] = []

    run_health = {
        "selection_mode": cfg.selection_mode,
        "consensus_seeds": consensus_seeds,
        "consensus_min_successes": min_success,
        "consensus_tie_break": cfg.consensus_tie_break,
        "seed_status": {},
        "global_warning_counts": {
            "optimization_warning": 0,
            "negative_variance_warning": 0,
            "other_warning": 0,
        },
        "functions_below_min_success": [],
    }

    for s in consensus_seeds:
        runs = seed_runs[s]
        seed_ok = sum(1 for fn in all_functions if runs[fn].success)
        seed_fail = len(all_functions) - seed_ok
        warn = {
            "optimization_warning": sum(r.warning_counts["optimization_warning"] for r in runs.values()),
            "negative_variance_warning": sum(r.warning_counts["negative_variance_warning"] for r in runs.values()),
            "other_warning": sum(r.warning_counts["other_warning"] for r in runs.values()),
        }
        run_health["seed_status"][str(s)] = {
            "n_function_success": int(seed_ok),
            "n_function_fail": int(seed_fail),
            "warning_counts": warn,
            "failures": {
                fn: {"error_type": r.error_type, "error_message": r.error_message}
                for fn, r in runs.items()
                if not r.success
            },
        }
        for k in run_health["global_warning_counts"]:
            run_health["global_warning_counts"][k] += int(warn[k])

    for fn in all_functions:
        per_seed = []
        for s in consensus_seeds:
            r = seed_runs[s][fn]
            seed_suggestions_rows.append(
                {
                    "function": fn,
                    "seed": int(s),
                    "status": "success" if r.success else "failed",
                    "portal_string": r.submission_portal_string if r.success else "",
                    "submission_score": r.submission_score if r.success else np.nan,
                    "mu_t": r.submission_mu_t if r.success else np.nan,
                    "error_type": r.error_type if not r.success else "",
                    "error_message": r.error_message if not r.success else "",
                }
            )
            if r.success:
                per_seed.append(
                    {
                        "seed": int(s),
                        "portal_string": str(r.submission_portal_string),
                        "submission_score": float(r.submission_score),
                        "mu_t": float(r.submission_mu_t),
                        "result": r,
                    }
                )

        n_success = len(per_seed)
        n_fail = len(consensus_seeds) - n_success
        unique_suggestions = len(set(p["portal_string"] for p in per_seed)) if per_seed else 0

        selected_result = None
        selected_portal = ""
        consensus_frequency = 0
        fallback_used = False

        if n_success == 0:
            run_health["functions_below_min_success"].append(fn)
            stability_rows.append(
                {
                    "function": fn,
                    "n_seed_success": 0,
                    "n_seed_fail": int(n_fail),
                    "unique_suggestions": 0,
                    "consensus_frequency": 0,
                    "consensus_frequency_ratio": 0.0,
                    "selected_portal_string": "",
                    "selected_seed": -1,
                    "fallback_used": True,
                    "fit_attempt_distribution": "{}",
                    "selection_mode": cfg.selection_mode,
                }
            )
            diagnostics[fn] = {
                "function": fn,
                "fit_failed": True,
                "error_type": "ModelFittingError",
                "error_message": "All seeds failed for this function.",
                "selection_mode": cfg.selection_mode,
            }
            portal_rows.append({"function": fn, "candidate": "chosen_submission", "portal_string": ""})
            continue

        # Consensus / fallback selection.
        if n_success < min_success:
            fallback_used = True
            run_health["functions_below_min_success"].append(fn)
            seed0 = next((p for p in per_seed if p["seed"] == int(cfg.seed)), None)
            pick = seed0 if seed0 is not None else per_seed[0]
            selected_result = pick["result"]
            selected_portal = pick["portal_string"]
            consensus_frequency = sum(1 for p in per_seed if p["portal_string"] == selected_portal)
        else:
            selected_portal, consensus_frequency, _, _ = _consensus_pick(per_seed, cfg.consensus_tie_break)
            candidates = [p for p in per_seed if p["portal_string"] == selected_portal]
            candidates.sort(key=lambda x: (-x["submission_score"], -x["mu_t"], x["seed"]))
            selected_result = candidates[0]["result"]

        assert selected_result is not None

        # Representative rows come from selected_result seed run.
        candidate_rows.extend(selected_result.candidate_rows)
        recommendation_rows.extend(selected_result.recommendation_rows)

        portal_rows.append(
            {
                "function": fn,
                "candidate": "chosen_submission",
                "portal_string": selected_portal,
            }
        )

        # mark any mismatch in representative recommendation rows.
        for rr in recommendation_rows[-len(selected_result.recommendation_rows):]:
            if rr["function"] == fn:
                rr["is_submission"] = bool(
                    rr["recommendation_type"] == "chosen_submission" and rr["portal_string"] == selected_portal
                )
                rr["selection_mode"] = cfg.selection_mode
                rr["consensus_frequency"] = int(consensus_frequency)
                rr["n_seed_success"] = int(n_success)

        fit_attempt_dist: dict[str, int] = {}
        for p in per_seed:
            d = p["result"].diagnostics or {}
            label = str(d.get("fit_attempt_label", "unknown"))
            fit_attempt_dist[label] = fit_attempt_dist.get(label, 0) + 1

        diag_payload = dict(selected_result.diagnostics or {})
        diag_payload.update(
            {
                "selection_mode": cfg.selection_mode,
                "n_seed_success": int(n_success),
                "n_seed_fail": int(n_fail),
                "unique_suggestions": int(unique_suggestions),
                "consensus_frequency": int(consensus_frequency),
                "consensus_frequency_ratio": float(consensus_frequency / n_success),
                "selected_portal_string": selected_portal,
                "selected_seed": int(selected_result.seed),
                "fallback_used": bool(fallback_used),
                "fit_attempt_distribution": fit_attempt_dist,
            }
        )
        diagnostics[fn] = diag_payload

        stability_rows.append(
            {
                "function": fn,
                "n_seed_success": int(n_success),
                "n_seed_fail": int(n_fail),
                "unique_suggestions": int(unique_suggestions),
                "consensus_frequency": int(consensus_frequency),
                "consensus_frequency_ratio": float(consensus_frequency / n_success),
                "selected_portal_string": selected_portal,
                "selected_seed": int(selected_result.seed),
                "fallback_used": bool(fallback_used),
                "fit_attempt_distribution": json.dumps(fit_attempt_dist, sort_keys=True),
                "selection_mode": cfg.selection_mode,
            }
        )

    candidates_df = pd.DataFrame(candidate_rows)
    recommendations_df = pd.DataFrame(recommendation_rows)
    portal_df = pd.DataFrame(portal_rows).sort_values("function").reset_index(drop=True)
    seed_suggestions_df = pd.DataFrame(seed_suggestions_rows).sort_values(["function", "seed"]).reset_index(drop=True)
    stability_df = pd.DataFrame(stability_rows).sort_values("function").reset_index(drop=True)

    run_health["global_function_fail_rate"] = float(
        sum(x["n_function_fail"] for x in run_health["seed_status"].values())
        / max(1, len(consensus_seeds) * len(all_functions))
    )

    if write_outputs:
        write_report_csvs(
            report_dir=report_dir,
            candidate_summary=candidates_df,
            recommendations=recommendations_df,
            portal_strings=portal_df,
            diagnostics_by_function=diagnostics,
            write_dashboard=write_dashboard,
            seed_suggestions=seed_suggestions_df,
            stability_summary=stability_df,
            run_health=run_health,
        )

    return candidates_df, recommendations_df, portal_df, diagnostics


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CapstoneBO+ trust-region and portfolio weekly pack")
    default_root = Path(__file__).resolve().parent
    ap.add_argument("--data_dir", default=str(default_root / "data"))
    ap.add_argument("--reports_dir", default=str(default_root / "reports"))
    ap.add_argument("--report_date", default=date.today().isoformat())
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    ap.add_argument("--use_output_power_transform", dest="use_output_power_transform", action="store_true")
    ap.add_argument("--no_output_power_transform", dest="use_output_power_transform", action="store_false")
    ap.set_defaults(use_output_power_transform=True)
    ap.add_argument("--output_transform_mode", choices=["power", "copula"], default="power")

    ap.add_argument("--use_input_warp", dest="use_input_warp", action="store_true")
    ap.add_argument("--no_input_warp", dest="use_input_warp", action="store_false")
    ap.set_defaults(use_input_warp=True)

    ap.add_argument("--kernel_mode", choices=["baseline", "lin_matern32_add"], default="lin_matern32_add")
    ap.add_argument("--acq_mode", default="mace_pareto_pool")

    ap.add_argument("--selection_mode", choices=["robust_consensus", "single_seed"], default="robust_consensus")
    ap.add_argument("--consensus_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--consensus_min_successes", type=int, default=3)
    ap.add_argument("--portfolio_primary", default="primary_pareto_sigma")
    ap.add_argument("--portfolio_when_stagnant", default="stagnation_hybrid")

    ap.add_argument("--n_sobol", type=int, default=4096)
    ap.add_argument("--n_sobol_global", type=int, default=3072)
    ap.add_argument("--n_sobol_tr", type=int, default=1024)
    ap.add_argument("--n_sobol_elite", type=int, default=512)
    ap.add_argument("--topk_per_acq", type=int, default=32)
    ap.add_argument("--ts_n_draws", type=int, default=1)
    ap.add_argument("--tr_stagnation_window", type=int, default=6)
    ap.add_argument("--tr_stagnation_rel_tol", type=float, default=0.02)
    ap.add_argument("--tr_stagnation_abs_floor", type=float, default=1e-15)
    ap.add_argument("--stagnation_hybrid_ucb_rel_tol", type=float, default=0.05)
    ap.add_argument("--decimals", type=int, default=6)
    ap.add_argument("--boundary_penalty_weight", type=float, default=0.10)
    ap.add_argument("--boundary_penalty_min_dim", type=int, default=4)
    ap.add_argument("--boundary_margin", type=float, default=0.02)
    ap.add_argument("--enable_robust_acq_noise", action="store_true")
    ap.add_argument("--enable_model_ensemble", action="store_true")

    ap.add_argument("--no_dashboard", action="store_true", help="Skip weekly_dashboard.html generation")

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    cfg = RebuildConfig(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        report_date=args.report_date,
        overwrite=bool(args.overwrite),
        seed=int(args.seed),
        dtype=args.dtype,
        use_output_power_transform=bool(args.use_output_power_transform),
        output_transform_mode=args.output_transform_mode,
        use_input_warp=bool(args.use_input_warp),
        kernel_mode=args.kernel_mode,
        acq_mode=args.acq_mode,
        selection_mode=args.selection_mode,
        consensus_seeds=tuple(int(x) for x in args.consensus_seeds),
        consensus_min_successes=int(args.consensus_min_successes),
        portfolio_primary=args.portfolio_primary,
        portfolio_when_stagnant=args.portfolio_when_stagnant,
        enable_model_ensemble=bool(args.enable_model_ensemble),
        enable_robust_acq_noise=bool(args.enable_robust_acq_noise),
        n_sobol=int(args.n_sobol),
        n_sobol_global=int(args.n_sobol_global),
        n_sobol_tr=int(args.n_sobol_tr),
        n_sobol_elite=int(args.n_sobol_elite),
        topk_per_acq=int(args.topk_per_acq),
        ts_n_draws=int(args.ts_n_draws),
        tr_stagnation_window=int(args.tr_stagnation_window),
        tr_stagnation_rel_tol=float(args.tr_stagnation_rel_tol),
        tr_stagnation_abs_floor=float(args.tr_stagnation_abs_floor),
        stagnation_hybrid_ucb_rel_tol=float(args.stagnation_hybrid_ucb_rel_tol),
        decimals=int(args.decimals),
        boundary_penalty_weight=float(args.boundary_penalty_weight),
        boundary_penalty_min_dim=int(args.boundary_penalty_min_dim),
        boundary_margin=float(args.boundary_margin),
    )

    run_weekly(
        cfg=cfg,
        write_outputs=True,
        write_dashboard=not bool(args.no_dashboard),
    )

    report_dir = Path(cfg.reports_dir) / cfg.report_date
    print(f"Wrote report to: {report_dir}")
    print("Key files:")
    print(f" - {report_dir / 'candidate_summary.csv'}")
    print(f" - {report_dir / 'recommendations.csv'}")
    print(f" - {report_dir / 'portal_strings.csv'}")
    print(f" - {report_dir / 'seed_suggestions.csv'}")
    print(f" - {report_dir / 'stability_summary.csv'}")
    print(f" - {report_dir / 'run_health.json'}")
    if not args.no_dashboard:
        print(f" - {report_dir / 'weekly_dashboard.html'}")


if __name__ == "__main__":
    main()
