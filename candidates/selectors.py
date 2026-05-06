from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acquisition.mace_pareto import pareto_front_indices, pareto_ranks
from config import RebuildConfig, sigma_quantile_schedule
from stagnation import scale_aware_tolerance


@dataclass
class SelectionResult:
    selected_index: int
    pareto_indices: np.ndarray
    pareto_ranks: np.ndarray
    sigma_quantile_used: float
    sigma_threshold: float
    candidate_set_after_sigma_gate: np.ndarray
    strategy_name: str = "primary_pareto_sigma"


def select_submission_from_pareto(
    *,
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    ei: np.ndarray,
    pi: np.ndarray,
    ucb: np.ndarray,
    n_points: int,
    mu_t_tiebreak: np.ndarray | None = None,
) -> SelectionResult:
    mu_t = np.asarray(mu_t, dtype=float)
    sigma_t = np.asarray(sigma_t, dtype=float)
    ei = np.asarray(ei, dtype=float)
    pi = np.asarray(pi, dtype=float)
    ucb = np.asarray(ucb, dtype=float)
    mu_tb = mu_t if mu_t_tiebreak is None else np.asarray(mu_t_tiebreak, dtype=float)

    objectives_min = np.column_stack((-ei, -pi, -ucb))
    pf = pareto_front_indices(objectives_min)
    ranks = pareto_ranks(objectives_min)

    q_sigma = float(sigma_quantile_schedule(int(n_points)))
    sigma_pf = sigma_t[pf]
    sigma_threshold = float(np.quantile(sigma_pf, q_sigma)) if sigma_pf.size else float("nan")
    gated = pf[sigma_t[pf] >= sigma_threshold] if sigma_pf.size else pf
    if gated.size == 0:
        gated = pf

    mu_best = np.max(mu_tb[gated])
    cand = gated[np.isclose(mu_tb[gated], mu_best)]
    if cand.size > 1:
        ei_best = np.max(ei[cand])
        cand = cand[np.isclose(ei[cand], ei_best)]
    if cand.size > 1:
        pi_best = np.max(pi[cand])
        cand = cand[np.isclose(pi[cand], pi_best)]
    selected = int(np.min(cand))

    return SelectionResult(
        selected_index=selected,
        pareto_indices=pf,
        pareto_ranks=ranks,
        sigma_quantile_used=q_sigma,
        sigma_threshold=sigma_threshold,
        candidate_set_after_sigma_gate=gated,
        strategy_name="primary_pareto_sigma",
    )


def select_portfolio(
    *,
    cfg: RebuildConfig,
    n_points: int,
    mu_t: np.ndarray,
    sigma_t: np.ndarray,
    ei: np.ndarray,
    pi: np.ndarray,
    ucb: np.ndarray,
    ts: np.ndarray,
    exploration_score: np.ndarray,
    stagnation_flag: bool,
    candidate_origin: np.ndarray,
    mu_t_tiebreak: np.ndarray | None = None,
    stagnation_scale: float = 0.0,
) -> tuple[SelectionResult, dict[str, int]]:
    primary = select_submission_from_pareto(
        mu_t=mu_t,
        sigma_t=sigma_t,
        ei=ei,
        pi=pi,
        ucb=ucb,
        n_points=n_points,
        mu_t_tiebreak=mu_t_tiebreak,
    )

    origin = np.asarray(candidate_origin, dtype=object)
    non_tr_idx = np.where(origin != "tr")[0]
    global_idx = non_tr_idx if non_tr_idx.size else np.arange(origin.shape[0])

    named = {
        "primary_pareto_sigma": int(primary.selected_index),
        "global_explore": int(global_idx[np.argmax(exploration_score[global_idx])]),
        "shadow_oldstyle": int(np.argmax(mu_t + 0.5 * sigma_t)),
    }
    tr_idx = np.where(origin == "tr")[0]
    named["trust_region_ts"] = int(tr_idx[np.argmax(ts[tr_idx])]) if tr_idx.size else int(np.argmax(ts))

    chosen_name = str(cfg.portfolio_when_stagnant if stagnation_flag else cfg.portfolio_primary)
    resolved_name = chosen_name
    if stagnation_flag and chosen_name == "stagnation_hybrid":
        if tr_idx.size == 0:
            resolved_name = "global_explore"
        elif non_tr_idx.size == 0:
            resolved_name = "trust_region_ts"
        else:
            hybrid_tol = scale_aware_tolerance(
                stagnation_scale,
                rel_tol=float(cfg.stagnation_hybrid_ucb_rel_tol),
                abs_floor=float(cfg.tr_stagnation_abs_floor),
            )
            tr_choice = int(named["trust_region_ts"])
            global_choice = int(named["global_explore"])
            resolved_name = "trust_region_ts" if float(ucb[tr_choice]) + hybrid_tol >= float(ucb[global_choice]) else "global_explore"

    primary.selected_index = int(named.get(resolved_name, primary.selected_index))
    primary.strategy_name = resolved_name
    return primary, named
