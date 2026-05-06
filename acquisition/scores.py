from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound

try:
    from botorch.acquisition import LogExpectedImprovement

    _HAS_LOG_EI = True
except Exception:
    LogExpectedImprovement = None
    _HAS_LOG_EI = False


@dataclass
class AcquisitionScores:
    mu_t: np.ndarray
    sigma_t: np.ndarray
    ei: np.ndarray
    log_ei: np.ndarray
    pi: np.ndarray
    ucb_by_beta: dict[float, np.ndarray]
    ucb_max: np.ndarray
    ucb_max_beta: np.ndarray
    maxvar: np.ndarray
    ts: np.ndarray
    ts_max: np.ndarray


def _eval_acqf(acqf, X_pool: torch.Tensor) -> np.ndarray:
    vals = acqf(X_pool.unsqueeze(-2))
    v = vals.detach().cpu().numpy().reshape(-1)
    v = np.asarray(v, dtype=float)
    return np.nan_to_num(v, nan=-1e30, posinf=1e30, neginf=-1e30)


def compute_acq_scores(
    model,
    y_t: torch.Tensor,
    X_pool: torch.Tensor,
    ucb_betas: tuple[float, ...],
    *,
    seed: int = 0,
    n_ts_draws: int = 1,
    robust_noise_frac: float = 0.0,
) -> AcquisitionScores:
    post = model.posterior(X_pool)
    mu_t = post.mean.detach().cpu().numpy().reshape(-1).astype(float)
    var_t = post.variance.detach().cpu().numpy().reshape(-1).astype(float)
    sigma_t = np.sqrt(np.maximum(var_t, 0.0))

    best_f_t = float(y_t.max().detach().cpu().item())

    if _HAS_LOG_EI and LogExpectedImprovement is not None:
        acq_logei = LogExpectedImprovement(model=model, best_f=best_f_t, maximize=True)
        log_ei = _eval_acqf(acq_logei, X_pool)
    else:
        acq_ei_fallback = ExpectedImprovement(model=model, best_f=best_f_t, maximize=True)
        ei_fallback = np.maximum(_eval_acqf(acq_ei_fallback, X_pool), 1e-30)
        log_ei = np.log(ei_fallback)

    if _HAS_LOG_EI and LogExpectedImprovement is not None:
        ei_raw = np.exp(np.clip(log_ei, -80.0, 80.0))
    else:
        acq_ei = ExpectedImprovement(model=model, best_f=best_f_t, maximize=True)
        ei_raw = _eval_acqf(acq_ei, X_pool)
        bad = ~np.isfinite(ei_raw) | (ei_raw <= 0)
        if np.any(bad):
            ei_raw[bad] = np.exp(np.clip(log_ei[bad], -80.0, 80.0))
        ei_raw = np.maximum(ei_raw, 0.0)

    acq_pi = ProbabilityOfImprovement(model=model, best_f=best_f_t, maximize=True)
    pi = np.clip(_eval_acqf(acq_pi, X_pool), 0.0, 1.0)

    ucb_by_beta: dict[float, np.ndarray] = {}
    for beta in ucb_betas:
        acq_ucb = UpperConfidenceBound(model=model, beta=float(beta), maximize=True)
        ucb_by_beta[float(beta)] = _eval_acqf(acq_ucb, X_pool)

    ucb_stack = np.column_stack([ucb_by_beta[b] for b in ucb_betas])
    max_pos = np.argmax(ucb_stack, axis=1)
    ucb_max = ucb_stack[np.arange(ucb_stack.shape[0]), max_pos]
    beta_arr = np.asarray(ucb_betas, dtype=float)
    ucb_max_beta = beta_arr[max_pos]

    rng = np.random.default_rng(int(seed))
    ts = mu_t + sigma_t * rng.standard_normal(size=mu_t.shape[0])
    if int(n_ts_draws) <= 1:
        ts_max = ts.copy()
    else:
        draws = rng.standard_normal(size=(int(n_ts_draws), mu_t.shape[0]))
        ts_max = np.max(mu_t[None, :] + sigma_t[None, :] * draws, axis=0)

    if robust_noise_frac > 0.0:
        scale = robust_noise_frac * max(1e-12, float(np.nanmedian(np.abs(ucb_max))))
        ucb_max = ucb_max + rng.normal(0.0, scale, size=ucb_max.shape[0])

    return AcquisitionScores(
        mu_t=mu_t,
        sigma_t=sigma_t,
        ei=ei_raw,
        log_ei=log_ei,
        pi=pi,
        ucb_by_beta=ucb_by_beta,
        ucb_max=ucb_max,
        ucb_max_beta=ucb_max_beta,
        maxvar=sigma_t,
        ts=ts,
        ts_max=ts_max,
    )
