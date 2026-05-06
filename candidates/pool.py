from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from acquisition.scores import AcquisitionScores, compute_acq_scores
from candidates.trust_region import TrustRegionState, sample_in_tr
from config import RebuildConfig


@dataclass
class CandidatePool:
    X_pool: torch.Tensor
    scores: AcquisitionScores
    priority_indices: np.ndarray
    candidate_origin: np.ndarray
    tr_state: TrustRegionState | None


def _topk_idx(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, values.shape[0])
    idx = np.argpartition(values, -k)[-k:]
    idx = idx[np.argsort(values[idx])[::-1]]
    return idx.astype(int)


def build_candidate_pool(
    model,
    y_t: torch.Tensor,
    X_obs: torch.Tensor,
    y_obs: torch.Tensor,
    d: int,
    cfg: RebuildConfig,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    tr_state: TrustRegionState | None = None,
) -> CandidatePool:
    pools: list[torch.Tensor] = []
    origins: list[np.ndarray] = []

    n_global = int(cfg.n_sobol_global) if int(cfg.n_sobol_global) > 0 else int(cfg.n_sobol)
    sobol = SobolEngine(dimension=d, scramble=True, seed=int(seed))
    X_global = sobol.draw(n_global).to(device=device, dtype=dtype)
    pools.append(X_global)
    origins.append(np.full(X_global.shape[0], "global", dtype=object))

    if bool(cfg.use_trust_region) and tr_state is not None and int(cfg.n_sobol_tr) > 0:
        X_tr = sample_in_tr(
            tr_state,
            int(cfg.n_sobol_tr),
            d,
            seed=int(seed) + 101,
            device=device,
            dtype=dtype,
            lower=float(cfg.lower),
            upper=float(cfg.upper),
        )
        if X_tr.numel() > 0:
            pools.append(X_tr)
            origins.append(np.full(X_tr.shape[0], "tr", dtype=object))

    if int(cfg.n_sobol_elite) > 0 and X_obs.shape[0] >= 3:
        y_np = y_obs.detach().cpu().numpy().reshape(-1)
        X_np = X_obs.detach().cpu().numpy()
        elite_n = max(1, int(np.ceil(float(cfg.elite_frac) * X_np.shape[0])))
        elite_idx = np.argsort(y_np)[-elite_n:]
        elite = X_np[elite_idx]
        lo = np.clip(np.min(elite, axis=0) - float(cfg.elite_margin), cfg.lower, cfg.upper)
        hi = np.clip(np.max(elite, axis=0) + float(cfg.elite_margin), cfg.lower, cfg.upper)
        if np.all(hi >= lo):
            sobol_elite = SobolEngine(dimension=d, scramble=True, seed=int(seed) + 202)
            unit = sobol_elite.draw(int(cfg.n_sobol_elite)).detach().cpu().numpy()
            pts = lo[None, :] + (hi - lo)[None, :] * unit
            X_elite = torch.tensor(pts, device=device, dtype=dtype)
            pools.append(X_elite)
            origins.append(np.full(X_elite.shape[0], "elite", dtype=object))

    X_pool = torch.cat(pools, dim=0)
    candidate_origin = np.concatenate(origins, axis=0)

    scores = compute_acq_scores(
        model=model,
        y_t=y_t,
        X_pool=X_pool,
        ucb_betas=cfg.ucb_betas,
        seed=seed,
        n_ts_draws=int(cfg.ts_n_draws),
        robust_noise_frac=float(cfg.robust_acq_noise_frac) if bool(cfg.enable_robust_acq_noise) else 0.0,
    )

    top_idx = set()
    for idx in _topk_idx(scores.log_ei, cfg.topk_per_acq):
        top_idx.add(int(idx))
    for idx in _topk_idx(scores.ei, cfg.topk_per_acq):
        top_idx.add(int(idx))
    for idx in _topk_idx(scores.pi, cfg.topk_per_acq):
        top_idx.add(int(idx))
    for idx in _topk_idx(scores.maxvar, cfg.topk_per_acq):
        top_idx.add(int(idx))
    for idx in _topk_idx(scores.ts_max, cfg.topk_per_acq):
        top_idx.add(int(idx))
    for beta in cfg.ucb_betas:
        for idx in _topk_idx(scores.ucb_by_beta[float(beta)], cfg.topk_per_acq):
            top_idx.add(int(idx))

    priority = np.array(sorted(top_idx), dtype=int)
    return CandidatePool(
        X_pool=X_pool,
        scores=scores,
        priority_indices=priority,
        candidate_origin=candidate_origin,
        tr_state=tr_state,
    )
