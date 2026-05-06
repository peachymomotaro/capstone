from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import inspect

import numpy as np
import torch
import torch.nn.functional as F
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.exceptions.errors import ModelFittingError
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator import settings as linop_settings

from config import RebuildConfig
from models.kernels import make_covar_module
from transforms.output_copula import OutputCopulaTransform
from transforms.output_power import OutputPowerTransform


try:
    from botorch.models.transforms.input import Warp as BoTorchWarp
    _HAS_BOTORCH_WARP = True
except Exception:
    BoTorchWarp = None
    _HAS_BOTORCH_WARP = False

try:
    from botorch.models.transforms.input import InputTransform
except Exception:
    InputTransform = torch.nn.Module


class KumaraswamyWarpFallback(InputTransform):
    """Fallback input warp if botorch Warp is unavailable.

    Applies w(x) = 1 - (1 - x^a)^b with learnable a, b > 0 via softplus.
    """

    def __init__(self, d: int, eps: float = 1e-7) -> None:
        super().__init__()
        self.d = int(d)
        self.eps = float(eps)
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

        self.raw_a = torch.nn.Parameter(torch.zeros(self.d))
        self.raw_b = torch.nn.Parameter(torch.zeros(self.d))

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a) + 1e-4

    @property
    def b(self) -> torch.Tensor:
        return F.softplus(self.raw_b) + 1e-4

    def _expand(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while p.ndim < x.ndim:
            p = p.unsqueeze(0)
        return p

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        Xc = X.clamp(self.eps, 1.0 - self.eps)
        a = self._expand(self.a, Xc)
        b = self._expand(self.b, Xc)
        return 1.0 - torch.pow(1.0 - torch.pow(Xc, a), b)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        Xc = X.clamp(self.eps, 1.0 - self.eps)
        a = self._expand(self.a, Xc)
        b = self._expand(self.b, Xc)
        return torch.pow(1.0 - torch.pow(1.0 - Xc, 1.0 / b), 1.0 / a)


@dataclass
class FittedWarpedGP:
    model: SingleTaskGP
    y_transform: object
    y_raw: torch.Tensor
    y_t: torch.Tensor
    warp_kind: str
    fit_attempt_index: int
    fit_attempt_label: str
    fit_attempt_count: int
    failed_attempts: list[dict[str, object]]
    model_variant_name: str = "primary"

    def diagnostics(self) -> dict:
        di: dict[str, object] = {
            "warp_kind": self.warp_kind,
            "output_transform_method": getattr(self.y_transform, "method", None),
            "output_lambda": getattr(self.y_transform, "lambda_", None),
            "output_sign_flip_applied": getattr(self.y_transform, "sign_flip_applied", False),
            "fit_attempt_index": self.fit_attempt_index,
            "fit_attempt_label": self.fit_attempt_label,
            "fit_attempt_count": self.fit_attempt_count,
            "fit_failed": False,
            "failed_attempts": self.failed_attempts,
            "model_variant_name": self.model_variant_name,
        }

        # Noise
        try:
            di["noise_est"] = float(self.model.likelihood.noise.detach().cpu().item())
        except Exception:
            di["noise_est"] = float("nan")

        # Lengthscales (if present)
        try:
            ls = (
                self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
            )
            di["len_min"] = float(np.min(ls))
            di["len_med"] = float(np.median(ls))
        except Exception:
            di["len_min"] = float("nan")
            di["len_med"] = float("nan")

        # Warp params
        try:
            it = self.model.input_transform
            if hasattr(it, "concentration0") and hasattr(it, "concentration1"):
                c0 = it.concentration0.detach().cpu().numpy().reshape(-1).tolist()
                c1 = it.concentration1.detach().cpu().numpy().reshape(-1).tolist()
                di["warp_a"] = c1
                di["warp_b"] = c0
            elif hasattr(it, "a") and hasattr(it, "b"):
                di["warp_a"] = it.a.detach().cpu().numpy().reshape(-1).tolist()
                di["warp_b"] = it.b.detach().cpu().numpy().reshape(-1).tolist()
            else:
                di["warp_a"] = []
                di["warp_b"] = []
        except Exception:
            di["warp_a"] = []
            di["warp_b"] = []

        return di


def _make_input_warp(d: int, use_input_warp: bool):
    if not use_input_warp:
        return None, "none"

    if _HAS_BOTORCH_WARP and BoTorchWarp is not None:
        indices = list(range(d))
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        attempts: list[tuple[tuple, dict]] = []

        # Signature-driven attempts for version compatibility.
        try:
            params = inspect.signature(BoTorchWarp).parameters
        except Exception:
            params = {}

        if "d" in params and "indices" in params:
            attempts.append(((), {"d": d, "indices": indices}))
            attempts.append(((), {"d": d, "indices": indices, "bounds": bounds}))
        if "indices" in params:
            attempts.append(((), {"indices": indices}))
            attempts.append(((), {"indices": indices, "bounds": bounds}))

        # Positional fallbacks for older/newer API variants.
        attempts.extend(
            [
                ((d, indices), {}),
                ((indices,), {}),
                ((indices, bounds), {}),
                ((d,), {}),
            ]
        )

        for args, kwargs in attempts:
            try:
                return BoTorchWarp(*args, **kwargs), "botorch_warp"
            except TypeError:
                continue
            except Exception:
                # If constructor exists but fails for non-signature reasons,
                # we'll gracefully fall back to custom warp.
                break

    return KumaraswamyWarpFallback(d=d), "fallback_kumaraswamy"


def fit_warped_gp(X: torch.Tensor, y: torch.Tensor, cfg: RebuildConfig) -> FittedWarpedGP:
    attempts = [
        ("primary", {}),
        ("no_input_warp", {"use_input_warp": False}),
        ("no_output_power_transform", {"use_output_power_transform": False}),
        ("baseline_kernel", {"kernel_mode": "baseline"}),
        (
            "baseline_no_transforms",
            {
                "kernel_mode": "baseline",
                "use_input_warp": False,
                "use_output_power_transform": False,
            },
        ),
        ("high_jitter", {"_jitter": 1e-4}),
        (
            "high_jitter_baseline_no_transforms",
            {
                "_jitter": 1e-3,
                "kernel_mode": "baseline",
                "use_input_warp": False,
                "use_output_power_transform": False,
            },
        ),
    ]

    max_attempts = max(1, int(cfg.fit_retry_max_attempts))
    attempts = attempts[:max_attempts]
    failures: list[dict[str, object]] = []

    for idx, (label, overrides) in enumerate(attempts):
        run_cfg = cfg
        jitter = float(overrides.get("_jitter", 0.0))
        cfg_overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        if cfg_overrides:
            run_cfg = dataclasses.replace(cfg, **cfg_overrides)
        try:
            fitted = _fit_warped_gp_once(X=X, y=y, cfg=run_cfg, jitter=jitter)
            fitted.fit_attempt_index = idx
            fitted.fit_attempt_label = label
            fitted.fit_attempt_count = len(attempts)
            fitted.failed_attempts = failures
            return fitted
        except Exception as exc:
            failures.append(
                {
                    "attempt_index": idx,
                    "attempt_label": label,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    msg = "; ".join(
        [f"{x['attempt_label']}:{x['error_type']}" for x in failures[-3:]]
    )
    raise ModelFittingError(f"All attempts to fit the model have failed. Recent failures: {msg}")


def _fit_warped_gp_once(X: torch.Tensor, y: torch.Tensor, cfg: RebuildConfig, jitter: float = 0.0) -> FittedWarpedGP:
    d = int(X.shape[-1])

    # Output transform fitted every run.
    y_tf = OutputCopulaTransform() if str(cfg.output_transform_mode) == "copula" else OutputPowerTransform()
    if cfg.use_output_power_transform:
        y_tf.fit(y.detach().cpu().numpy().reshape(-1))
        y_t_np = y_tf.transform(y.detach().cpu().numpy().reshape(-1))
        y_t = torch.tensor(y_t_np, dtype=X.dtype, device=X.device).unsqueeze(-1)
    else:
        y_tf.method = "identity"
        y_tf.lambda_ = None
        y_tf.sign_flip_applied = False
        y_tf.fitted = True
        y_tf._identity = True
        y_tf._pt = None
        y_t = y

    input_warp, warp_kind = _make_input_warp(d=d, use_input_warp=cfg.use_input_warp)
    covar_module = make_covar_module(kernel_mode=cfg.kernel_mode, d=d)

    kwargs = {
        "input_transform": input_warp,
        "outcome_transform": Standardize(m=1),
        "covar_module": covar_module,
    }

    # Remove None to keep botorch happy.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model = SingleTaskGP(train_X=X, train_Y=y_t, **kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if jitter > 0.0:
        with linop_settings.cholesky_jitter(float(jitter)):
            fit_gpytorch_mll(mll)
    else:
        fit_gpytorch_mll(mll)

    return FittedWarpedGP(
        model=model,
        y_transform=y_tf,
        y_raw=y,
        y_t=y_t,
        warp_kind=warp_kind,
        fit_attempt_index=0,
        fit_attempt_label="primary",
        fit_attempt_count=1,
        failed_attempts=[],
        model_variant_name=f"{cfg.kernel_mode}:{cfg.output_transform_mode}",
    )
