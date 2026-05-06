from __future__ import annotations

from gpytorch.kernels import AdditiveKernel, LinearKernel, MaternKernel, ScaleKernel


def make_covar_module(kernel_mode: str, d: int):
    mode = str(kernel_mode).lower()

    if mode == "baseline":
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))

    if mode == "lin_matern32_add":
        add = AdditiveKernel(
            LinearKernel(ard_num_dims=d),
            MaternKernel(nu=1.5, ard_num_dims=d),
        )
        return ScaleKernel(add)

    raise ValueError(f"Unknown kernel_mode='{kernel_mode}'.")
