from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import PowerTransformer


@dataclass
class OutputPowerTransform:
    method: str | None = None
    lambda_: float | None = None
    sign_flip_applied: bool = False
    fitted: bool = False
    _pt: PowerTransformer | None = None
    _identity: bool = False

    def fit(self, y_np: np.ndarray) -> "OutputPowerTransform":
        y = np.asarray(y_np, dtype=float).reshape(-1, 1)
        if y.shape[0] < 2:
            self._identity = True
            self.fitted = True
            self.method = "identity"
            self.lambda_ = None
            self.sign_flip_applied = False
            self._pt = None
            return self

        all_pos = bool(np.all(y > 0.0))
        all_neg = bool(np.all(y < 0.0))

        self.sign_flip_applied = False

        if all_pos:
            method = "box-cox"
            y_fit = y
        elif all_neg:
            method = "box-cox"
            y_fit = -y
            self.sign_flip_applied = True
        else:
            method = "yeo-johnson"
            y_fit = y

        try:
            pt = PowerTransformer(method=method, standardize=False)
            pt.fit(y_fit)
        except Exception:
            # Safety fallback
            method = "yeo-johnson"
            self.sign_flip_applied = False
            y_fit = y
            pt = PowerTransformer(method=method, standardize=False)
            pt.fit(y_fit)

        self.method = method
        self.lambda_ = float(pt.lambdas_[0])
        self._pt = pt
        self.fitted = True
        self._identity = False
        return self

    def transform(self, y_np: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputPowerTransform must be fit before transform().")
        y = np.asarray(y_np, dtype=float).reshape(-1, 1)
        if self._identity:
            return y.reshape(-1)

        assert self._pt is not None
        if self.sign_flip_applied:
            y = -y
        yt = self._pt.transform(y)
        if self.sign_flip_applied:
            # Preserve "larger is better" semantics for maximization code paths.
            yt = -yt
        return yt.reshape(-1)

    def inverse_transform(self, y_t_np: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputPowerTransform must be fit before inverse_transform().")
        yt = np.asarray(y_t_np, dtype=float).reshape(-1, 1)
        if self._identity:
            return yt.reshape(-1)

        assert self._pt is not None
        if self.sign_flip_applied:
            yt = -yt
        y = self._pt.inverse_transform(yt)
        if self.sign_flip_applied:
            y = -y
        return y.reshape(-1)
