from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import QuantileTransformer


@dataclass
class OutputCopulaTransform:
    method: str | None = None
    fitted: bool = False
    _qt: QuantileTransformer | None = None
    _identity: bool = False

    def fit(self, y_np: np.ndarray) -> "OutputCopulaTransform":
        y = np.asarray(y_np, dtype=float).reshape(-1, 1)
        if y.shape[0] < 4:
            self.method = "identity"
            self.fitted = True
            self._identity = True
            self._qt = None
            return self

        qt = QuantileTransformer(
            n_quantiles=min(32, y.shape[0]),
            output_distribution="normal",
            random_state=0,
            subsample=int(1e9),
        )
        qt.fit(y)
        self.method = "copula_normal"
        self.fitted = True
        self._identity = False
        self._qt = qt
        return self

    def transform(self, y_np: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputCopulaTransform must be fit before transform().")
        y = np.asarray(y_np, dtype=float).reshape(-1, 1)
        if self._identity:
            return y.reshape(-1)
        assert self._qt is not None
        return self._qt.transform(y).reshape(-1)

    def inverse_transform(self, y_t_np: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("OutputCopulaTransform must be fit before inverse_transform().")
        yt = np.asarray(y_t_np, dtype=float).reshape(-1, 1)
        if self._identity:
            return yt.reshape(-1)
        assert self._qt is not None
        return self._qt.inverse_transform(yt).reshape(-1)
