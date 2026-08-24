"""HAR-RV (Heterogeneous Autoregressive Realized Variance) baseline.

Corsi (2009) model. Forecasts next-day realized variance as a linear
combination of past daily, weekly, and monthly RV averages:

    RV_{t+1} = beta_0 + beta_d * RV_d + beta_w * RV_w + beta_m * RV_m + e

where:
    RV_d  = RV_{t}           (yesterday's RV)
    RV_w  = mean(RV_{t-4:t}) (last 5 days)
    RV_m  = mean(RV_{t-21:t})(last 22 days)

Fit uses OLS (sklearn LinearRegression).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class HARRV:
    """OLS HAR-RV forecaster.

    Attributes
    ----------
    coef_:
        Fitted [beta_d, beta_w, beta_m] after calling :meth:`fit`.
    intercept_:
        Fitted intercept.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_X(rv_series: pd.Series | np.ndarray) -> np.ndarray:
        """Build design matrix [RV_d, RV_w, RV_m] from a RV_1d series.

        Parameters
        ----------
        rv_series:
            Daily realized variance series (squared daily returns or
            rolling 1-day RV).

        Returns
        -------
        np.ndarray
            Shape (n, 3); first 21 rows have NaN and are stripped by caller.
        """
        rv = pd.Series(rv_series).reset_index(drop=True)
        rv_d = rv.shift(1)
        rv_w = rv.rolling(5).mean().shift(1)
        rv_m = rv.rolling(22).mean().shift(1)
        return np.column_stack([rv_d, rv_w, rv_m])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, rv_series: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> "HARRV":
        """Fit the HAR-RV model.

        Parameters
        ----------
        rv_series:
            Daily realized variance series (full history).
        target:
            Next-day realized variance targets aligned with *rv_series*.

        Returns
        -------
        self
        """
        X = self._build_X(rv_series)
        y = np.asarray(target, dtype=np.float64)

        # Drop rows with NaN from rolling windows
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        self._model.fit(X[valid], y[valid])
        self.coef_ = self._model.coef_
        self.intercept_ = float(self._model.intercept_)
        return self

    def predict(self, rv_series: pd.Series | np.ndarray) -> np.ndarray:
        """Predict next-day RV for each row in *rv_series*.

        Parameters
        ----------
        rv_series:
            Daily realized variance series to forecast from.

        Returns
        -------
        np.ndarray
            Predictions aligned with *rv_series*; first 21 entries are NaN.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = self._build_X(rv_series)
        preds = np.full(len(rv_series), np.nan)
        valid = ~np.isnan(X).any(axis=1)
        preds[valid] = self._model.predict(X[valid])
        # Clip to non-negative variance
        preds = np.where(np.isnan(preds), np.nan, np.maximum(preds, 0.0))
        return preds

    def predict_from_components(
        self,
        rv_d: float | np.ndarray,
        rv_w: float | np.ndarray,
        rv_m: float | np.ndarray,
    ) -> np.ndarray:
        """Predict from pre-computed HAR components.

        Parameters
        ----------
        rv_d:
            Yesterday's realized variance.
        rv_w:
            Five-day average realized variance.
        rv_m:
            Twenty-two-day average realized variance.

        Returns
        -------
        np.ndarray
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict_from_components().")
        X = np.column_stack([rv_d, rv_w, rv_m])
        return np.maximum(self._model.predict(X), 0.0)

    def summary(self) -> str:
        """Return a human-readable coefficient table.

        Returns
        -------
        str
        """
        if self.coef_ is None:
            return "Model not fitted."
        lines = [
            "HAR-RV coefficients",
            f"  intercept : {self.intercept_:.6f}",
            f"  beta_d    : {self.coef_[0]:.6f}",
            f"  beta_w    : {self.coef_[1]:.6f}",
            f"  beta_m    : {self.coef_[2]:.6f}",
        ]
        return "\n".join(lines)
