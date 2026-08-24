"""GARCH(1,1) baseline using the arch library.

Fits a GARCH(1,1) model on log-returns and produces one-step-ahead
conditional variance forecasts. The arch library uses percent returns
internally, so we scale log-returns by 100 on entry and descale forecasts
back to return-squared units on exit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class GARCH11:
    """GARCH(1,1) wrapper around the arch library.

    Parameters
    ----------
    p:
        GARCH lag order (default 1).
    q:
        ARCH lag order (default 1).

    Attributes
    ----------
    result_:
        arch model fit result after calling :meth:`fit`.
    """

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self.p = p
        self.q = q
        self.result_ = None

    def fit(
        self,
        log_returns: pd.Series | np.ndarray,
        disp: bool = False,
    ) -> "GARCH11":
        """Fit GARCH(p, q) to log-returns.

        Parameters
        ----------
        log_returns:
            Daily log-return series (in natural units, not percent).
        disp:
            Whether to display arch optimiser output.

        Returns
        -------
        self
        """
        from arch import arch_model  # type: ignore[import]

        r = pd.Series(np.asarray(log_returns, dtype=np.float64)) * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(r, vol="Garch", p=self.p, q=self.q, rescale=False)
            self.result_ = am.fit(disp="off" if not disp else "final", show_warning=False)
        return self

    def forecast_variance(
        self,
        log_returns: pd.Series | np.ndarray,
        horizon: int = 1,
    ) -> np.ndarray:
        """Produce one-step-ahead in-sample conditional variance forecasts.

        Re-fits the model on the provided returns and returns the conditional
        variance series (in return-squared units, not percent-squared).

        Parameters
        ----------
        log_returns:
            Full return series to forecast over.
        horizon:
            Steps ahead (default 1 for next-day).

        Returns
        -------
        np.ndarray
            Conditional variance in daily return-squared units, aligned with
            the *log_returns* index. First entry is NaN (no prior info).
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before forecast_variance().")

        # Conditional variances from the in-sample fit (percent-squared)
        cond_vol_pct = self.result_.conditional_volatility
        # Convert back to natural units: (sigma_pct / 100)^2
        cond_var = (cond_vol_pct / 100.0) ** 2

        # Align with the full return series length
        n = len(log_returns)
        result_n = len(cond_var)

        if result_n >= n:
            return cond_var.values[-n:]

        # Pad front with NaN if result is shorter
        padded = np.full(n, np.nan)
        padded[n - result_n :] = cond_var.values
        return padded

    def predict_next(self) -> float:
        """Return the one-step-ahead conditional variance for t+1.

        Must be called after :meth:`fit`. Uses the arch forecast method.

        Returns
        -------
        float
            One-step variance forecast in return-squared units.
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before predict_next().")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.result_.forecast(horizon=1, reindex=False)
        var_pct_sq = float(fc.variance.values[-1, 0])
        return var_pct_sq / 10_000.0  # convert percent-squared → return-squared

    def params_summary(self) -> str:
        """Return a one-line parameter summary.

        Returns
        -------
        str
        """
        if self.result_ is None:
            return "Model not fitted."
        p = self.result_.params
        return (
            f"GARCH({self.p},{self.q}) | "
            f"omega={p.get('omega', float('nan')):.2e} "
            f"alpha={p.get('alpha[1]', float('nan')):.4f} "
            f"beta={p.get('beta[1]', float('nan')):.4f}"
        )
