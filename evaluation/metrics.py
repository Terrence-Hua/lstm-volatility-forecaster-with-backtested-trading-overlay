"""Volatility forecast evaluation metrics.

QLIKE (Quasi-Likelihood) is the standard loss function for volatility
model comparison:

    QLIKE(sigma2, rv) = log(sigma2) + rv / sigma2

where sigma2 is the forecast variance and rv is the realised variance.
Lower QLIKE is better.

MSE on variance:
    MSE(sigma2, rv) = (sigma2 - rv)^2

HMSE (heteroskedasticity-adjusted MSE):
    HMSE(sigma2, rv) = (1 - rv/sigma2)^2

All functions clip forecasts to a small positive floor before applying log.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-10  # floor for variance forecasts to avoid log(0)


def qlike(
    forecasts: np.ndarray,
    realized: np.ndarray,
    clip_floor: float = _EPS,
) -> float:
    """Quasi-likelihood loss for volatility forecast evaluation.

    Parameters
    ----------
    forecasts:
        Predicted variance series (sigma2_hat). Shape (N,).
    realized:
        Realised variance series (proxy). Shape (N,).
    clip_floor:
        Minimum value forecasts are clipped to before log. Prevents -inf.

    Returns
    -------
    float
        Mean QLIKE over all valid (non-NaN) observations.

    Notes
    -----
    Patton (2011) shows QLIKE is a consistent loss function under
    misspecified volatility models.
    """
    fc = np.asarray(forecasts, dtype=np.float64)
    rv = np.asarray(realized, dtype=np.float64)

    valid = ~(np.isnan(fc) | np.isnan(rv))
    fc_v = np.clip(fc[valid], clip_floor, None)
    rv_v = rv[valid]

    return float(np.mean(np.log(fc_v) + rv_v / fc_v))


def mse(
    forecasts: np.ndarray,
    realized: np.ndarray,
) -> float:
    """Mean squared error on variance forecasts.

    Parameters
    ----------
    forecasts:
        Predicted variance series. Shape (N,).
    realized:
        Realised variance series. Shape (N,).

    Returns
    -------
    float
        Mean MSE over valid observations.
    """
    fc = np.asarray(forecasts, dtype=np.float64)
    rv = np.asarray(realized, dtype=np.float64)

    valid = ~(np.isnan(fc) | np.isnan(rv))
    return float(np.mean((fc[valid] - rv[valid]) ** 2))


def hmse(
    forecasts: np.ndarray,
    realized: np.ndarray,
    clip_floor: float = _EPS,
) -> float:
    """Heteroskedasticity-adjusted MSE.

    Scales each squared error by 1/forecast^2, down-weighting large
    forecasts relative to raw MSE.

    Parameters
    ----------
    forecasts:
        Predicted variance series. Shape (N,).
    realized:
        Realised variance series. Shape (N,).
    clip_floor:
        Minimum forecast value to avoid division by zero.

    Returns
    -------
    float
        Mean HMSE over valid observations.
    """
    fc = np.asarray(forecasts, dtype=np.float64)
    rv = np.asarray(realized, dtype=np.float64)

    valid = ~(np.isnan(fc) | np.isnan(rv))
    fc_v = np.clip(fc[valid], clip_floor, None)
    rv_v = rv[valid]

    return float(np.mean((1.0 - rv_v / fc_v) ** 2))


def mae(
    forecasts: np.ndarray,
    realized: np.ndarray,
) -> float:
    """Mean absolute error on variance forecasts.

    Parameters
    ----------
    forecasts:
        Predicted variance series. Shape (N,).
    realized:
        Realised variance series. Shape (N,).

    Returns
    -------
    float
    """
    fc = np.asarray(forecasts, dtype=np.float64)
    rv = np.asarray(realized, dtype=np.float64)

    valid = ~(np.isnan(fc) | np.isnan(rv))
    return float(np.mean(np.abs(fc[valid] - rv[valid])))


def score_all(
    forecasts: np.ndarray,
    realized: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics and return as a dict.

    Parameters
    ----------
    forecasts:
        Predicted variance series. Shape (N,).
    realized:
        Realised variance series. Shape (N,).

    Returns
    -------
    dict[str, float]
        Keys: 'qlike', 'mse', 'hmse', 'mae'.
    """
    return {
        "qlike": qlike(forecasts, realized),
        "mse": mse(forecasts, realized),
        "hmse": hmse(forecasts, realized),
        "mae": mae(forecasts, realized),
    }
