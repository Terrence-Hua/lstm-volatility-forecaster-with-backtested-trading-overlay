"""Synthetic financial data generator.

Simulates daily price, log-return, and realized-vol series using a
discrete-time Heston stochastic-volatility model. Output is a DataFrame
that matches the schema expected by data/features.py and the rest of the
pipeline, so every component works offline without market data.

Usage
-----
>>> from data.generate import make_synthetic_data
>>> df = make_synthetic_data(n=2500, seed=42)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_data(
    n: int = 2500,
    seed: int = 42,
    kappa: float = 3.0,
    theta: float = 0.04,
    xi: float = 0.4,
    rho: float = -0.7,
    s0: float = 100.0,
    v0: float = 0.04,
    dt: float = 1 / 252,
) -> pd.DataFrame:
    """Generate a synthetic daily OHLCV + VIX-proxy DataFrame.

    Uses an Euler-Maruyama discretisation of the Heston (1993) model.

    Parameters
    ----------
    n:
        Number of trading days to generate.
    seed:
        Random seed for reproducibility.
    kappa:
        Mean-reversion speed of variance.
    theta:
        Long-run mean variance.
    xi:
        Volatility of variance (vol-of-vol).
    rho:
        Correlation between price and variance Brownian motions.
    s0:
        Initial stock price.
    v0:
        Initial variance.
    dt:
        Time step in years (1/252 for daily).

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, volume, log_return, vix_proxy.
        Index is integer; date column holds business-day timestamps.
    """
    rng = np.random.default_rng(seed)

    prices = np.empty(n + 1)
    variances = np.empty(n + 1)
    prices[0] = s0
    variances[0] = v0

    sqrt_dt = np.sqrt(dt)

    for t in range(n):
        v_t = max(variances[t], 1e-8)
        z1 = rng.standard_normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal()

        prices[t + 1] = prices[t] * np.exp(
            (-0.5 * v_t) * dt + np.sqrt(v_t) * sqrt_dt * z1
        )
        variances[t + 1] = max(
            variances[t]
            + kappa * (theta - v_t) * dt
            + xi * np.sqrt(v_t) * sqrt_dt * z2,
            1e-8,
        )

    close = prices[1:]
    open_ = prices[:-1]

    # Intraday range: open-to-close range scaled by daily vol estimate
    intraday_noise = rng.standard_normal((n, 2)) * np.sqrt(variances[1:, None]) * 0.5
    high = np.maximum(close, open_) * (1 + np.abs(intraday_noise[:, 0]))
    low = np.minimum(close, open_) * (1 - np.abs(intraday_noise[:, 1]))

    volume = rng.integers(1_000_000, 10_000_000, size=n)

    log_returns = np.log(prices[1:] / prices[:-1])

    # VIX proxy: 30-day forward annualised vol (use centred window, then shift)
    ann_vol = np.sqrt(variances[1:] * 252) * 100  # percent, like VIX
    # Add a bit of noise and a risk premium
    vix_proxy = ann_vol + rng.standard_normal(n) * 1.5 + 2.0
    vix_proxy = np.maximum(vix_proxy, 5.0)

    dates = pd.bdate_range(start="2010-01-04", periods=n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "log_return": log_returns,
            "vix_proxy": vix_proxy,
            "true_variance": variances[1:],
        }
    )
    return df


def make_synthetic_data_from_config(cfg: dict) -> pd.DataFrame:
    """Construct synthetic data using values from the YAML config dict.

    Parameters
    ----------
    cfg:
        The full config dict (root level).

    Returns
    -------
    pd.DataFrame
        See `make_synthetic_data`.
    """
    data_cfg = cfg.get("data", {})
    n = data_cfg.get("synthetic_n", 2500)
    seed = data_cfg.get("synthetic_seed", 42)
    return make_synthetic_data(n=n, seed=seed)
