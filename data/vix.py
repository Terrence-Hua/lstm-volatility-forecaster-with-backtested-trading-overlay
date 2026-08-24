"""VIX proxy construction and implied-vol spread utilities.

When real VIX data is unavailable (offline mode), a proxy is generated from
the conditional variance estimated by a rolling-window EWMA.

The vol spread is the key signal for the straddle strategy:
    spread = implied_daily_var - realized_daily_var

A positive spread means options are pricing more vol than has been realised —
conditions under which selling volatility (short straddle) tends to be
profitable, and vice versa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_variance(log_returns: pd.Series, halflife: int = 20) -> pd.Series:
    """EWMA variance estimate with *halflife* days.

    Parameters
    ----------
    log_returns:
        Daily log-return series.
    halflife:
        Decay halflife in trading days.

    Returns
    -------
    pd.Series
        Daily conditional variance estimate (same length as input).
    """
    return log_returns.pow(2).ewm(halflife=halflife).mean()


def vix_proxy_from_returns(
    log_returns: pd.Series,
    halflife: int = 20,
    risk_premium: float = 0.02,
    noise_std: float = 0.005,
    seed: int = 0,
) -> pd.Series:
    """Build a VIX-like series from log-returns when real VIX is absent.

    The proxy is an annualised volatility (in percent, like VIX):
        VIX_proxy = 100 * sqrt(252 * EWMA_variance) + risk_premium + noise

    Parameters
    ----------
    log_returns:
        Daily log-return series.
    halflife:
        EWMA halflife in days.
    risk_premium:
        Additive constant (annualised vol units) representing the variance
        risk premium. Default 0.02 (2 vol points).
    noise_std:
        Standard deviation of Gaussian noise added to the proxy.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pd.Series
        Annualised volatility proxy in percent (index preserved).
    """
    rng = np.random.default_rng(seed)
    ewma_var = ewma_variance(log_returns, halflife=halflife)
    ann_vol = 100.0 * np.sqrt(252.0 * ewma_var)
    noise = pd.Series(rng.standard_normal(len(log_returns)) * noise_std * 100, index=log_returns.index)
    proxy = ann_vol + risk_premium * 100 + noise
    return proxy.clip(lower=5.0)


def add_vix_proxy(df: pd.DataFrame, halflife: int = 20) -> pd.DataFrame:
    """Fill NaN values in the ``vix_proxy`` column using EWMA.

    If ``vix_proxy`` is entirely NaN (synthetic or offline mode), generates
    the full column. Otherwise, forward-fills gaps and replaces remaining
    NaNs with the EWMA estimate.

    Parameters
    ----------
    df:
        DataFrame with ``log_return`` and ``vix_proxy`` columns.
    halflife:
        EWMA halflife passed to :func:`vix_proxy_from_returns`.

    Returns
    -------
    pd.DataFrame
        Input df with ``vix_proxy`` fully populated.
    """
    df = df.copy()
    if df["vix_proxy"].isna().all():
        df["vix_proxy"] = vix_proxy_from_returns(df["log_return"], halflife=halflife).values
    else:
        ewma_fallback = vix_proxy_from_returns(df["log_return"], halflife=halflife)
        df["vix_proxy"] = df["vix_proxy"].fillna(ewma_fallback)
    return df


def compute_vol_spread(
    df: pd.DataFrame,
    rv_col: str = "rv_1d",
) -> pd.Series:
    """Compute the daily implied-minus-realised variance spread.

    Converts VIX (annualised % vol) to daily variance units, then subtracts
    realised variance.

    Parameters
    ----------
    df:
        Feature DataFrame with ``vix_proxy`` and *rv_col* columns.
    rv_col:
        Column name of the realised variance to subtract.

    Returns
    -------
    pd.Series
        Daily variance spread. Positive means implied > realised.
    """
    implied_daily_var = (df["vix_proxy"] / 100.0) ** 2 / 252.0
    spread = implied_daily_var - df[rv_col]
    return spread.rename("vol_spread")
