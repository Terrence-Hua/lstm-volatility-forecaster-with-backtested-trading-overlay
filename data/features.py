"""Feature engineering for volatility forecasting.

Builds the feature matrix from a raw OHLCV + vix_proxy DataFrame.
Features:
- Realized variance estimates at multiple horizons (RV_1, RV_5, RV_22)
- HAR-RV components (same as above, used by the HAR-RV baseline)
- Lagged log-returns
- Calendar dummies (day-of-week, month)
- VIX proxy (implied vol level)
- Vol spread: vix_proxy / sqrt(252) - sqrt(RV_1) (daily annualised units)

All targets and features are in daily variance units unless otherwise noted.

Usage
-----
>>> from data.generate import make_synthetic_data
>>> from data.features import build_features
>>> df = make_synthetic_data(n=500)
>>> feat = build_features(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _realized_variance(log_returns: pd.Series, window: int) -> pd.Series:
    """Rolling sum of squared log-returns over *window* days.

    Parameters
    ----------
    log_returns:
        Daily log-return series (NaN-free expected).
    window:
        Lookback window in trading days.

    Returns
    -------
    pd.Series
        Rolling realized variance; first ``window - 1`` rows are NaN.
    """
    return (log_returns**2).rolling(window).sum()


def _har_rv_components(rv1: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute daily, weekly, and monthly HAR-RV components from RV_1.

    HAR-RV decomposes volatility into:
    - Daily:   RV_{t-1}
    - Weekly:  average of RV_{t-1} to RV_{t-5}
    - Monthly: average of RV_{t-1} to RV_{t-22}

    Parameters
    ----------
    rv1:
        Daily realized variance series (window=1 squared return).

    Returns
    -------
    (rv_d, rv_w, rv_m): tuple of pd.Series
        Lagged daily, mean-weekly, mean-monthly components.
    """
    rv_d = rv1.shift(1)
    rv_w = rv1.rolling(5).mean().shift(1)
    rv_m = rv1.rolling(22).mean().shift(1)
    return rv_d, rv_w, rv_m


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    rv_windows: list[int] | None = None,
    return_lags: list[int] | None = None,
    add_calendar: bool = True,
) -> pd.DataFrame:
    """Construct the full feature matrix.

    Parameters
    ----------
    df:
        Raw DataFrame with at least columns: log_return, vix_proxy, date.
    rv_windows:
        Rolling windows (trading days) for realized variance. Defaults to
        [1, 5, 22].
    return_lags:
        Lagged return lags to include. Defaults to [1, 2, 5].
    add_calendar:
        Whether to add day-of-week and month dummy columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix. Target column is ``rv_target`` (next-day RV_1).
        NaN rows (from rolling windows) are dropped and index is reset.
    """
    if rv_windows is None:
        rv_windows = [1, 5, 22]
    if return_lags is None:
        return_lags = [1, 2, 5]

    out = df.copy()

    # ---- Realized variance at each horizon --------------------------------
    for w in rv_windows:
        out[f"rv_{w}d"] = _realized_variance(out["log_return"], window=w)

    # ---- HAR-RV components (from RV_1) ------------------------------------
    rv1_col = out["log_return"] ** 2  # single-period squared return
    rv_d, rv_w, rv_m = _har_rv_components(rv1_col)
    out["har_rv_d"] = rv_d
    out["har_rv_w"] = rv_w
    out["har_rv_m"] = rv_m

    # ---- Target: next-day realized variance --------------------------------
    out["rv_target"] = rv1_col.shift(-1)

    # ---- Lagged returns ----------------------------------------------------
    for lag in return_lags:
        out[f"ret_lag{lag}"] = out["log_return"].shift(lag)

    # ---- VIX proxy (already in df) ----------------------------------------
    # Convert VIX from annualised % vol to daily variance for comparability
    out["vix_var"] = (out["vix_proxy"] / 100) ** 2 / 252

    # ---- Vol spread: implied daily var - realised daily var ----------------
    out["vol_spread"] = out["vix_var"] - out["rv_1d"] if "rv_1d" in out else (
        out["vix_var"] - out.get("rv_1d", pd.Series(np.nan, index=out.index))
    )
    # Use the freshly computed rv_1d
    if "rv_1d" in out.columns:
        out["vol_spread"] = out["vix_var"] - out["rv_1d"]

    # ---- Calendar dummies --------------------------------------------------
    if add_calendar:
        dates = pd.to_datetime(out["date"])
        dow_dummies = pd.get_dummies(dates.dt.dayofweek, prefix="dow", dtype=float)
        # Drop one level to avoid multicollinearity
        dow_dummies = dow_dummies.iloc[:, 1:]
        month_dummies = pd.get_dummies(dates.dt.month, prefix="mon", dtype=float)
        month_dummies = month_dummies.iloc[:, 1:]
        out = pd.concat([out, dow_dummies, month_dummies], axis=1)

    # ---- Drop NaN rows from rolling windows --------------------------------
    out = out.dropna().reset_index(drop=True)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes metadata and target).

    Parameters
    ----------
    df:
        Output of :func:`build_features`.

    Returns
    -------
    list[str]
    """
    exclude = {
        "date", "open", "high", "low", "close", "volume",
        "log_return", "true_variance", "rv_target",
    }
    return [c for c in df.columns if c not in exclude]


def get_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract X (features) and y (target) as numpy arrays.

    Parameters
    ----------
    df:
        Output of :func:`build_features`.

    Returns
    -------
    (X, y): tuple[np.ndarray, np.ndarray]
        X has shape (n_samples, n_features); y has shape (n_samples,).
    """
    cols = feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["rv_target"].to_numpy(dtype=np.float32)
    return X, y
