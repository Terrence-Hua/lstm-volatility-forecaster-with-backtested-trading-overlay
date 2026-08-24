"""Market data fetcher using yfinance.

Downloads OHLCV and VIX data for a given ticker and date range.
Falls back to synthetic data when the download fails (e.g., in offline
environments).

Usage
-----
>>> from data.fetch import fetch_market_data
>>> df = fetch_market_data("SPY", "2015-01-01", "2024-12-31")
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_market_data(
    ticker: str,
    start: str,
    end: str,
    vix_ticker: str = "^VIX",
) -> pd.DataFrame:
    """Fetch daily OHLCV for *ticker* and add a vix_proxy column.

    Parameters
    ----------
    ticker:
        Equity ticker (e.g. "SPY").
    start:
        ISO date string, inclusive (e.g. "2015-01-01").
    end:
        ISO date string, exclusive (e.g. "2024-12-31").
    vix_ticker:
        VIX symbol. Default is "^VIX".

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, volume, log_return, vix_proxy.
        Sorted ascending by date, NaN rows dropped.

    Raises
    ------
    RuntimeError
        If the download returns an empty DataFrame.
    """
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from exc

    logger.info("Downloading %s from %s to %s", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {ticker} ({start} to {end}). "
            "Check your internet connection or set data.synthetic=true in the config."
        )

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.rename(columns={"adj close": "close"})
    raw.index.name = "date"
    raw = raw.reset_index()

    # Log returns
    raw["log_return"] = (raw["close"] / raw["close"].shift(1)).pipe(
        lambda s: s.apply(lambda x: float("nan") if x <= 0 else __import__("math").log(x))
    )

    # Fetch VIX
    logger.info("Downloading VIX (%s)", vix_ticker)
    try:
        vix_raw = yf.download(
            vix_ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = [c[0].lower() for c in vix_raw.columns]
        else:
            vix_raw.columns = [c.lower() for c in vix_raw.columns]
        vix_raw.index.name = "date"
        vix_series = vix_raw["close"].rename("vix_proxy").reset_index()
        raw = raw.merge(vix_series, on="date", how="left")
    except Exception:  # noqa: BLE001
        logger.warning("VIX download failed; using NaN placeholder for vix_proxy")
        raw["vix_proxy"] = float("nan")

    raw = raw.dropna(subset=["log_return"])
    raw = raw.sort_values("date").reset_index(drop=True)
    return raw


def load_data(cfg: dict) -> pd.DataFrame:
    """Load data according to the config dict.

    If ``data.synthetic`` is true, generates synthetic data.
    Otherwise, downloads via yfinance.

    Parameters
    ----------
    cfg:
        Root-level config dict.

    Returns
    -------
    pd.DataFrame
    """
    from data.generate import make_synthetic_data_from_config

    data_cfg = cfg.get("data", {})
    if data_cfg.get("synthetic", False):
        logger.info("Using synthetic data (data.synthetic=true)")
        return make_synthetic_data_from_config(cfg)

    return fetch_market_data(
        ticker=data_cfg.get("ticker", "SPY"),
        start=data_cfg.get("start", "2015-01-01"),
        end=data_cfg.get("end", "2024-12-31"),
    )
