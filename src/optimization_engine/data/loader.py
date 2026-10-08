"""Data loaders for prices/returns from Excel, CSV, or synthetic samples."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from optimization_engine._optional import require

#: The synthetic universe: ``name -> (group, annual return, annual vol)``.
#: Module-level so the app and the ingest provider can offer it as a default
#: universe without restating a list that would then drift out of step.
SAMPLE_ASSETS: dict[str, tuple[str, float, float]] = {
    "US_Equity":     ("Equity",       0.08,  0.18),
    "Intl_Equity":   ("Equity",       0.07,  0.20),
    "EM_Equity":     ("Equity",       0.09,  0.24),
    "Real_Estate":   ("Alternatives", 0.06,  0.19),
    "Commodities":   ("Alternatives", 0.04,  0.22),
    "Infra":         ("Alternatives", 0.07,  0.16),
    "Gold":          ("Alternatives", 0.04,  0.15),
    "US_Treasuries": ("FixedIncome",  0.03,  0.07),
    "TIPS":          ("FixedIncome",  0.03,  0.06),
    "IG_Credit":     ("FixedIncome",  0.04,  0.08),
    "HY_Credit":     ("FixedIncome",  0.05,  0.11),
    "EM_Debt":       ("FixedIncome",  0.05,  0.10),
    "Cash":          ("FixedIncome",  0.025, 0.005),
}

#: Just the names, in generation order.
SAMPLE_UNIVERSE: tuple[str, ...] = tuple(SAMPLE_ASSETS)

#: Last date of every :func:`sample_dataset` panel. Fixed rather than "today"
#: so that the seed alone decides the panel: with a moving end date the values
#: stayed put but every calendar-driven step — a monthly rebalance, a holdout
#: cut-off — landed on different rows each day, and a backtest's
#: ``result_hash`` changed under an unchanged ``spec_hash``.
SAMPLE_END = pd.Timestamp("2025-12-31")


def load_prices(
    path: str | Path,
    sheet_name: str | int | None = "Precios",
    index_col: str | int | None = 0,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Load a price panel from an Excel or CSV file.

    Args:
        path: The file to read. The reader follows the extension.
        sheet_name: Worksheet to read from an Excel workbook. Ignored for CSV.
        index_col: Which column holds the dates.
        date_format: An explicit ``strftime`` pattern for parsing the index.
            ``None`` lets pandas infer it.

    Returns:
        Prices with a ``DatetimeIndex``, sorted ascending, one column per
        asset, with entirely-empty rows dropped.

    Raises:
        ValueError: If the extension is neither a spreadsheet nor a CSV.
        MissingDependencyError: If an ``.xlsx`` is read without openpyxl.
            Install it with ``finport-optengine[excel]``.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls", ".xlsm"}:
        require("openpyxl", extra="excel", purpose="reading Excel workbooks")
        df = pd.read_excel(p, sheet_name=sheet_name, index_col=index_col, parse_dates=True)
    elif suf == ".csv":
        df = pd.read_csv(p, index_col=index_col, parse_dates=True)
    elif suf == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file extension: {suf}")
    if date_format:
        df.index = pd.to_datetime(df.index, format=date_format)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.dropna(how="all")


def prices_to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Convert a price panel to periodic returns.

    Args:
        prices: Prices indexed by date, one column per asset.
        log: Return log differences instead of simple percentage changes.
            Log returns aggregate additively across time, which is convenient
            for estimation; simple returns aggregate across assets, which is
            what a portfolio needs. The engine optimizes on simple returns.

    Returns:
        A frame one row shorter than ``prices``, with rows that are entirely
        missing dropped. Individual gaps are left as NaN for the alignment
        step to deal with explicitly: an interior missing price costs *two*
        returns, the gap and the period after it. ``fill_method=None`` is
        passed explicitly because pandas 2.0-2.2 forward-fill by default and
        3.0 does not — left implicit, the same panel would give a different
        return series depending on which pandas was installed, with the gap
        becoming a 0% period followed by the two-period move booked as one.
    """
    if log:
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change(fill_method=None).dropna(how="all")


def sample_dataset(
    n_periods: int = 252 * 8,
    seed: int = 42,
    assets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Synthetic but realistic-ish price panel for tutorials and tests.

    Generates correlated daily log-returns from a multivariate normal with
    a hand-tuned covariance structure across asset classes, then exponentiates
    to a price series starting at 100.

    Args:
        n_periods: How many business days to generate. Defaults to eight
            years.
        seed: Seed for the generator. The same seed gives the same panel.
        assets: Which of the built-in asset names to include. ``None`` gives
            the full multi-asset universe.

    Returns:
        Prices indexed by business day and ending on :data:`SAMPLE_END`, one
        column per asset, all starting at 100. The same arguments give the
        same panel, dates included, whatever day it is built.
    """
    rng = np.random.default_rng(seed)

    default_assets = SAMPLE_ASSETS

    if assets is not None:
        keys = list(assets)
        meta = {k: default_assets[k] for k in keys if k in default_assets}
    else:
        meta = default_assets
        keys = list(meta.keys())

    n = len(keys)
    mu = np.array([meta[k][1] / 252 for k in keys])
    sigma = np.array([meta[k][2] / np.sqrt(252) for k in keys])

    groups = [meta[k][0] for k in keys]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] == groups[j]:
                base = 0.7 if groups[i] == "Equity" else 0.55
            elif {"Equity", "Alternatives"} == {groups[i], groups[j]}:
                base = 0.45
            elif {"Equity", "FixedIncome"} == {groups[i], groups[j]}:
                base = -0.05
            else:
                base = 0.2
            jitter = rng.uniform(-0.05, 0.05)
            corr[i, j] = corr[j, i] = float(np.clip(base + jitter, -0.95, 0.95))

    cov = corr * np.outer(sigma, sigma)
    cov = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 1e-10, None)
    cov = (eigvec * eigval) @ eigvec.T

    log_rets = rng.multivariate_normal(mu, cov, size=n_periods)
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    dates = pd.bdate_range(end=SAMPLE_END, periods=n_periods)
    return pd.DataFrame(prices, index=dates, columns=keys)
