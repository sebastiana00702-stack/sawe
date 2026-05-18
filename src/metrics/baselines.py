"""Personal rolling baselines and deviation metrics.

Implements the HRV / RHR / respiratory-rate "Derived metrics" from
framework.md §9 plus the deviation forms the threshold table in §9 reasons
over (HRV z-score, HRV % deviation, RHR delta, respiratory-rate deviation).

Design rules (shared by ``src/metrics/load.py``):

* Functions accept a WhoopDaily-shaped :class:`pandas.DataFrame`. ``date``
  may be a column or the index.
* The input frame is **never mutated**.
* Every function returns a :class:`pandas.Series` indexed by a sorted
  ``DatetimeIndex`` so metrics from different functions align by calendar
  date regardless of input row order.
* Insufficient history yields ``NaN`` (rolling ``min_periods`` defaults to
  the full window, matching the framework reference code). A zero standard
  deviation yields ``NaN`` here (z-score is undefined); monotony in
  ``load.py`` is the only metric that returns ``inf`` for zero SD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# rMSSD trend window (framework.md §8: "Use 7-day rolling mean").
_HRV_SHORT = 7
# Personal baseline / dispersion window (framework.md §9 derived metrics
# use a 28-day rolling mean & SD for the HRV z-score and RHR delta).
_BASELINE = 28


def _metric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return ``df[column]`` as a date-indexed, chronologically sorted Series.

    ``date`` may be a column or the frame index. The result is indexed by a
    ``DatetimeIndex`` (sorted ascending) and carries float dtype so rolling
    statistics behave consistently. The input frame is copied out via
    ``to_numpy`` and never mutated.

    Shared by :mod:`src.metrics.load`.
    """
    if column not in df.columns:
        raise KeyError(f"DataFrame is missing required column {column!r}")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"].to_numpy())
    else:
        dates = pd.to_datetime(df.index.to_numpy())

    series = pd.Series(
        df[column].to_numpy(dtype="float64", na_value=np.nan),
        index=pd.DatetimeIndex(dates, name="date"),
        name=column,
    )
    return series.sort_index(kind="stable")


# --------------------------------------------------------------------------
# HRV (rMSSD)
# --------------------------------------------------------------------------

def hrv_baseline_7d(df: pd.DataFrame) -> pd.Series:
    """Rolling 7-day mean of ``hrv_rmssd`` (framework.md §8 trend baseline)."""
    return (
        _metric_series(df, "hrv_rmssd")
        .rolling(_HRV_SHORT)
        .mean()
        .rename("hrv_baseline_7d")
    )


def hrv_baseline_28d(df: pd.DataFrame) -> pd.Series:
    """Rolling 28-day mean of ``hrv_rmssd`` (personal baseline)."""
    return (
        _metric_series(df, "hrv_rmssd")
        .rolling(_BASELINE)
        .mean()
        .rename("hrv_baseline_28d")
    )


def hrv_sd_28d(df: pd.DataFrame) -> pd.Series:
    """Rolling 28-day sample SD of ``hrv_rmssd`` (ddof=1, pandas default)."""
    return (
        _metric_series(df, "hrv_rmssd")
        .rolling(_BASELINE)
        .std()
        .rename("hrv_sd_28d")
    )


def hrv_zscore(df: pd.DataFrame) -> pd.Series:
    """(today − 28-day mean) / 28-day SD.

    Used by the §9 overreaching detector ("HRV today < baseline − 1 SD").
    A zero SD makes the z-score undefined, so it returns ``NaN`` (not
    ``inf``); insufficient history likewise returns ``NaN``.
    """
    hrv = _metric_series(df, "hrv_rmssd")
    mean28 = hrv.rolling(_BASELINE).mean()
    sd28 = hrv.rolling(_BASELINE).std()
    z = (hrv - mean28) / sd28
    z = z.where(sd28 != 0)  # sd == 0 -> undefined -> NaN
    z = z.replace([np.inf, -np.inf], np.nan)
    return z.rename("hrv_zscore")


def hrv_pct_deviation(df: pd.DataFrame) -> pd.Series:
    """(today − 7-day mean) / 7-day mean × 100, as a percentage.

    A readability-friendly companion to the z-score; HRV rMSSD is strictly
    positive so the 7-day mean is non-zero in practice, but a zero/NaN mean
    is guarded to ``NaN`` for consistency.
    """
    hrv = _metric_series(df, "hrv_rmssd")
    mean7 = hrv.rolling(_HRV_SHORT).mean()
    pct = (hrv - mean7) / mean7 * 100.0
    pct = pct.where(mean7 != 0)
    pct = pct.replace([np.inf, -np.inf], np.nan)
    return pct.rename("hrv_pct_deviation")


# --------------------------------------------------------------------------
# Resting heart rate
# --------------------------------------------------------------------------

def rhr_baseline_7d(df: pd.DataFrame) -> pd.Series:
    """Rolling 7-day mean of ``rhr`` (framework.md §8 RHR baseline)."""
    return (
        _metric_series(df, "rhr")
        .rolling(_HRV_SHORT)
        .mean()
        .rename("rhr_baseline_7d")
    )


def rhr_delta(df: pd.DataFrame) -> pd.Series:
    """today RHR − 28-day rolling mean (framework.md §9 ``rhr_delta``).

    The §9 threshold table flags ``rhr_delta`` > 5 (caution) and > 7
    (illness suspect when paired with an HRV crash).
    """
    rhr = _metric_series(df, "rhr")
    return (rhr - rhr.rolling(_BASELINE).mean()).rename("rhr_delta")


# --------------------------------------------------------------------------
# Respiratory rate (illness early-warning, framework.md §8)
# --------------------------------------------------------------------------

def resp_rate_baseline(df: pd.DataFrame) -> pd.Series:
    """Rolling 28-day mean of ``respiratory_rate``."""
    return (
        _metric_series(df, "respiratory_rate")
        .rolling(_BASELINE)
        .mean()
        .rename("resp_rate_baseline")
    )


def resp_rate_deviation(df: pd.DataFrame) -> pd.Series:
    """today respiratory rate − 28-day baseline.

    The §8 illness signal is a sustained +2 br/min deviation.
    """
    rr = _metric_series(df, "respiratory_rate")
    return (rr - rr.rolling(_BASELINE).mean()).rename("resp_rate_deviation")
