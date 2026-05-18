"""Training-load metrics: acute/chronic load, ACWR, monotony, strain.

Implements the load "Derived metrics" from framework.md §9. ACWR drives the
§9 / §11 safety gate (sweet spot 0.8–1.3, hard stop > 1.5; Gabbett 2016,
Maupin 2020). Monotony / strain are Foster's (Med Sci Sports Exerc
1998;30(7):1164–8) load-distribution metrics — monotony > 2.0 is a §9
deload trigger.

Conventions match :mod:`src.metrics.baselines`: input frame is never
mutated; results are date-indexed Series; insufficient history is ``NaN``.
Monotony is the one metric where a zero SD returns ``inf`` (degenerate but
maximally "monotonous" load), mirroring the framework reference
``return m / s if s > 0 else float("inf")``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.baselines import _metric_series

# Acute = last 7 days; chronic = last 28 days (framework.md §8/§9).
_ACUTE = 7
_CHRONIC = 28
# Chronic strain sum is normalised to a weekly equivalent (framework.md §9:
# "28-day rolling sum divided by 4").
_CHRONIC_WEEKS = 4
# Default Foster monotony/strain window (framework.md §9 uses a 7-day week).
_MONOTONY_WINDOW = 7


def acute_load(df: pd.DataFrame) -> pd.Series:
    """7-day rolling sum of ``day_strain`` (ACWR numerator, fatigue proxy)."""
    return (
        _metric_series(df, "day_strain")
        .rolling(_ACUTE)
        .sum()
        .rename("acute_load")
    )


def chronic_load(df: pd.DataFrame) -> pd.Series:
    """28-day rolling sum of ``day_strain`` / 4 (weekly-equivalent fitness)."""
    return (
        _metric_series(df, "day_strain")
        .rolling(_CHRONIC)
        .sum()
        .div(_CHRONIC_WEEKS)
        .rename("chronic_load")
    )


def acwr(df: pd.DataFrame) -> pd.Series:
    """Acute:chronic workload ratio = ``acute_load`` / ``chronic_load``.

    Sweet spot 0.8–1.3; > 1.5 = danger (framework.md §9/§11). A zero
    chronic load (no logged strain in the 28-day window) makes the ratio
    undefined and returns ``NaN``.
    """
    acute = acute_load(df)
    chronic = chronic_load(df)
    ratio = acute / chronic
    ratio = ratio.where(chronic != 0)  # zero chronic load -> undefined
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.rename("acwr")


def training_monotony(df: pd.DataFrame, window: int = _MONOTONY_WINDOW) -> pd.Series:
    """Foster monotony: rolling mean(day_strain) / SD(day_strain).

    > 2.0 over the past 7 days is a §9 deload trigger. Per the framework
    reference, a zero SD (perfectly flat load) returns ``inf``; insufficient
    history returns ``NaN``.
    """
    day_strain = _metric_series(df, "day_strain")
    mean_ = day_strain.rolling(window).mean()
    sd = day_strain.rolling(window).std()
    monotony = mean_ / sd
    # sd == 0 -> inf (framework `else float("inf")`); sd is NaN when there
    # is insufficient history, and NaN == 0 is False so those stay NaN.
    monotony = monotony.mask(sd == 0, np.inf)
    return monotony.rename("training_monotony")


def training_strain(df: pd.DataFrame, window: int = _MONOTONY_WINDOW) -> pd.Series:
    """Foster strain: rolling weekly load × monotony (framework.md §9).

    Tracked week-over-week as an overall load-stress signal.
    """
    day_strain = _metric_series(df, "day_strain")
    weekly_load = day_strain.rolling(window).sum()
    monotony = training_monotony(df, window)
    return (weekly_load * monotony).rename("training_strain")
