"""Unit + integration tests for the Phase 2 derived metrics.

Strategy:

* Hand-calculated expected values on tiny fixtures for the 7-day metrics
  (means / sums small enough to verify by arithmetic).
* An independent NumPy reference (``ddof=1`` sample SD, matching pandas
  ``rolling().std()``) for the 28-day window metrics.
* Explicit edge cases: empty frame, single row, exactly-at-window-boundary,
  zero-SD (NaN for z-score, ``inf`` for monotony), input not mutated, and
  unsorted input still aligned by calendar date.
* An integration test over ``data/fake_whoop.csv`` asserting the metrics
  land in physiologically sensible ranges (framework.md §8/§9).
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    acute_load,
    acwr,
    chronic_load,
    hrv_baseline_7d,
    hrv_baseline_28d,
    hrv_pct_deviation,
    hrv_sd_28d,
    hrv_zscore,
    resp_rate_baseline,
    resp_rate_deviation,
    rhr_baseline_7d,
    rhr_delta,
    training_monotony,
    training_strain,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CSV = REPO_ROOT / "data" / "fake_whoop.csv"

START = date(2026, 1, 1)


def _df(**columns: list) -> pd.DataFrame:
    """Build a WhoopDaily-shaped frame with a ``date`` column.

    Only the columns a metric reads are required; ``date`` is generated as
    consecutive days so row order == calendar order unless reordered.
    """
    n = len(next(iter(columns.values())))
    dates = [START + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({"date": dates, **columns})


# Tiny hand-checkable fixture: arithmetic progressions.
HRV_10 = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]
RHR_10 = [48, 50, 52, 54, 56, 58, 60, 62, 64, 66]
STRAIN_10 = [10, 12, 8, 14, 6, 16, 4, 18, 2, 20]


# --------------------------------------------------------------------------
# 7-day baselines — hand-calculated
# --------------------------------------------------------------------------

def test_hrv_baseline_7d_handcalc():
    s = hrv_baseline_7d(_df(hrv_rmssd=HRV_10)).reset_index(drop=True)
    assert s.iloc[:6].isna().all()                 # insufficient history
    assert s.iloc[6] == pytest.approx(462 / 7)     # mean(60..72) = 66.0
    assert s.iloc[7] == pytest.approx(476 / 7)     # mean(62..74) = 68.0
    assert s.iloc[9] == pytest.approx(504 / 7)     # mean(66..78) = 72.0
    assert s.name == "hrv_baseline_7d"


def test_rhr_baseline_7d_handcalc():
    s = rhr_baseline_7d(_df(rhr=RHR_10)).reset_index(drop=True)
    assert s.iloc[5] != s.iloc[5] or pd.isna(s.iloc[5])  # NaN before window
    assert s.iloc[6] == pytest.approx(378 / 7)     # mean(48..60) = 54.0
    assert s.iloc[9] == pytest.approx(420 / 7)     # mean(54..66) = 60.0


def test_hrv_pct_deviation_handcalc():
    s = hrv_pct_deviation(_df(hrv_rmssd=HRV_10)).reset_index(drop=True)
    # idx6: today 72, 7d mean 66 -> (72-66)/66*100
    assert s.iloc[6] == pytest.approx((72 - 66) / 66 * 100)
    # idx9: today 78, 7d mean 72 -> (78-72)/72*100
    assert s.iloc[9] == pytest.approx((78 - 72) / 72 * 100)
    assert s.iloc[:6].isna().all()


def test_acute_load_handcalc():
    s = acute_load(_df(day_strain=STRAIN_10)).reset_index(drop=True)
    assert s.iloc[:6].isna().all()
    assert s.iloc[6] == pytest.approx(10 + 12 + 8 + 14 + 6 + 16 + 4)   # 70
    assert s.iloc[7] == pytest.approx(12 + 8 + 14 + 6 + 16 + 4 + 18)   # 78
    assert s.iloc[9] == pytest.approx(14 + 6 + 16 + 4 + 18 + 2 + 20)   # 80


def test_training_monotony_and_strain_handcalc_small_window():
    df = _df(day_strain=STRAIN_10)
    mono = training_monotony(df, window=3).reset_index(drop=True)
    strain = training_strain(df, window=3).reset_index(drop=True)
    # window=3 at idx2 -> [10, 12, 8]: mean 10, sample SD = sqrt(8/2) = 2
    assert mono.iloc[:2].isna().all()
    assert mono.iloc[2] == pytest.approx(10 / 2.0)          # 5.0
    # strain = weekly_load * monotony = sum([10,12,8]) * 5.0 = 150
    assert strain.iloc[2] == pytest.approx(30 * 5.0)
    # window=3 at idx3 -> [12, 8, 14], verified against NumPy (ddof=1)
    win = np.array([12, 8, 14], dtype=float)
    expected = win.mean() / win.std(ddof=1)
    assert mono.iloc[3] == pytest.approx(expected)


# --------------------------------------------------------------------------
# 28-day window metrics — NumPy reference (ddof=1 matches pandas rolling.std)
# --------------------------------------------------------------------------

def _deterministic_hrv(n: int) -> list:
    rng = np.random.default_rng(7)
    return list(np.round(68 + 6 * np.sin(np.arange(n) / 4.0) + rng.normal(0, 4, n), 2))


def test_hrv_28d_baseline_sd_zscore_numpy_reference():
    vals = _deterministic_hrv(30)
    df = _df(hrv_rmssd=vals)
    arr = np.asarray(vals, dtype=float)

    base = hrv_baseline_28d(df).reset_index(drop=True)
    sd = hrv_sd_28d(df).reset_index(drop=True)
    z = hrv_zscore(df).reset_index(drop=True)

    # First defined value is exactly at the 28-day boundary (idx 27).
    assert base.iloc[26] != base.iloc[26] or pd.isna(base.iloc[26])
    assert sd.iloc[:27].isna().all()

    for i in (27, 28, 29):
        window = arr[i - 27 : i + 1]               # last 28 samples
        assert base.iloc[i] == pytest.approx(window.mean())
        assert sd.iloc[i] == pytest.approx(window.std(ddof=1))
        assert z.iloc[i] == pytest.approx(
            (arr[i] - window.mean()) / window.std(ddof=1)
        )


def test_rhr_delta_numpy_reference():
    rng = np.random.default_rng(3)
    vals = list(48 + rng.integers(-4, 5, 30))
    df = _df(rhr=vals)
    arr = np.asarray(vals, dtype=float)
    d = rhr_delta(df).reset_index(drop=True)
    assert d.iloc[:27].isna().all()
    for i in (27, 29):
        assert d.iloc[i] == pytest.approx(arr[i] - arr[i - 27 : i + 1].mean())


def test_resp_rate_baseline_and_deviation_numpy_reference():
    rng = np.random.default_rng(5)
    vals = list(np.round(14.5 + rng.normal(0, 0.4, 30), 2))
    df = _df(respiratory_rate=vals)
    arr = np.asarray(vals, dtype=float)
    base = resp_rate_baseline(df).reset_index(drop=True)
    dev = resp_rate_deviation(df).reset_index(drop=True)
    assert base.iloc[:27].isna().all()
    for i in (27, 29):
        window = arr[i - 27 : i + 1]
        assert base.iloc[i] == pytest.approx(window.mean())
        assert dev.iloc[i] == pytest.approx(arr[i] - window.mean())


def test_acwr_numpy_reference():
    rng = np.random.default_rng(11)
    vals = list(np.round(np.clip(rng.normal(10, 3, 30), 0, 21), 1))
    df = _df(day_strain=vals)
    arr = np.asarray(vals, dtype=float)
    a = acwr(df).reset_index(drop=True)
    assert a.iloc[:27].isna().all()                # needs 28d chronic
    for i in (27, 29):
        acute = arr[i - 6 : i + 1].sum()           # 7-day sum
        chronic = arr[i - 27 : i + 1].sum() / 4    # weekly-equivalent
        assert a.iloc[i] == pytest.approx(acute / chronic)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_empty_frame_returns_empty_series():
    empty = pd.DataFrame({"date": [], "hrv_rmssd": [], "day_strain": [],
                          "rhr": [], "respiratory_rate": []})
    for fn in (hrv_baseline_7d, hrv_zscore, hrv_pct_deviation):
        out = fn(empty)
        assert isinstance(out, pd.Series) and out.empty
    assert acwr(empty).empty
    assert training_monotony(empty).empty


def test_single_row_is_all_nan():
    df = _df(hrv_rmssd=[70.0], day_strain=[10.0], rhr=[50],
             respiratory_rate=[14.0])
    assert pd.isna(hrv_baseline_7d(df).iloc[0])
    assert pd.isna(acute_load(df).iloc[0])
    assert pd.isna(training_monotony(df).iloc[0])
    assert pd.isna(hrv_zscore(df).iloc[0])


def test_exactly_at_window_boundary():
    # 7 rows: idx 0-5 NaN, idx 6 is the first defined 7-day mean.
    df = _df(hrv_rmssd=HRV_10[:7])
    s = hrv_baseline_7d(df).reset_index(drop=True)
    assert s.iloc[:6].isna().all()
    assert s.iloc[6] == pytest.approx(sum(HRV_10[:7]) / 7)


def test_zero_sd_zscore_is_nan_not_inf():
    df = _df(hrv_rmssd=[55.0] * 28)               # perfectly flat -> SD 0
    z = hrv_zscore(df).reset_index(drop=True)
    assert pd.isna(z.iloc[26])                     # insufficient history
    assert pd.isna(z.iloc[27])                     # 0/0, undefined -> NaN
    assert not np.isinf(z.to_numpy(dtype=float)).any()
    assert hrv_sd_28d(df).reset_index(drop=True).iloc[27] == pytest.approx(0.0)


def test_zero_sd_monotony_is_inf():
    df = _df(day_strain=[12.0] * 10)              # flat load -> SD 0
    mono = training_monotony(df, window=5).reset_index(drop=True)
    assert mono.iloc[:4].isna().all()             # insufficient history
    assert np.isinf(mono.iloc[4]) and mono.iloc[4] > 0
    # strain = weekly_load * inf -> inf when weekly load > 0
    strain = training_strain(df, window=5).reset_index(drop=True)
    assert np.isinf(strain.iloc[4])


def test_acwr_zero_chronic_load_is_nan():
    df = _df(day_strain=[0.0] * 30)               # no load at all
    a = acwr(df).reset_index(drop=True)
    assert pd.isna(a.iloc[29])                     # 0/0 -> NaN, not inf
    assert not np.isinf(a.to_numpy(dtype=float)).any()


def test_input_frame_not_mutated():
    df = _df(hrv_rmssd=HRV_10, rhr=RHR_10, day_strain=STRAIN_10,
             respiratory_rate=[14.0] * 10)
    before = copy.deepcopy(df)
    for fn in (hrv_baseline_7d, hrv_baseline_28d, hrv_sd_28d, hrv_zscore,
               hrv_pct_deviation, rhr_baseline_7d, rhr_delta,
               resp_rate_baseline, resp_rate_deviation, acute_load,
               chronic_load, acwr, training_monotony, training_strain):
        fn(df)
    pd.testing.assert_frame_equal(df, before)


def test_unsorted_input_is_aligned_by_calendar_date():
    df = _df(hrv_rmssd=HRV_10)
    shuffled = df.iloc[[9, 0, 5, 2, 7, 1, 8, 3, 6, 4]].reset_index(drop=True)
    ordered = hrv_baseline_7d(df)
    out = hrv_baseline_7d(shuffled)
    # Result is indexed by a sorted DatetimeIndex regardless of row order.
    assert out.index.is_monotonic_increasing
    assert isinstance(out.index, pd.DatetimeIndex)
    pd.testing.assert_series_equal(out, ordered)


# --------------------------------------------------------------------------
# Integration: metrics over the generated 90-day dataset
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not FAKE_CSV.exists(),
    reason="run scripts/generate_fake_whoop.py first",
)
def test_fake_whoop_metrics_in_sensible_ranges():
    df = pd.read_csv(FAKE_CSV)
    assert len(df) == 90

    a = acwr(df).dropna()
    m = training_monotony(df).dropna()
    z = hrv_zscore(df).dropna()

    # First defined values land exactly on the window boundaries.
    assert acute_load(df).reset_index(drop=True).first_valid_index() == 6
    assert chronic_load(df).reset_index(drop=True).first_valid_index() == 27

    # ACWR: strictly positive, normal weeks well under the 1.5 danger gate;
    # nothing should approach the implausible >3 region (framework.md §9).
    assert (a > 0).all()
    assert a.max() < 3.0
    assert 0.8 <= a.median() <= 1.4

    # Foster monotony: 1-5 is the typical band (>2 is the deload trigger);
    # the synthetic athlete has variety so it stays modest.
    assert np.isfinite(m).all()
    assert m.min() >= 0.5
    assert (m <= 5).mean() >= 0.9
    assert 1.0 <= m.median() <= 4.0

    # HRV z-scores: mean near zero, essentially all within +/- 3 SD even
    # with the injected illness/bad-recovery windows.
    assert abs(z.mean()) < 1.0
    assert (z.abs() <= 3).mean() >= 0.95
    assert z.abs().max() < 5.0
