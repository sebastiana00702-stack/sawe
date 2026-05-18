"""Phase 5 orchestrator tests: state helpers, recommender pipeline.

Covers framework.md §9 (decision logic), §10 (the override branches in
action), and §12 (the deliverable JSON shape):

* :mod:`src.agent.state` — ``trailing_streak`` / ``consecutive_red_streak``
  / ``days_since_last_hard`` at their boundaries, and the classifier
  delegation (incl. the documented ``"borderline"`` vs spec ``"fresh"``
  reconciliation).
* :func:`recommend_daily_workout` — every §9 override branch produces the
  right action/type/intensity, the result always validates against the
  *unchanged* Phase 1 schema, §11 medical flags ride through verbatim, and
  a mis-dated plan fails loudly.
* An integration sweep over ``data/fake_whoop.csv``.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.agent import (
    classify_readiness,
    classify_training_state,
    consecutive_red_streak,
    days_since_last_hard,
    recommend_daily_workout,
    trailing_streak,
)
from src.agent.state import HARD_DAY_STRAIN_MIN, NO_RECENT_HARD
from src.models import Recommendation, RunnerProfile, WhoopDaily
from src.models.weekly_plan import PlannedDay, WeeklyPlan
from src.rules import thresholds as T

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CSV = REPO_ROOT / "data" / "fake_whoop.csv"

TODAY = date(2026, 5, 18)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def whoop(d: date = TODAY, **over) -> WhoopDaily:
    """A clean all-green WHOOP day; override one field to isolate a path."""
    base = dict(
        date=d,
        recovery_score=80,
        hrv_rmssd=65.0,
        rhr=48,
        sleep_performance=0.95,
        sleep_hours=8.0,
        sleep_need_hours=8.0,
        rem_min=100,
        sws_min=90,
        light_min=260,
        day_strain=9.0,
        respiratory_rate=14.0,
        zone_minutes={"Z1": 40.0},
        journal={},
    )
    base.update(over)
    return WhoopDaily(**base)


# A mild 7-day strain cycle: SD > 0 (no spurious monotony=inf) and the
# 28-day sum keeps ACWR ≈ 1.0 (→ training_state "functional").
_STRAIN_CYCLE = [9.0, 10.0, 11.0, 10.0, 9.0, 12.0, 8.0]


def history(n: int = 35, end: date = TODAY - timedelta(days=1), **const) -> pd.DataFrame:
    """``n`` calm WhoopDaily rows ending ``end`` (the day before today)."""
    rows = []
    for i in range(n):
        d = end - timedelta(days=n - 1 - i)
        row = whoop(
            d,
            day_strain=_STRAIN_CYCLE[i % 7],
            hrv_rmssd=65.0 + (i % 3) - 1,   # tiny variation → SD > 0
            rhr=48 + (i % 2),
            respiratory_rate=14.0 + 0.1 * (i % 2),
        ).model_dump()
        row.update(const)
        rows.append(row)
    return pd.DataFrame(rows)


def plan_with(
    ptype: str,
    d: date = TODAY,
    duration_min: int | None = 40,
    structure: str | None = "planned session",
) -> WeeklyPlan:
    return WeeklyPlan(
        week_starting=d,
        athlete_id="u_001",
        tier="intermediate",
        goal="10K",
        phase="build",
        planned_mpw=32.0,
        days=[
            PlannedDay(
                date=d, type=ptype, duration_min=duration_min,
                structure=structure,
            )
        ],
        deload_due_in_weeks=2,
        acwr_target=(0.9, 1.2),
    )


PROFILE = RunnerProfile(
    tier="intermediate", goal="10K", age=30, sex="M",
    current_mpw=32.0, longest_recent_run_mi=14.0,
)


def _roundtrips(rec: Recommendation) -> None:
    """Every recommendation must validate against the unchanged schema."""
    Recommendation.model_validate(rec.model_dump())


# ==========================================================================
# trailing_streak
# ==========================================================================

def test_trailing_streak_counts_only_the_tail_run():
    assert trailing_streak([1, 1, 0, 1, 1, 1], lambda x: x == 1) == 3
    assert trailing_streak([1, 1, 1], lambda x: x == 1) == 3
    assert trailing_streak([1, 0], lambda x: x == 1) == 0
    assert trailing_streak([], lambda x: True) == 0


# ==========================================================================
# consecutive_red_streak — Red band (<= 33) at the tail
# ==========================================================================

def test_consecutive_red_streak_boundary_and_order():
    red = T.RECOVERY_RED_MAX           # 33, still Red
    not_red = T.RECOVERY_RED_MAX + 1   # 34, Yellow
    df = history(10)
    df.loc[df.index[-3:], "recovery_score"] = red
    assert consecutive_red_streak(df) == 3
    df.loc[df.index[-1], "recovery_score"] = not_red
    assert consecutive_red_streak(df) == 0          # tail no longer Red
    # 34 is the first non-Red value (boundary).
    df2 = history(5)
    df2["recovery_score"] = not_red
    assert consecutive_red_streak(df2) == 0
    df2["recovery_score"] = red
    assert consecutive_red_streak(df2) == 5
    # Unsorted input is sorted by date first.
    shuffled = df2.sample(frac=1.0, random_state=1)
    assert consecutive_red_streak(shuffled) == 5
    assert consecutive_red_streak(pd.DataFrame()) == 0


# ==========================================================================
# days_since_last_hard — calendar-diff, gap-robust
# ==========================================================================

def test_days_since_last_hard_basic_and_gaps():
    df = history(10)
    df["day_strain"] = 6.0                       # nothing hard
    assert days_since_last_hard(df) == NO_RECENT_HARD
    assert days_since_last_hard(pd.DataFrame()) == NO_RECENT_HARD

    # Last completed day hard → 1 (today = last_date + 1).
    df.loc[df.index[-1], "day_strain"] = HARD_DAY_STRAIN_MIN
    assert days_since_last_hard(df) == 1

    # Hard 3 calendar days before today (matches §12 "3 days ago").
    df["day_strain"] = 6.0
    df.loc[df.index[-3], "day_strain"] = HARD_DAY_STRAIN_MIN
    assert days_since_last_hard(df) == 3

    # Boundary: just below the strain cutoff is NOT hard.
    df["day_strain"] = 6.0
    df.loc[df.index[-1], "day_strain"] = HARD_DAY_STRAIN_MIN - 0.01
    assert days_since_last_hard(df) == NO_RECENT_HARD

    # Calendar diff, not row count: drop the (today-2) row, then mark the
    # (today-3) row hard. ref = last row (today-1); today = ref + 1, so the
    # last hard session is 3 calendar days before today.
    gapped = df.drop(df.index[-2]).copy()
    gapped["day_strain"] = 6.0
    gapped.loc[gapped.index[-2], "day_strain"] = HARD_DAY_STRAIN_MIN
    assert days_since_last_hard(gapped) == 3


def test_days_since_last_hard_zone_based_detection():
    """A short, sharp session: low whole-day strain but Z3+ minutes."""
    df = history(6)
    df["day_strain"] = 6.0
    last = df.index[-1]
    df.at[last, "zone_minutes"] = {"Z1": 30.0, "Z3": 8.0, "Z4": 6.0}  # 14 ≥ 12
    assert days_since_last_hard(df) == 1


# ==========================================================================
# Classifier delegation (+ "fresh"/"borderline" reconciliation)
# ==========================================================================

def test_classifiers_delegate_to_framework_section9():
    assert classify_readiness(67, 7.0, 3.0) == "high"
    assert classify_readiness(40, 8.0, 0.0) == "low"
    assert classify_training_state(1.0, None, None, 0) == "functional"
    assert classify_training_state(1.0, None, None, 2) == "overreached"
    # Spec asked for "fresh"; framework §9 + the frozen Phase 1 schema use
    # "borderline" for the 1.3 < ACWR <= 1.5 zone. We keep the schema
    # literal so the recommendation still validates.
    assert classify_training_state(1.4, None, None, 0) == "borderline"
    assert classify_training_state(1.4, None, None, 0) in set(
        Recommendation.model_fields["training_state"].annotation.__args__
    )


# ==========================================================================
# recommend_daily_workout — §9 override branches
# ==========================================================================

def test_clean_green_easy_day_proceeds():
    rec = recommend_daily_workout(whoop(), history(), PROFILE, plan_with("easy"))
    assert rec.training_state == "functional"
    assert rec.readiness == "high"
    assert rec.recommendation.type == "easy"
    assert rec.recommendation.duration_min == 40
    assert rec.recommendation.intensity_target.pace_label == "E"
    assert rec.recommendation.warmup and rec.recommendation.cooldown
    assert rec.requires_medical_review is False
    assert "All safety gates clear." in rec.recommendation.rationale
    _roundtrips(rec)


def test_recovery_red_forces_rest_or_walk():
    rec = recommend_daily_workout(
        whoop(recovery_score=T.RECOVERY_RED_MAX), history(), PROFILE,
        plan_with("vo2max"),
    )
    assert rec.recommendation.type == "rest_or_walk"
    assert rec.recommendation.duration_min == 0
    assert rec.recommendation.intensity_target is None
    assert rec.recommendation.warmup is None      # no running mobility
    # Rest-day alternatives are the gentle set.
    assert {a.type for a in rec.recommendation.alternatives} == {
        "rest_or_walk", "cross_train",
    }
    _roundtrips(rec)


def test_yellow_recovery_downgrades_quality_session():
    rec = recommend_daily_workout(
        whoop(recovery_score=60), history(), PROFILE, plan_with("vo2max"),
    )
    # §9: recovery < 67 downgrades a quality session; vo2max → easy+strides.
    assert rec.recommendation.type == "easy_with_strides"
    assert "Downgraded vo2max" in rec.recommendation.structure
    assert rec.recommendation.duration_min == 40   # type drop, load 1.0
    _roundtrips(rec)


def test_acwr_hard_stop_halves_load():
    # Build chronic load so ACWR > 1.5: a long calm base, then a spike.
    hist = history(35)
    hist.loc[hist.index[-7:], "day_strain"] = 19.0   # acute spike
    rec = recommend_daily_workout(
        whoop(day_strain=19.0), hist, PROFILE, plan_with("vo2max", duration_min=60),
    )
    assert rec.training_state == "strained"
    assert rec.recommendation.type == "easy_with_strides"
    assert rec.recommendation.duration_min == 30     # 60 × 0.5
    _roundtrips(rec)


def test_sleep_below_5h_rests():
    rec = recommend_daily_workout(
        whoop(sleep_hours=4.5), history(), PROFILE, plan_with("easy"),
    )
    assert rec.recommendation.type == "rest_or_walk"
    assert rec.recommendation.duration_min == 0
    _roundtrips(rec)


def test_medical_red_flag_rides_through_verbatim():
    rec = recommend_daily_workout(
        whoop(journal={"chest_pain": True}), history(), PROFILE,
        plan_with("easy"),
    )
    assert rec.requires_medical_review is True
    assert any("medical" in f and "Chest pain" in f for f in rec.flags)
    # The session still proceeds (easy) but the flag is never paraphrased.
    assert rec.recommendation.type == "easy"
    _roundtrips(rec)


def test_vo2max_spacing_converts_to_easy_strides():
    hist = history(35)
    hist.loc[hist.index[-1], "day_strain"] = 16.0   # hard yesterday
    rec = recommend_daily_workout(
        whoop(), hist, PROFILE, plan_with("vo2max"),
    )
    assert rec.recommendation.type == "easy_with_strides"
    assert "Converted vo2max" in rec.recommendation.structure
    _roundtrips(rec)


def test_missing_planned_day_raises():
    plan = plan_with("easy", d=TODAY + timedelta(days=3))
    with pytest.raises(ValueError, match="no day for"):
        recommend_daily_workout(whoop(), history(), PROFILE, plan)


def test_downgrade_path_keys_are_threshold_anchored():
    rec = recommend_daily_workout(whoop(), history(), PROFILE, plan_with("easy"))
    dp = rec.recommendation.downgrade_path
    assert f"if_recovery_drops_below_{T.READINESS_MODERATE_RECOVERY}" in dp
    assert f"if_sleep_below_{int(T.SLEEP_DOWNGRADE_BELOW_H)}h" in dp
    assert "if_two_consecutive_reds" in dp


def test_rationale_has_section12_style_lines():
    rec = recommend_daily_workout(whoop(), history(), PROFILE, plan_with("easy"))
    text = " | ".join(rec.recommendation.rationale)
    assert "Recovery 80% (Green)" in text
    assert "Sleep 8 h (above 6 h floor)" in text
    assert "in sweet spot (0.8-1.3)" in text


# ==========================================================================
# Integration sweep over the synthetic dataset
# ==========================================================================

@pytest.mark.skipif(not FAKE_CSV.exists(), reason="run scripts/generate_fake_whoop.py")
def test_integration_sweep_over_fake_whoop():
    df = pd.read_csv(FAKE_CSV)
    df["zone_minutes"] = df["zone_minutes"].map(json.loads)
    df["journal"] = df["journal"].map(
        lambda s: json.loads(s) if isinstance(s, str) else {}
    )

    def _to_whoop(row: pd.Series) -> WhoopDaily:
        # CSV serialises absent optionals as NaN; coerce back to None
        # per-record (mirrors scripts/generate_fake_whoop.py:_validate).
        rec = row.to_dict()
        for opt in ("workout_strain", "workout_hr_mean", "workout_hr_max",
                    "skin_temp_dev_c"):
            v = rec.get(opt)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                rec[opt] = None
        return WhoopDaily.model_validate(rec)

    # Walk the last 30 days; each must yield a schema-valid recommendation.
    seen_actions = set()
    for i in range(len(df) - 30, len(df)):
        hist = df.iloc[:i].reset_index(drop=True)
        today = _to_whoop(df.iloc[i])
        plan = plan_with(
            "vo2max", d=today.date, duration_min=50,
        )
        rec = recommend_daily_workout(today, hist, PROFILE, plan)
        _roundtrips(rec)
        assert rec.training_state in set(
            Recommendation.model_fields["training_state"].annotation.__args__
        )
        # Rest actions never carry a prescribed effort window.
        if rec.recommendation.type in {"rest", "rest_or_walk"}:
            assert rec.recommendation.intensity_target is None
        seen_actions.add(rec.recommendation.type)

    # The illness window (days 60-64) sits inside the sweep, so at least
    # one day must have been pulled off the planned vo2max.
    assert seen_actions != {"vo2max"}
