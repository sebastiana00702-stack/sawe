"""Schema validation tests for the Phase 1 Pydantic models.

Each model is checked for: (a) a known-good payload validates, and (b)
out-of-range values, bad literals, missing required fields, and unexpected
extra fields all raise ``ValidationError``. Threshold boundaries from
framework.md §8/§9 (recovery 0-100, strain 0-21, sleep performance 0-1) are
asserted exactly at the edge.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models import (
    Recommendation,
    RunnerProfile,
    WeeklyPlan,
    WhoopDaily,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CSV = REPO_ROOT / "data" / "fake_whoop.csv"


# --------------------------------------------------------------------------
# WhoopDaily
# --------------------------------------------------------------------------

def valid_whoop_kwargs() -> dict:
    return dict(
        date=date(2026, 5, 18),
        recovery_score=58,
        hrv_rmssd=64.2,
        rhr=49,
        sleep_performance=0.88,
        sleep_hours=7.4,
        sleep_need_hours=8.0,
        rem_min=95,
        sws_min=80,
        light_min=270,
        day_strain=12.5,
        workout_strain=10.1,
        workout_hr_mean=148,
        workout_hr_max=176,
        zone_minutes={"Z1": 35.0, "Z2": 6.0, "Z3": 8.0, "Z4": 2.0, "Z5": 0.0},
        respiratory_rate=14.6,
        skin_temp_dev_c=0.1,
        journal={"soreness": 1},
    )


def test_whoop_daily_valid():
    w = WhoopDaily(**valid_whoop_kwargs())
    assert w.recovery_score == 58
    assert w.workout_strain == 10.1


def test_whoop_daily_optionals_default_none():
    kw = valid_whoop_kwargs()
    for k in ("workout_strain", "workout_hr_mean", "workout_hr_max",
              "skin_temp_dev_c"):
        kw.pop(k)
    kw.pop("journal")
    w = WhoopDaily(**kw)
    assert w.workout_strain is None
    assert w.skin_temp_dev_c is None
    assert w.journal == {}


@pytest.mark.parametrize("recovery", [0, 100])
def test_whoop_daily_recovery_boundaries_ok(recovery):
    WhoopDaily(**{**valid_whoop_kwargs(), "recovery_score": recovery})


@pytest.mark.parametrize("recovery", [-1, 101, 150])
def test_whoop_daily_recovery_out_of_range(recovery):
    with pytest.raises(ValidationError):
        WhoopDaily(**{**valid_whoop_kwargs(), "recovery_score": recovery})


def test_whoop_daily_day_strain_boundary_and_overflow():
    WhoopDaily(**{**valid_whoop_kwargs(), "day_strain": 21.0})  # ok at ceiling
    with pytest.raises(ValidationError):
        WhoopDaily(**{**valid_whoop_kwargs(), "day_strain": 21.5})


@pytest.mark.parametrize("hrv", [0, -5.0])
def test_whoop_daily_hrv_must_be_positive(hrv):
    with pytest.raises(ValidationError):
        WhoopDaily(**{**valid_whoop_kwargs(), "hrv_rmssd": hrv})


def test_whoop_daily_sleep_performance_out_of_range():
    with pytest.raises(ValidationError):
        WhoopDaily(**{**valid_whoop_kwargs(), "sleep_performance": 1.5})


def test_whoop_daily_missing_required_field():
    kw = valid_whoop_kwargs()
    del kw["rhr"]
    with pytest.raises(ValidationError):
        WhoopDaily(**kw)


def test_whoop_daily_extra_field_forbidden():
    with pytest.raises(ValidationError):
        WhoopDaily(**{**valid_whoop_kwargs(), "unexpected": 1})


def test_whoop_daily_bad_zone_key():
    with pytest.raises(ValidationError):
        WhoopDaily(**{**valid_whoop_kwargs(), "zone_minutes": {"Z9": 5.0}})


def test_whoop_daily_negative_zone_minutes():
    with pytest.raises(ValidationError):
        WhoopDaily(
            **{**valid_whoop_kwargs(), "zone_minutes": {"Z1": -3.0}}
        )


# --------------------------------------------------------------------------
# RunnerProfile
# --------------------------------------------------------------------------

def valid_profile_kwargs() -> dict:
    return dict(
        tier="intermediate",
        goal="10K",
        age=28,
        sex="M",
        current_mpw=32.0,
        longest_recent_run_mi=12.0,
    )


def test_runner_profile_valid():
    p = RunnerProfile(**valid_profile_kwargs())
    assert p.tier == "intermediate"


def test_runner_profile_hrmax_formula_and_measured():
    p = RunnerProfile(**{**valid_profile_kwargs(), "age": 30})
    assert p.hrmax() == round(208 - 0.7 * 30)  # 187
    p2 = RunnerProfile(**{**valid_profile_kwargs(), "hrmax_measured": 195})
    assert p2.hrmax() == 195


def test_runner_profile_bad_tier_literal():
    with pytest.raises(ValidationError):
        RunnerProfile(**{**valid_profile_kwargs(), "tier": "elite"})


def test_runner_profile_bad_goal_literal():
    with pytest.raises(ValidationError):
        RunnerProfile(**{**valid_profile_kwargs(), "goal": "ultra"})


def test_runner_profile_age_out_of_range():
    with pytest.raises(ValidationError):
        RunnerProfile(**{**valid_profile_kwargs(), "age": 3})


def test_runner_profile_missing_required_field():
    kw = valid_profile_kwargs()
    del kw["current_mpw"]
    with pytest.raises(ValidationError):
        RunnerProfile(**kw)


# --------------------------------------------------------------------------
# Recommendation  (mirrors framework.md §12 example JSON)
# --------------------------------------------------------------------------

def valid_recommendation_kwargs() -> dict:
    return dict(
        date=date(2026, 5, 18),
        athlete_id="u_001",
        training_state="functional",
        readiness="moderate",
        recommendation=dict(
            type="threshold_intervals",
            duration_min=50,
            structure="10 min wu + 4x8 min @ T + 10 min cd",
            intensity_target={"hr_pct_max": [88, 92], "rpe": 7,
                              "pace_label": "T"},
            rationale=["Recovery 58% (Yellow)", "ACWR 1.12 in sweet spot"],
            downgrade_path={"if_recovery_drops_below_50": "40 min easy"},
            alternatives=[
                {"type": "easy_run", "duration_min": 45},
                {"type": "cross_train", "modality": "bike",
                 "duration_min": 60},
            ],
            warmup="5-10 min easy jog, leg swings, 4 strides 80m",
            cooldown="10 min easy jog, static stretch 30s each",
        ),
        flags=[],
    )


def test_recommendation_valid_and_defaults():
    r = Recommendation(**valid_recommendation_kwargs())
    assert r.requires_medical_review is False
    assert "Not medical advice" in r.recommendation.disclaimer
    assert r.recommendation.intensity_target.hr_pct_max == (88, 92)


def test_recommendation_bad_training_state():
    kw = valid_recommendation_kwargs()
    kw["training_state"] = "great"
    with pytest.raises(ValidationError):
        Recommendation(**kw)


def test_recommendation_bad_readiness():
    kw = valid_recommendation_kwargs()
    kw["readiness"] = "fine"
    with pytest.raises(ValidationError):
        Recommendation(**kw)


def test_recommendation_rpe_out_of_range():
    kw = valid_recommendation_kwargs()
    kw["recommendation"]["intensity_target"]["rpe"] = 12
    with pytest.raises(ValidationError):
        Recommendation(**kw)


def test_recommendation_missing_workout():
    kw = valid_recommendation_kwargs()
    del kw["recommendation"]
    with pytest.raises(ValidationError):
        Recommendation(**kw)


# --------------------------------------------------------------------------
# WeeklyPlan  (mirrors framework.md §12 example JSON)
# --------------------------------------------------------------------------

def valid_weekly_plan_kwargs() -> dict:
    return dict(
        week_starting=date(2026, 5, 18),
        athlete_id="u_001",
        tier="intermediate",
        goal="10K",
        phase="specific_prep",
        planned_mpw=32,
        days=[
            {"date": "2026-05-18", "type": "easy", "duration_min": 40},
            {"date": "2026-05-19", "type": "vo2max",
             "structure": "4x4 min @ 90-95% HRmax"},
            {"date": "2026-05-22", "type": "rest"},
            {"date": "2026-05-24", "type": "long_run", "duration_min": 90},
        ],
        deload_due_in_weeks=2,
        acwr_target=[0.9, 1.2],
    )


def test_weekly_plan_valid():
    plan = WeeklyPlan(**valid_weekly_plan_kwargs())
    assert plan.acwr_target == (0.9, 1.2)
    assert plan.days[1].type == "vo2max"


def test_weekly_plan_bad_workout_type():
    kw = valid_weekly_plan_kwargs()
    kw["days"][0]["type"] = "crossfit"
    with pytest.raises(ValidationError):
        WeeklyPlan(**kw)


def test_weekly_plan_negative_mpw():
    with pytest.raises(ValidationError):
        WeeklyPlan(**{**valid_weekly_plan_kwargs(), "planned_mpw": -5})


def test_weekly_plan_inverted_acwr_band():
    with pytest.raises(ValidationError):
        WeeklyPlan(**{**valid_weekly_plan_kwargs(), "acwr_target": [1.4, 0.9]})


def test_weekly_plan_requires_at_least_one_day():
    with pytest.raises(ValidationError):
        WeeklyPlan(**{**valid_weekly_plan_kwargs(), "days": []})


# --------------------------------------------------------------------------
# Integration: the generated CSV must round-trip through WhoopDaily
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not FAKE_CSV.exists(),
    reason="run scripts/generate_fake_whoop.py first",
)
def test_generated_csv_all_rows_validate():
    import pandas as pd

    df = pd.read_csv(FAKE_CSV)
    assert len(df) == 90
    for rec in df.to_dict(orient="records"):
        rec = dict(rec)
        rec["zone_minutes"] = json.loads(rec["zone_minutes"])
        rec["journal"] = json.loads(rec["journal"])
        for opt in ("workout_strain", "workout_hr_mean", "workout_hr_max",
                    "skin_temp_dev_c"):
            if pd.isna(rec[opt]):
                rec[opt] = None
        WhoopDaily.model_validate(rec)

    assert (df["recovery_score"] < 34).sum() >= 3  # injected bad/illness days
