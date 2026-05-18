"""Boundary tests for the Phase 3 safety gates.

Per ``CLAUDE.md``: every rule has a unit test pinned to its *exact*
threshold. Each test asserts both sides of the boundary (the value that
must NOT trip and the next value that must), because the §9 reference
pseudocode mixes strict (``<``/``>``) and non-strict (``>=``/``<=``)
comparisons and getting one wrong is a silent safety failure.

Helpers build a clean "all-green" day / empty history so each test
isolates a single gate by overriding one field.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.whoop_daily import WhoopDaily
from src.rules import thresholds as T
from src.rules.gates import (
    FLAG_ACWR_DETRAIN,
    FLAG_ACWR_HARD_STOP,
    FLAG_FULL_REST,
    FLAG_HRV_LOW_PERSIST,
    FLAG_ILLNESS,
    FLAG_MANDATORY_REST,
    FLAG_MED_CARDIAC,
    FLAG_MED_PAIN,
    FLAG_MED_REDS,
    FLAG_MED_RHR_PERSIST,
    FLAG_MONOTONY_HIGH,
    FLAG_NFOR,
    FLAG_OVERREACHING,
    FLAG_RECOVERY_RED,
    FLAG_RECOVERY_YELLOW,
    FLAG_RHR_CAUTION,
    FLAG_SLEEP_DOWNGRADE,
    FLAG_SLEEP_PERFORMANCE,
    FLAG_SLEEP_REST,
    FLAG_SPACING,
    Flag,
    MetricHistory,
    choose_workout,
    classify_recovery,
    deload_due,
    deload_triggers,
    downgrade_workout_type,
    evaluate_safety_gates,
    next_week_target_mpw,
    overreaching,
    readiness,
    training_state,
)


def whoop(**over) -> WhoopDaily:
    """A clean all-green WHOOP day; override one field to isolate a gate."""
    base = dict(
        date=date(2026, 5, 18),
        recovery_score=80,
        hrv_rmssd=65.0,
        rhr=48,
        sleep_performance=0.95,
        sleep_hours=8.0,
        sleep_need_hours=8.0,
        rem_min=100,
        sws_min=90,
        light_min=260,
        day_strain=10.0,
        respiratory_rate=14.0,
        zone_minutes={"Z1": 40.0},
        journal={},
    )
    base.update(over)
    return WhoopDaily(**base)


def hist(**over) -> MetricHistory:
    return MetricHistory(**over)


def codes(report) -> set[str]:
    return set(report.codes())


# ==========================================================================
# classify_recovery — bands 0-33 / 34-66 / 67-100 (§8)
# ==========================================================================

def test_classify_recovery_band_boundaries():
    assert classify_recovery(0) == "red"
    assert classify_recovery(T.RECOVERY_RED_MAX) == "red"          # 33
    assert classify_recovery(T.RECOVERY_RED_MAX + 1) == "yellow"   # 34
    assert classify_recovery(T.RECOVERY_YELLOW_MAX) == "yellow"    # 66
    assert classify_recovery(T.RECOVERY_GREEN_MIN) == "green"      # 67
    assert classify_recovery(100) == "green"


# ==========================================================================
# readiness() — §9, mirrored exactly
# ==========================================================================

def test_readiness_high_boundary():
    # high needs rec>=67 AND sleep>=7 AND known rhr_delta<=3.
    assert readiness(67, 7.0, 3.0) == "high"
    assert readiness(66, 7.0, 3.0) == "moderate"          # rec one below
    assert readiness(67, 6.99, 3.0) == "moderate"         # sleep one below
    assert readiness(67, 7.0, 3.01) == "moderate"         # rhr one above
    assert readiness(67, 7.0, None) == "moderate"         # rhr unknown


def test_readiness_moderate_low_verylow_boundaries():
    assert readiness(50, 6.0, None) == "moderate"
    assert readiness(49, 6.0, None) == "low"              # rec<50
    assert readiness(50, 5.99, None) == "low"             # sleep<6 -> low
    assert readiness(T.READINESS_LOW_RECOVERY, 0.0, None) == "low"   # 34
    assert readiness(T.READINESS_LOW_RECOVERY - 1, 9.0, 0.0) == "very_low"


# ==========================================================================
# training_state() — §9, mirrored exactly
# ==========================================================================

def test_training_state_overreached_branches():
    assert training_state(1.0, None, None, 2) == "overreached"   # red_streak
    assert training_state(1.0, None, None, 1) != "overreached"
    # hrv_z<-1 AND rhr_delta>5 (both strict)
    assert training_state(1.0, -1.0, 6.0, 0) != "overreached"    # z not < -1
    assert training_state(1.0, -1.01, 5.0, 0) != "overreached"   # rhr not >5
    assert training_state(1.0, -1.01, 5.01, 0) == "overreached"


def test_training_state_acwr_bands():
    assert training_state(None, None, None, 0) == "borderline"   # acwr unknown
    assert training_state(T.ACWR_HARD_STOP, None, None, 0) == "borderline"   # 1.5 not >1.5
    assert training_state(T.ACWR_HARD_STOP + 0.01, None, None, 0) == "strained"
    assert training_state(T.ACWR_SWEET_HIGH, None, None, 0) == "functional"   # 1.3
    assert training_state(T.ACWR_SWEET_HIGH + 0.01, None, None, 0) == "borderline"
    assert training_state(T.ACWR_SWEET_LOW, None, None, 0) == "functional"    # 0.8
    assert training_state(T.ACWR_SWEET_LOW - 0.01, None, None, 0) == "detraining"


# ==========================================================================
# evaluate_safety_gates — one boundary per flag
# ==========================================================================

def test_full_rest_triggers_boundaries():
    assert FLAG_FULL_REST not in codes(
        evaluate_safety_gates(whoop(recovery_score=T.RECOVERY_FULL_REST_BELOW), hist())
    )                                                             # 20 not <20
    assert FLAG_FULL_REST in codes(
        evaluate_safety_gates(whoop(recovery_score=T.RECOVERY_FULL_REST_BELOW - 1), hist())
    )                                                             # 19
    assert FLAG_FULL_REST not in codes(
        evaluate_safety_gates(whoop(sleep_hours=T.SLEEP_FULL_REST_BELOW_H), hist())
    )                                                             # 4.0 not <4
    assert FLAG_FULL_REST in codes(
        evaluate_safety_gates(whoop(sleep_hours=3.99), hist())
    )
    assert FLAG_FULL_REST in codes(
        evaluate_safety_gates(whoop(journal={"fever": True}), hist())
    )
    assert FLAG_FULL_REST in codes(
        evaluate_safety_gates(whoop(journal={"sharp_pain_altering_gait": True}), hist())
    )


def test_recovery_band_flags_boundary():
    assert FLAG_RECOVERY_RED in codes(
        evaluate_safety_gates(whoop(recovery_score=T.RECOVERY_RED_MAX), hist())
    )                                                             # 33
    c34 = codes(evaluate_safety_gates(whoop(recovery_score=34), hist()))
    assert FLAG_RECOVERY_RED not in c34 and FLAG_RECOVERY_YELLOW in c34
    c66 = codes(evaluate_safety_gates(whoop(recovery_score=T.RECOVERY_YELLOW_MAX), hist()))
    assert FLAG_RECOVERY_YELLOW in c66                            # 66
    c67 = codes(evaluate_safety_gates(whoop(recovery_score=T.RECOVERY_GREEN_MIN), hist()))
    assert FLAG_RECOVERY_YELLOW not in c67 and FLAG_RECOVERY_RED not in c67


def test_sleep_flags_boundaries():
    # 4.99 -> rest floor (and not full rest, since >= 4).
    c = codes(evaluate_safety_gates(whoop(sleep_hours=4.99), hist()))
    assert FLAG_SLEEP_REST in c and FLAG_FULL_REST not in c
    # 5.0 -> not rest floor (<5 strict), but < 6 -> downgrade.
    c = codes(evaluate_safety_gates(whoop(sleep_hours=T.SLEEP_REST_BELOW_H), hist()))
    assert FLAG_SLEEP_REST not in c and FLAG_SLEEP_DOWNGRADE in c
    # 5.99 -> downgrade; 6.0 -> neither.
    assert FLAG_SLEEP_DOWNGRADE in codes(
        evaluate_safety_gates(whoop(sleep_hours=5.99), hist())
    )
    c6 = codes(evaluate_safety_gates(whoop(sleep_hours=T.SLEEP_DOWNGRADE_BELOW_H), hist()))
    assert FLAG_SLEEP_REST not in c6 and FLAG_SLEEP_DOWNGRADE not in c6


def test_sleep_performance_boundary():
    assert FLAG_SLEEP_PERFORMANCE not in codes(
        evaluate_safety_gates(
            whoop(sleep_performance=T.SLEEP_PERFORMANCE_DOWNGRADE_BELOW), hist()
        )
    )                                                             # 0.85 not <0.85
    assert FLAG_SLEEP_PERFORMANCE in codes(
        evaluate_safety_gates(whoop(sleep_performance=0.8499), hist())
    )


def test_mandatory_rest_red_streak_boundary():
    assert FLAG_MANDATORY_REST not in codes(
        evaluate_safety_gates(whoop(), hist(red_streak=1))
    )
    assert FLAG_MANDATORY_REST in codes(
        evaluate_safety_gates(whoop(), hist(red_streak=T.RED_STREAK_MANDATORY_REST))
    )


def test_illness_rhr_hrv_boundary():
    # rhr_delta>7 AND hrv_z<-1 (both strict).
    assert FLAG_ILLNESS not in codes(
        evaluate_safety_gates(whoop(), hist(rhr_delta=T.RHR_DELTA_ILLNESS_REST, hrv_z=-2.0))
    )                                                             # 7.0 not >7
    assert FLAG_ILLNESS not in codes(
        evaluate_safety_gates(whoop(), hist(rhr_delta=7.01, hrv_z=T.HRV_Z_LOW))
    )                                                             # z -1.0 not < -1
    assert FLAG_ILLNESS in codes(
        evaluate_safety_gates(whoop(), hist(rhr_delta=7.01, hrv_z=-1.01))
    )


def test_illness_rhr_rr_boundary():
    assert FLAG_ILLNESS not in codes(
        evaluate_safety_gates(
            whoop(), hist(rhr_delta=5.01, resp_rate_delta=T.RESP_RATE_DELTA_ILLNESS)
        )
    )                                                             # rr 2.0 not >2
    assert FLAG_ILLNESS in codes(
        evaluate_safety_gates(whoop(), hist(rhr_delta=5.01, resp_rate_delta=2.01))
    )


def test_rhr_caution_boundary_without_illness():
    # No HRV / RR data -> isolated RHR caution, not illness.
    assert FLAG_RHR_CAUTION not in codes(
        evaluate_safety_gates(whoop(), hist(rhr_delta=T.RHR_DELTA_CAUTION))
    )                                                             # 5.0 not >5
    c = codes(evaluate_safety_gates(whoop(), hist(rhr_delta=5.01)))
    assert FLAG_RHR_CAUTION in c and FLAG_ILLNESS not in c


def test_hrv_low_persist_boundary():
    assert FLAG_HRV_LOW_PERSIST not in codes(
        evaluate_safety_gates(whoop(), hist(hrv_low_streak_days=T.HRV_LOW_PERSIST_DAYS - 1))
    )                                                             # 2
    assert FLAG_HRV_LOW_PERSIST in codes(
        evaluate_safety_gates(whoop(), hist(hrv_low_streak_days=T.HRV_LOW_PERSIST_DAYS))
    )                                                             # 3


def test_acwr_flag_boundaries():
    assert FLAG_ACWR_HARD_STOP not in codes(
        evaluate_safety_gates(whoop(), hist(acwr=T.ACWR_HARD_STOP))
    )                                                             # 1.5 not >1.5
    assert FLAG_ACWR_HARD_STOP in codes(
        evaluate_safety_gates(whoop(), hist(acwr=1.51))
    )
    assert FLAG_ACWR_DETRAIN not in codes(
        evaluate_safety_gates(whoop(), hist(acwr=T.ACWR_SWEET_LOW))
    )                                                             # 0.8 not <0.8
    assert FLAG_ACWR_DETRAIN in codes(
        evaluate_safety_gates(whoop(), hist(acwr=0.79))
    )


def test_monotony_boundary():
    assert FLAG_MONOTONY_HIGH not in codes(
        evaluate_safety_gates(whoop(), hist(monotony=T.MONOTONY_HIGH))
    )                                                             # 2.0 not >2.0
    assert FLAG_MONOTONY_HIGH in codes(
        evaluate_safety_gates(whoop(), hist(monotony=2.01))
    )


def test_spacing_days_since_hard_boundary():
    assert FLAG_SPACING not in codes(
        evaluate_safety_gates(whoop(), hist(days_since_hard=T.DAYS_SINCE_HARD_MIN))
    )                                                             # 2 not <2
    assert FLAG_SPACING in codes(
        evaluate_safety_gates(whoop(), hist(days_since_hard=1))
    )
    assert FLAG_SPACING not in codes(
        evaluate_safety_gates(whoop(), hist(days_since_hard=None))
    )                                                             # unknown -> silent


def test_nfor_persistence_boundary():
    assert FLAG_NFOR not in codes(
        evaluate_safety_gates(whoop(), hist(poor_recovery_persist_days=T.NFOR_PERSIST_DAYS))
    )                                                             # 14 not >14
    assert FLAG_NFOR in codes(
        evaluate_safety_gates(whoop(), hist(poor_recovery_persist_days=15))
    )


# ==========================================================================
# §11 red flags -> requires_medical_review
# ==========================================================================

@pytest.mark.parametrize("journal_key", ["chest_pain", "syncope", "palpitations"])
def test_cardiac_red_flag_forces_medical(journal_key):
    r = evaluate_safety_gates(whoop(journal={journal_key: True}), hist())
    assert r.requires_medical_review is True
    assert FLAG_MED_CARDIAC in codes(r)


def test_rhr_persistent_red_flag_boundary():
    assert FLAG_MED_RHR_PERSIST not in codes(
        evaluate_safety_gates(
            whoop(), hist(rhr_red_flag_streak_days=T.RHR_RED_FLAG_PERSIST_DAYS)
        )
    )                                                             # 7 not >7
    r = evaluate_safety_gates(whoop(), hist(rhr_red_flag_streak_days=8))
    assert FLAG_MED_RHR_PERSIST in codes(r) and r.requires_medical_review


def test_red_s_red_flag_boundary():
    # hrv_z < -2 (strict) AND a RED-S correlate.
    assert FLAG_MED_REDS not in codes(
        evaluate_safety_gates(
            whoop(journal={"weight_loss": True}), hist(hrv_z=T.HRV_Z_RED_FLAG)
        )
    )                                                             # -2.0 not < -2
    r = evaluate_safety_gates(
        whoop(journal={"amenorrhea": True}), hist(hrv_z=-2.01)
    )
    assert FLAG_MED_REDS in codes(r) and r.requires_medical_review
    # HRV crash without a correlate is not the RED-S red flag.
    assert FLAG_MED_REDS not in codes(
        evaluate_safety_gates(whoop(), hist(hrv_z=-3.0))
    )


def test_pain_red_flag():
    r = evaluate_safety_gates(whoop(journal={"pain_worsens_during_run": True}), hist())
    assert FLAG_MED_PAIN in codes(r) and r.requires_medical_review


def test_clean_day_needs_no_medical_review():
    r = evaluate_safety_gates(whoop(), hist())
    assert r.flags == [] and r.requires_medical_review is False


# ==========================================================================
# choose_workout — §9 ordered override tree + §11 ceilings
# ==========================================================================

def test_choose_full_rest_precedes_everything():
    d = choose_workout("vo2max", whoop(recovery_score=19), hist(acwr=2.0, red_streak=3))
    assert d.action == "full_rest" and d.load_factor == 0.0


def test_choose_recovery_red_rest_or_walk_boundary():
    assert choose_workout("easy", whoop(recovery_score=T.RECOVERY_RED_MAX), hist()).action == "rest_or_walk"
    assert choose_workout("easy", whoop(recovery_score=T.RECOVERY_REST_BELOW), hist()).action != "rest_or_walk"


def test_choose_sleep_rest_boundary():
    # 4.5 h: below the 5 h rest floor, above the 4 h full-rest floor.
    assert choose_workout("easy", whoop(sleep_hours=4.5), hist()).action == "rest_or_walk"
    assert choose_workout("easy", whoop(sleep_hours=T.SLEEP_REST_BELOW_H), hist()).action != "rest_or_walk"


def test_choose_mandatory_rest_and_illness():
    assert choose_workout("easy", whoop(), hist(red_streak=2)).action == "mandatory_rest"
    d = choose_workout("vo2max", whoop(), hist(rhr_delta=7.01, hrv_z=-1.01))
    assert d.action == "illness_rest" and d.load_factor == 0.0


def test_choose_acwr_hard_stop_halves_load_and_precedes_recovery():
    # ACWR>1.5 is checked before the recovery<67 downgrade; load halved.
    d = choose_workout("vo2max", whoop(recovery_score=60), hist(acwr=1.51))
    assert d.action == "downgrade"
    assert d.load_factor == pytest.approx(T.ACWR_DOWNGRADE_FACTOR)
    assert d.resulting_type == "easy_with_strides"
    # 1.5 exactly does NOT hard-stop.
    assert choose_workout("easy", whoop(), hist(acwr=T.ACWR_HARD_STOP)).action == "proceed"


def test_choose_recovery_moderate_downgrade_boundary():
    # recovery<67 downgrades a quality session; easy is untouched.
    d = choose_workout("threshold", whoop(recovery_score=66), hist())
    assert d.action == "downgrade" and d.resulting_type == "easy" and d.load_factor == 1.0
    assert choose_workout("easy", whoop(recovery_score=66), hist()).action == "proceed"
    assert choose_workout("threshold", whoop(recovery_score=67), hist()).action == "proceed"


def test_choose_sleep_downgrade_boundary():
    # green recovery, sleep just under 6 -> downgrade vo2max only.
    assert choose_workout("vo2max", whoop(sleep_hours=5.99), hist()).action == "downgrade"
    assert choose_workout("vo2max", whoop(sleep_hours=T.SLEEP_DOWNGRADE_BELOW_H), hist()).action == "proceed"
    assert choose_workout("easy", whoop(sleep_hours=5.5), hist()).action == "proceed"


def test_choose_hrv_low_cuts_load_30pct():
    d = choose_workout("long_run", whoop(), hist(hrv_low_streak_days=3))
    assert d.action == "downgrade"
    assert d.load_factor == pytest.approx(1.0 - T.HRV_LOW_LOAD_CUT_APPLIED)  # 0.70
    # Below the 3-day persistence it does not fire.
    assert choose_workout("long_run", whoop(), hist(hrv_low_streak_days=2)).action == "proceed"


def test_choose_sprint_blocked_by_soreness_boundary():
    d = choose_workout(
        "speed", whoop(journal={"hamstring_calf_soreness": T.SORENESS_SPRINT_BLOCK}), hist()
    )
    assert d.action == "convert_easy_strides" and d.resulting_type == "easy_with_strides"
    assert choose_workout(
        "speed",
        whoop(journal={"hamstring_calf_soreness": T.SORENESS_SPRINT_BLOCK - 1}),
        hist(),
    ).action == "proceed"


def test_choose_racepace_blocked_after_two_yellows_unless_rhr_baseline():
    # Green today but 2 prior Yellows + RHR above baseline -> block race-pace.
    assert choose_workout(
        "threshold", whoop(), hist(yellow_streak=2, rhr_delta=1.0)
    ).action == "downgrade"
    # RHR at baseline (<=0) lifts the block.
    assert choose_workout(
        "threshold", whoop(), hist(yellow_streak=2, rhr_delta=T.RHR_DELTA_AT_BASELINE)
    ).action == "proceed"
    # Unknown RHR is treated as "not at baseline" -> still blocked.
    assert choose_workout(
        "threshold", whoop(), hist(yellow_streak=2, rhr_delta=None)
    ).action == "downgrade"
    # One Yellow is below the 2-streak ceiling.
    assert choose_workout(
        "threshold", whoop(), hist(yellow_streak=1, rhr_delta=5.0)
    ).action == "proceed"


def test_choose_vo2max_spacing_boundary():
    assert choose_workout("vo2max", whoop(), hist(days_since_hard=1)).action == "convert_easy_strides"
    assert choose_workout("vo2max", whoop(), hist(days_since_hard=T.DAYS_SINCE_HARD_MIN)).action == "proceed"
    # Spacing only converts the VO2max session, not e.g. an easy day.
    assert choose_workout("easy", whoop(), hist(days_since_hard=1)).action == "proceed"


def test_choose_green_proceeds_and_propagates_medical():
    d = choose_workout("easy", whoop(), hist())
    assert d.action == "proceed" and d.load_factor == 1.0 and d.resulting_type == "easy"
    assert d.requires_medical_review is False
    # A §11 red flag rides through even when the session proceeds.
    d2 = choose_workout("easy", whoop(journal={"chest_pain": True}), hist())
    assert d2.requires_medical_review is True
    assert FLAG_MED_CARDIAC in {f.code for f in d2.flags}


# ==========================================================================
# downgrade_workout_type
# ==========================================================================

def test_downgrade_workout_type_mapping():
    assert downgrade_workout_type("vo2max") == "easy_with_strides"
    assert downgrade_workout_type("speed") == "easy_with_strides"
    assert downgrade_workout_type("threshold") == "easy"
    assert downgrade_workout_type("tempo") == "easy"
    assert downgrade_workout_type("long_run") == "easy"
    assert downgrade_workout_type("easy") == "easy"               # unchanged
    assert downgrade_workout_type("rest") == "rest"


# ==========================================================================
# overreaching() — §6 all-of detector
# ==========================================================================

def test_overreaching_requires_all_components():
    ok = dict(hrv_low_streak_days=3, rhr_7d_vs_28d=5.01, red_streak=2)
    assert overreaching(hist(**ok)) is True
    assert overreaching(hist(**{**ok, "hrv_low_streak_days": 2})) is False
    assert overreaching(hist(**{**ok, "rhr_7d_vs_28d": T.OVERREACH_RHR_DELTA})) is False  # 5.0 not >5
    assert overreaching(hist(**{**ok, "rhr_7d_vs_28d": None})) is False
    # Streak branch is OR: 2 Reds or 5 Yellows.
    assert overreaching(hist(hrv_low_streak_days=3, rhr_7d_vs_28d=6.0, red_streak=1, yellow_streak=5)) is True
    assert overreaching(hist(hrv_low_streak_days=3, rhr_7d_vs_28d=6.0, red_streak=1, yellow_streak=4)) is False


def test_overreaching_surfaces_as_flag():
    r = evaluate_safety_gates(
        whoop(), hist(hrv_low_streak_days=3, rhr_7d_vs_28d=6.0, red_streak=2)
    )
    assert FLAG_OVERREACHING in codes(r)


# ==========================================================================
# deload_triggers / deload_due — §9 (fire any)
# ==========================================================================

def test_deload_build_weeks_boundary():
    assert "build_weeks_complete" not in deload_triggers(
        hist(builds_completed=T.BUILD_WEEKS_BEFORE_DELOAD - 1)
    )
    assert "build_weeks_complete" in deload_triggers(
        hist(builds_completed=T.BUILD_WEEKS_BEFORE_DELOAD)
    )


def test_deload_monotony_boundary():
    assert "monotony_high" not in deload_triggers(hist(monotony=T.MONOTONY_HIGH))
    assert "monotony_high" in deload_triggers(hist(monotony=2.01))


def test_deload_yellow_cluster_boundary():
    assert "yellow_cluster" not in deload_triggers(
        hist(yellow_count_7d=T.YELLOW_DELOAD_COUNT - 1)
    )
    assert "yellow_cluster" in deload_triggers(
        hist(yellow_count_7d=T.YELLOW_DELOAD_COUNT)
    )


def test_deload_hrv_rhr_persist_boundary():
    assert "hrv_rhr_persist" not in deload_triggers(
        hist(hrv_low_streak_days=3, rhr_delta=T.OVERREACH_RHR_DELTA)   # 5.0 not >5
    )
    assert "hrv_rhr_persist" in deload_triggers(
        hist(hrv_low_streak_days=3, rhr_delta=5.01)
    )
    assert deload_due(hist(hrv_low_streak_days=3, rhr_delta=5.01)) is True
    assert deload_due(hist()) is False


# ==========================================================================
# next_week_target_mpw — §9, mirrored exactly
# ==========================================================================

def test_next_week_target_mpw_branches():
    base = 40.0
    # Deload wins regardless of ACWR.
    assert next_week_target_mpw(base, 1.0, deload_due=True) == pytest.approx(
        base * T.MPW_DELOAD_FACTOR
    )
    assert next_week_target_mpw(base, 0.5, deload_due=True) == pytest.approx(
        base * T.MPW_DELOAD_FACTOR
    )
    # acwr>1.3 -> hold; exactly 1.3 -> normal +7%.
    assert next_week_target_mpw(base, 1.31, False) == pytest.approx(base)
    assert next_week_target_mpw(base, T.ACWR_SWEET_HIGH, False) == pytest.approx(
        base * T.MPW_RAMP_NORMAL_FACTOR
    )
    # acwr<0.8 -> +10%; exactly 0.8 -> normal +7%.
    assert next_week_target_mpw(base, 0.79, False) == pytest.approx(
        base * T.MPW_RAMP_DETRAIN_FACTOR
    )
    assert next_week_target_mpw(base, T.ACWR_SWEET_LOW, False) == pytest.approx(
        base * T.MPW_RAMP_NORMAL_FACTOR
    )
    # Unknown ACWR -> gentle default +7%.
    assert next_week_target_mpw(base, None, False) == pytest.approx(
        base * T.MPW_RAMP_NORMAL_FACTOR
    )
    # Both ramps stay within the §11 10% soft cap.
    assert T.MPW_RAMP_DETRAIN_FACTOR <= 1.0 + T.MPW_SOFT_CAP_PCT + 1e-9
    assert T.MPW_RAMP_NORMAL_FACTOR <= 1.0 + T.MPW_SOFT_CAP_PCT + 1e-9


# ==========================================================================
# Flag rendering (surfaced verbatim on Recommendation)
# ==========================================================================

def test_flag_render_format():
    f = Flag(code="x", severity="rest", message="rest now")
    assert f.render() == "[rest] rest now"


def test_combined_bad_day_is_full_rest_with_flags():
    """A clearly unsafe day: many gates trip, action is full rest, medical set."""
    w = whoop(
        recovery_score=15,
        sleep_hours=3.5,
        sleep_performance=0.4,
        journal={"chest_pain": True, "fever": True},
    )
    h = hist(red_streak=3, acwr=1.9, monotony=2.5, rhr_delta=9.0, hrv_z=-2.5)
    d = choose_workout("vo2max", w, h)
    assert d.action == "full_rest" and d.load_factor == 0.0
    assert d.requires_medical_review is True
    assert {FLAG_FULL_REST, FLAG_RECOVERY_RED, FLAG_MANDATORY_REST,
            FLAG_MED_CARDIAC}.issubset({f.code for f in d.flags})
