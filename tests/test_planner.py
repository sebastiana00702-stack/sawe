"""Phase 4 planner tests: templates, progression, and the plan generator.

Covers the framework.md §3/§6/§10 requirements:

* every (tier, goal) template emits a valid 7-day sequence for weeks 1/4/8,
* :func:`generate_weekly_plan` output validates against the *unchanged*
  Phase 1 :class:`~src.models.WeeklyPlan` schema,
* deload weeks carry strictly less volume than the surrounding build weeks
  (framework.md §10 "every 4th week … volume −25–35%"),
* the long run is 20–35% of weekly running volume, capped at 2.5 h
  (Daniels 25–30%, framework.md §6),
* the §2 warm-up/cooldown sidecars are attached to every running day.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.models import RunnerProfile, WeeklyPlan
from src.models.weekly_plan import WorkoutType as ModelWorkoutType
from src.planner import (
    COOLDOWN_STR,
    DEFAULT_ACWR_TARGET,
    MODEL_WORKOUT_TYPE,
    TEMPLATES,
    WARMUP_STR,
    IntensityTarget,
    WorkoutType,
    build_week,
    generate_weekly_plan,
    get_template,
    to_model_type,
)
from src.planner.templates import (
    DELOAD_CUT,
    LONG_RUN_CAP_MIN,
    _mpw_sequence,
    is_deload_week,
    is_taper_week,
    planned_mpw,
)

# The ten framework.md §10 templates, by (tier, goal).
ALL_COMBOS = sorted(TEMPLATES.keys())
TEST_WEEKS = [1, 4, 8]

# Coarse Phase 1 model literal set (kept here so the test fails loudly if
# the schema and the planner's mapping ever drift).
MODEL_TYPES = set(ModelWorkoutType.__args__)  # type: ignore[attr-defined]

_PROFILE_DEFAULTS = dict(age=30, sex="M", longest_recent_run_mi=10.0)


def _profile(tier: str, goal: str) -> RunnerProfile:
    return RunnerProfile(
        tier=tier, goal=goal, current_mpw=20.0, **_PROFILE_DEFAULTS
    )


# --------------------------------------------------------------------------
# Registry / mapping sanity
# --------------------------------------------------------------------------

def test_all_ten_framework_templates_present():
    assert set(TEMPLATES) == {
        ("beginner", "health"),
        ("novice", "health"),
        ("intermediate", "health"),
        ("advanced", "health"),
        ("intermediate", "speed"),     # VO2max focus
        ("advanced", "speed"),         # speed-endurance block
        ("intermediate", "5K"),
        ("advanced", "10K"),
        ("intermediate", "HM"),
        ("advanced", "marathon"),
    }


def test_every_workout_type_maps_to_a_valid_model_literal():
    # Mapping is total over the enum and lands inside the Phase 1 literal.
    assert set(MODEL_WORKOUT_TYPE) == set(WorkoutType)
    for wt in WorkoutType:
        assert to_model_type(wt) in MODEL_TYPES


def test_get_template_falls_back_to_same_tier_general_health():
    # (intermediate, marathon) has no dedicated template -> intermediate GH.
    spec = get_template("intermediate", "marathon")
    assert spec is TEMPLATES[("intermediate", "health")]


def test_get_template_raises_when_nothing_matches():
    with pytest.raises(KeyError):
        get_template("competitive", "marathon")


def test_intensity_target_bridges_to_schema_model():
    it = IntensityTarget(hr_pct_max=(88, 92), rpe=7, pace_label="T")
    m = it.to_model()
    assert m.hr_pct_max == (88, 92)
    assert m.rpe == 7 and m.pace_label == "T"
    assert "88-92% max" in it.describe()


# --------------------------------------------------------------------------
# Every template: valid 7-day sequence for weeks 1, 4, 8
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
@pytest.mark.parametrize("week", TEST_WEEKS)
def test_template_builds_seven_valid_slots(tier, goal, week):
    slots = build_week(tier, goal, week)
    assert len(slots) == 7
    for s in slots:
        assert isinstance(s.workout, WorkoutType)
        assert isinstance(s.intensity, IntensityTarget)
        if s.duration_min is not None:
            assert s.duration_min >= 1


@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
@pytest.mark.parametrize("week", TEST_WEEKS)
def test_generate_weekly_plan_validates_against_schema(tier, goal, week):
    start = date(2026, 5, 18)
    plan = generate_weekly_plan(_profile(tier, goal), week, start)

    assert isinstance(plan, WeeklyPlan)
    # Round-trips through the Pydantic schema unchanged.
    WeeklyPlan.model_validate(plan.model_dump())

    assert plan.week_starting == start
    assert plan.tier == tier and plan.goal == goal
    assert len(plan.days) == 7
    assert [d.date for d in plan.days] == [
        start + timedelta(days=i) for i in range(7)
    ]
    assert plan.acwr_target == DEFAULT_ACWR_TARGET
    assert plan.planned_mpw >= 0
    assert plan.deload_due_in_weeks >= 0
    assert plan.phase in {"base", "build", "peak", "deload", "taper"}


def test_generate_weekly_plan_clamps_overrun_week():
    spec = TEMPLATES[("intermediate", "5K")]
    plan = generate_weekly_plan(
        _profile("intermediate", "5K"), 999, date(2026, 5, 18)
    )
    # Clamped to the final (taper) week, still a valid plan.
    assert plan.phase == "taper"
    assert plan.planned_mpw == round(
        planned_mpw(spec, spec.total_weeks), 1
    )


# --------------------------------------------------------------------------
# Progression: ~10% build, deload 25–35% cut below the build weeks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
def test_deload_weeks_have_lower_volume_than_build_weeks(tier, goal):
    spec = TEMPLATES[(tier, goal)]
    seq = _mpw_sequence(spec)
    deloads = [
        w for w in range(1, spec.total_weeks + 1) if is_deload_week(spec, w)
    ]
    assert deloads, f"{spec.name} should have at least one deload week"
    for dw in deloads:
        prev_build = seq[dw - 2]  # week dw-1 is always a build week
        cut = seq[dw - 1]
        assert cut < prev_build, (
            f"{spec.name} wk{dw} deload {cut} not below build {prev_build}"
        )
        # Volume cut inside the framework.md §10 25–35% band (≈30%).
        ratio = cut / prev_build
        assert 0.65 <= ratio <= 0.75, f"{spec.name} wk{dw} cut ratio {ratio}"
        assert abs(ratio - (1.0 - DELOAD_CUT)) <= 0.02


@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
def test_build_weeks_grow_about_ten_percent_and_respect_peak(tier, goal):
    spec = TEMPLATES[(tier, goal)]
    seq = _mpw_sequence(spec)
    for w in range(2, spec.total_weeks + 1):
        if is_deload_week(spec, w) or is_taper_week(spec, w):
            continue
        if is_deload_week(spec, w - 1) or is_taper_week(spec, w - 1):
            continue  # step measured off the previous *build* week only
        prev, cur = seq[w - 2], seq[w - 1]
        assert cur >= prev  # monotone build
        # Never more than ~10% week-on-week (the soft framework.md §2 rule).
        assert cur <= round(prev * 1.10, 1) + 1e-9
        assert cur <= spec.peak_mpw


@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
def test_taper_weeks_reduce_volume(tier, goal):
    spec = TEMPLATES[(tier, goal)]
    if not spec.taper_factors:
        pytest.skip(f"{spec.name} has no taper")
    seq = _mpw_sequence(spec)
    peak = max(seq)
    for i in range(len(spec.taper_factors)):
        wk = spec.total_weeks - len(spec.taper_factors) + 1 + i
        assert seq[wk - 1] < peak


# --------------------------------------------------------------------------
# Long run is 20–35% of weekly running volume, capped at 2.5 h
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
@pytest.mark.parametrize("week", TEST_WEEKS)
def test_long_run_within_expected_fraction_of_weekly_volume(tier, goal, week):
    plan = generate_weekly_plan(
        _profile(tier, goal), week, date(2026, 5, 18)
    )
    run_minutes = [
        d.duration_min
        for d in plan.days
        if d.duration_min is not None and d.type != "rest"
    ]
    assert run_minutes, f"{tier}/{goal} wk{week} has no running minutes"
    longest = max(run_minutes)
    total = sum(run_minutes)
    frac = longest / total
    assert 0.20 <= frac <= 0.35, (
        f"{tier}/{goal} wk{week}: long run {frac:.0%} of weekly volume"
    )
    assert longest <= LONG_RUN_CAP_MIN  # 2.5 h absolute cap (framework.md §6)


# --------------------------------------------------------------------------
# Mobility sidecars attached to every running day (framework.md §2/§10)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
@pytest.mark.parametrize("week", TEST_WEEKS)
def test_mobility_attached_to_every_running_day(tier, goal, week):
    plan = generate_weekly_plan(
        _profile(tier, goal), week, date(2026, 5, 18)
    )
    running_days = [
        d for d in plan.days if d.type not in {"rest", "cross_train"}
    ]
    assert running_days, f"{tier}/{goal} wk{week} has no running days"
    for d in running_days:
        assert d.structure is not None
        assert "Warmup:" in d.structure
        assert "Cooldown:" in d.structure
        assert COOLDOWN_STR in d.structure
    # Rest days never carry running mobility.
    for d in plan.days:
        if d.type == "rest":
            assert "Warmup:" not in (d.structure or "")


def test_mobility_constants_are_nonempty_and_quality_adds_strides():
    from src.planner import QUALITY_WARMUP_STR

    assert WARMUP_STR and COOLDOWN_STR
    assert QUALITY_WARMUP_STR.startswith(WARMUP_STR)
    assert "strides" in QUALITY_WARMUP_STR
    # A VO2max day (hard) gets the strides warm-up; an easy day does not.
    plan = generate_weekly_plan(
        _profile("intermediate", "speed"), 2, date(2026, 5, 18)
    )
    vo2 = next(d for d in plan.days if d.type == "vo2max")
    assert "strides 80m" in vo2.structure


# --------------------------------------------------------------------------
# Schema guard: the planner must not emit a type outside the model literal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tier,goal", ALL_COMBOS)
def test_every_emitted_day_type_is_a_valid_model_literal(tier, goal):
    for week in range(1, TEMPLATES[(tier, goal)].total_weeks + 1):
        plan = generate_weekly_plan(
            _profile(tier, goal), week, date(2026, 5, 18)
        )
        for d in plan.days:
            assert d.type in MODEL_TYPES


def test_planneddays_reject_unknown_type_via_schema():
    # Guard that the schema itself still forbids junk (regression anchor).
    with pytest.raises(ValidationError):
        WeeklyPlan(
            week_starting=date(2026, 5, 18),
            athlete_id="u",
            tier="intermediate",
            goal="5K",
            phase="base",
            planned_mpw=20,
            days=[{"date": "2026-05-18", "type": "parkour"}],
            deload_due_in_weeks=1,
            acwr_target=[0.9, 1.2],
        )
