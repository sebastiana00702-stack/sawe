"""Per-(tier, goal) weekly templates and block progression (framework.md
§3, §6, §10).

Each template is a :class:`TemplateSpec`: a tier/goal, a block length, a
base→peak weekly-mileage envelope, and a ``builder`` that lays out the seven
day microcycle for a given week. Templates are **pure data + simple
progression** — no WHOOP/safety logic (that is Phase 3, applied in Phase 5).

Progression rules (framework.md §2 "10% rule", §6 ≤10% week-on-week, §9/§10
deload, §6 long-run proportion):

* Build weeks ramp ~10% week-on-week (:data:`RAMP`) toward the block peak.
* Every :data:`DEFAULT_DELOAD_EVERY`-th week is a deload: volume cut
  :data:`DELOAD_CUT` (30%, inside the framework.md §10 "25–35%" band),
  intensity preserved.
* Race blocks taper the final weeks via :attr:`TemplateSpec.taper_factors`.
* The long run is ~:data:`LONG_RUN_FRACTION` of weekly running volume
  (Daniels 25–30%), hard-capped at :data:`LONG_RUN_CAP_MIN` (2.5 h).

Day durations scale with that week's volume factor, so a single builder
expresses the whole block (early weeks shorter, peak weeks longest, deload
and taper weeks lighter).

The ten templates implemented (framework.md §10): True Beginner / Novice /
Intermediate / Advanced general health, Intermediate VO2max focus, Advanced
speed-endurance, 5K (intermediate), 10K (advanced), Half-marathon
(intermediate, Pfitzinger-style), Marathon (advanced, 18 wk). VO2max-focus
and speed-endurance are keyed under ``goal="speed"`` (the only non-race
fitness-focus goal in the Phase 1 ``RunnerProfile`` literal), differentiated
by tier exactly as framework.md §10 names them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.models.runner_profile import Goal, Tier
from src.planner.workout_types import (
    EASY,
    GENERAL_HEALTH_Z2,
    INTERVAL,
    MARATHON,
    REPETITION,
    REST_INTENSITY,
    THRESHOLD,
    IntensityTarget,
    WorkoutType,
)

# --------------------------------------------------------------------------
# Progression constants (framework.md §2/§6/§9/§10)
# --------------------------------------------------------------------------

#: ~10% week-on-week build (framework.md §2 "10% rule" as a soft guide).
RAMP = 1.10
#: Deload volume cut — 30%, inside the framework.md §10 "25–35%" band.
DELOAD_CUT = 0.30
#: Every 4th week is a deload (framework.md §9/§10).
DEFAULT_DELOAD_EVERY = 4
#: Long run as a fraction of weekly running volume (Daniels 25–30%, §6).
LONG_RUN_FRACTION = 0.28
#: Absolute long-run ceiling: 2.5 h (framework.md §6).
LONG_RUN_CAP_MIN = 150
#: Default ACWR sweet-spot band the plan advertises (framework.md §11; the
#: §12 example weekly JSON uses [0.9, 1.2]).
DEFAULT_ACWR_TARGET: tuple[float, float] = (0.9, 1.2)


@dataclass(frozen=True)
class PlannedSlot:
    """One day of a template, before a calendar date is attached.

    :func:`src.planner.weekly_plan.generate_weekly_plan` turns this into a
    schema :class:`src.models.PlannedDay` (mapping the fine
    :class:`WorkoutType` to the coarse model literal and folding
    ``intensity`` + mobility into the day's structure text).
    """

    workout: WorkoutType
    duration_min: Optional[int] = None
    structure: Optional[str] = None
    intensity: IntensityTarget = REST_INTENSITY
    modality: Optional[str] = None


# A builder takes (week_number, spec) and returns exactly 7 slots (Mon→Sun).
Builder = Callable[[int, "TemplateSpec"], list[PlannedSlot]]


@dataclass(frozen=True)
class TemplateSpec:
    """A named (tier, goal) block: length, mileage envelope, day builder."""

    name: str
    tier: Tier
    goal: Goal
    total_weeks: int
    base_mpw: float
    peak_mpw: float
    builder: Builder
    deload_every: int = DEFAULT_DELOAD_EVERY
    #: Volume factors (× peak) for the final ``len`` weeks (race tapers).
    taper_factors: tuple[float, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Block-progression maths (shared by every builder + the plan generator)
# --------------------------------------------------------------------------

def clamp_week(spec: TemplateSpec, week: int) -> int:
    """Clamp ``week`` into ``[1, total_weeks]`` (callers may over-run a block)."""
    if week < 1:
        return 1
    if week > spec.total_weeks:
        return spec.total_weeks
    return week


def is_taper_week(spec: TemplateSpec, week: int) -> bool:
    """True for the final ``len(taper_factors)`` weeks of a race block."""
    n = len(spec.taper_factors)
    return n > 0 and week > spec.total_weeks - n


def is_deload_week(spec: TemplateSpec, week: int) -> bool:
    """Every ``deload_every``-th week that is not already a taper week."""
    if is_taper_week(spec, week):
        return False
    return week % spec.deload_every == 0


def _mpw_sequence(spec: TemplateSpec) -> list[float]:
    """Full week-1..N planned-mileage curve.

    Build weeks ramp ~10% off the previous *build* week toward the peak;
    deload weeks dip to ``(1 - DELOAD_CUT)`` of the preceding build week
    (so a deload is always strictly below its neighbours and progression
    resumes from the pre-deload level); taper weeks are ``factor × peak``.
    """
    seq: list[float] = []
    prev_build = spec.base_mpw
    n_taper = len(spec.taper_factors)
    build_cutoff = spec.total_weeks - n_taper
    for wk in range(1, spec.total_weeks + 1):
        if wk > build_cutoff:
            factor = spec.taper_factors[wk - build_cutoff - 1]
            seq.append(round(spec.peak_mpw * factor, 1))
        elif wk % spec.deload_every == 0:
            seq.append(round(prev_build * (1.0 - DELOAD_CUT), 1))
        else:
            cur = (
                spec.base_mpw
                if wk == 1
                else min(spec.peak_mpw, round(prev_build * RAMP, 1))
            )
            prev_build = cur
            seq.append(cur)
    return seq


def planned_mpw(spec: TemplateSpec, week: int) -> float:
    """Planned miles/week for ``week`` on this block's progression curve."""
    return _mpw_sequence(spec)[clamp_week(spec, week) - 1]


def volume_factor(spec: TemplateSpec, week: int) -> float:
    """``planned_mpw(week) / peak`` ∈ (0, 1] — scales day durations."""
    if spec.peak_mpw <= 0:
        return 1.0
    return planned_mpw(spec, week) / spec.peak_mpw


def phase_for(spec: TemplateSpec, week: int) -> str:
    """One of ``base``/``build``/``peak``/``deload``/``taper`` for ``week``."""
    week = clamp_week(spec, week)
    if is_taper_week(spec, week):
        return "taper"
    if is_deload_week(spec, week):
        return "deload"
    build_total = spec.total_weeks - len(spec.taper_factors)
    if week <= build_total / 3:
        return "base"
    if week <= 2 * build_total / 3:
        return "build"
    return "peak"


def deload_due_in_weeks(spec: TemplateSpec, week: int) -> int:
    """Weeks until the next deload (0 if this week is one / none remain)."""
    week = clamp_week(spec, week)
    if is_deload_week(spec, week):
        return 0
    for ahead in range(1, spec.total_weeks - week + 1):
        if is_deload_week(spec, week + ahead):
            return ahead
    return 0


def _scale(base_min: int, spec: TemplateSpec, week: int) -> int:
    """Scale a peak-week duration by this week's volume factor (min 1)."""
    return max(1, round(base_min * volume_factor(spec, week)))


def _long_minutes(other_running_min: int) -> int:
    """Long-run minutes ≈ :data:`LONG_RUN_FRACTION` of weekly volume.

    Derived from the rest of the week's running minutes so the long run is
    always ~28% of the total (Daniels 25–30%, framework.md §6), then capped
    at :data:`LONG_RUN_CAP_MIN` (2.5 h).
    """
    f = LONG_RUN_FRACTION
    raw = round(other_running_min * f / (1.0 - f))
    return max(1, min(LONG_RUN_CAP_MIN, raw))


def _finalize_long(
    slots: list[PlannedSlot], long_index: int
) -> list[PlannedSlot]:
    """Set ``slots[long_index]``'s duration from the other running minutes."""
    other = sum(
        s.duration_min or 0
        for i, s in enumerate(slots)
        if i != long_index and s.workout is not WorkoutType.REST
    )
    long_slot = slots[long_index]
    slots[long_index] = PlannedSlot(
        workout=long_slot.workout,
        duration_min=_long_minutes(other),
        structure=long_slot.structure,
        intensity=long_slot.intensity,
        modality=long_slot.modality,
    )
    return slots


# Shared single-day slot factories ----------------------------------------

def _rest() -> PlannedSlot:
    return PlannedSlot(WorkoutType.REST, structure="Rest day", intensity=REST_INTENSITY)


def _easy(minutes: int, intensity: IntensityTarget = EASY) -> PlannedSlot:
    return PlannedSlot(
        WorkoutType.EASY,
        duration_min=minutes,
        structure=f"{minutes} min easy, conversational",
        intensity=intensity,
    )


def _easy_strides(minutes: int) -> PlannedSlot:
    return PlannedSlot(
        WorkoutType.STRIDES,
        duration_min=minutes,
        structure=f"{minutes} min easy + 4-6 x 20s strides @ ~90% (framework.md §2/§5)",
        intensity=EASY,
    )


# ==========================================================================
# Builders — general health (framework.md §3 per-tier weekly structure)
# ==========================================================================

def _true_beginner_health(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """True Beginner run-walk (framework.md §3/§10 wk1 1:2x8 → wk8 20' cont.)."""
    # Run interval grows 1 min (wk1) → 20 min continuous (wk8).
    run_int = min(20, round(1 + (clamp_week(spec, week) - 1) * (19 / 7)))
    reps = max(1, round(24 / max(run_int, 1)))
    if run_int >= 20:
        rw = f"Run {run_int} min continuous"
    else:
        rw = f"{run_int} min run / 2 min walk x {reps}"
    walk = _scale(35, spec, week)
    slots = [
        PlannedSlot(WorkoutType.WALK, duration_min=walk,
                    structure=f"Brisk walk {walk} min", intensity=GENERAL_HEALTH_Z2),
        _rest(),
        PlannedSlot(WorkoutType.EASY_SHORT, duration_min=_scale(28, spec, week),
                    structure=rw, intensity=GENERAL_HEALTH_Z2),
        _rest(),
        PlannedSlot(WorkoutType.WALK, duration_min=walk,
                    structure=f"Brisk walk {walk} min", intensity=GENERAL_HEALTH_Z2),
        PlannedSlot(WorkoutType.EASY_SHORT, duration_min=_scale(28, spec, week),
                    structure=rw, intensity=GENERAL_HEALTH_Z2),
        PlannedSlot(WorkoutType.EASY_LONG, structure=f"Longer aerobic: {rw}, then "
                    f"easy walk to finish", intensity=GENERAL_HEALTH_Z2),
    ]
    return _finalize_long(slots, 6)


def _novice_health(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Novice general health: 3-4 conversational runs, strides wk5+ (§3/§10)."""
    w = clamp_week(spec, week)
    thu = (
        _easy_strides(_scale(35, spec, week))
        if w >= 5
        else _easy(_scale(35, spec, week), GENERAL_HEALTH_Z2)
    )
    slots = [
        _rest(),
        _easy(_scale(35, spec, week), GENERAL_HEALTH_Z2),
        PlannedSlot(WorkoutType.WALK, duration_min=_scale(30, spec, week),
                    structure="Optional brisk walk / cross", intensity=GENERAL_HEALTH_Z2),
        thu,
        _rest(),
        _easy(_scale(35, spec, week), GENERAL_HEALTH_Z2),
        PlannedSlot(WorkoutType.EASY_LONG,
                    structure="Long easy run, conversational (cap ~45 min by wk 8)",
                    intensity=GENERAL_HEALTH_Z2),
    ]
    return _finalize_long(slots, 6)


def _intermediate_health(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Intermediate general health: 4-5 days, 1x strides, long run (§3)."""
    slots = [
        _easy(_scale(45, spec, week)),
        _easy(_scale(45, spec, week)),
        _easy(_scale(40, spec, week)),
        _easy_strides(_scale(45, spec, week)),
        _rest(),
        _easy(_scale(40, spec, week)),
        PlannedSlot(WorkoutType.LONG_RUN,
                    structure="Long run, easy aerobic (Z1-Z2)", intensity=EASY),
    ]
    return _finalize_long(slots, 6)


def _advanced_health(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Advanced general health: 5-6 days, 2x strides, one light tempo (§3)."""
    slots = [
        _easy(_scale(55, spec, week)),
        _easy_strides(_scale(55, spec, week)),
        _easy(_scale(50, spec, week)),
        PlannedSlot(WorkoutType.TEMPO, duration_min=_scale(50, spec, week),
                    structure="15 min easy + 20 min steady tempo @ T + 10 min easy",
                    intensity=THRESHOLD),
        _easy_strides(_scale(45, spec, week)),
        _rest(),
        PlannedSlot(WorkoutType.LONG_RUN,
                    structure="Long run, easy aerobic (Z1-Z2)", intensity=EASY),
    ]
    return _finalize_long(slots, 6)


# ==========================================================================
# Builders — fitness focus blocks (framework.md §10)
# ==========================================================================

def _intermediate_vo2max(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Intermediate VO2max focus — framework.md §10 table verbatim."""
    slots = [
        _easy_strides(_scale(35, spec, week)),
        PlannedSlot(WorkoutType.VO2MAX_4X4, duration_min=_scale(50, spec, week),
                    structure="10 min wu + 4x4 min @ 90-95% HRmax, 3 min jog "
                              "+ 10 min cd (Helgerud)", intensity=INTERVAL),
        _easy(_scale(40, spec, week)),
        PlannedSlot(WorkoutType.TEMPO, duration_min=_scale(40, spec, week),
                    structure="10 min wu + 20-30 min @ T-pace + 10 min cd",
                    intensity=THRESHOLD),
        _rest(),
        PlannedSlot(WorkoutType.HILL_SPRINTS, duration_min=_scale(55, spec, week),
                    structure="6x8 s hill sprints, 2+ min recovery (Magness) "
                              "+ 45 min easy", intensity=REPETITION),
        PlannedSlot(WorkoutType.LONG_RUN,
                    structure="75-90 min long run, aerobic (Z1-Z2)",
                    intensity=EASY),
    ]
    return _finalize_long(slots, 6)


def _advanced_speed_endurance(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Advanced speed-endurance block — framework.md §10 (4-wk block, here
    progressed over 8 weeks)."""
    slots = [
        _easy(_scale(50, spec, week)),
        PlannedSlot(WorkoutType.SPEED_ENDURANCE, duration_min=_scale(55, spec, week),
                    structure="wu + 8x400 m @ 5K pace, 60 s jog + cd",
                    intensity=REPETITION),
        _easy(_scale(45, spec, week)),
        PlannedSlot(WorkoutType.VO2MAX_5X1K, duration_min=_scale(55, spec, week),
                    structure="wu + 5x1 km @ 10K pace, 90 s jog + cd",
                    intensity=INTERVAL),
        _rest(),
        PlannedSlot(WorkoutType.HILL_SPRINTS, duration_min=_scale(75, spec, week),
                    structure="8x8 s hill sprints, full recovery + 60 min easy",
                    intensity=REPETITION),
        PlannedSlot(WorkoutType.LONG_RUN_WITH_MP,
                    structure="90-120 min long, last 20 min @ marathon pace",
                    intensity=MARATHON),
    ]
    return _finalize_long(slots, 6)


# ==========================================================================
# Builders — race blocks (framework.md §10)
# ==========================================================================

def _five_k_intermediate(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """5K (intermediate, 8 wk): base → VO2max emphasis → R-pace sharpen
    → race-week taper (framework.md §10)."""
    w = clamp_week(spec, week)
    if w <= 3:  # base
        tue = PlannedSlot(WorkoutType.TEMPO, duration_min=_scale(45, spec, week),
                          structure="10 min wu + 20 min @ T + 10 min cd",
                          intensity=THRESHOLD)
        thu = _easy_strides(_scale(40, spec, week))
    elif w <= 6:  # VO2max emphasis
        tue = PlannedSlot(WorkoutType.VO2MAX_4X4, duration_min=_scale(50, spec, week),
                          structure="wu + 4x4 min @ I-pace, 3 min jog + cd",
                          intensity=INTERVAL)
        thu = PlannedSlot(WorkoutType.VO2MAX_5X1K, duration_min=_scale(50, spec, week),
                          structure="wu + 5x1 km @ I-pace, 2-3 min jog + cd",
                          intensity=INTERVAL)
    elif w == 7:  # sharpen
        tue = PlannedSlot(WorkoutType.SPEED_ENDURANCE, duration_min=_scale(45, spec, week),
                          structure="wu + 6x200 m + 4x400 m @ R-pace, full rec + cd",
                          intensity=REPETITION)
        thu = _easy_strides(_scale(35, spec, week))
    else:  # race week taper, keep one sharpener
        tue = PlannedSlot(WorkoutType.SPEED_ENDURANCE, duration_min=_scale(35, spec, week),
                          structure="wu + 4x200 m @ R-pace, full recovery + cd",
                          intensity=REPETITION)
        thu = _easy(_scale(30, spec, week))
    slots = [
        _easy(_scale(40, spec, week)),
        tue,
        _easy(_scale(35, spec, week)),
        thu,
        _rest(),
        _easy(_scale(35, spec, week)),
        PlannedSlot(WorkoutType.LONG_RUN, structure="Long run, easy aerobic",
                    intensity=EASY),
    ]
    return _finalize_long(slots, 6)


def _ten_k_advanced(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """10K (advanced, 12 wk): T cruise + VO2max + hills + MP long
    (framework.md §10)."""
    w = clamp_week(spec, week)
    thu = (
        PlannedSlot(WorkoutType.VO2MAX_4X4, duration_min=_scale(55, spec, week),
                    structure="wu + 4x4 min @ I-pace, 3 min jog + cd",
                    intensity=INTERVAL)
        if w % 2 == 0
        else PlannedSlot(WorkoutType.VO2MAX_5X1K, duration_min=_scale(55, spec, week),
                         structure="wu + 6x1 km @ I-pace, 90 s jog + cd",
                         intensity=INTERVAL)
    )
    long_mp = 7 <= w <= 10
    long_slot = (
        PlannedSlot(WorkoutType.LONG_RUN_WITH_MP,
                    structure="Long run 14-18 mi w/ 4-6 mi @ MP+10s",
                    intensity=MARATHON)
        if long_mp
        else PlannedSlot(WorkoutType.LONG_RUN, structure="Long run, easy aerobic",
                         intensity=EASY)
    )
    slots = [
        _easy(_scale(50, spec, week)),
        PlannedSlot(WorkoutType.THRESHOLD_INTERVALS, duration_min=_scale(55, spec, week),
                    structure="wu + 4x8 min @ T, 2 min jog + cd (cruise intervals)",
                    intensity=THRESHOLD),
        _easy(_scale(45, spec, week)),
        thu,
        _rest(),
        PlannedSlot(WorkoutType.HILL_SPRINTS, duration_min=_scale(50, spec, week),
                    structure="6-8x8 s hill sprints + 40 min easy",
                    intensity=REPETITION),
        long_slot,
    ]
    return _finalize_long(slots, 6)


def _half_marathon_intermediate(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Half-marathon (intermediate, 12 wk, Pfitzinger-style): mid-week
    medium-long + Sunday long + cumulative T (framework.md §10)."""
    slots = [
        _easy(_scale(40, spec, week)),
        PlannedSlot(WorkoutType.THRESHOLD_INTERVALS, duration_min=_scale(55, spec, week),
                    structure="wu + 25-40 min cumulative @ T "
                              "(e.g. 4x10 min, 2 min jog) + cd",
                    intensity=THRESHOLD),
        PlannedSlot(WorkoutType.EASY_LONG, duration_min=_scale(80, spec, week),
                    structure="Mid-week medium-long 8-12 mi, endurance pace "
                              "(Pfitzinger)", intensity=EASY),
        _easy(_scale(40, spec, week)),
        _rest(),
        _easy_strides(_scale(40, spec, week)),
        PlannedSlot(WorkoutType.LONG_RUN,
                    structure="Sunday long 12-14 mi with progressive segments",
                    intensity=EASY),
    ]
    return _finalize_long(slots, 6)


def _marathon_advanced(week: int, spec: TemplateSpec) -> list[PlannedSlot]:
    """Marathon (advanced, 18 wk): Endurance → LT+Endurance → Race Prep →
    Taper, MP work in the long run during race prep (framework.md §10)."""
    w = clamp_week(spec, week)
    race_prep = 13 <= w <= 16
    thu = (
        PlannedSlot(WorkoutType.VO2MAX_4X4, duration_min=_scale(55, spec, week),
                    structure="wu + 4x4 min @ I-pace, 3 min jog + cd",
                    intensity=INTERVAL)
        if w % 2 == 0
        else PlannedSlot(WorkoutType.HILL_SPRINTS, duration_min=_scale(55, spec, week),
                         structure="8x8 s hill sprints + 40 min easy",
                         intensity=REPETITION)
    )
    long_slot = (
        PlannedSlot(WorkoutType.LONG_RUN_WITH_MP,
                    structure="Long run to 20-22 mi with 12-16 mi @ marathon pace",
                    intensity=MARATHON)
        if race_prep
        else PlannedSlot(WorkoutType.LONG_RUN,
                         structure="Long run, easy endurance (Z1-Z2)",
                         intensity=EASY)
    )
    slots = [
        _easy(_scale(50, spec, week)),
        PlannedSlot(WorkoutType.THRESHOLD_INTERVALS, duration_min=_scale(55, spec, week),
                    structure="wu + LT intervals 4-5x6-8 min @ T, 2 min jog + cd",
                    intensity=THRESHOLD),
        PlannedSlot(WorkoutType.EASY_LONG, duration_min=_scale(80, spec, week),
                    structure="Mid-week medium-long 11-15 mi, endurance pace "
                              "(Pfitzinger)", intensity=EASY),
        thu,
        _rest(),
        _easy(_scale(45, spec, week)),
        long_slot,
    ]
    return _finalize_long(slots, 6)


# ==========================================================================
# Registry + lookup
# ==========================================================================

TEMPLATES: dict[tuple[Tier, Goal], TemplateSpec] = {
    ("beginner", "health"): TemplateSpec(
        "True Beginner — General Health", "beginner", "health",
        total_weeks=8, base_mpw=3, peak_mpw=10, builder=_true_beginner_health),
    ("novice", "health"): TemplateSpec(
        "Novice — General Health", "novice", "health",
        total_weeks=8, base_mpw=10, peak_mpw=20, builder=_novice_health),
    ("intermediate", "health"): TemplateSpec(
        "Intermediate — General Health", "intermediate", "health",
        total_weeks=12, base_mpw=20, peak_mpw=30, builder=_intermediate_health),
    ("advanced", "health"): TemplateSpec(
        "Advanced — General Health", "advanced", "health",
        total_weeks=12, base_mpw=30, peak_mpw=50, builder=_advanced_health),
    ("intermediate", "speed"): TemplateSpec(
        "Intermediate — VO2max Focus", "intermediate", "speed",
        total_weeks=8, base_mpw=25, peak_mpw=35, builder=_intermediate_vo2max),
    ("advanced", "speed"): TemplateSpec(
        "Advanced — Speed-Endurance Block", "advanced", "speed",
        total_weeks=8, base_mpw=35, peak_mpw=50,
        builder=_advanced_speed_endurance),
    ("intermediate", "5K"): TemplateSpec(
        "5K Plan (Intermediate)", "intermediate", "5K",
        total_weeks=8, base_mpw=22, peak_mpw=30, builder=_five_k_intermediate,
        taper_factors=(0.5,)),
    ("advanced", "10K"): TemplateSpec(
        "10K Plan (Advanced)", "advanced", "10K",
        total_weeks=12, base_mpw=35, peak_mpw=50, builder=_ten_k_advanced,
        taper_factors=(0.7, 0.5)),
    ("intermediate", "HM"): TemplateSpec(
        "Half-Marathon Plan (Intermediate, Pfitzinger)", "intermediate", "HM",
        total_weeks=12, base_mpw=25, peak_mpw=35,
        builder=_half_marathon_intermediate, taper_factors=(0.7, 0.5)),
    ("advanced", "marathon"): TemplateSpec(
        "Marathon Plan (Advanced, 18 wk)", "advanced", "marathon",
        total_weeks=18, base_mpw=35, peak_mpw=55, builder=_marathon_advanced,
        taper_factors=(0.7, 0.5)),
}


def get_template(tier: Tier, goal: Goal) -> TemplateSpec:
    """Resolve the template for ``(tier, goal)``.

    Exact match first; otherwise fall back to the same tier's general-health
    block (a always-safe aerobic default), else raise so a missing
    combination is loud rather than silently mis-prescribed.
    """
    if (tier, goal) in TEMPLATES:
        return TEMPLATES[(tier, goal)]
    if (tier, "health") in TEMPLATES:
        return TEMPLATES[(tier, "health")]
    raise KeyError(f"No template for tier={tier!r}, goal={goal!r}")


def build_week(tier: Tier, goal: Goal, week: int) -> list[PlannedSlot]:
    """Seven :class:`PlannedSlot`s for ``week`` of the (tier, goal) block."""
    spec = get_template(tier, goal)
    slots = spec.builder(clamp_week(spec, week), spec)
    if len(slots) != 7:
        raise AssertionError(
            f"{spec.name} week {week} produced {len(slots)} days, expected 7"
        )
    return slots
