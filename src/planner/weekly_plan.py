"""Turn a runner profile + week number into a schema :class:`WeeklyPlan`.

This is the seam between the planner's fine-grained, dateless templates
(:mod:`src.planner.templates`) and the Phase 1 output schema. It:

* resolves the (tier, goal) :class:`~src.planner.templates.TemplateSpec`,
* expands the seven :class:`~src.planner.templates.PlannedSlot`s for the
  requested week and stamps consecutive calendar dates onto them,
* maps each fine :class:`~src.planner.workout_types.WorkoutType` onto the
  coarse :data:`src.models.weekly_plan.WorkoutType` literal (so the result
  validates against the *unchanged* Phase 1 schema), folding the intensity
  target and the framework.md §2 warm-up/cooldown into the day's
  ``structure`` text — :class:`~src.models.PlannedDay` is a fixed
  ``extra="forbid"`` schema with no intensity/mobility fields, and §2 is
  explicit those are sidecar *text*, not separate workouts,
* reads ``planned_mpw`` / ``phase`` / ``deload_due_in_weeks`` off the
  block's progression curve and advertises the default ACWR sweet-spot band.

No WHOOP/safety logic here — downgrades and gates are Phase 3, applied to
this plan in Phase 5.
"""

from __future__ import annotations

from datetime import date, timedelta

from src.models import RunnerProfile
from src.models.weekly_plan import PlannedDay, WeeklyPlan
from src.planner.mobility import cooldown_for, warmup_for
from src.planner.templates import (
    DEFAULT_ACWR_TARGET,
    PlannedSlot,
    build_week,
    deload_due_in_weeks,
    get_template,
    phase_for,
    planned_mpw,
)
from src.planner.workout_types import to_model_type

#: Default athlete id when the caller does not supply one (RunnerProfile has
#: no id field; the orchestrator passes the real one in Phase 5).
DEFAULT_ATHLETE_ID = "athlete"


def _compose_structure(slot: PlannedSlot) -> str:
    """Workout text + intensity tag + §2 warm-up/cooldown sidecars.

    Mobility is appended for every running day (framework.md §2/§10 "agent
    appends ``dynamic_warmup_str`` and ``cooldown_str``"); rest and
    cross-training carry none.
    """
    pieces = []
    if slot.structure:
        pieces.append(slot.structure)
    tag = slot.intensity.describe()
    if tag:
        pieces.append(tag)
    text = " ".join(pieces)

    warmup = warmup_for(slot.workout)
    cooldown = cooldown_for(slot.workout)
    if warmup and cooldown:
        text = f"{text} | Warmup: {warmup} | Cooldown: {cooldown}"
    return text


def generate_weekly_plan(
    profile: RunnerProfile,
    week_number: int,
    start_date: date,
    athlete_id: str = DEFAULT_ATHLETE_ID,
) -> WeeklyPlan:
    """Build the validated :class:`WeeklyPlan` for ``week_number``.

    ``week_number`` is 1-based within the block; values past the block
    length are clamped to the final week (callers may legitimately ask for
    "this week" of a finished block).
    """
    spec = get_template(profile.tier, profile.goal)
    slots = build_week(profile.tier, profile.goal, week_number)

    days: list[PlannedDay] = []
    for offset, slot in enumerate(slots):
        days.append(
            PlannedDay(
                date=start_date + timedelta(days=offset),
                type=to_model_type(slot.workout),
                duration_min=slot.duration_min,
                structure=_compose_structure(slot),
                modality=slot.modality,
            )
        )

    return WeeklyPlan(
        week_starting=start_date,
        athlete_id=athlete_id,
        tier=profile.tier,
        goal=profile.goal,
        phase=phase_for(spec, week_number),
        planned_mpw=round(planned_mpw(spec, week_number), 1),
        days=days,
        deload_due_in_weeks=deload_due_in_weeks(spec, week_number),
        acwr_target=DEFAULT_ACWR_TARGET,
    )
