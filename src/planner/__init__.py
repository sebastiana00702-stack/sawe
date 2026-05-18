"""Weekly plan templates and the plan generator (framework.md §3/§6/§10).

Pure data + simple block progression — no WHOOP/safety logic (Phase 3,
applied in Phase 5). Re-exported so callers can
``from src.planner import generate_weekly_plan, WorkoutType, ...``.
"""

from src.planner.mobility import (
    COOLDOWN_STR,
    QUALITY_WARMUP_STR,
    STRETCHING_REMINDER,
    WARMUP_STR,
    cooldown_for,
    warmup_for,
)
from src.planner.templates import (
    DEFAULT_ACWR_TARGET,
    PlannedSlot,
    TemplateSpec,
    TEMPLATES,
    build_week,
    deload_due_in_weeks,
    get_template,
    is_deload_week,
    phase_for,
    planned_mpw,
    volume_factor,
)
from src.planner.weekly_plan import generate_weekly_plan
from src.planner.workout_types import (
    HARD_WORKOUTS,
    IntensityTarget,
    MODEL_WORKOUT_TYPE,
    WorkoutType,
    to_model_type,
)

__all__ = [
    # workout vocabulary
    "WorkoutType",
    "IntensityTarget",
    "MODEL_WORKOUT_TYPE",
    "HARD_WORKOUTS",
    "to_model_type",
    # mobility sidecars
    "WARMUP_STR",
    "QUALITY_WARMUP_STR",
    "COOLDOWN_STR",
    "STRETCHING_REMINDER",
    "warmup_for",
    "cooldown_for",
    # templates + progression
    "TemplateSpec",
    "PlannedSlot",
    "TEMPLATES",
    "DEFAULT_ACWR_TARGET",
    "get_template",
    "build_week",
    "planned_mpw",
    "phase_for",
    "volume_factor",
    "is_deload_week",
    "deload_due_in_weeks",
    # plan generator
    "generate_weekly_plan",
]
