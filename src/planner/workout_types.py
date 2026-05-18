"""Workout vocabulary the weekly templates speak.

The Pydantic :data:`src.models.weekly_plan.WorkoutType` literal is a *coarse*
plan-output vocabulary ("vo2max", "speed", …) mirroring the framework.md §12
example JSON. Templates need to be more specific than that — a "vo2max" day
in a 5K block is a 4×4, in a 10K block a 5×1 km, in a 5K sharpen week a
30/30. :class:`WorkoutType` here is that *fine-grained* internal vocabulary
covering every session named anywhere in framework.md §3–§6/§10.

Each fine type maps back onto exactly one coarse model literal via
:data:`MODEL_WORKOUT_TYPE` / :func:`to_model_type`, so a generated
:class:`~src.models.WeeklyPlan` still validates against the unchanged Phase 1
schema (the specific variant is preserved in the day's ``structure`` text).

:class:`IntensityTarget` is the planner-native effort window (Daniels §7
zones). It is a plain frozen dataclass so templates stay pure data with no
Pydantic dependency; :meth:`IntensityTarget.to_model` bridges to the
schema's :class:`src.models.IntensityTarget` for the Phase 5 recommendation
layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.models import IntensityTarget as ModelIntensityTarget
from src.models.weekly_plan import WorkoutType as ModelWorkoutType


class WorkoutType(str, Enum):
    """Every session type referenced in framework.md §3–§6 and §10.

    ``str``-valued so the enum serialises to its framework name and reads
    naturally in structure text and tests.
    """

    REST = "rest"
    WALK = "walk"

    # Easy aerobic continuum (framework.md §3/§7 Zone E).
    EASY_SHORT = "easy_short"
    EASY = "easy"
    EASY_LONG = "easy_long"

    # Long runs (framework.md §6 — Daniels/Pfitzinger).
    LONG_RUN = "long_run"
    LONG_RUN_WITH_MP = "long_run_with_mp"  # marathon-pace segment embedded

    # Threshold / tempo (framework.md §6/§7 Zone T).
    TEMPO = "tempo"
    THRESHOLD_INTERVALS = "threshold_intervals"

    # VO2max variants (framework.md §4 — Helgerud / Billat).
    VO2MAX_4X4 = "vo2max_4x4"
    VO2MAX_5X1K = "vo2max_5x1k"
    VO2MAX_30_30 = "vo2max_30_30"

    # Neuromuscular / speed (framework.md §5 — Magness).
    HILL_SPRINTS = "hill_sprints"
    STRIDES = "strides"
    SPEED_ENDURANCE = "speed_endurance"  # 150–400 m repeats, R-pace

    CROSS_TRAIN = "cross_train"


# --------------------------------------------------------------------------
# Fine type -> coarse Phase 1 model literal. Every fine type collapses onto
# exactly one src.models.weekly_plan.WorkoutType so a WeeklyPlan validates
# unchanged; the specific variant survives in PlannedDay.structure.
# --------------------------------------------------------------------------

MODEL_WORKOUT_TYPE: dict[WorkoutType, ModelWorkoutType] = {
    WorkoutType.REST: "rest",
    WorkoutType.WALK: "rest_or_walk",
    WorkoutType.EASY_SHORT: "easy",
    WorkoutType.EASY: "easy",
    WorkoutType.EASY_LONG: "easy",
    WorkoutType.LONG_RUN: "long_run",
    WorkoutType.LONG_RUN_WITH_MP: "long_run",
    WorkoutType.TEMPO: "tempo",
    WorkoutType.THRESHOLD_INTERVALS: "threshold",
    WorkoutType.VO2MAX_4X4: "vo2max",
    WorkoutType.VO2MAX_5X1K: "vo2max",
    WorkoutType.VO2MAX_30_30: "vo2max",
    WorkoutType.HILL_SPRINTS: "speed",
    WorkoutType.STRIDES: "easy_with_strides",
    WorkoutType.SPEED_ENDURANCE: "speed",
    WorkoutType.CROSS_TRAIN: "cross_train",
}

#: Fine types that are genuinely hard quality sessions (framework.md §11
#: 48 h spacing). Used by templates to avoid stacking hard days.
HARD_WORKOUTS: frozenset[WorkoutType] = frozenset(
    {
        WorkoutType.TEMPO,
        WorkoutType.THRESHOLD_INTERVALS,
        WorkoutType.VO2MAX_4X4,
        WorkoutType.VO2MAX_5X1K,
        WorkoutType.VO2MAX_30_30,
        WorkoutType.SPEED_ENDURANCE,
        WorkoutType.LONG_RUN_WITH_MP,
    }
)

#: Types that carry no running and therefore get no warmup/cooldown.
NON_RUNNING: frozenset[WorkoutType] = frozenset(
    {WorkoutType.REST, WorkoutType.CROSS_TRAIN}
)


def to_model_type(workout: WorkoutType) -> ModelWorkoutType:
    """Coarse Phase 1 :data:`ModelWorkoutType` for a fine ``workout``."""
    return MODEL_WORKOUT_TYPE[workout]


@dataclass(frozen=True)
class IntensityTarget:
    """Prescribed effort window (framework.md §7 Daniels/Seiler zones).

    ``hr_pct_max`` is an inclusive ``(low, high)`` % of HRmax. ``rpe`` is a
    single representative Borg CR-10 value. ``pace_label`` is the Daniels
    letter (E/M/T/I/R) or a plain label. All optional so a rest slot can
    carry ``IntensityTarget()``.
    """

    hr_pct_max: Optional[tuple[int, int]] = None
    rpe: Optional[int] = None
    pace_label: Optional[str] = None

    def to_model(self) -> ModelIntensityTarget:
        """Bridge to the Phase 1 schema model (for the Phase 5 agent)."""
        return ModelIntensityTarget(
            hr_pct_max=self.hr_pct_max,
            rpe=self.rpe,
            pace_label=self.pace_label,
        )

    def describe(self) -> str:
        """Compact ``[HR 88-92% max, RPE 7, T]`` tag for structure text."""
        parts: list[str] = []
        if self.hr_pct_max is not None:
            parts.append(f"HR {self.hr_pct_max[0]}-{self.hr_pct_max[1]}% max")
        if self.rpe is not None:
            parts.append(f"RPE {self.rpe}")
        if self.pace_label is not None:
            parts.append(self.pace_label)
        return f"[{', '.join(parts)}]" if parts else ""


# --------------------------------------------------------------------------
# Daniels/Seiler intensity presets (framework.md §3 Zone-2 anchor, §7 table).
# %HRmax bands and RPE come straight from the §7 zone table.
# --------------------------------------------------------------------------

#: framework.md §3 general-health Zone-2 anchor: 60–70% HRmax, RPE 3–4.
GENERAL_HEALTH_Z2 = IntensityTarget(hr_pct_max=(60, 70), rpe=4, pace_label="Z2")
#: §7 Easy (E): 65–79% HRmax, RPE 3–4.
EASY = IntensityTarget(hr_pct_max=(65, 79), rpe=4, pace_label="E")
#: §7 Marathon (M): 80–89% HRmax, RPE 5–6.
MARATHON = IntensityTarget(hr_pct_max=(80, 89), rpe=6, pace_label="M")
#: §7 Threshold (T): 88–92% HRmax, RPE 7.
THRESHOLD = IntensityTarget(hr_pct_max=(88, 92), rpe=7, pace_label="T")
#: §7 Interval (I) / §4 VO2max: 95–100% HRmax, RPE 8–9.
INTERVAL = IntensityTarget(hr_pct_max=(95, 100), rpe=9, pace_label="I")
#: §7 Repetition (R): >100% HRmax effort, RPE 9–10 (neuromuscular).
REPETITION = IntensityTarget(hr_pct_max=(100, 100), rpe=10, pace_label="R")
#: Rest / recovery walk — no prescribed effort window.
REST_INTENSITY = IntensityTarget()
