"""Weekly plan.

The week-ahead training template the planner emits, mirroring the example
JSON in ``docs/framework.md`` §12. ``tier`` and ``goal`` reuse the literals
from :mod:`src.models.runner_profile` so a plan can never describe a tier the
profile model would reject.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.runner_profile import Goal, Tier

# Workout types used across the framework.md §10/§12 example plans.
WorkoutType = Literal[
    "easy",
    "easy_with_strides",
    "long_run",
    "tempo",
    "threshold",
    "vo2max",
    "speed",
    "cross_train",
    "rest",
    "rest_or_walk",
]


class PlannedDay(BaseModel):
    """One scheduled day within a :class:`WeeklyPlan`."""

    model_config = ConfigDict(extra="forbid")

    date: date
    type: WorkoutType
    duration_min: Optional[int] = Field(default=None, ge=0)
    structure: Optional[str] = None
    modality: Optional[str] = Field(
        default=None, description="e.g. bike/swim for cross_train days"
    )


class WeeklyPlan(BaseModel):
    """A seven-ish-day plan with its load target and deload countdown."""

    model_config = ConfigDict(extra="forbid")

    week_starting: date
    athlete_id: str
    tier: Tier
    goal: Goal
    phase: str = Field(description="e.g. base/specific_prep/taper")
    planned_mpw: float = Field(ge=0, le=250)
    days: list[PlannedDay] = Field(min_length=1)
    deload_due_in_weeks: int = Field(ge=0)
    # [low, high] ACWR sweet-spot band, e.g. [0.8, 1.3] (framework.md §11).
    acwr_target: tuple[float, float]

    @model_validator(mode="after")
    def _validate_acwr_band(self) -> "WeeklyPlan":
        low, high = self.acwr_target
        if low < 0 or high < 0:
            raise ValueError("acwr_target values must be >= 0")
        if low > high:
            raise ValueError(
                f"acwr_target low ({low}) must be <= high ({high})"
            )
        return self
