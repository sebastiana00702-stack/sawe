"""Runner profile.

Static (slowly-changing) athlete context, mirroring the ``RunnerProfile``
schema in ``docs/framework.md`` §9. Tier and goal drive which weekly template
the planner selects; ``hrmax`` exposes the framework.md §9 derived metric
(Tanaka: 208 - 0.7 * age) so HR zone math has a single source of truth.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Shared literals — also consumed by WeeklyPlan so a plan's tier/goal can
# never drift from a profile's.
Tier = Literal["beginner", "novice", "intermediate", "advanced", "competitive"]
Goal = Literal["health", "5K", "10K", "HM", "marathon", "speed"]
Sex = Literal["M", "F", "other"]


class RunnerProfile(BaseModel):
    """Athlete tier, goal, and physiology used to select and scale plans."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier
    goal: Goal
    age: int = Field(ge=5, le=100)
    sex: Sex

    # Lab/field-measured HRmax overrides the age formula when present.
    hrmax_measured: Optional[int] = Field(default=None, ge=100, le=230)
    vdot: Optional[float] = Field(default=None, gt=0, le=90)

    current_mpw: float = Field(ge=0, le=250, description="Current miles per week")
    longest_recent_run_mi: float = Field(ge=0, le=100)

    target_race_date: Optional[date] = None

    def hrmax(self) -> int:
        """Estimated max HR (framework.md §9 derived metric).

        Uses the measured value when available, else Tanaka et al. 2001:
        ``HRmax ≈ 208 - 0.7 * age``. The classical ``220 - age`` is avoided
        because it over-estimates older adults.
        """
        if self.hrmax_measured is not None:
            return self.hrmax_measured
        return round(208 - 0.7 * self.age)
