"""Daily recommendation.

Output model for the agent's ``/daily_recommendation`` decision, mirroring
the example JSON in ``docs/framework.md`` §12. ``training_state`` and
``readiness`` use the exact classification literals from §9.

Per framework.md §1/§11 every recommendation carries a ``disclaimer`` and a
``requires_medical_review`` flag that the rules layer forces ``True`` when a
red-flag threshold trips. The default disclaimer is hard-coded here so it can
never be omitted.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# framework.md §9 training_state() outcomes.
TrainingState = Literal[
    "overreached", "strained", "detraining", "functional", "borderline"
]
# framework.md §9 readiness() outcomes.
Readiness = Literal["high", "moderate", "low", "very_low"]

DEFAULT_DISCLAIMER = (
    "Not medical advice. Stop if pain, chest discomfort, or dizziness."
)


class IntensityTarget(BaseModel):
    """Prescribed effort window for a workout (framework.md §7 zones)."""

    model_config = ConfigDict(extra="forbid")

    # [low%, high%] of HRmax, e.g. [88, 92] for threshold work.
    hr_pct_max: Optional[tuple[int, int]] = None
    rpe: Optional[int] = Field(default=None, ge=1, le=10)
    pace_label: Optional[str] = Field(
        default=None, description="Daniels pace label, e.g. E/M/T/I/R"
    )


class Alternative(BaseModel):
    """A swap the athlete may take instead of the primary workout."""

    model_config = ConfigDict(extra="forbid")

    type: str
    duration_min: Optional[int] = Field(default=None, ge=0)
    modality: Optional[str] = Field(
        default=None, description="e.g. bike/swim/elliptical for cross-training"
    )


class WorkoutRecommendation(BaseModel):
    """The prescribed session plus its downgrade path and sidecar text."""

    model_config = ConfigDict(extra="forbid")

    type: str
    duration_min: Optional[int] = Field(default=None, ge=0)
    structure: Optional[str] = None
    intensity_target: Optional[IntensityTarget] = None

    rationale: list[str] = Field(default_factory=list)
    # Maps a condition key -> what the workout becomes if it trips.
    downgrade_path: dict[str, str] = Field(default_factory=dict)
    alternatives: list[Alternative] = Field(default_factory=list)

    # Mobility sidecars the agent always surfaces (framework.md §2).
    warmup: Optional[str] = None
    cooldown: Optional[str] = None

    disclaimer: str = DEFAULT_DISCLAIMER


class Recommendation(BaseModel):
    """Top-level daily recommendation returned by the agent."""

    model_config = ConfigDict(extra="forbid")

    date: date
    athlete_id: str
    training_state: TrainingState
    readiness: Readiness
    recommendation: WorkoutRecommendation

    # Surfaced safety/illness flags (never paraphrased away — framework.md §12).
    flags: list[str] = Field(default_factory=list)
    # Forced True by the rules layer when a red-flag threshold trips (§11).
    requires_medical_review: bool = False
