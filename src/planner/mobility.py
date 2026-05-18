"""Mobility / warm-up / cooldown sidecars (framework.md §2).

framework.md §2 is explicit that these are *sidecar text fields on the plan
output, not separate workouts* and that the agent appends a dynamic warm-up
and a cooldown to every running session (also §10: "Every plan: agent
appends ``dynamic_warmup_str`` and ``cooldown_str``"). No strength
programming — text reminders only.

The exact strings mirror the framework.md §12 example recommendation JSON.
:func:`warmup_for` / :func:`cooldown_for` pick the right text for a workout
(quality sessions get pre-session strides per §2; rest and cross-training
carry no running mobility).
"""

from __future__ import annotations

from typing import Optional

from src.planner.workout_types import HARD_WORKOUTS, NON_RUNNING, WorkoutType

#: §2 pre-run dynamic warm-up (no strides) — every easy/long running day.
WARMUP_STR = (
    "5-10 min easy jog, leg swings 10/side, walking lunges 10/side, "
    "A-skips 2x20m, B-skips 2x20m"
)

#: §2: "4x 60-80 m strides before any quality session" — appended to the
#: dynamic warm-up ahead of threshold/VO2max/speed work.
QUALITY_WARMUP_STR = WARMUP_STR + ", 4 strides 80m"

#: §2 post-run cooldown (also the §12 example JSON cooldown).
COOLDOWN_STR = (
    "5-10 min easy jog/walk, standing calf/hamstring/quad/hip-flexor "
    "20-30s each side"
)

#: Optional extra reminder the agent may surface (framework.md §2).
STRETCHING_REMINDER = (
    "Hold each stretch 20-30s, no bouncing; add foam-rolling of "
    "calves/quads/IT band if same-site soreness lingers."
)


def warmup_for(workout: WorkoutType) -> Optional[str]:
    """Dynamic warm-up text for ``workout`` (``None`` for non-running days).

    Quality sessions get the strides-inclusive variant (framework.md §2
    "4x 60-80 m strides before any quality session").
    """
    if workout in NON_RUNNING:
        return None
    if workout in HARD_WORKOUTS:
        return QUALITY_WARMUP_STR
    return WARMUP_STR


def cooldown_for(workout: WorkoutType) -> Optional[str]:
    """Cooldown text for ``workout`` (``None`` for non-running days)."""
    if workout in NON_RUNNING:
        return None
    return COOLDOWN_STR
