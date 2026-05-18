"""Deterministic safety rules for the Sawe running-coach agent.

Numeric thresholds (``thresholds``) and the §9/§11 gate logic (``gates``),
per framework.md. Re-exported here so callers can
``from src.rules import evaluate_safety_gates, choose_workout, ...``.

``thresholds`` is also importable as a module
(``from src.rules import thresholds as T``) — it is the single source of
truth every gate, the planner, and the tests share.
"""

from src.rules import thresholds
from src.rules.gates import (
    Flag,
    MetricHistory,
    SafetyReport,
    WorkoutDecision,
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

__all__ = [
    "thresholds",
    "Flag",
    "SafetyReport",
    "WorkoutDecision",
    "MetricHistory",
    "classify_recovery",
    "readiness",
    "training_state",
    "evaluate_safety_gates",
    "choose_workout",
    "downgrade_workout_type",
    "deload_triggers",
    "deload_due",
    "overreaching",
    "next_week_target_mpw",
]
