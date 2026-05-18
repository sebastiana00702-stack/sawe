"""The Phase 5 orchestrator — the deterministic recommendation engine.

:func:`recommend_daily_workout` is the single entry point framework.md §12
calls ``/daily_recommendation``. It folds the four prior phases together,
in the framework.md §9 order, into one validated
:class:`~src.models.Recommendation`:

    a. Phase 2 metrics over the rolling window (history + today appended,
       so today's z-score / ACWR / deltas are included).
    b. Phase 3 streak/counter bookkeeping → :class:`MetricHistory`.
    c. §9 classifiers (Phase 3 via :mod:`src.agent.state`).
    d. §9/§11 safety gates + the ordered override tree
       (:func:`src.rules.gates.choose_workout`).
    e. Today's planned session looked up from the Phase 4
       :class:`~src.models.WeeklyPlan`.
    f. The decision (downgrade / convert / rest / proceed) rendered into a
       concrete :class:`~src.models.WorkoutRecommendation` with intensity,
       a deterministic downgrade path, 2-3 alternatives, the framework.md
       §2 warm-up/cooldown sidecars, the surfaced flags, and the forced
       ``requires_medical_review`` bit.
    g. The completed :class:`~src.models.Recommendation`.

Per ``CLAUDE.md`` everything here is deterministic — no LLM, no training
logic invented locally. The §9 numeric decisions all live in Phase 3; this
module only *sequences* them and shapes the output schema. The rationale
prose (Phase 5's :mod:`src.agent.rationale`) is the only LLM-replaceable
layer and is still computed deterministically.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.agent.rationale import build_rationale
from src.agent.state import (
    classify_readiness,
    classify_training_state,
    consecutive_red_streak,
    days_since_last_hard,
    trailing_streak,
)
from src.metrics import (
    acwr,
    hrv_zscore,
    rhr_baseline_7d,
    rhr_delta,
    resp_rate_deviation,
    training_monotony,
)
from src.metrics.baselines import _BASELINE, _metric_series
from src.models import (
    Alternative,
    IntensityTarget,
    Recommendation,
    RunnerProfile,
    WeeklyPlan,
    WhoopDaily,
    WorkoutRecommendation,
)
from src.models.weekly_plan import PlannedDay
from src.planner.mobility import COOLDOWN_STR, QUALITY_WARMUP_STR, WARMUP_STR
from src.planner.workout_types import EASY, INTERVAL, REPETITION, THRESHOLD
from src.rules import thresholds as T
from src.rules.gates import MetricHistory, WorkoutDecision, choose_workout

# --------------------------------------------------------------------------
# Coarse-type → effort window (framework.md §7 Daniels zones). The Phase 4
# IntensityTarget presets are reused so the zone numbers have one source.
# rest / rest_or_walk / cross_train carry no prescribed effort.
# --------------------------------------------------------------------------

_COARSE_INTENSITY: dict[str, object] = {
    "easy": EASY,
    "easy_with_strides": EASY,
    "long_run": EASY,            # §6 long runs are Z1-Z2 aerobic
    "tempo": THRESHOLD,
    "threshold": THRESHOLD,
    "vo2max": INTERVAL,
    "speed": REPETITION,
}

#: §9 actions that mean "no training stimulus today".
_REST_ACTIONS = frozenset(
    {"full_rest", "rest_or_walk", "mandatory_rest", "illness_rest"}
)
#: Coarse types that get no running warm-up/cooldown (framework.md §2).
_NON_RUNNING = frozenset({"rest", "rest_or_walk", "cross_train"})
#: Coarse quality types — get the strides-inclusive §2 warm-up.
_QUALITY = frozenset({"vo2max", "threshold", "tempo", "speed"})


# --------------------------------------------------------------------------
# Frame plumbing
# --------------------------------------------------------------------------

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``date`` column (promoting a date index)."""
    if "date" in df.columns:
        return df.copy()
    out = df.copy()
    out.insert(0, "date", pd.to_datetime(out.index))
    return out.reset_index(drop=True)


def _combined_frame(today: WhoopDaily, history_df: pd.DataFrame) -> pd.DataFrame:
    """History with today's row appended — the window the metrics roll over.

    Today must be included so its own ACWR / HRV z-score / deltas are the
    most-recent values the gates read. The input frame is not mutated.
    """
    today_row = pd.DataFrame([today.model_dump()])
    if history_df is None or history_df.empty:
        return today_row
    hist = _ensure_date_column(history_df)
    return pd.concat([hist, today_row], ignore_index=True)


def _last_known(series: pd.Series) -> Optional[float]:
    """Most-recent value of a date-sorted metric Series, NaN → ``None``."""
    if series.empty:
        return None
    v = series.iloc[-1]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _is_yellow(score: object) -> bool:
    return T.RECOVERY_RED_MAX < int(score) <= T.RECOVERY_YELLOW_MAX


def _build_metric_history(
    today: WhoopDaily, history_df: pd.DataFrame
) -> MetricHistory:
    """Phase 2 scalars + Phase 3 streak counters for one decision.

    Unknown rolling values stay ``None`` (insufficient history) so the §9
    rules never assert danger from missing data. Streaks are measured on
    the *combined* window (they must include today); ``days_since_hard`` is
    measured on *history only* — today's session has not happened yet.
    """
    combined = _combined_frame(today, history_df)

    # --- rolling scalars (today = the last, most-recent row) ---
    rhr_d = _last_known(rhr_delta(combined))
    hrv_z = _last_known(hrv_zscore(combined))
    rr_d = _last_known(resp_rate_deviation(combined))
    acwr_now = _last_known(acwr(combined))
    monotony = _last_known(training_monotony(combined))

    # RHR 7-day mean vs 28-day mean (framework.md §6 overreaching input).
    rhr_7d = _last_known(rhr_baseline_7d(combined))
    rhr_28d = _last_known(
        _metric_series(combined, "rhr").rolling(_BASELINE).mean()
    )
    rhr_7d_vs_28d = (
        rhr_7d - rhr_28d
        if rhr_7d is not None and rhr_28d is not None
        else None
    )

    # --- streaks on the combined window (must include today) ---
    recoveries = (
        _metric_series(combined, "recovery_score").tolist()
    )
    red_streak = consecutive_red_streak(combined)
    yellow_streak = trailing_streak(recoveries, _is_yellow)
    yellow_count_7d = sum(
        1 for s in recoveries[-7:] if _is_yellow(s)
    )

    hrv_z_series = hrv_zscore(combined).tolist()
    hrv_low_streak = trailing_streak(
        hrv_z_series,
        lambda z: z is not None
        and not (isinstance(z, float) and math.isnan(z))
        and z < T.HRV_Z_LOW,
    )

    rhr_delta_series = rhr_delta(combined).tolist()
    rhr_red_flag_streak = trailing_streak(
        rhr_delta_series,
        lambda d: d is not None
        and not (isinstance(d, float) and math.isnan(d))
        and d > T.RHR_DELTA_RED_FLAG,
    )

    # NFOR clock proxy: consecutive sub-moderate-recovery days (framework
    # §11 "recovery <50 … escalate"). A genuine "despite reduced load"
    # ledger needs cross-week state the single-day call does not own.
    poor_recovery_persist = trailing_streak(
        recoveries, lambda s: int(s) < T.READINESS_MODERATE_RECOVERY
    )

    return MetricHistory(
        rhr_delta=rhr_d,
        hrv_z=hrv_z,
        resp_rate_delta=rr_d,
        acwr=acwr_now,
        monotony=monotony,
        red_streak=red_streak,
        yellow_streak=yellow_streak,
        yellow_count_7d=yellow_count_7d,
        days_since_hard=days_since_last_hard(
            _ensure_date_column(history_df)
            if history_df is not None and not history_df.empty
            else pd.DataFrame()
        ),
        hrv_low_streak_days=hrv_low_streak,
        rhr_7d_vs_28d=rhr_7d_vs_28d,
        # Cross-week build-completion ledger is a scheduler concern, not a
        # single-day input; left 0 (deload-by-builds simply will not fire
        # here — the other deload triggers are fully WHOOP-derived).
        builds_completed=0,
        rhr_red_flag_streak_days=rhr_red_flag_streak,
        poor_recovery_persist_days=poor_recovery_persist,
    )


# --------------------------------------------------------------------------
# Decision → concrete workout
# --------------------------------------------------------------------------

def _scaled_duration(
    planned_min: Optional[int], load_factor: float
) -> Optional[int]:
    """Scale a planned duration by the §9 load factor (>=1.0 → unchanged)."""
    if planned_min is None:
        return None
    if load_factor >= 1.0:
        return planned_min
    return max(1, round(planned_min * load_factor))


def _intensity_for(coarse_type: str) -> Optional[IntensityTarget]:
    preset = _COARSE_INTENSITY.get(coarse_type)
    return preset.to_model() if preset is not None else None


def _mobility_for(coarse_type: str) -> tuple[Optional[str], Optional[str]]:
    """(warmup, cooldown) §2 sidecars for a coarse workout type."""
    if coarse_type in _NON_RUNNING:
        return None, None
    warmup = QUALITY_WARMUP_STR if coarse_type in _QUALITY else WARMUP_STR
    return warmup, COOLDOWN_STR


def _structure_text(
    action: str,
    planned: PlannedDay,
    decision: WorkoutDecision,
    resulting_type: str,
    duration_min: Optional[int],
) -> str:
    """Deterministic structure prose for the resolved session."""
    reason = decision.rationale[0] if decision.rationale else ""
    dur = f"{duration_min} min" if duration_min else "the planned time"

    if action == "proceed":
        return planned.structure or f"{planned.type} as planned"
    if action == "full_rest":
        return f"Full rest — {reason} No training stimulus today."
    if action == "mandatory_rest":
        return f"Mandatory rest day — {reason}"
    if action == "illness_rest":
        return (
            f"Rest and run a sick-check — {reason} See a clinician if "
            f"symptoms persist."
        )
    if action == "rest_or_walk":
        return f"Rest or an easy 20-30 min walk — {reason}"
    if action == "convert_easy_strides":
        return (
            f"Converted {planned.type} → easy + strides ({reason}) "
            f"{dur} easy @ E + 4-6 x 20 s strides."
        )
    # downgrade
    return (
        f"Downgraded {planned.type} → {resulting_type} ({reason}) "
        f"{dur} at the lower {resulting_type} intensity."
    )


def _alternatives(
    action: str, planned: PlannedDay
) -> list[Alternative]:
    """2-3 deterministic backup options (framework.md §12)."""
    if action in _REST_ACTIONS:
        return [
            Alternative(type="rest_or_walk", duration_min=20),
            Alternative(type="cross_train", modality="bike", duration_min=30),
        ]
    easy_min = planned.duration_min or 40
    return [
        Alternative(type="easy", duration_min=easy_min),
        Alternative(type="cross_train", modality="bike", duration_min=60),
        Alternative(type="rest"),
    ]


def _downgrade_path() -> dict[str, str]:
    """The conditional fallbacks (framework.md §12 ``downgrade_path``).

    Deterministic and keyed off the named §9 thresholds so the conditions
    can never drift from the gates that enforce them.
    """
    return {
        f"if_recovery_drops_below_{T.READINESS_MODERATE_RECOVERY}": (
            "convert to easy Z1-Z2 aerobic"
        ),
        f"if_sleep_below_{int(T.SLEEP_DOWNGRADE_BELOW_H)}h": (
            "downgrade any quality session / shorten to easy"
        ),
        f"if_rhr_above_baseline_{int(T.RHR_DELTA_CAUTION)}bpm": (
            "drop intensity — easy or rest, no quality"
        ),
        "if_two_consecutive_reds": "mandatory rest day",
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def recommend_daily_workout(
    today: WhoopDaily,
    history_df: pd.DataFrame,
    profile: RunnerProfile,
    weekly_plan: WeeklyPlan,
) -> Recommendation:
    """Produce the validated daily :class:`~src.models.Recommendation`.

    Pipeline a–g in the module docstring / framework.md §9 order. Raises
    :class:`ValueError` if ``weekly_plan`` has no entry for ``today.date``
    (loud is correct — a silently mis-dated plan must not be prescribed).
    """
    # (a–b) metrics + streak context
    mh = _build_metric_history(today, history_df)

    # (c) §9 classifiers (Phase 3, exact)
    state = classify_training_state(
        mh.acwr, mh.hrv_z, mh.rhr_delta, mh.red_streak
    )
    ready = classify_readiness(
        today.recovery_score, today.sleep_hours, mh.rhr_delta
    )

    # (e) today's planned session
    planned = next(
        (d for d in weekly_plan.days if d.date == today.date), None
    )
    if planned is None:
        raise ValueError(
            f"WeeklyPlan starting {weekly_plan.week_starting} has no day "
            f"for {today.date}; cannot recommend without a planned session."
        )

    # (d) §9/§11 gates + ordered override tree
    decision: WorkoutDecision = choose_workout(planned.type, today, mh)
    action = decision.action
    resulting_type: str = decision.resulting_type or planned.type

    # (f) render the decision into a concrete session
    if action in _REST_ACTIONS:
        duration_min: Optional[int] = 0
    else:
        duration_min = _scaled_duration(
            planned.duration_min, decision.load_factor
        )

    rationale = build_rationale(today, state, ready, mh, decision)
    warmup, cooldown = _mobility_for(resulting_type)

    workout = WorkoutRecommendation(
        type=resulting_type,
        duration_min=duration_min,
        structure=_structure_text(
            action, planned, decision, resulting_type, duration_min
        ),
        intensity_target=(
            None if action in _REST_ACTIONS
            else _intensity_for(resulting_type)
        ),
        rationale=rationale,
        downgrade_path=_downgrade_path(),
        alternatives=_alternatives(action, planned),
        warmup=warmup,
        cooldown=cooldown,
        # disclaimer left to the schema default (framework.md §1/§11 —
        # it can never be omitted).
    )

    # (g) the completed recommendation
    return Recommendation(
        date=today.date,
        athlete_id=weekly_plan.athlete_id,
        training_state=state,
        readiness=ready,
        recommendation=workout,
        flags=[f.render() for f in decision.flags],
        requires_medical_review=decision.requires_medical_review,
    )
