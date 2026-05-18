"""Deterministic safety gates, classifiers, and downgrade/deload logic.

This is the §9 "Agent Decision Logic (CORE)" and the §11 hard ceilings,
implemented exactly as the framework pseudocode specifies (strict ``<``/``>``
for the override tree, non-strict ``>=``/``<=`` for the classifiers). It is
the deterministic heart of the agent — per ``CLAUDE.md`` the LLM layer never
touches anything here.

Separation of concerns:

* :mod:`src.metrics` (Phase 2) turns a WhoopDaily history into rolling
  scalars (``acwr``, ``hrv_zscore``, ``rhr_delta`` …).
* The orchestrator (Phase 5) folds those scalars plus streak bookkeeping
  (consecutive Reds, Yellow cluster, days since a hard session) into a
  :class:`MetricHistory`.
* This module consumes only :class:`~src.models.WhoopDaily` (today's raw row)
  and :class:`MetricHistory` (the precomputed context) — **no pandas here**,
  so every rule is a pure function pinned to a constant in
  :mod:`src.rules.thresholds` and unit-tested at its exact boundary.

The public surface is:

* classifiers — :func:`classify_recovery`, :func:`readiness`,
  :func:`training_state`
* :func:`evaluate_safety_gates` — the §12-recommended single
  ``(WhoopDaily, history) -> SafetyReport`` flag generator
* :func:`choose_workout` — the §9 ordered override tree
* :func:`deload_triggers` / :func:`deload_due` / :func:`overreaching`
* :func:`next_week_target_mpw` — §9 ACWR-aware mileage progression
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.models.recommendation import Readiness, TrainingState
from src.models.weekly_plan import WorkoutType
from src.models.whoop_daily import WhoopDaily
from src.rules import thresholds as T

# --------------------------------------------------------------------------
# Workout-type sets the §9 override tree branches on. The framework
# pseudocode uses {"VO2max","threshold","long_run"} and
# {"VO2max","threshold","sprint"}; mapped onto this project's WorkoutType
# literal ("sprint" -> "speed"). ``tempo`` is grouped with ``threshold`` —
# it is the same T-pace quality session (framework.md §6/§10) and
# downgrading it under low recovery is the safety-first reading of §9.
# --------------------------------------------------------------------------

#: §9: ``recovery < 67`` downgrades these (quality / long aerobic) sessions.
RECOVERY_DOWNGRADE_TYPES: frozenset[WorkoutType] = frozenset(
    {"vo2max", "threshold", "tempo", "long_run"}
)
#: §9: ``sleep < 6 h`` downgrades these (quality / neuromuscular) sessions.
SLEEP_DOWNGRADE_TYPES: frozenset[WorkoutType] = frozenset(
    {"vo2max", "threshold", "tempo", "speed"}
)
#: Any genuinely hard session — used by the §11 "no hard workout on a
#: 2-Red streak" / 48 h spacing ceilings.
HARD_TYPES: frozenset[WorkoutType] = frozenset(
    {"vo2max", "threshold", "tempo", "long_run", "speed"}
)
#: §11 hard ceiling: race-pace work blocked after 2 Yellows unless RHR is
#: at baseline.
RACE_PACE_TYPES: frozenset[WorkoutType] = frozenset(
    {"vo2max", "threshold", "tempo", "speed"}
)


# --------------------------------------------------------------------------
# Output models
# --------------------------------------------------------------------------

FlagSeverity = Literal["info", "caution", "downgrade", "rest", "medical"]

# Stable flag codes (so tests and the orchestrator never match on prose).
FLAG_FULL_REST = "full_rest"
FLAG_RECOVERY_RED = "recovery_red"
FLAG_SLEEP_REST = "sleep_below_rest_floor"
FLAG_SLEEP_DOWNGRADE = "sleep_below_downgrade_floor"
FLAG_SLEEP_PERFORMANCE = "sleep_performance_low"
FLAG_RECOVERY_YELLOW = "recovery_yellow"
FLAG_MANDATORY_REST = "two_consecutive_reds"
FLAG_ILLNESS = "illness_suspect"
FLAG_RHR_CAUTION = "rhr_elevated"
FLAG_HRV_LOW_PERSIST = "hrv_low_3d"
FLAG_ACWR_HARD_STOP = "acwr_above_1_5"
FLAG_ACWR_DETRAIN = "acwr_detraining"
FLAG_MONOTONY_HIGH = "monotony_high"
FLAG_SPACING = "hard_session_too_soon"
FLAG_SORENESS_SPRINT_BLOCK = "sprint_blocked_soreness"
FLAG_RACEPACE_BLOCK = "racepace_blocked_yellow_streak"
FLAG_OVERREACHING = "overreaching"
FLAG_NFOR = "nfor_suspect"
# Red flags -> requires_medical_review (framework.md §11).
FLAG_MED_CARDIAC = "red_flag_cardiac"
FLAG_MED_RHR_PERSIST = "red_flag_rhr_persistent"
FLAG_MED_REDS = "red_flag_red_s_suspect"
FLAG_MED_PAIN = "red_flag_pain_worsening"
FLAG_MED_ILLNESS = "red_flag_persistent_illness"


class Flag(BaseModel):
    """One tripped threshold. ``code`` is stable; ``message`` is human text.

    Surfaced verbatim on :class:`~src.models.Recommendation` — per
    framework.md §12 red flags must never be paraphrased away, so the
    orchestrator renders :meth:`render`, not a re-summary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: FlagSeverity
    message: str

    def render(self) -> str:
        """``"[severity] message"`` for ``Recommendation.flags`` (list[str])."""
        return f"[{self.severity}] {self.message}"


class SafetyReport(BaseModel):
    """Every tripped threshold for a day, plus the hard medical bit."""

    model_config = ConfigDict(extra="forbid")

    flags: list[Flag] = Field(default_factory=list)
    #: Forced ``True`` by any §11 red flag; the orchestrator copies this
    #: onto the Recommendation and the LLM layer may never clear it.
    requires_medical_review: bool = False

    def codes(self) -> list[str]:
        return [f.code for f in self.flags]


WorkoutAction = Literal[
    "proceed",            # green light, run the planned session
    "downgrade",          # lower intensity / volume (load_factor applies)
    "convert_easy_strides",  # quality -> easy + strides (spacing)
    "rest_or_walk",       # §9 recovery/sleep override
    "mandatory_rest",     # §9/§11 two consecutive Reds
    "illness_rest",       # §9 RHR+HRV / RHR+RR illness signature
    "full_rest",          # §11 hard full-rest trigger
]


class WorkoutDecision(BaseModel):
    """What the §9 override tree decided about today's planned session."""

    model_config = ConfigDict(extra="forbid")

    action: WorkoutAction
    #: Multiplier on the planned session's load. 1.0 = as planned,
    #: 0.5 = §9 ACWR>1.5 halving, 0.0 = any rest action.
    load_factor: float = Field(ge=0.0, le=1.0)
    #: The substituted WorkoutType when the session is changed (None when
    #: the planned type stands or is simply scaled).
    resulting_type: Optional[WorkoutType] = None
    rationale: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    requires_medical_review: bool = False


# --------------------------------------------------------------------------
# Precomputed context the rules consume
# --------------------------------------------------------------------------

class MetricHistory(BaseModel):
    """Rolling scalars (from :mod:`src.metrics`) + streak bookkeeping.

    Optional float fields are ``None`` when there is not yet enough history
    (the metrics layer yields ``NaN``; the orchestrator passes ``None``).
    A rule whose input is unknown does not fire — the agent never asserts
    danger from missing data, and conversely never green-lights past a
    *known* breach.
    """

    model_config = ConfigDict(extra="forbid")

    # --- rolling deviations (src.metrics) ---
    rhr_delta: Optional[float] = None          # today RHR - 28d mean
    hrv_z: Optional[float] = None              # today's 28d z-score
    resp_rate_delta: Optional[float] = None    # today RR - 28d mean
    acwr: Optional[float] = None
    monotony: Optional[float] = None

    # --- streaks / counts (orchestrator bookkeeping) ---
    red_streak: int = Field(default=0, ge=0)        # consec Red incl. today
    yellow_streak: int = Field(default=0, ge=0)     # consec Yellow incl. today
    yellow_count_7d: int = Field(default=0, ge=0)   # Yellows in trailing 7d
    days_since_hard: Optional[int] = Field(default=None, ge=0)
    hrv_low_streak_days: int = Field(default=0, ge=0)  # consec HRV<base-1SD
    rhr_7d_vs_28d: Optional[float] = None           # RHR 7d mean - 28d mean
    builds_completed: int = Field(default=0, ge=0)  # consec build weeks
    rhr_red_flag_streak_days: int = Field(default=0, ge=0)  # consec dRHR>+10
    poor_recovery_persist_days: int = Field(default=0, ge=0)  # NFOR clock


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _known(x: Optional[float]) -> bool:
    """True iff ``x`` is a real number (not ``None`` / NaN)."""
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _journal_flag(w: WhoopDaily, key: str) -> bool:
    """Truthy WHOOP-Journal boolean (absent key -> ``False``)."""
    return bool(w.journal.get(key, False))


def _journal_score(w: WhoopDaily, key: str) -> Optional[int]:
    """A 0-5 WHOOP-Journal self-report, or ``None`` if not logged."""
    v = w.journal.get(key)
    return int(v) if isinstance(v, (int, float)) else None


# --------------------------------------------------------------------------
# Classifiers (framework.md §8 / §9)
# --------------------------------------------------------------------------

RecoveryBand = Literal["red", "yellow", "green"]


def classify_recovery(recovery_score: int) -> RecoveryBand:
    """WHOOP band: Red ``<= 33``, Yellow ``34..66``, Green ``>= 67`` (§8)."""
    if recovery_score <= T.RECOVERY_RED_MAX:
        return "red"
    if recovery_score >= T.RECOVERY_GREEN_MIN:
        return "green"
    return "yellow"


def readiness(
    recovery_score: int,
    sleep_hours: float,
    rhr_delta: Optional[float],
) -> Readiness:
    """§9 ``readiness()``, mirrored exactly.

    ``high`` needs a known ``rhr_delta <= 3``; when it is unknown the day
    cannot be certified ``high`` and falls through to ``moderate`` (which
    does not test RHR).
    """
    if (
        recovery_score >= T.READINESS_HIGH_RECOVERY
        and sleep_hours >= T.READINESS_HIGH_SLEEP_H
        and _known(rhr_delta)
        and rhr_delta <= T.RHR_DELTA_HIGH_READINESS
    ):
        return "high"
    if (
        recovery_score >= T.READINESS_MODERATE_RECOVERY
        and sleep_hours >= T.READINESS_MODERATE_SLEEP_H
    ):
        return "moderate"
    if recovery_score >= T.READINESS_LOW_RECOVERY:
        return "low"
    return "very_low"


def training_state(
    acwr: Optional[float],
    hrv_z: Optional[float],
    rhr_delta: Optional[float],
    red_streak: int,
) -> TrainingState:
    """§9 ``training_state()``, mirrored exactly.

    The overreached branch needs no ACWR; if ACWR is still unknown after
    that, the day is ``borderline`` (cannot place it on the ACWR scale).
    """
    if red_streak >= T.RED_STREAK_MANDATORY_REST or (
        _known(hrv_z)
        and hrv_z < T.HRV_Z_LOW
        and _known(rhr_delta)
        and rhr_delta > T.RHR_DELTA_CAUTION
    ):
        return "overreached"
    if not _known(acwr):
        return "borderline"
    if acwr > T.ACWR_HARD_STOP:
        return "strained"
    if acwr < T.ACWR_SWEET_LOW:
        return "detraining"
    if T.ACWR_SWEET_LOW <= acwr <= T.ACWR_SWEET_HIGH:
        return "functional"
    return "borderline"


# --------------------------------------------------------------------------
# Red flags -> medical referral (framework.md §11)
# --------------------------------------------------------------------------

def _medical_flags(w: WhoopDaily, h: MetricHistory) -> list[Flag]:
    """§11 red flags. Any one forces ``requires_medical_review=True``."""
    flags: list[Flag] = []

    if (
        _journal_flag(w, "chest_pain")
        or _journal_flag(w, "syncope")
        or _journal_flag(w, "palpitations")
    ):
        flags.append(Flag(
            code=FLAG_MED_CARDIAC,
            severity="medical",
            message=(
                "Chest pain / syncope / palpitations reported during "
                "exercise — stop and seek medical evaluation."
            ),
        ))

    # RHR persistently > +10 bpm above baseline for > 7 days.
    if h.rhr_red_flag_streak_days > T.RHR_RED_FLAG_PERSIST_DAYS:
        flags.append(Flag(
            code=FLAG_MED_RHR_PERSIST,
            severity="medical",
            message=(
                f"Resting HR >+{T.RHR_DELTA_RED_FLAG:.0f} bpm above baseline "
                f"for {h.rhr_red_flag_streak_days} days "
                f"(> {T.RHR_RED_FLAG_PERSIST_DAYS}) — medical referral."
            ),
        ))

    # HRV crash > 2 SD below baseline + RED-S correlates.
    if (
        _known(h.hrv_z)
        and h.hrv_z < T.HRV_Z_RED_FLAG
        and (
            _journal_flag(w, "weight_loss")
            or _journal_flag(w, "amenorrhea")
            or _journal_flag(w, "mood_disturbance")
        )
    ):
        flags.append(Flag(
            code=FLAG_MED_REDS,
            severity="medical",
            message=(
                "HRV crash >2 SD below baseline with weight loss / "
                "amenorrhea / mood disturbance — suspect RED-S, refer."
            ),
        ))

    if _journal_flag(w, "pain_worsens_during_run"):
        flags.append(Flag(
            code=FLAG_MED_PAIN,
            severity="medical",
            message="Pain that worsens during a run — stop and assess.",
        ))

    if _journal_flag(w, "persistent_illness"):
        flags.append(Flag(
            code=FLAG_MED_ILLNESS,
            severity="medical",
            message="Persistent illness symptoms — see a clinician.",
        ))

    return flags


# --------------------------------------------------------------------------
# The single safety-gate evaluator (framework.md §12 recommendation #2)
# --------------------------------------------------------------------------

def evaluate_safety_gates(w: WhoopDaily, h: MetricHistory) -> SafetyReport:
    """Enumerate every tripped §9/§11 threshold for one day.

    Independent of what was planned — it reports *signals*. The chosen
    action lives in :func:`choose_workout`. ``requires_medical_review`` is
    forced ``True`` by any §11 red flag (never clearable downstream).
    """
    flags: list[Flag] = []

    # ---- §11 hard full-rest triggers ----
    full_rest_reasons = []
    if w.recovery_score < T.RECOVERY_FULL_REST_BELOW:
        full_rest_reasons.append(
            f"recovery {w.recovery_score} < {T.RECOVERY_FULL_REST_BELOW}"
        )
    if w.sleep_hours < T.SLEEP_FULL_REST_BELOW_H:
        full_rest_reasons.append(
            f"sleep {w.sleep_hours:g} h < {T.SLEEP_FULL_REST_BELOW_H:g} h"
        )
    if _journal_flag(w, "fever"):
        full_rest_reasons.append("fever reported")
    if _journal_flag(w, "sharp_pain_altering_gait"):
        full_rest_reasons.append("sharp pain altering gait")
    if full_rest_reasons:
        flags.append(Flag(
            code=FLAG_FULL_REST,
            severity="rest",
            message="Full rest: " + "; ".join(full_rest_reasons) + ".",
        ))

    # ---- §9 recovery band ----
    band = classify_recovery(w.recovery_score)
    if band == "red":
        flags.append(Flag(
            code=FLAG_RECOVERY_RED,
            severity="rest",
            message=(
                f"Recovery {w.recovery_score}% is Red "
                f"(<= {T.RECOVERY_RED_MAX}%) — rest or walk only."
            ),
        ))
    elif band == "yellow":
        flags.append(Flag(
            code=FLAG_RECOVERY_YELLOW,
            severity="caution",
            message=(
                f"Recovery {w.recovery_score}% is Yellow — moderate "
                f"ceiling, no hard session."
            ),
        ))

    # ---- §9 sleep ----
    if w.sleep_hours < T.SLEEP_REST_BELOW_H:
        flags.append(Flag(
            code=FLAG_SLEEP_REST,
            severity="rest",
            message=(
                f"Sleep {w.sleep_hours:g} h < {T.SLEEP_REST_BELOW_H:g} h "
                f"hard floor — rest (no interval session)."
            ),
        ))
    elif w.sleep_hours < T.SLEEP_DOWNGRADE_BELOW_H:
        flags.append(Flag(
            code=FLAG_SLEEP_DOWNGRADE,
            severity="downgrade",
            message=(
                f"Sleep {w.sleep_hours:g} h < {T.SLEEP_DOWNGRADE_BELOW_H:g} h "
                f"— downgrade any quality session."
            ),
        ))
    if w.sleep_performance < T.SLEEP_PERFORMANCE_DOWNGRADE_BELOW:
        flags.append(Flag(
            code=FLAG_SLEEP_PERFORMANCE,
            severity="downgrade",
            message=(
                f"Sleep performance {w.sleep_performance:.0%} < "
                f"{T.SLEEP_PERFORMANCE_DOWNGRADE_BELOW:.0%} — downgrade."
            ),
        ))

    # ---- §9 / §11 two consecutive Reds ----
    if h.red_streak >= T.RED_STREAK_MANDATORY_REST:
        flags.append(Flag(
            code=FLAG_MANDATORY_REST,
            severity="rest",
            message=(
                f"{h.red_streak} consecutive Red recoveries "
                f"(>= {T.RED_STREAK_MANDATORY_REST}) — mandatory rest day."
            ),
        ))

    # ---- §9 illness signatures ----
    rhr_hrv_illness = (
        _known(h.rhr_delta)
        and h.rhr_delta > T.RHR_DELTA_ILLNESS_REST
        and _known(h.hrv_z)
        and h.hrv_z < T.HRV_Z_LOW
    )
    rhr_rr_illness = (
        _known(h.rhr_delta)
        and h.rhr_delta > T.RHR_DELTA_CAUTION
        and _known(h.resp_rate_delta)
        and h.resp_rate_delta > T.RESP_RATE_DELTA_ILLNESS
    )
    if rhr_hrv_illness or rhr_rr_illness:
        flags.append(Flag(
            code=FLAG_ILLNESS,
            severity="rest",
            message=(
                "Illness signature (elevated RHR with HRV crash or raised "
                "respiratory rate) — rest and run a sick-check."
            ),
        ))
    elif _known(h.rhr_delta) and h.rhr_delta > T.RHR_DELTA_CAUTION:
        flags.append(Flag(
            code=FLAG_RHR_CAUTION,
            severity="caution",
            message=(
                f"RHR +{h.rhr_delta:.1f} bpm > +{T.RHR_DELTA_CAUTION:g} "
                f"vs baseline — caution, no intensity."
            ),
        ))

    # ---- §9 sustained low HRV ----
    if h.hrv_low_streak_days >= T.HRV_LOW_PERSIST_DAYS:
        flags.append(Flag(
            code=FLAG_HRV_LOW_PERSIST,
            severity="downgrade",
            message=(
                f"HRV below baseline -1 SD for {h.hrv_low_streak_days} days "
                f"(>= {T.HRV_LOW_PERSIST_DAYS}) — reduce load "
                f"{T.HRV_LOW_LOAD_CUT_MIN:.0%}-{T.HRV_LOW_LOAD_CUT_MAX:.0%}."
            ),
        ))

    # ---- §9 / §11 ACWR ----
    if _known(h.acwr):
        if h.acwr > T.ACWR_HARD_STOP:
            flags.append(Flag(
                code=FLAG_ACWR_HARD_STOP,
                severity="downgrade",
                message=(
                    f"ACWR {h.acwr:.2f} > {T.ACWR_HARD_STOP:g} — hold/cut "
                    f"load (halve the session)."
                ),
            ))
        elif h.acwr < T.ACWR_SWEET_LOW:
            flags.append(Flag(
                code=FLAG_ACWR_DETRAIN,
                severity="info",
                message=(
                    f"ACWR {h.acwr:.2f} < {T.ACWR_SWEET_LOW:g} — "
                    f"detraining zone, room to build."
                ),
            ))

    # ---- §9 monotony ----
    if _known(h.monotony) and h.monotony > T.MONOTONY_HIGH:
        flags.append(Flag(
            code=FLAG_MONOTONY_HIGH,
            severity="caution",
            message=(
                f"Training monotony {h.monotony:.2f} > {T.MONOTONY_HIGH:g} "
                f"(Foster) — force variety / easy days."
            ),
        ))

    # ---- §9 hard-session spacing ----
    if (
        h.days_since_hard is not None
        and h.days_since_hard < T.DAYS_SINCE_HARD_MIN
    ):
        flags.append(Flag(
            code=FLAG_SPACING,
            severity="caution",
            message=(
                f"Only {h.days_since_hard} day(s) since the last hard "
                f"session (< {T.DAYS_SINCE_HARD_MIN}) — no quality today."
            ),
        ))

    # ---- §6 overreaching detector ----
    if overreaching(h):
        flags.append(Flag(
            code=FLAG_OVERREACHING,
            severity="downgrade",
            message=(
                "Poor-adaptation pattern (sustained low HRV, raised RHR, "
                "Red/Yellow streak) — prescribe a 5-7 day deload, "
                "50% volume, no intensity."
            ),
        ))

    # ---- §11 NFOR persistence ----
    if h.poor_recovery_persist_days > T.NFOR_PERSIST_DAYS:
        flags.append(Flag(
            code=FLAG_NFOR,
            severity="caution",
            message=(
                f"Poor recovery markers persisting "
                f"{h.poor_recovery_persist_days} days "
                f"(> {T.NFOR_PERSIST_DAYS}) despite reduced load — "
                f"suspect non-functional overreaching; consider "
                f"medical/coach review."
            ),
        ))

    # ---- §11 red flags -> medical ----
    med = _medical_flags(w, h)
    flags.extend(med)

    return SafetyReport(flags=flags, requires_medical_review=bool(med))


# --------------------------------------------------------------------------
# The §9 ordered override tree
# --------------------------------------------------------------------------

def downgrade_workout_type(planned: WorkoutType) -> WorkoutType:
    """Deterministic intensity drop for a downgraded session (§9 ``downgrade``).

    Quality work collapses to its safest aerobic equivalent; VO2max/speed
    keep a neuromuscular touch via strides (framework.md §5/§9).
    """
    mapping: dict[WorkoutType, WorkoutType] = {
        "vo2max": "easy_with_strides",
        "speed": "easy_with_strides",
        "threshold": "easy",
        "tempo": "easy",
        "long_run": "easy",
    }
    return mapping.get(planned, planned)


def choose_workout(
    planned_type: WorkoutType,
    w: WhoopDaily,
    h: MetricHistory,
) -> WorkoutDecision:
    """§9 ``choose_workout`` override tree (+ §11 ceilings), in order.

    Returns the *decision* (action + load factor + substituted type), not a
    full plan — wiring it to the planner's concrete session is Phase 4/5.
    Every flag from :func:`evaluate_safety_gates` rides along, and any §11
    red flag forces ``requires_medical_review``.
    """
    report = evaluate_safety_gates(w, h)
    codes = set(report.codes())

    def decide(
        action: WorkoutAction,
        load_factor: float,
        rationale: str,
        resulting_type: Optional[WorkoutType] = None,
    ) -> WorkoutDecision:
        return WorkoutDecision(
            action=action,
            load_factor=load_factor,
            resulting_type=resulting_type,
            rationale=[rationale],
            flags=report.flags,
            requires_medical_review=report.requires_medical_review,
        )

    # 0) §11 hard full-rest trigger — overrides everything.
    if FLAG_FULL_REST in codes:
        return decide("full_rest", 0.0, "§11 full-rest trigger.", "rest")

    # 1) §9 hard safety overrides, in framework order.
    if w.recovery_score < T.RECOVERY_REST_BELOW:
        return decide(
            "rest_or_walk", 0.0,
            f"Recovery {w.recovery_score}% < {T.RECOVERY_REST_BELOW} (Red).",
            "rest_or_walk",
        )
    if w.sleep_hours < T.SLEEP_REST_BELOW_H:
        return decide(
            "rest_or_walk", 0.0,
            f"Sleep {w.sleep_hours:g} h < {T.SLEEP_REST_BELOW_H:g} h.",
            "rest_or_walk",
        )
    if h.red_streak >= T.RED_STREAK_MANDATORY_REST:
        return decide(
            "mandatory_rest", 0.0,
            f"{h.red_streak} consecutive Reds "
            f"(>= {T.RED_STREAK_MANDATORY_REST}).",
            "rest",
        )
    if FLAG_ILLNESS in codes:
        return decide(
            "illness_rest", 0.0,
            "Illness signature (RHR+HRV or RHR+RR) — rest and sick-check.",
            "rest",
        )
    if _known(h.acwr) and h.acwr > T.ACWR_HARD_STOP:
        return decide(
            "downgrade", T.ACWR_DOWNGRADE_FACTOR,
            f"ACWR {h.acwr:.2f} > {T.ACWR_HARD_STOP:g} — halve load.",
            downgrade_workout_type(planned_type),
        )

    # 2) Moderate downgrades.
    if (
        w.recovery_score < T.RECOVERY_MODERATE_BELOW
        and planned_type in RECOVERY_DOWNGRADE_TYPES
    ):
        return decide(
            "downgrade", 1.0,
            f"Recovery {w.recovery_score}% < {T.RECOVERY_MODERATE_BELOW} "
            f"(not Green) — downgrade {planned_type}.",
            downgrade_workout_type(planned_type),
        )
    if (
        w.sleep_hours < T.SLEEP_DOWNGRADE_BELOW_H
        and planned_type in SLEEP_DOWNGRADE_TYPES
    ):
        return decide(
            "downgrade", 1.0,
            f"Sleep {w.sleep_hours:g} h < {T.SLEEP_DOWNGRADE_BELOW_H:g} h "
            f"— downgrade {planned_type}.",
            downgrade_workout_type(planned_type),
        )
    # §9 sustained-low-HRV load reduction (20-30%, conservative end applied).
    if (
        h.hrv_low_streak_days >= T.HRV_LOW_PERSIST_DAYS
        and planned_type in HARD_TYPES
    ):
        return decide(
            "downgrade", 1.0 - T.HRV_LOW_LOAD_CUT_APPLIED,
            f"HRV low {h.hrv_low_streak_days}d "
            f"(>= {T.HRV_LOW_PERSIST_DAYS}) — cut load "
            f"{T.HRV_LOW_LOAD_CUT_APPLIED:.0%}.",
            downgrade_workout_type(planned_type),
        )

    # 3) §11 hard ceilings on quality work.
    # No NEW sprint introduction with hamstring/calf soreness >= 3/5.
    sore = _journal_score(w, "hamstring_calf_soreness")
    if (
        planned_type == "speed"
        and sore is not None
        and sore >= T.SORENESS_SPRINT_BLOCK
    ):
        return decide(
            "convert_easy_strides", 1.0,
            f"Hamstring/calf soreness {sore}/5 "
            f"(>= {T.SORENESS_SPRINT_BLOCK}) — no new sprinting.",
            "easy_with_strides",
        )
    # No race-pace after 2 consecutive Yellows unless RHR at baseline.
    if (
        planned_type in RACE_PACE_TYPES
        and h.yellow_streak >= T.YELLOW_STREAK_RACEPACE_BLOCK
        and not (
            _known(h.rhr_delta)
            and h.rhr_delta <= T.RHR_DELTA_AT_BASELINE
        )
    ):
        return decide(
            "downgrade", 1.0,
            f"{h.yellow_streak} consecutive Yellows "
            f"(>= {T.YELLOW_STREAK_RACEPACE_BLOCK}) and RHR not at baseline "
            f"— no race-pace work.",
            downgrade_workout_type(planned_type),
        )

    # 4) §9 hard-session spacing: VO2max needs >= 2 days since the last hard.
    if (
        planned_type == "vo2max"
        and h.days_since_hard is not None
        and h.days_since_hard < T.DAYS_SINCE_HARD_MIN
    ):
        return decide(
            "convert_easy_strides", 1.0,
            f"Only {h.days_since_hard} day(s) since last hard "
            f"(< {T.DAYS_SINCE_HARD_MIN}) — easy + strides instead.",
            "easy_with_strides",
        )

    # 5) Green light.
    return decide("proceed", 1.0, "All safety gates clear.", planned_type)


# --------------------------------------------------------------------------
# Deload triggers & overreaching detector (framework.md §6 / §9)
# --------------------------------------------------------------------------

def overreaching(h: MetricHistory) -> bool:
    """§6 poor-adaptation detector — fires only when ALL hold.

    HRV 7-day mean < 28-day mean -1 SD for 3+ consecutive days, AND RHR
    7-day mean > 28-day mean +5 bpm, AND (2 consecutive Reds OR sustained
    Yellow 5+ days). Components with no data cannot satisfy the AND, so the
    detector stays silent rather than false-firing.
    """
    hrv_persist = h.hrv_low_streak_days >= T.OVERREACH_HRV_PERSIST_DAYS
    rhr_high = _known(h.rhr_7d_vs_28d) and h.rhr_7d_vs_28d > T.OVERREACH_RHR_DELTA
    streak = (
        h.red_streak >= T.OVERREACH_RED_STREAK
        or h.yellow_streak >= T.OVERREACH_YELLOW_DAYS
    )
    return hrv_persist and rhr_high and streak


def deload_triggers(h: MetricHistory) -> list[str]:
    """§9 deload triggers — returns the code of every trigger that fired."""
    fired: list[str] = []
    if h.builds_completed >= T.BUILD_WEEKS_BEFORE_DELOAD:
        fired.append("build_weeks_complete")
    if _known(h.monotony) and h.monotony > T.MONOTONY_HIGH:
        fired.append("monotony_high")
    if h.yellow_count_7d >= T.YELLOW_DELOAD_COUNT:
        fired.append("yellow_cluster")
    if (
        h.hrv_low_streak_days >= T.OVERREACH_HRV_PERSIST_DAYS
        and _known(h.rhr_delta)
        and h.rhr_delta > T.OVERREACH_RHR_DELTA
    ):
        fired.append("hrv_rhr_persist")
    if overreaching(h):
        fired.append("overreaching")
    return fired


def deload_due(h: MetricHistory) -> bool:
    """True when any §9 deload trigger has fired."""
    return bool(deload_triggers(h))


# --------------------------------------------------------------------------
# ACWR-aware mileage progression (framework.md §9)
# --------------------------------------------------------------------------

def next_week_target_mpw(
    current_mpw: float,
    acwr_now: Optional[float],
    deload_due: bool,
) -> float:
    """§9 ``next_week_target_mpw``, mirrored exactly.

    A pending deload always wins (60%). Otherwise: ACWR > 1.3 holds,
    ACWR < 0.8 ramps +10%, anything else (incl. unknown ACWR — gentle by
    default) ramps the standard +7%. Both ramps respect the §11 10% soft
    cap.
    """
    if deload_due:
        return current_mpw * T.MPW_DELOAD_FACTOR
    if _known(acwr_now) and acwr_now > T.MPW_HOLD_ABOVE_ACWR:
        return current_mpw * T.MPW_HOLD_FACTOR
    if _known(acwr_now) and acwr_now < T.MPW_RAMP_BELOW_ACWR:
        return current_mpw * T.MPW_RAMP_DETRAIN_FACTOR
    return current_mpw * T.MPW_RAMP_NORMAL_FACTOR
