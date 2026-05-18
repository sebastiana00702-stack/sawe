"""Deterministic rationale builder (framework.md §12, no LLM).

``CLAUDE.md`` and framework.md §12 are explicit: the *plain-English "why
this workout"* layer is the one piece that an LLM may later personalise —
but the numeric decision and the safety story must stay deterministic and
must never be paraphrased away. So this module emits the rationale as a
list of fixed, templated strings (mirroring the §12 example JSON:
``"Recovery 58% (Yellow), within moderate band"``, ``"ACWR 1.12 in sweet
spot"``, ``"Last hard session 3 days ago"``, ``"Sleep 6.8 h (above 6h
floor)"``). A future LLM layer would *restyle* these lines; it would not
compute them.

:func:`build_rationale` is pure: inputs in, ``list[str]`` out. It reads
only the already-classified state, the precomputed
:class:`~src.rules.gates.MetricHistory` scalars, today's raw row, and the
:class:`~src.rules.gates.WorkoutDecision` the §9 tree produced. Lines are
ordered facts → conclusion: the readiness/load facts first, then the
decisive override reason(s) from the decision itself.
"""

from __future__ import annotations

from src.models.recommendation import Readiness, TrainingState
from src.models.whoop_daily import WhoopDaily
from src.rules import thresholds as T
from src.rules.gates import MetricHistory, WorkoutDecision, classify_recovery

_BAND_TITLE = {"red": "Red", "yellow": "Yellow", "green": "Green"}
_BAND_GLOSS = {
    "red": "rest/walk only",
    "yellow": "within the moderate band, no hard session",
    "green": "primed, plan as scheduled",
}


def _known(x) -> bool:
    return x is not None and not (isinstance(x, float) and x != x)  # NaN-safe


def _recovery_line(w: WhoopDaily) -> str:
    band = classify_recovery(w.recovery_score)
    return (
        f"Recovery {w.recovery_score}% ({_BAND_TITLE[band]}), "
        f"{_BAND_GLOSS[band]}"
    )


def _sleep_line(w: WhoopDaily) -> str:
    h = w.sleep_hours
    if h < T.SLEEP_REST_BELOW_H:
        return (
            f"Sleep {h:g} h below the {T.SLEEP_REST_BELOW_H:g} h hard floor "
            f"— rest (no intervals)"
        )
    if h < T.SLEEP_DOWNGRADE_BELOW_H:
        return (
            f"Sleep {h:g} h below the {T.SLEEP_DOWNGRADE_BELOW_H:g} h floor "
            f"— downgrade any quality"
        )
    return f"Sleep {h:g} h (above {T.SLEEP_DOWNGRADE_BELOW_H:g} h floor)"


def _acwr_line(acwr: float) -> str:
    if acwr > T.ACWR_HARD_STOP:
        return f"ACWR {acwr:.2f} > {T.ACWR_HARD_STOP:g} — hold/cut load"
    if acwr < T.ACWR_SWEET_LOW:
        return (
            f"ACWR {acwr:.2f} < {T.ACWR_SWEET_LOW:g} — detraining zone, "
            f"room to build"
        )
    if acwr <= T.ACWR_SWEET_HIGH:
        return (
            f"ACWR {acwr:.2f} in sweet spot "
            f"({T.ACWR_SWEET_LOW:g}-{T.ACWR_SWEET_HIGH:g})"
        )
    return (
        f"ACWR {acwr:.2f} elevated (above the {T.ACWR_SWEET_HIGH:g} "
        f"sweet-spot ceiling)"
    )


def build_rationale(
    today: WhoopDaily,
    training_state: TrainingState,
    readiness: Readiness,
    mh: MetricHistory,
    decision: WorkoutDecision,
) -> list[str]:
    """The ordered, deterministic "why" lines for the recommendation.

    Only lines whose underlying signal is *known* are emitted (the agent
    never narrates data it does not have). The decision's own rationale
    (the §9 override reason, or "All safety gates clear.") is appended last
    as the conclusion.
    """
    lines: list[str] = [_recovery_line(today), _sleep_line(today)]

    if _known(mh.acwr):
        lines.append(_acwr_line(mh.acwr))

    if _known(mh.hrv_z):
        if mh.hrv_low_streak_days >= T.HRV_LOW_PERSIST_DAYS:
            lines.append(
                f"HRV {mh.hrv_z:+.2f} SD, below baseline for "
                f"{mh.hrv_low_streak_days} days "
                f"(>= {T.HRV_LOW_PERSIST_DAYS}) — reduce load"
            )
        elif mh.hrv_z < T.HRV_Z_LOW:
            lines.append(
                f"HRV {mh.hrv_z:+.2f} SD below baseline (single day — "
                f"treat as noise until it persists)"
            )
        else:
            lines.append(f"HRV {mh.hrv_z:+.2f} SD, within normal band")

    if _known(mh.rhr_delta):
        if mh.rhr_delta > T.RHR_DELTA_CAUTION:
            lines.append(
                f"RHR +{mh.rhr_delta:.1f} bpm > +{T.RHR_DELTA_CAUTION:g} "
                f"vs 28-day baseline — caution, no intensity"
            )
        else:
            lines.append(
                f"RHR {mh.rhr_delta:+.1f} bpm vs 28-day baseline "
                f"(within range)"
            )

    if mh.days_since_hard is not None:
        if mh.days_since_hard < T.DAYS_SINCE_HARD_MIN:
            lines.append(
                f"Last hard session {mh.days_since_hard} day(s) ago "
                f"(< {T.DAYS_SINCE_HARD_MIN} — too soon for quality)"
            )
        elif mh.days_since_hard >= 0 and mh.days_since_hard < 90:
            lines.append(f"Last hard session {mh.days_since_hard} days ago")
        else:
            lines.append("No hard session in the recent window")

    lines.append(
        f"Training state {training_state}; readiness {readiness}"
    )

    # The §9 override conclusion (already a templated string from the gate).
    lines.extend(decision.rationale)
    return lines
