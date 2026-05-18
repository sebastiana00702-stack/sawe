"""Pure transforms: raw WHOOP JSON → validated :class:`WhoopDaily`.

WHOOP models a day as three linked records — a physiological *cycle*
(strain), a *recovery* (HRV/RHR/recovery score), and a *sleep* (stages,
need, respiratory rate) — plus zero or more *workouts*. This module merges
them onto the framework.md §9 ``WhoopDaily`` shape.

Everything here is a **pure function**: no network, no clock, no globals.
:func:`merge_history` is the one orchestration helper — it calls the
injected client's already-fetched-data methods and then folds the JSON
together; the HTTP itself lives entirely in :mod:`src.ingest.whoop_client`.

Field-mapping notes (framework.md §8 semantics):

* ``hrv_rmssd_milli`` is already milliseconds rMSSD → ``hrv_rmssd``.
* ``sleep_hours`` is *actual* sleep (light+SWS+REM), not time in bed.
* ``sleep_need_hours`` is WHOOP's composite need (baseline + debt +
  strain − nap credit).
* ``skin_temp_dev_c`` is intentionally ``None``: WHOOP's API exposes an
  *absolute* skin temperature, never the personal-baseline *deviation*
  the framework defines, and a single record cannot establish a baseline.
* ``journal`` is ``{}``: WHOOP's API exposes no public journal collection.

v2 note: WHOOP's v2 resource IDs (``cycle_id``, ``sleep_id``,
``workout_id``) are UUID strings rather than v1 integers. The
recovery→cycle and recovery→sleep links below are plain equality lookups,
so they are unaffected — string keys compare just as integer keys did.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

from src.ingest.whoop_client import WhoopError
from src.models import WhoopDaily

_MS_PER_MIN = 60_000.0
_MS_PER_HOUR = 3_600_000.0

# WHOOP reports six effort zones (zone_zero is below Z1 / inactive); the
# agent's schema only models Z1-Z5, so zone_zero is dropped.
_ZONE_MAP = {
    "zone_one_milli": "Z1",
    "zone_two_milli": "Z2",
    "zone_three_milli": "Z3",
    "zone_four_milli": "Z4",
    "zone_five_milli": "Z5",
}


class NormalizationError(WhoopError):
    """A day's WHOOP records are too incomplete to form a valid WhoopDaily.

    Raised (and skipped by :func:`merge_history`) when a record is unscored
    / missing its ``score`` block, or a required physiological field is
    absent — never to mask a genuine mapping bug.
    """


# --------------------------------------------------------------------------
# Small parsing helpers
# --------------------------------------------------------------------------

def _scored(record: Optional[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Return ``record['score']`` or raise if the record is unusable."""
    if not record:
        raise NormalizationError(f"missing {kind} record")
    if record.get("score_state") not in (None, "SCORED"):
        raise NormalizationError(
            f"{kind} record is {record.get('score_state')}, not SCORED"
        )
    score = record.get("score")
    if not isinstance(score, dict):
        raise NormalizationError(f"{kind} record has no score block")
    return score


def _require(value: Any, kind: str, field: str) -> Any:
    if value is None:
        raise NormalizationError(f"{kind} record missing {field}")
    return value


def _parse_instant(value: str) -> datetime:
    """Parse a WHOOP RFC 3339 timestamp (``...Z`` or ``+00:00``)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_offset(offset: Optional[str]) -> timedelta:
    """``"-05:00"`` / ``"+01:30"`` → a :class:`timedelta` (default UTC)."""
    if not offset:
        return timedelta(0)
    sign = -1 if offset.startswith("-") else 1
    body = offset.lstrip("+-")
    hh, _, mm = body.partition(":")
    return sign * timedelta(hours=int(hh), minutes=int(mm or 0))


def _local_date(iso_start: str, tz_offset: Optional[str]) -> date:
    """Calendar date of an instant in the athlete's local timezone.

    WHOOP timestamps are UTC; the day a cycle "belongs to" is its local
    start date, so the offset is applied before taking ``.date()``.
    """
    return (_parse_instant(iso_start) + _parse_offset(tz_offset)).date()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# Single-day normalization
# --------------------------------------------------------------------------

def normalize_whoop_day(
    recovery_json: dict[str, Any],
    cycle_json: dict[str, Any],
    sleep_json: dict[str, Any],
    workouts_json: Optional[list[dict[str, Any]]] = None,
) -> WhoopDaily:
    """Merge one day's recovery/cycle/sleep/workout JSON into a WhoopDaily.

    Pure. Raises :class:`NormalizationError` if the recovery/cycle/sleep
    triple is incomplete; the result is always schema-validated before it
    is returned (a :class:`pydantic.ValidationError` here means WHOOP sent
    a physiologically impossible value, which must surface, not be hidden).
    """
    rec = _scored(recovery_json, "recovery")
    cyc = _scored(cycle_json, "cycle")
    slp = _scored(sleep_json, "sleep")

    day = _local_date(
        _require(cycle_json.get("start"), "cycle", "start"),
        cycle_json.get("timezone_offset"),
    )

    # --- recovery ---------------------------------------------------------
    recovery_score = int(
        round(_require(rec.get("recovery_score"), "recovery", "recovery_score"))
    )
    hrv = float(
        _require(rec.get("hrv_rmssd_milli"), "recovery", "hrv_rmssd_milli")
    )
    if hrv <= 0:
        raise NormalizationError("recovery hrv_rmssd_milli is non-positive")
    rhr = int(
        round(
            _require(
                rec.get("resting_heart_rate"), "recovery",
                "resting_heart_rate",
            )
        )
    )

    # --- sleep ------------------------------------------------------------
    stages = _require(slp.get("stage_summary"), "sleep", "stage_summary")
    rem_ms = stages.get("total_rem_sleep_time_milli", 0) or 0
    sws_ms = stages.get("total_slow_wave_sleep_time_milli", 0) or 0
    light_ms = stages.get("total_light_sleep_time_milli", 0) or 0
    asleep_ms = rem_ms + sws_ms + light_ms

    need = slp.get("sleep_needed") or {}
    need_ms = (
        (need.get("baseline_milli", 0) or 0)
        + (need.get("need_from_sleep_debt_milli", 0) or 0)
        + (need.get("need_from_recent_strain_milli", 0) or 0)
        + (need.get("need_from_recent_nap_milli", 0) or 0)
    )

    sleep_perf = _require(
        slp.get("sleep_performance_percentage"),
        "sleep", "sleep_performance_percentage",
    )
    respiratory_rate = float(
        _require(slp.get("respiratory_rate"), "sleep", "respiratory_rate")
    )

    # --- cycle / workouts -------------------------------------------------
    day_strain = float(_require(cyc.get("strain"), "cycle", "strain"))

    (
        workout_strain,
        workout_hr_mean,
        workout_hr_max,
        zone_minutes,
    ) = _summarize_workouts(workouts_json or [])

    return WhoopDaily(
        date=day,
        recovery_score=int(_clamp(recovery_score, 0, 100)),
        hrv_rmssd=_clamp(hrv, 0.001, 300.0),
        rhr=int(_clamp(rhr, 20, 120)),
        sleep_performance=_clamp(sleep_perf / 100.0, 0.0, 1.0),
        sleep_hours=_clamp(asleep_ms / _MS_PER_HOUR, 0.0, 24.0),
        sleep_need_hours=_clamp(need_ms / _MS_PER_HOUR, 0.0, 24.0),
        rem_min=max(0, round(rem_ms / _MS_PER_MIN)),
        sws_min=max(0, round(sws_ms / _MS_PER_MIN)),
        light_min=max(0, round(light_ms / _MS_PER_MIN)),
        day_strain=_clamp(day_strain, 0.0, 21.0),
        workout_strain=workout_strain,
        workout_hr_mean=workout_hr_mean,
        workout_hr_max=workout_hr_max,
        zone_minutes=zone_minutes,
        respiratory_rate=_clamp(respiratory_rate, 0.001, 40.0),
        # WHOOP exposes absolute skin temp, not a baseline deviation — the
        # framework's field is a deviation, so it stays None (see module
        # docstring). journal has no public collection endpoint.
        skin_temp_dev_c=None,
        journal={},
    )


def _summarize_workouts(
    workouts: list[dict[str, Any]],
) -> tuple[Optional[float], Optional[int], Optional[int], dict[str, float]]:
    """Collapse a day's workouts into the WhoopDaily workout fields.

    HR mean/max and ``workout_strain`` come from the day's *hardest* scored
    workout (highest strain); zone minutes are *summed* across all of them
    so the daily polarized-distribution math sees the full day's load.
    Returns all-``None``/empty when there is no scored workout.
    """
    zone_minutes: dict[str, float] = {}
    hardest: Optional[dict[str, Any]] = None
    hardest_strain = -1.0

    for w in workouts:
        if w.get("score_state") not in (None, "SCORED"):
            continue
        score = w.get("score")
        if not isinstance(score, dict):
            continue

        for ms_key, zlabel in _ZONE_MAP.items():
            ms = (score.get("zone_duration") or {}).get(ms_key, 0) or 0
            if ms:
                zone_minutes[zlabel] = round(
                    zone_minutes.get(zlabel, 0.0) + ms / _MS_PER_MIN, 2
                )

        strain = score.get("strain")
        if strain is not None and float(strain) > hardest_strain:
            hardest_strain = float(strain)
            hardest = score

    if hardest is None:
        return None, None, None, zone_minutes

    hr_mean = hardest.get("average_heart_rate")
    hr_max = hardest.get("max_heart_rate")
    return (
        round(_clamp(float(hardest_strain), 0.0, 21.0), 4),
        int(hr_mean) if hr_mean is not None else None,
        int(hr_max) if hr_max is not None else None,
        zone_minutes,
    )


# --------------------------------------------------------------------------
# History merge (orchestrates the client; no HTTP of its own)
# --------------------------------------------------------------------------

def merge_history(
    start_date: date,
    end_date: date,
    client: Any,
) -> list[WhoopDaily]:
    """Fetch + merge ``[start_date, end_date]`` into a clean time series.

    Pulls the four collections via ``client`` (the client owns all I/O),
    links recovery→cycle→sleep by WHOOP's ``cycle_id``/``sleep_id``,
    attaches that day's workouts, and normalizes each day. Days whose
    triple is incomplete/unscored are skipped (a partial WHOOP day is
    expected, not an error); the result is de-duplicated by date and
    sorted ascending so the recommender's rolling windows are chronological.
    """
    recoveries = client.get_recovery(start_date, end_date)
    cycles = client.get_cycles(start_date, end_date)
    sleeps = client.get_sleep(start_date, end_date)
    workouts = client.get_workouts(start_date, end_date)

    cycle_by_id = {c.get("id"): c for c in cycles if c.get("id") is not None}
    sleep_by_id = {s.get("id"): s for s in sleeps if s.get("id") is not None}

    workouts_by_day: dict[date, list[dict[str, Any]]] = {}
    for w in workouts:
        if not w.get("start"):
            continue
        wd = _local_date(w["start"], w.get("timezone_offset"))
        workouts_by_day.setdefault(wd, []).append(w)

    by_date: dict[date, WhoopDaily] = {}
    for recovery in recoveries:
        cycle = cycle_by_id.get(recovery.get("cycle_id"))
        sleep = sleep_by_id.get(recovery.get("sleep_id"))
        if cycle is None or sleep is None or not cycle.get("start"):
            continue
        day = _local_date(cycle["start"], cycle.get("timezone_offset"))
        try:
            whoop_day = normalize_whoop_day(
                recovery, cycle, sleep, workouts_by_day.get(day, [])
            )
        except NormalizationError:
            continue
        # Later cycles win if WHOOP returns more than one for a calendar
        # day (e.g. naps split a cycle); recoveries arrive newest-last.
        by_date[whoop_day.date] = whoop_day

    return [by_date[d] for d in sorted(by_date)]
