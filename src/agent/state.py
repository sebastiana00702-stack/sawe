"""Training-state / readiness classification + streak bookkeeping.

This is the first half of the Phase 5 orchestrator: it turns the rolling
scalars from :mod:`src.metrics` and the raw WHOOP history into the small
set of *classifications* and *streak counts* that the §9 decision tree
(:mod:`src.rules.gates`) consumes.

Two layers live here:

* **Classifiers** — :func:`classify_training_state` /
  :func:`classify_readiness` are thin, intentional delegations to the
  framework.md §9 reference implementations already mirrored *exactly* in
  :mod:`src.rules.gates` (Phase 3, boundary-tested in ``tests/test_rules``).
  Phase 5 must not re-derive §9 logic — duplicating it would let the two
  copies drift, and ``CLAUDE.md`` makes framework.md §9 the single source
  of truth. They are re-exposed here only so the orchestrator has one
  import surface.

  *Spec reconciliation*: the Phase 5 brief writes
  ``classify_training_state(...) -> Literal["fresh","functional",
  "strained","overreached","detraining"]``. Framework.md §9
  ``training_state()`` (authoritative per ``CLAUDE.md``) and the frozen
  Phase 1 :class:`~src.models.Recommendation` schema both use
  ``"borderline"`` — *not* ``"fresh"`` — for the ``1.3 < ACWR <= 1.5``
  zone, and the schema is ``extra="forbid"``. We keep the framework /
  schema literal (:data:`~src.models.recommendation.TrainingState`) so the
  recommendation still validates and the §9 boundary tests still hold;
  ``"borderline"`` *is* the "neither fresh-functional nor overreached"
  bucket the brief's ``"fresh"`` was gesturing at.

* **Streak helpers** — :func:`days_since_last_hard` and
  :func:`consecutive_red_streak` (plus the generic
  :func:`trailing_streak`) read a WhoopDaily-shaped history frame. They
  are pure, never mutate the frame, and tolerate a ``date`` column or a
  date index, gaps in the calendar, and unsorted input.

"Hard session" is not in the §9 numeric threshold table, so by the
``CLAUDE.md`` rule it is *not* a ``src.rules.thresholds`` constant — it is
an orchestration heuristic defined (and documented) here, anchored to the
framework.md §7 polarised zones (Z3+ = above LT2) and the §8 day-strain
load proxy.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Optional

import pandas as pd

from src.rules.gates import classify_recovery
from src.rules.gates import readiness as _gate_readiness
from src.rules.gates import training_state as _gate_training_state
from src.models.recommendation import Readiness, TrainingState

# --------------------------------------------------------------------------
# "Hard session" detection heuristic (framework.md §7 / §8 / §11)
#
# §9 references ``hist.days_since_hard`` as a *given* input; it never
# defines how a day is judged hard from WHOOP data, so this is an
# orchestration choice, not a §9 safety threshold (hence here, not in
# ``src.rules.thresholds``). A day counts as hard if EITHER:
#   * day_strain is in the quality/long-run band — the §8 load proxy
#     cleanly separates easy/recovery (~7-8) from quality (~13+) and long
#     runs (~16+); long runs are Z1-Z2 yet still "hard" for the §11 48 h
#     spacing rule, and only day_strain captures them, OR
#   * enough minutes were spent above LT2 (Z3+Z4+Z5) — catches a short,
#     sharp quality session that barely moved whole-day strain.
# --------------------------------------------------------------------------

#: §8 day-strain at/above which a day is treated as a hard session.
HARD_DAY_STRAIN_MIN = 11.0
#: Combined Z3+Z4+Z5 minutes (framework.md §7 "above LT2") that mark a
#: quality session regardless of whole-day strain.
HARD_INTENSITY_MIN_MINUTES = 12.0
#: Returned by :func:`days_since_last_hard` when the history holds no hard
#: session (or is empty): a value large enough to never trip the §9
#: ``days_since_hard < 2`` spacing gate.
NO_RECENT_HARD = 99

_HIGH_ZONES = ("Z3", "Z4", "Z5")


# --------------------------------------------------------------------------
# Classifiers — delegate to the §9-faithful Phase 3 implementations
# --------------------------------------------------------------------------

def classify_training_state(
    acwr: Optional[float],
    hrv_z: Optional[float],
    rhr_delta: Optional[float],
    red_streak: int,
) -> TrainingState:
    """Framework.md §9 ``training_state()`` (see module docstring re ``fresh``).

    Delegates to the boundary-tested :func:`src.rules.gates.training_state`
    so the §9 logic has exactly one implementation.
    """
    return _gate_training_state(acwr, hrv_z, rhr_delta, red_streak)


def classify_readiness(
    recovery_score: int,
    sleep_hours: float,
    rhr_delta: Optional[float],
) -> Readiness:
    """Framework.md §9 ``readiness()`` — delegates to Phase 3 (one source)."""
    return _gate_readiness(recovery_score, sleep_hours, rhr_delta)


# --------------------------------------------------------------------------
# History-frame helpers
# --------------------------------------------------------------------------

def _sorted_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Chronologically sorted *copy* (``date`` column or index), never mutated.

    A stable sort preserves input order among equal dates so the "trailing"
    helpers stay deterministic.
    """
    if "date" in df.columns:
        keyed = df.assign(_d=pd.to_datetime(df["date"]))
        return keyed.sort_values("_d", kind="stable").drop(columns="_d")
    return df.sort_index(kind="stable")


def trailing_streak(values: Iterable, predicate: Callable[[object], bool]) -> int:
    """Length of the run of ``predicate``-true items at the *end* of ``values``.

    ``values`` must already be in chronological order. The count stops at
    the first item (scanning backwards) that fails ``predicate``.
    """
    seq = list(values)
    n = 0
    for v in reversed(seq):
        if predicate(v):
            n += 1
        else:
            break
    return n


def consecutive_red_streak(history_df: pd.DataFrame) -> int:
    """Trailing run of Red-recovery days, ending at the most recent row.

    Red is the framework.md §8 band (recovery ``<= 33``); the band test is
    delegated to :func:`src.rules.gates.classify_recovery` so the cutoff is
    not re-typed here. Returns ``0`` for an empty frame or when the latest
    day is not Red.
    """
    if history_df.empty or "recovery_score" not in history_df.columns:
        return 0
    df = _sorted_by_date(history_df)
    return trailing_streak(
        df["recovery_score"].tolist(),
        lambda r: classify_recovery(int(r)) == "red",
    )


def _high_intensity_minutes(row: "pd.Series") -> float:
    """Z3+Z4+Z5 minutes for a row; tolerant of dict / JSON string / missing.

    ``zone_minutes`` is a dict after ``WhoopDaily.model_dump()`` but a JSON
    string when the frame came from ``data/fake_whoop.csv``; either parses,
    anything else (or absent) contributes 0.0 so the strain branch still
    decides.
    """
    zm = row.get("zone_minutes") if hasattr(row, "get") else None
    if isinstance(zm, str):
        try:
            zm = json.loads(zm)
        except (ValueError, TypeError):
            return 0.0
    if not isinstance(zm, dict):
        return 0.0
    return float(sum(float(zm.get(z, 0.0) or 0.0) for z in _HIGH_ZONES))


def _is_hard_day(row: "pd.Series") -> bool:
    """§7/§8 hard-session heuristic (see the module-level constants block)."""
    strain = row.get("day_strain")
    if strain is not None and not pd.isna(strain) and float(strain) >= HARD_DAY_STRAIN_MIN:
        return True
    return _high_intensity_minutes(row) >= HARD_INTENSITY_MIN_MINUTES


def days_since_last_hard(history_df: pd.DataFrame) -> int:
    """Calendar days from the last hard session to *today*.

    The frame is the completed history **up to and including the most
    recent day** (the day before the one being recommended), so the result
    is measured against ``today = last_date + 1 day``:

    * last completed day was hard            -> ``1``
    * hard session 3 calendar days back      -> ``3`` (matches the §12
      example "Last hard session 3 days ago")

    Calendar diff (not row count) makes it robust to missing days. Returns
    :data:`NO_RECENT_HARD` when the frame is empty or holds no hard day, so
    the §9 ``days_since_hard < 2`` spacing gate never spuriously fires.
    """
    if history_df.empty:
        return NO_RECENT_HARD
    df = _sorted_by_date(history_df).reset_index(drop=True)
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        dates = pd.to_datetime(pd.Series(df.index))
    ref = dates.iloc[-1]

    last_hard: Optional[pd.Timestamp] = None
    for i in range(len(df)):
        if _is_hard_day(df.iloc[i]):
            last_hard = dates.iloc[i]
    if last_hard is None:
        return NO_RECENT_HARD
    return int((ref - last_hard).days) + 1
