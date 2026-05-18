"""Disk-cached WHOOP history loader.

:func:`load_whoop_history` is the drop-in replacement for the Phase 1-5
``pd.read_csv('data/fake_whoop.csv')`` call: it returns the *same* one-row-
per-day DataFrame the recommender already consumes (columns = the
:class:`~src.models.WhoopDaily` fields, ``date`` as a real ``date``,
``zone_minutes``/``journal`` as dicts), so no metric/rule/planner code
changes.

A small JSON cache (``data/whoop_cache.json``, 6-hour TTL) sits in front of
the WHOOP API so iterative dev work does not hammer (or rate-limit) the
account. The cache key includes ``days_back`` so a wider request is not
served a narrower cached window.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.ingest.normalizer import merge_history
from src.ingest.whoop_client import WhoopClient
from src.models import WhoopDaily

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "whoop_cache.json"
CACHE_TTL = timedelta(hours=6)
CACHE_SCHEMA = 1

#: Column order mirrors ``WhoopDaily`` so an empty result still has the
#: shape the recommender expects.
_COLUMNS = list(WhoopDaily.model_fields.keys())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Cache I/O
# --------------------------------------------------------------------------

def _read_cache(path: Path) -> Optional[dict[str, Any]]:
    """Load the cache file, treating any corruption as a cache miss."""
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(blob, dict) or blob.get("schema") != CACHE_SCHEMA:
        return None
    return blob


def _cache_fresh(blob: dict[str, Any], days_back: int, now: datetime) -> bool:
    """True if the cache is within TTL and covers at least ``days_back``."""
    try:
        fetched_at = datetime.fromisoformat(blob["fetched_at"])
        cached_days = int(blob["days_back"])
    except (KeyError, ValueError, TypeError):
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return cached_days >= days_back and (now - fetched_at) < CACHE_TTL


def _write_cache(
    path: Path, days: list[WhoopDaily], days_back: int, now: datetime
) -> None:
    """Atomically persist the fetched series (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "schema": CACHE_SCHEMA,
        "fetched_at": now.isoformat(),
        "days_back": days_back,
        "records": [d.model_dump(mode="json") for d in days],
    }
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with open(fd, "w") as fh:
            json.dump(blob, fh)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------

def _to_frame(days: list[WhoopDaily], days_back: int) -> pd.DataFrame:
    """WhoopDaily list → the recommender-shaped, date-sorted DataFrame.

    Trimmed to the most-recent ``days_back`` calendar days so a wider cache
    still honours a narrower request.
    """
    if not days:
        return pd.DataFrame(columns=_COLUMNS)

    ordered = sorted(days, key=lambda d: d.date)
    cutoff = ordered[-1].date - timedelta(days=days_back - 1)
    ordered = [d for d in ordered if d.date >= cutoff]

    frame = pd.DataFrame([d.model_dump() for d in ordered])
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def load_whoop_history(
    days_back: int = 90,
    *,
    client: Optional[Any] = None,
    cache_path: Path = CACHE_PATH,
    use_cache: bool = True,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    """Return the last ``days_back`` days of WHOOP data as a DataFrame.

    Serves a fresh on-disk cache when one exists (<6 h old and at least as
    wide as ``days_back``); otherwise fetches via WHOOP and refreshes the
    cache. ``client`` is injectable for tests; when omitted a
    :class:`WhoopClient` is constructed from the environment and closed
    after use. Drop-in for ``pd.read_csv('data/fake_whoop.csv')``.
    """
    now = now or _now()

    if use_cache:
        blob = _read_cache(cache_path)
        if blob is not None and _cache_fresh(blob, days_back, now):
            cached = [
                WhoopDaily.model_validate(r) for r in blob.get("records", [])
            ]
            return _to_frame(cached, days_back)

    owns_client = client is None
    if owns_client:
        client = WhoopClient()
    try:
        end = now.date()
        start = end - timedelta(days=days_back - 1)
        days = merge_history(start, end, client)
    finally:
        if owns_client and hasattr(client, "close"):
            client.close()

    if use_cache:
        _write_cache(cache_path, days, days_back, now)
    return _to_frame(days, days_back)
