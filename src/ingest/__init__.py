"""WHOOP ingestion layer (Phase 7).

Replaces the synthetic ``data/fake_whoop.csv`` source with the live WHOOP
developer API. The boundary is deliberately narrow so the Phase 1-5
recommender / rules / metrics / planner code is untouched:

* :mod:`src.ingest.whoop_client` — OAuth 2.0 + REST I/O (the only place
  network calls happen).
* :mod:`src.ingest.normalizer` — *pure* transforms from raw WHOOP JSON
  into validated :class:`~src.models.WhoopDaily` rows.
* :mod:`src.ingest.loaders` — :func:`load_whoop_history`, a disk-cached
  drop-in replacement for ``pd.read_csv('data/fake_whoop.csv')``.
* :mod:`src.ingest.auth_setup` — one-time interactive OAuth bootstrap
  (run manually; never imported by the service).
"""

from src.ingest.normalizer import (
    NormalizationError,
    merge_history,
    normalize_whoop_day,
)
from src.ingest.whoop_client import (
    WhoopAPIError,
    WhoopAuthError,
    WhoopClient,
    WhoopError,
    WhoopNetworkError,
    WhoopRateLimitError,
)

__all__ = [
    "WhoopClient",
    "WhoopError",
    "WhoopAuthError",
    "WhoopRateLimitError",
    "WhoopAPIError",
    "WhoopNetworkError",
    "NormalizationError",
    "normalize_whoop_day",
    "merge_history",
]
