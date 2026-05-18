"""Thin OAuth 2.0 + REST wrapper over the WHOOP developer API.

This is the **only** module that performs network I/O. It owns the access
token lifecycle (refresh-on-401, with WHOOP's rotating refresh tokens),
respects the documented rate limit (sleep + retry on 429), walks the
``next_token`` pagination, and turns every failure mode into a typed
:class:`WhoopError` subclass with an actionable message — so the pure
normalizer and the loader never have to reason about HTTP.

Credentials come exclusively from the environment (``WHOOP_CLIENT_ID``,
``WHOOP_CLIENT_SECRET``, ``WHOOP_REFRESH_TOKEN``), optionally hydrated from
a ``.env`` file via python-dotenv. Nothing is ever hard-coded; the
constructor only takes explicit values so tests can inject fakes.

WHOOP API reference: https://developer.whoop.com/api
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Endpoints / constants (single source of truth)
# --------------------------------------------------------------------------

#: Developer API root. All data collections hang off this base.
WHOOP_API_BASE = "https://api.prod.whoop.com/developer"
#: OAuth 2.0 token endpoint (lives outside the ``/developer`` namespace).
WHOOP_TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
#: OAuth 2.0 authorization endpoint (used only by :mod:`auth_setup`).
WHOOP_AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"

# v1 collection paths. The Phase 7 brief lists these as /v1/recovery,
# /v1/cycle, /v1/sleep, /v1/workout; the live API namespaces the two
# activity collections under /v1/activity/*. Named here so a path change
# is one edit and the test fixtures key off the same constants.
RECOVERY_PATH = "/v1/recovery"
CYCLE_PATH = "/v1/cycle"
SLEEP_PATH = "/v1/activity/sleep"
WORKOUT_PATH = "/v1/activity/workout"

#: Scopes requested at authorization time. ``offline`` is what makes WHOOP
#: issue a refresh token at all (without it there is nothing to persist).
WHOOP_SCOPES = (
    "offline read:recovery read:cycles read:sleep "
    "read:workout read:profile"
)

#: WHOOP caps collection pages at 25 records.
MAX_PAGE_LIMIT = 25
#: How many times a 429 is retried before giving up.
MAX_RATE_LIMIT_RETRIES = 4
#: Cap on a single backoff sleep (seconds) so a bad Retry-After can't hang.
MAX_BACKOFF_SECONDS = 60.0
DEFAULT_TIMEOUT = 30.0


# --------------------------------------------------------------------------
# Typed errors — every failure mode the loader/API layer must distinguish
# --------------------------------------------------------------------------

class WhoopError(RuntimeError):
    """Base for every WHOOP ingestion failure."""


class WhoopAuthError(WhoopError):
    """Missing credentials or an unrecoverable OAuth failure.

    Raised when env vars are absent, the refresh token is rejected, or a
    request still 401s *after* a forced token refresh (so the caller knows
    re-running ``auth_setup`` is required — not just a retry).
    """


class WhoopRateLimitError(WhoopError):
    """Rate limited (429) and the retry budget was exhausted."""


class WhoopNetworkError(WhoopError):
    """The HTTP request never produced a response (DNS, TLS, timeout)."""


class WhoopAPIError(WhoopError):
    """WHOOP returned an unexpected non-2xx status (4xx/5xx other than the
    ones handled specially)."""


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------

def _as_date(value: date) -> date:
    """Coerce a ``date``/``datetime`` to a plain ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"expected date/datetime, got {type(value)!r}")


def _iso_z(dt: datetime) -> str:
    """RFC 3339 UTC timestamp with a trailing ``Z`` (WHOOP's format)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _window(start: date, end: date) -> tuple[str, str]:
    """Inclusive ``[start 00:00, end+1d 00:00)`` UTC range as ISO strings.

    WHOOP filters on the record ``start`` instant; widening the upper bound
    to the start of the day *after* ``end`` guarantees the whole ``end`` day
    is captured regardless of the athlete's timezone offset.
    """
    s = _as_date(start)
    e = _as_date(end)
    lo = datetime.combine(s, dtime.min, tzinfo=timezone.utc)
    hi = datetime.combine(e, dtime.min, tzinfo=timezone.utc) + timedelta(days=1)
    return _iso_z(lo), _iso_z(hi)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class WhoopClient:
    """Authenticated WHOOP REST client.

    Parameters mirror the three required env vars; passing them explicitly
    (as the tests do) bypasses the environment entirely. ``transport`` lets
    a caller inject an :class:`httpx.MockTransport` so no real network is
    touched. ``sleep`` is injectable so 429 backoff is instant under test.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
        max_rate_limit_retries: int = MAX_RATE_LIMIT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        load_env: bool = True,
    ) -> None:
        if load_env and transport is None:
            # Hydrate a local .env when running for real; tests inject a
            # transport and explicit creds, so skip the side effect there.
            load_dotenv()

        self.client_id = client_id or os.environ.get("WHOOP_CLIENT_ID")
        self.client_secret = (
            client_secret or os.environ.get("WHOOP_CLIENT_SECRET")
        )
        self.refresh_token = (
            refresh_token or os.environ.get("WHOOP_REFRESH_TOKEN")
        )

        missing = [
            name
            for name, value in (
                ("WHOOP_CLIENT_ID", self.client_id),
                ("WHOOP_CLIENT_SECRET", self.client_secret),
                ("WHOOP_REFRESH_TOKEN", self.refresh_token),
            )
            if not value
        ]
        if missing:
            raise WhoopAuthError(
                "Missing WHOOP credentials: "
                + ", ".join(missing)
                + ". Run `python -m src.ingest.auth_setup` once to create "
                "them, or set them in the environment / .env."
            )

        self._access_token: Optional[str] = None
        self._sleep = sleep
        self._max_rate_limit_retries = max_rate_limit_retries
        self._http = httpx.Client(
            base_url=WHOOP_API_BASE, timeout=timeout, transport=transport
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "WhoopClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- OAuth -------------------------------------------------------------

    def _refresh_access_token(self) -> None:
        """Exchange the stored refresh token for a fresh access token.

        WHOOP rotates refresh tokens, so the new one (when returned) is kept
        for the next refresh within this process's lifetime. A 4xx here is
        unrecoverable (bad/expired credentials) → :class:`WhoopAuthError`.
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            # Re-asserting offline keeps a rotating refresh token flowing.
            "scope": "offline",
        }
        try:
            resp = self._http.post(WHOOP_TOKEN_URL, data=data)
        except httpx.HTTPError as exc:
            raise WhoopNetworkError(
                f"Could not reach WHOOP token endpoint: {exc}"
            ) from exc

        if resp.status_code in (400, 401, 403):
            raise WhoopAuthError(
                "WHOOP rejected the refresh token "
                f"(HTTP {resp.status_code}). It is likely expired or "
                "revoked — re-run `python -m src.ingest.auth_setup`."
            )
        if resp.status_code != 200:
            raise WhoopAPIError(
                f"Unexpected HTTP {resp.status_code} from the WHOOP token "
                f"endpoint: {resp.text[:300]}"
            )

        try:
            payload = resp.json()
            self._access_token = payload["access_token"]
        except (ValueError, KeyError) as exc:
            raise WhoopAPIError(
                f"Malformed WHOOP token response: {exc}"
            ) from exc

        rotated = payload.get("refresh_token")
        if rotated:
            self.refresh_token = rotated

    # -- request plumbing --------------------------------------------------

    @staticmethod
    def _backoff_seconds(resp: httpx.Response, attempt: int) -> float:
        """Seconds to wait before retrying a 429.

        Honour ``Retry-After`` (delta-seconds form) when present; otherwise
        exponential backoff (1, 2, 4, 8 ...), capped.
        """
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), MAX_BACKOFF_SECONDS)
            except ValueError:
                pass  # HTTP-date form — fall through to backoff
        return min(2.0 ** (attempt - 1), MAX_BACKOFF_SECONDS)

    def _authorized_get(
        self, path: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """GET ``path`` with the bearer token, refreshing once on 401 and
        backing off on 429. Returns the decoded JSON body."""
        if self._access_token is None:
            self._refresh_access_token()

        refreshed = False
        rl_attempts = 0
        while True:
            headers = {"Authorization": f"Bearer {self._access_token}"}
            try:
                resp = self._http.get(path, params=params, headers=headers)
            except httpx.HTTPError as exc:
                raise WhoopNetworkError(
                    f"WHOOP request to {path} failed: {exc}"
                ) from exc

            if resp.status_code == 401:
                if refreshed:
                    raise WhoopAuthError(
                        f"WHOOP still returned 401 for {path} after a token "
                        "refresh — credentials are invalid; re-run "
                        "`python -m src.ingest.auth_setup`."
                    )
                refreshed = True
                self._refresh_access_token()
                continue

            if resp.status_code == 429:
                rl_attempts += 1
                if rl_attempts > self._max_rate_limit_retries:
                    raise WhoopRateLimitError(
                        f"WHOOP rate limit not cleared for {path} after "
                        f"{self._max_rate_limit_retries} retries."
                    )
                self._sleep(self._backoff_seconds(resp, rl_attempts))
                continue

            if resp.status_code >= 400:
                raise WhoopAPIError(
                    f"WHOOP API error {resp.status_code} for {path}: "
                    f"{resp.text[:300]}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise WhoopAPIError(
                    f"WHOOP returned non-JSON for {path}: {exc}"
                ) from exc

    def _collect(
        self, path: str, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """Page through ``path`` over ``[start_date, end_date]`` and return
        the flattened ``records`` list."""
        start_iso, end_iso = _window(start_date, end_date)
        params: dict[str, Any] = {
            "start": start_iso,
            "end": end_iso,
            "limit": MAX_PAGE_LIMIT,
        }
        records: list[dict[str, Any]] = []
        while True:
            body = self._authorized_get(path, params)
            records.extend(body.get("records", []) or [])
            next_token = body.get("next_token")
            if not next_token:
                return records
            params = {**params, "nextToken": next_token}

    # -- public data methods ----------------------------------------------

    def get_recovery(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """All recovery records whose cycle starts in the window."""
        return self._collect(RECOVERY_PATH, start_date, end_date)

    def get_cycles(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """All physiological cycle records in the window."""
        return self._collect(CYCLE_PATH, start_date, end_date)

    def get_sleep(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """All sleep activity records in the window."""
        return self._collect(SLEEP_PATH, start_date, end_date)

    def get_workouts(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """All workout activity records in the window."""
        return self._collect(WORKOUT_PATH, start_date, end_date)
