"""Phase 7 WHOOP ingestion tests.

All HTTP is mocked with :class:`httpx.MockTransport` against realistic
WHOOP JSON (shapes per https://developer.whoop.com/api) — **no real API
calls**. Covers:

* normalizer: full mapping, no-workout day, unscored → skip, tz-offset
  date, ``merge_history`` linking/sorting;
* client: OAuth refresh on 401, 401-after-refresh → auth error, 429
  backoff + exhaustion, ``next_token`` pagination, per-endpoint paths,
  missing-credential failure, network error;
* loader: 15-minute disk cache (hit / stale / bypass / force-refresh);
* API: ``GET /me/today`` happy path, ``?refresh`` propagation, the
  ``meta.data_freshness`` staleness indicator, and WHOOP-error mapping.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ingest import loaders
from src.ingest.loaders import load_whoop_history
from src.ingest.normalizer import (
    NormalizationError,
    merge_history,
    normalize_whoop_day,
)
from src.ingest.whoop_client import (
    WHOOP_TOKEN_URL,
    WhoopAuthError,
    WhoopClient,
    WhoopError,
    WhoopNetworkError,
    WhoopRateLimitError,
)
from src.models import WhoopDaily

CREDS = dict(
    client_id="cid", client_secret="csecret", refresh_token="rtoken"
)


def uid(seed: int) -> str:
    """A deterministic v2-style UUID string from a small int seed.

    WHOOP v2 resource IDs are UUID strings, not v1 integers. The tests
    still pass terse int seeds for readability; equal seeds must map to
    equal UUIDs so the recovery→cycle / recovery→sleep equality joins in
    ``merge_history`` keep lining up exactly as they did with int IDs.
    """
    return str(uuid.UUID(int=seed))


# ==========================================================================
# WHOOP JSON fixture builders
# ==========================================================================

def recovery_rec(
    cycle_id: str,
    sleep_id: str,
    *,
    recovery: int = 70,
    hrv: float = 65.0,
    rhr: int = 48,
    state: str = "SCORED",
) -> dict:
    return {
        "cycle_id": cycle_id,
        "sleep_id": sleep_id,
        "user_id": 1,
        "created_at": "2026-05-18T11:00:00.000Z",
        "updated_at": "2026-05-18T11:05:00.000Z",
        "score_state": state,
        "score": (
            None
            if state != "SCORED"
            else {
                "user_calibrating": False,
                "recovery_score": recovery,
                "resting_heart_rate": rhr,
                "hrv_rmssd_milli": hrv,
                "spo2_percentage": 96.0,
                "skin_temp_celsius": 33.5,
            }
        ),
    }


def cycle_rec(
    cycle_id: str,
    day: date,
    *,
    strain: float = 10.0,
    tz: str = "+00:00",
    start_hour: int = 6,
) -> dict:
    return {
        "id": cycle_id,
        "user_id": 1,
        "start": f"{day.isoformat()}T{start_hour:02d}:00:00.000Z",
        "end": f"{day.isoformat()}T22:00:00.000Z",
        "timezone_offset": tz,
        "score_state": "SCORED",
        "score": {
            "strain": strain,
            "kilojoule": 8000.0,
            "average_heart_rate": 60,
            "max_heart_rate": 150,
        },
    }


def sleep_rec(
    sleep_id: str,
    *,
    perf: float = 90.0,
    rr: float = 14.5,
    rem: int = 90,
    sws: int = 80,
    light: int = 300,
    need_h: float = 8.0,
) -> dict:
    m = 60_000
    return {
        "id": sleep_id,
        "user_id": 1,
        "start": "2026-05-17T23:00:00.000Z",
        "end": "2026-05-18T07:00:00.000Z",
        "timezone_offset": "+00:00",
        "nap": False,
        "score_state": "SCORED",
        "score": {
            "stage_summary": {
                "total_in_bed_time_milli": (rem + sws + light + 20) * m,
                "total_awake_time_milli": 20 * m,
                "total_no_data_time_milli": 0,
                "total_light_sleep_time_milli": light * m,
                "total_slow_wave_sleep_time_milli": sws * m,
                "total_rem_sleep_time_milli": rem * m,
                "sleep_cycle_count": 4,
                "disturbance_count": 8,
            },
            "sleep_needed": {
                "baseline_milli": int(need_h * 3_600_000),
                "need_from_sleep_debt_milli": 0,
                "need_from_recent_strain_milli": 0,
                "need_from_recent_nap_milli": 0,
            },
            "respiratory_rate": rr,
            "sleep_performance_percentage": perf,
            "sleep_consistency_percentage": 85.0,
            "sleep_efficiency_percentage": 92.0,
        },
    }


def workout_rec(
    day: date,
    *,
    strain: float = 8.0,
    hr_mean: int = 150,
    hr_max: int = 175,
    z1: int = 20,
    z3: int = 10,
) -> dict:
    m = 60_000
    return {
        "id": uid(day.toordinal()),
        "user_id": 1,
        "start": f"{day.isoformat()}T17:00:00.000Z",
        "end": f"{day.isoformat()}T18:00:00.000Z",
        "timezone_offset": "+00:00",
        "sport_id": 0,
        "score_state": "SCORED",
        "score": {
            "strain": strain,
            "average_heart_rate": hr_mean,
            "max_heart_rate": hr_max,
            "kilojoule": 1500.0,
            "percent_recorded": 100.0,
            "distance_meter": 8000.0,
            "altitude_gain_meter": 20.0,
            "altitude_change_meter": 0.0,
            "zone_duration": {
                "zone_zero_milli": 5 * m,
                "zone_one_milli": z1 * m,
                "zone_two_milli": 0,
                "zone_three_milli": z3 * m,
                "zone_four_milli": 0,
                "zone_five_milli": 0,
            },
        },
    }


# ==========================================================================
# normalizer
# ==========================================================================

def test_normalize_full_day_maps_every_field():
    d = date(2026, 5, 18)
    wd = normalize_whoop_day(
        recovery_rec(uid(1), uid(1), recovery=72, hrv=66.5, rhr=47),
        cycle_rec(uid(1), d, strain=12.3),
        sleep_rec(uid(1), perf=88.0, rr=15.0, rem=95, sws=85, light=310, need_h=8.2),
        [workout_rec(d, strain=9.4, hr_mean=152, hr_max=178, z1=25, z3=12)],
    )
    assert isinstance(wd, WhoopDaily)
    assert wd.date == d
    assert wd.recovery_score == 72
    assert wd.hrv_rmssd == 66.5
    assert wd.rhr == 47
    assert wd.sleep_performance == pytest.approx(0.88)
    assert wd.sleep_hours == pytest.approx((95 + 85 + 310) / 60.0, abs=1e-6)
    assert wd.sleep_need_hours == pytest.approx(8.2, abs=1e-6)
    assert (wd.rem_min, wd.sws_min, wd.light_min) == (95, 85, 310)
    assert wd.day_strain == pytest.approx(12.3)
    assert wd.workout_strain == pytest.approx(9.4)
    assert wd.workout_hr_mean == 152
    assert wd.workout_hr_max == 178
    assert wd.zone_minutes == {"Z1": 25.0, "Z3": 12.0}
    assert wd.respiratory_rate == pytest.approx(15.0)
    # WHOOP exposes absolute skin temp, not the framework's deviation.
    assert wd.skin_temp_dev_c is None
    assert wd.journal == {}


def test_normalize_no_workout_day_defaults_workout_fields():
    d = date(2026, 5, 18)
    wd = normalize_whoop_day(
        recovery_rec(uid(2), uid(2)), cycle_rec(uid(2), d), sleep_rec(uid(2)), []
    )
    assert wd.workout_strain is None
    assert wd.workout_hr_mean is None
    assert wd.workout_hr_max is None
    assert wd.zone_minutes == {}
    # Still a fully valid WhoopDaily.
    WhoopDaily.model_validate(wd.model_dump())


def test_normalize_summarizes_hardest_workout_and_sums_zones():
    d = date(2026, 5, 18)
    wd = normalize_whoop_day(
        recovery_rec(uid(3), uid(3)),
        cycle_rec(uid(3), d),
        sleep_rec(uid(3)),
        [
            workout_rec(d, strain=5.0, hr_mean=130, hr_max=150, z1=10, z3=0),
            workout_rec(d, strain=11.0, hr_mean=165, hr_max=185, z1=5, z3=8),
        ],
    )
    # HR + strain come from the hardest (strain=11) workout...
    assert wd.workout_strain == pytest.approx(11.0)
    assert wd.workout_hr_mean == 165
    assert wd.workout_hr_max == 185
    # ...zone minutes are summed across both.
    assert wd.zone_minutes == {"Z1": 15.0, "Z3": 8.0}


def test_normalize_unscored_recovery_raises():
    d = date(2026, 5, 18)
    with pytest.raises(NormalizationError):
        normalize_whoop_day(
            recovery_rec(uid(4), uid(4), state="PENDING_SCORE"),
            cycle_rec(uid(4), d),
            sleep_rec(uid(4)),
            [],
        )


def test_normalize_buckets_cycle_by_nearest_local_day_of_wake():
    # A cycle ``start`` is the *wake* instant. 02:00 UTC at -05:00 is
    # 21:00 the previous local evening — a late-evening wake belongs to
    # the day the athlete wakes *into* (the nearest local day, here the
    # 18th), matching WHOOP's own per-day labels. The naive local .date()
    # (the 17th) is exactly the collision bug _whoop_day fixes.
    rec = recovery_rec(uid(5), uid(5))
    cyc = cycle_rec(uid(5), date(2026, 5, 18), tz="-05:00", start_hour=2)
    wd = normalize_whoop_day(rec, cyc, sleep_rec(uid(5)), [])
    assert wd.date == date(2026, 5, 18)


def test_normalize_morning_wake_stays_on_that_local_day():
    # Counterpart: a normal early-morning wake (06:00 local) stays put —
    # the noon pivot only rolls the late-evening straddle cases forward.
    rec = recovery_rec(uid(15), uid(15))
    cyc = cycle_rec(uid(15), date(2026, 5, 18), tz="-05:00", start_hour=11)
    wd = normalize_whoop_day(rec, cyc, sleep_rec(uid(15)), [])
    # 11:00 UTC - 05:00 = 06:00 local on the 18th → the 18th.
    assert wd.date == date(2026, 5, 18)


def test_normalize_nonpositive_hrv_raises():
    d = date(2026, 5, 18)
    with pytest.raises(NormalizationError):
        normalize_whoop_day(
            recovery_rec(uid(6), uid(6), hrv=0.0), cycle_rec(uid(6), d), sleep_rec(uid(6)), []
        )


# ==========================================================================
# merge_history (uses a fake client; client owns I/O, merge is pure logic)
# ==========================================================================

class FakeClient:
    """Stand-in WhoopClient returning canned record lists."""

    def __init__(self, recoveries, cycles, sleeps, workouts):
        self._r, self._c, self._s, self._w = (
            recoveries,
            cycles,
            sleeps,
            workouts,
        )
        self.closed = False

    def get_recovery(self, s, e):
        return self._r

    def get_cycles(self, s, e):
        return self._c

    def get_sleep(self, s, e):
        return self._s

    def get_workouts(self, s, e):
        return self._w

    def close(self):
        self.closed = True


def test_merge_history_links_sorts_and_skips_incomplete():
    d1, d2, d3 = date(2026, 5, 16), date(2026, 5, 17), date(2026, 5, 18)
    client = FakeClient(
        recoveries=[
            recovery_rec(uid(30), uid(30), recovery=60),  # d3 (out of order first)
            recovery_rec(uid(10), uid(10), recovery=80),  # d1
            recovery_rec(uid(20), uid(20), state="UNSCORABLE"),  # d2 → skipped
            recovery_rec(uid(40), uid(40)),  # cycle 40 missing → skipped
        ],
        cycles=[
            cycle_rec(uid(10), d1),
            cycle_rec(uid(20), d2),
            cycle_rec(uid(30), d3),
        ],
        sleeps=[sleep_rec(uid(10)), sleep_rec(uid(20)), sleep_rec(uid(30))],
        workouts=[workout_rec(d1, strain=7.0)],
    )
    series = merge_history(d1, d3, client)

    assert [w.date for w in series] == [d1, d3]  # sorted, d2/40 skipped
    assert series[0].recovery_score == 80
    assert series[0].workout_strain == pytest.approx(7.0)
    assert series[1].recovery_score == 60
    assert series[1].workout_strain is None


# --------------------------------------------------------------------------
# Regression: real WHOOP v2 shapes, wake times straddling local midnight.
#
# These builders deliberately do NOT use the tidy uid()/cycle_rec() helpers
# — those mask the production bug. Real WHOOP v2: cycle ``id`` /
# recovery ``cycle_id`` are **integers**; sleep ``id`` / recovery
# ``sleep_id`` are **UUID strings**; ``timezone_offset`` is a real offset
# ("-04:00"); a cycle ``start`` is the *wake* instant, which drifts around
# local midnight; today's still-running cycle has ``end: null`` but is
# already ``SCORED`` with a partial strain. Verbatim from the production
# /v2 dump that reproduced the 05-15 + 05-18 drop.
# --------------------------------------------------------------------------

_TZ = "-04:00"


def _pcycle(cid: int, start: str, end, strain: float) -> dict:
    """Production-shaped cycle: int id, real tz, ``end=None`` when open."""
    return {
        "id": cid,
        "user_id": 1,
        "start": start,
        "end": end,
        "timezone_offset": _TZ,
        "score_state": "SCORED",
        "score": {
            "strain": strain,
            "kilojoule": 8000.0,
            "average_heart_rate": 60,
            "max_heart_rate": 150,
        },
    }


def _precovery(cid: int, sid: str, created: str, recovery: int) -> dict:
    """Production-shaped recovery: int cycle_id, UUID sleep_id."""
    return {
        "cycle_id": cid,
        "sleep_id": sid,
        "user_id": 1,
        "created_at": created,
        "updated_at": created,
        "score_state": "SCORED",
        "score": {
            "user_calibrating": False,
            "recovery_score": recovery,
            "resting_heart_rate": 48,
            "hrv_rmssd_milli": 65.0,
            "spo2_percentage": 96.0,
            "skin_temp_celsius": 33.5,
        },
    }


def _psleep(sid: str, start: str, *, nap: bool = False) -> dict:
    s = sleep_rec(sid)
    s["start"] = start
    s["timezone_offset"] = _TZ
    s["nap"] = nap
    return s


# UUIDs/ids/timestamps verbatim from the production diagnostic dump.
_S14 = "5d265630-d328-4ae6-a862-cf6c1ae0b1c7"
_S15 = "c3712649-e9dd-4694-ac51-8116d50fc9e8"
_S16 = "867509f4-b9de-468c-840f-305d96d9319f"
_S17 = "69f5de0d-3552-46f7-bcc2-64517549daf5"
_SNAP = "c4e6f6ce-2485-4931-bdbf-e01e51d338ca"
_S18 = "2002e494-c2f3-4d30-869e-9bf29227ef1d"


def test_merge_history_keeps_every_cycle_when_wake_straddles_midnight():
    # The exact production failure. Five SCORED recoveries (WHOOP returns
    # them newest-first). Wake instants for the 05-15 and 05-18 cycles fall
    # at 23:53 and 22:57 *local* (UTC-4) — i.e. the previous calendar day —
    # so the old local-date bucketing collapsed 05-15→05-14 and 05-18→05-17
    # and silently overwrote, dropping today's recovery (89) entirely.
    client = FakeClient(
        recoveries=[  # newest-first, as the WHOOP collection returns them
            _precovery(1506892357, _S18, "2026-05-18T12:41:27.547Z", 89),
            _precovery(1505059111, _S17, "2026-05-17T20:56:00.882Z", 6),
            _precovery(1502437776, _S16, "2026-05-16T13:38:19.461Z", 62),
            _precovery(1500170316, _S15, "2026-05-15T12:45:34.761Z", 55),
            _precovery(1498069420, _S14, "2026-05-14T13:53:38.846Z", 48),
        ],
        cycles=[
            _pcycle(1498069420, "2026-05-14T04:42:46.090Z",
                    "2026-05-15T03:53:46.180Z", 13.1),
            _pcycle(1500170316, "2026-05-15T03:53:46.180Z",
                    "2026-05-16T05:27:50.100Z", 9.2),
            _pcycle(1502437776, "2026-05-16T05:27:50.100Z",
                    "2026-05-17T08:08:56.900Z", 16.55),
            _pcycle(1505059111, "2026-05-17T08:08:56.900Z",
                    "2026-05-18T02:57:32.380Z", 10.38),
            # Today: still running — no end, but already SCORED.
            _pcycle(1506892357, "2026-05-18T02:57:32.380Z", None, 4.5),
        ],
        sleeps=[
            _psleep(_S14, "2026-05-14T04:42:46.090Z"),
            _psleep(_S15, "2026-05-15T03:53:46.180Z"),
            _psleep(_S16, "2026-05-16T05:27:50.100Z"),
            _psleep(_S17, "2026-05-17T08:08:56.900Z"),
            _psleep(_SNAP, "2026-05-17T16:08:26.730Z", nap=True),  # unlinked
            _psleep(_S18, "2026-05-18T02:57:32.380Z"),
        ],
        workouts=[],
    )
    series = merge_history(date(2026, 5, 12), date(2026, 5, 18), client)

    # Every distinct WHOOP cycle survives as its own day — nothing dropped.
    assert [w.date for w in series] == [
        date(2026, 5, 14),
        date(2026, 5, 15),
        date(2026, 5, 16),
        date(2026, 5, 17),
        date(2026, 5, 18),
    ]
    assert [w.recovery_score for w in series] == [48, 55, 62, 6, 89]
    # Today (05-18): still-running cycle, end=None, but SCORED with a
    # partial strain — it must be present and carry that strain.
    assert series[-1].date == date(2026, 5, 18)
    assert series[-1].day_strain == pytest.approx(4.5)
    WhoopDaily.model_validate(series[-1].model_dump())


# ==========================================================================
# WhoopClient — transport-mocked OAuth / rate-limit / pagination
# ==========================================================================

def _ok_token(rt: str = "rt2") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "access_token": "AT",
            "refresh_token": rt,
            "expires_in": 3600,
            "token_type": "bearer",
        },
    )


def _is_token(req: httpx.Request) -> bool:
    return req.url.path.endswith("oauth2/token")


def make_client(handler, **kw) -> WhoopClient:
    return WhoopClient(
        **CREDS,
        transport=httpx.MockTransport(handler),
        sleep=kw.pop("sleep", lambda _s: None),
        load_env=False,
        **kw,
    )


def test_client_refreshes_token_on_401_then_retries():
    calls = {"token": 0, "recovery": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            calls["token"] += 1
            return _ok_token()
        calls["recovery"] += 1
        if calls["recovery"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(
            200, json={"records": [recovery_rec(uid(1), uid(1))], "next_token": None}
        )

    client = make_client(handler)
    out = client.get_recovery(date(2026, 5, 1), date(2026, 5, 18))

    assert len(out) == 1
    assert calls["recovery"] == 2  # 401 then a successful retry
    assert calls["token"] == 2  # initial mint + post-401 refresh


def test_client_401_after_refresh_raises_auth_error():
    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        return httpx.Response(401, json={"error": "bad"})

    with pytest.raises(WhoopAuthError, match="re-run"):
        make_client(handler).get_recovery(
            date(2026, 5, 1), date(2026, 5, 18)
        )


def test_client_refresh_rejected_raises_auth_error():
    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(200, json={"records": []})

    with pytest.raises(WhoopAuthError, match="expired or revoked"):
        make_client(handler).get_cycles(
            date(2026, 5, 1), date(2026, 5, 18)
        )


def test_client_persists_rotated_refresh_token_to_env(tmp_path):
    # auth_setup wrote this: comments + creds + unrelated profile keys.
    env = tmp_path / ".env"
    env.write_text(
        "# WHOOP API credentials.\n"
        "WHOOP_CLIENT_ID=cid\n"
        "WHOOP_REFRESH_TOKEN=rtoken\n"
        "# trailing comment\n"
        "SAWE_TIER=intermediate\n"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            # WHOOP rotates: the response carries a brand-new refresh token.
            return _ok_token("new_token")
        return httpx.Response(
            200, json={"records": [recovery_rec(uid(1), uid(1))]}
        )

    client = make_client(handler, env_path=env)
    client.get_recovery(date(2026, 5, 1), date(2026, 5, 18))

    # 1. In-memory state rotated, so this process keeps working.
    assert client.refresh_token == "new_token"

    # 2. .env rewritten so the *next* process does too — old token gone.
    contents = env.read_text()
    assert "WHOOP_REFRESH_TOKEN=new_token\n" in contents
    assert "rtoken" not in contents

    # 3. Every other line preserved byte-for-byte (it is an in-place edit,
    #    not a regenerated file).
    assert "# WHOOP API credentials.\n" in contents
    assert "WHOOP_CLIENT_ID=cid\n" in contents
    assert "# trailing comment\n" in contents
    assert "SAWE_TIER=intermediate\n" in contents

    # 4. Atomic swap left no partial temp file behind.
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


def test_client_does_not_rewrite_env_when_token_unchanged(tmp_path):
    # WHOOP may return the *same* refresh token (or none) — then there is
    # nothing to persist and the file must be left strictly untouched.
    env = tmp_path / ".env"
    env.write_text("WHOOP_REFRESH_TOKEN=rtoken\n")
    before = env.stat().st_mtime_ns

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token("rtoken")  # unchanged
        return httpx.Response(200, json={"records": []})

    client = make_client(handler, env_path=env)
    client.get_recovery(date(2026, 5, 1), date(2026, 5, 18))

    assert client.refresh_token == "rtoken"
    assert env.read_text() == "WHOOP_REFRESH_TOKEN=rtoken\n"
    assert env.stat().st_mtime_ns == before  # not rewritten at all


def test_rotated_token_visible_to_a_second_client_in_same_process(
    monkeypatch,
):
    # Phase 7.5 footgun: WHOOP single-uses the old refresh token on
    # rotation. If only self.refresh_token / .env are updated, a SECOND
    # WhoopClient built from the environment later in this same process
    # reads the now-dead token back (load_dotenv won't override an
    # already-set var) and hard-fails its first refresh. Rotation must
    # also publish the rotated token to os.environ. env_path is None here
    # (load_env=False), so this also covers the env-var-only deployment
    # path where _persist_refresh_token has no file to write.
    monkeypatch.setenv("WHOOP_CLIENT_ID", "cid")
    monkeypatch.setenv("WHOOP_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("WHOOP_REFRESH_TOKEN", "rtoken")

    def rotating(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token("rotated_token")  # WHOOP rotates on every use
        return httpx.Response(200, json={"records": []})

    c1 = WhoopClient(
        transport=httpx.MockTransport(rotating),
        sleep=lambda _s: None,
        load_env=False,
    )
    c1.get_recovery(date(2026, 5, 1), date(2026, 5, 18))
    assert c1.refresh_token == "rotated_token"

    # The single-used token must be gone from the process environment...
    assert os.environ["WHOOP_REFRESH_TOKEN"] == "rotated_token"

    # ...so a sibling client built from the env starts on the live token,
    # not WHOOP's already-invalidated one (pre-fix: this was "rtoken").
    c2 = WhoopClient(
        transport=httpx.MockTransport(rotating),
        sleep=lambda _s: None,
        load_env=False,
    )
    assert c2.refresh_token == "rotated_token"


def test_client_429_backs_off_then_succeeds():
    slept: list[float] = []
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(
                429, headers={"Retry-After": "1"}, json={"error": "slow"}
            )
        return httpx.Response(200, json={"records": [recovery_rec(uid(1), uid(1))]})

    client = make_client(handler, sleep=slept.append)
    out = client.get_recovery(date(2026, 5, 1), date(2026, 5, 18))

    assert len(out) == 1
    assert slept == [1.0]  # honoured Retry-After before the retry


def test_client_429_exhaustion_raises_rate_limit_error():
    slept: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        return httpx.Response(429, json={"error": "slow"})

    client = make_client(
        handler, sleep=slept.append, max_rate_limit_retries=3
    )
    with pytest.raises(WhoopRateLimitError, match="3 retries"):
        client.get_sleep(date(2026, 5, 1), date(2026, 5, 18))
    assert len(slept) == 3  # backed off exactly the retry budget


def test_client_paginates_next_token():
    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        if "nextToken" not in req.url.params:
            return httpx.Response(
                200,
                json={"records": [recovery_rec(uid(1), uid(1))], "next_token": "pg2"},
            )
        return httpx.Response(
            200, json={"records": [recovery_rec(uid(2), uid(2))], "next_token": None}
        )

    out = make_client(handler).get_recovery(
        date(2026, 5, 1), date(2026, 5, 18)
    )
    assert [r["cycle_id"] for r in out] == [uid(1), uid(2)]


def test_client_methods_hit_expected_paths():
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        seen.append(req.url.path)
        return httpx.Response(200, json={"records": []})

    c = make_client(handler)
    s, e = date(2026, 5, 1), date(2026, 5, 18)
    c.get_recovery(s, e)
    c.get_cycles(s, e)
    c.get_sleep(s, e)
    c.get_workouts(s, e)

    assert seen == [
        "/developer/v2/recovery",
        "/developer/v2/cycle",
        "/developer/v2/activity/sleep",
        "/developer/v2/activity/workout",
    ]


def test_client_missing_credentials_raises_auth_error(monkeypatch):
    for var in (
        "WHOOP_CLIENT_ID",
        "WHOOP_CLIENT_SECRET",
        "WHOOP_REFRESH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(WhoopAuthError, match="Missing WHOOP credentials"):
        WhoopClient(load_env=False)


def test_client_network_error_is_typed():
    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        raise httpx.ConnectError("dns boom")

    with pytest.raises(WhoopNetworkError, match="failed"):
        make_client(handler).get_workouts(
            date(2026, 5, 1), date(2026, 5, 18)
        )


def test_client_passes_window_params():
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if _is_token(req):
            return _ok_token()
        captured.update(dict(req.url.params))
        return httpx.Response(200, json={"records": []})

    make_client(handler).get_recovery(date(2026, 5, 17), date(2026, 5, 18))
    assert captured["start"] == "2026-05-17T00:00:00.000000Z"
    # Upper bound widened to the start of the day after `end`.
    assert captured["end"] == "2026-05-19T00:00:00.000000Z"
    assert captured["limit"] == "25"


# ==========================================================================
# loaders — disk cache
# ==========================================================================

def _client_for(days: list[date]) -> FakeClient:
    return FakeClient(
        recoveries=[recovery_rec(uid(i), uid(i)) for i, _ in enumerate(days, 1)],
        cycles=[cycle_rec(uid(i), d) for i, d in enumerate(days, 1)],
        sleeps=[sleep_rec(uid(i)) for i, _ in enumerate(days, 1)],
        workouts=[],
    )


def test_loader_fetches_then_serves_fresh_cache(tmp_path):
    cache = tmp_path / "whoop_cache.json"
    days = [date(2026, 5, 17), date(2026, 5, 18)]
    now = datetime(2026, 5, 18, 12, tzinfo=timezone.utc)

    df1 = load_whoop_history(
        2, client=_client_for(days), cache_path=cache, now=now
    )
    assert list(df1["date"]) == days
    assert cache.exists()

    # Within TTL → cache hit, the client is never touched.
    class Boom:
        def __getattr__(self, _n):
            raise AssertionError("cache miss: client should not be called")

    df2 = load_whoop_history(
        2,
        client=Boom(),
        cache_path=cache,
        now=now + timedelta(minutes=10),  # still inside the 15-min TTL
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_loader_refetches_when_cache_stale(tmp_path):
    cache = tmp_path / "whoop_cache.json"
    days = [date(2026, 5, 17), date(2026, 5, 18)]
    now = datetime(2026, 5, 18, 12, tzinfo=timezone.utc)
    load_whoop_history(2, client=_client_for(days), cache_path=cache, now=now)

    fresh = _client_for(days)
    load_whoop_history(
        2,
        client=fresh,
        cache_path=cache,
        now=now + timedelta(minutes=20),  # past the 15-minute TTL
    )
    # Stale → the new client was actually consulted (and the loader
    # closed nothing it did not own).
    assert fresh.closed is False


def test_loader_force_refresh_bypasses_fresh_cache(tmp_path):
    cache = tmp_path / "whoop_cache.json"
    days = [date(2026, 5, 17), date(2026, 5, 18)]
    now = datetime(2026, 5, 18, 12, tzinfo=timezone.utc)

    load_whoop_history(2, client=_client_for(days), cache_path=cache, now=now)
    assert cache.exists()  # a still-fresh cache entry now exists

    # force_refresh must re-fetch even though the cache is fresh: a client
    # that explodes on any access proves the cache read was skipped.
    class Boom:
        def __getattr__(self, _n):
            raise AssertionError("force_refresh did not bypass the cache")

    with pytest.raises(AssertionError, match="did not bypass"):
        load_whoop_history(
            2,
            client=Boom(),
            cache_path=cache,
            force_refresh=True,
            now=now + timedelta(minutes=1),
        )

    # ...and the freshly fetched result is written back, so the next
    # normal call within TTL is a cache hit again (distinct from
    # use_cache=False, which would never repopulate the cache).
    fresh = _client_for(days)
    df = load_whoop_history(
        2,
        client=fresh,
        cache_path=cache,
        force_refresh=True,
        now=now + timedelta(minutes=2),
    )
    assert list(df["date"]) == days
    df_hit = load_whoop_history(
        2, client=Boom(), cache_path=cache, now=now + timedelta(minutes=5)
    )
    pd.testing.assert_frame_equal(df, df_hit)


def test_loader_use_cache_false_bypasses(tmp_path):
    cache = tmp_path / "whoop_cache.json"
    days = [date(2026, 5, 18)]
    now = datetime(2026, 5, 18, 12, tzinfo=timezone.utc)
    load_whoop_history(1, client=_client_for(days), cache_path=cache, now=now)

    used = _client_for(days)
    load_whoop_history(
        1, client=used, cache_path=cache, use_cache=False, now=now
    )
    # Bypassing the cache must not even read it (no exception) and must
    # produce the recommender-shaped frame.
    df = load_whoop_history(
        1, client=_client_for(days), cache_path=cache,
        use_cache=False, now=now,
    )
    assert list(df.columns) == list(WhoopDaily.model_fields.keys())


def test_loader_corrupt_cache_is_a_miss(tmp_path):
    cache = tmp_path / "whoop_cache.json"
    cache.write_text("{ not json")
    days = [date(2026, 5, 18)]
    now = datetime(2026, 5, 18, 12, tzinfo=timezone.utc)
    df = load_whoop_history(
        1, client=_client_for(days), cache_path=cache, now=now
    )
    assert len(df) == 1


# ==========================================================================
# API — GET /me/today
# ==========================================================================

def _history_frame(n: int = 35) -> pd.DataFrame:
    end = date(2026, 5, 18)
    rows = []
    for i in range(n):
        d = end - timedelta(days=n - 1 - i)
        rows.append(
            WhoopDaily(
                date=d,
                recovery_score=78,
                hrv_rmssd=64.0 + (i % 3),
                rhr=48 + (i % 2),
                sleep_performance=0.93,
                sleep_hours=7.8,
                sleep_need_hours=8.0,
                rem_min=95,
                sws_min=85,
                light_min=300,
                day_strain=[9.0, 11.0, 8.0, 12.0, 7.0, 13.0, 10.0][i % 7],
                respiratory_rate=14.2,
                zone_minutes={"Z1": 35.0},
                journal={},
            ).model_dump()
        )
    return pd.DataFrame(rows)


@pytest.fixture
def api_client():
    from src.api.main import app

    return TestClient(app)


def test_me_today_returns_recommendation(api_client, monkeypatch):
    monkeypatch.setattr(
        "src.api.main.load_whoop_history", lambda *_a, **_k: _history_frame()
    )
    resp = api_client.get("/me/today")
    assert resp.status_code == 200, resp.text

    from src.models import Recommendation

    payload = resp.json()
    assert set(payload) == {"meta", "recommendation"}
    rec = Recommendation.model_validate(payload["recommendation"])
    assert rec.date == date(2026, 5, 18)
    assert rec.recommendation.disclaimer  # §11 always present


def test_me_today_empty_history_returns_503(api_client, monkeypatch):
    monkeypatch.setattr(
        "src.api.main.load_whoop_history",
        lambda *_a, **_k: pd.DataFrame(
            columns=list(WhoopDaily.model_fields.keys())
        ),
    )
    resp = api_client.get("/me/today")
    assert resp.status_code == 503
    assert "auth_setup" in resp.json()["detail"]


@pytest.mark.parametrize(
    "exc, status",
    [
        (WhoopAuthError("creds expired"), 401),
        (WhoopRateLimitError("too many"), 429),
        (WhoopError("upstream 500"), 502),
    ],
)
def test_me_today_maps_whoop_errors(api_client, monkeypatch, exc, status):
    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr("src.api.main.load_whoop_history", boom)
    resp = api_client.get("/me/today")
    assert resp.status_code == status
    assert "detail" in resp.json()


def test_me_today_meta_marks_fresh_when_whoop_synced_today(
    api_client, monkeypatch
):
    # Freshest WHOOP row == wall-clock day → not stale.
    monkeypatch.setattr(
        "src.api.main.load_whoop_history", lambda *_a, **_k: _history_frame()
    )
    monkeypatch.setattr(
        "src.api.main._wall_clock_today", lambda: date(2026, 5, 18)
    )
    resp = api_client.get("/me/today")
    assert resp.status_code == 200, resp.text

    fresh = resp.json()["meta"]["data_freshness"]
    assert fresh["wall_clock_today"] == "2026-05-18"
    assert fresh["latest_whoop_date"] == "2026-05-18"
    assert fresh["days_behind"] == 0
    assert fresh["is_stale"] is False
    assert "current" in fresh["note"]


def test_me_today_meta_flags_stale_when_whoop_not_synced(
    api_client, monkeypatch
):
    # Freshest WHOOP row is 2026-05-17 but the wall clock is 2026-05-18:
    # WHOOP has not synced today's recovery yet → one day behind, stale.
    stale = _history_frame()
    stale["date"] = stale["date"].map(lambda d: d - timedelta(days=1))
    monkeypatch.setattr(
        "src.api.main.load_whoop_history", lambda *_a, **_k: stale
    )
    monkeypatch.setattr(
        "src.api.main._wall_clock_today", lambda: date(2026, 5, 18)
    )
    resp = api_client.get("/me/today")
    assert resp.status_code == 200, resp.text

    payload = resp.json()
    fresh = payload["meta"]["data_freshness"]
    assert fresh["wall_clock_today"] == "2026-05-18"
    assert fresh["latest_whoop_date"] == "2026-05-17"
    assert fresh["days_behind"] == 1
    assert fresh["is_stale"] is True
    assert "2026-05-17" in fresh["note"]
    # The recommendation itself is unchanged — still dated to the WHOOP
    # record, proving data_freshness is metadata only (CLAUDE.md).
    assert payload["recommendation"]["date"] == "2026-05-17"


def test_me_today_refresh_query_propagates_force_refresh(
    api_client, monkeypatch
):
    seen: dict = {}

    def fake_loader(days_back=90, **kwargs):
        seen["force_refresh"] = kwargs.get("force_refresh", False)
        return _history_frame()

    monkeypatch.setattr("src.api.main.load_whoop_history", fake_loader)
    monkeypatch.setattr(
        "src.api.main._wall_clock_today", lambda: date(2026, 5, 18)
    )

    api_client.get("/me/today")
    assert seen["force_refresh"] is False

    api_client.get("/me/today?refresh=true")
    assert seen["force_refresh"] is True
