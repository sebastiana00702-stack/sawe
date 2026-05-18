"""Phase 6 FastAPI service-layer tests.

Drives the thin HTTP wrapper with FastAPI's :class:`TestClient` and
asserts it returns the *unchanged* Phase 1 domain models (framework.md
§12 shapes) under a meta block:

* ``GET  /healthz``               → 200, exact liveness shape.
* ``POST /daily_recommendation``  → 200 + a complete, schema-valid
  :class:`~src.models.Recommendation`; 422 when fields are missing.
* ``POST /weekly_plan``           → 200 + a 7-day
  :class:`~src.models.WeeklyPlan`.
* Both endpoints over the synthetic ``data/fake_whoop.csv`` scenario.

The API must not reshape the recommender/planner output, so each body is
re-validated against the original domain schema.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import AGENT_VERSION
from src.models import Recommendation, WeeklyPlan, WhoopDaily

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CSV = REPO_ROOT / "data" / "fake_whoop.csv"

TODAY = date(2026, 5, 18)


# --------------------------------------------------------------------------
# JSON body builders
# --------------------------------------------------------------------------

def whoop_json(d: date = TODAY, **over) -> dict:
    """A clean all-green WhoopDaily, JSON-encoded for the request body."""
    base = dict(
        date=d,
        recovery_score=80,
        hrv_rmssd=65.0,
        rhr=48,
        sleep_performance=0.95,
        sleep_hours=8.0,
        sleep_need_hours=8.0,
        rem_min=100,
        sws_min=90,
        light_min=260,
        day_strain=9.0,
        respiratory_rate=14.0,
        zone_minutes={"Z1": 40.0},
        journal={},
    )
    base.update(over)
    return WhoopDaily(**base).model_dump(mode="json")


_STRAIN_CYCLE = [9.0, 10.0, 11.0, 10.0, 9.0, 12.0, 8.0]


def history_json(n: int = 35, end: date = TODAY - timedelta(days=1)) -> list[dict]:
    """``n`` calm WhoopDaily rows ending the day before today."""
    rows = []
    for i in range(n):
        d = end - timedelta(days=n - 1 - i)
        rows.append(
            whoop_json(
                d,
                day_strain=_STRAIN_CYCLE[i % 7],
                hrv_rmssd=65.0 + (i % 3) - 1,
                rhr=48 + (i % 2),
                respiratory_rate=14.0 + 0.1 * (i % 2),
            )
        )
    return rows


PROFILE = {
    "tier": "intermediate",
    "goal": "10K",
    "age": 30,
    "sex": "M",
    "current_mpw": 32.0,
    "longest_recent_run_mi": 14.0,
}


# ==========================================================================
# /healthz
# ==========================================================================

def test_healthz_returns_ok_shape():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "0.5.0"}
    assert AGENT_VERSION == "0.5.0"


# ==========================================================================
# /daily_recommendation
# ==========================================================================

def test_daily_recommendation_valid_returns_complete_recommendation():
    body = {
        "today": whoop_json(),
        "history": history_json(),
        "profile": PROFILE,
    }
    resp = client.post("/daily_recommendation", json=body)
    assert resp.status_code == 200, resp.text

    payload = resp.json()
    assert set(payload) == {"meta", "recommendation"}
    assert payload["meta"]["agent_version"] == "0.5.0"
    assert payload["meta"]["timestamp"]

    # API must not reshape: the body still validates against the
    # *unchanged* Phase 1 schema.
    rec = Recommendation.model_validate(payload["recommendation"])
    assert rec.date == TODAY
    assert rec.training_state in {
        "overreached", "strained", "detraining", "functional", "borderline",
    }
    assert rec.readiness in {"high", "moderate", "low", "very_low"}
    assert rec.recommendation.type
    assert rec.recommendation.disclaimer  # §11 always present
    assert isinstance(rec.requires_medical_review, bool)


def test_daily_recommendation_missing_fields_returns_422():
    # No `profile` → request-body validation fails.
    resp = client.post(
        "/daily_recommendation",
        json={"today": whoop_json(), "history": history_json()},
    )
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_daily_recommendation_honors_optional_week_number():
    body = {
        "today": whoop_json(),
        "history": history_json(),
        "profile": PROFILE,
        "week_number": 3,
    }
    resp = client.post("/daily_recommendation", json=body)
    assert resp.status_code == 200, resp.text
    Recommendation.model_validate(resp.json()["recommendation"])


# ==========================================================================
# /weekly_plan
# ==========================================================================

def test_weekly_plan_valid_returns_seven_days():
    body = {
        "profile": PROFILE,
        "week_number": 1,
        "start_date": TODAY.isoformat(),
    }
    resp = client.post("/weekly_plan", json=body)
    assert resp.status_code == 200, resp.text

    payload = resp.json()
    assert set(payload) == {"meta", "weekly_plan"}
    plan = WeeklyPlan.model_validate(payload["weekly_plan"])
    assert len(plan.days) == 7
    assert plan.days[0].date == TODAY
    assert plan.days[-1].date == TODAY + timedelta(days=6)
    assert plan.tier == "intermediate"
    assert plan.goal == "10K"


def test_weekly_plan_missing_fields_returns_422():
    resp = client.post("/weekly_plan", json={"profile": PROFILE})
    assert resp.status_code == 422


# ==========================================================================
# Synthetic data scenario (data/fake_whoop.csv)
# ==========================================================================

@pytest.mark.skipif(
    not FAKE_CSV.exists(), reason="run scripts/generate_fake_whoop.py"
)
def test_endpoints_handle_fake_whoop_scenario():
    df = pd.read_csv(FAKE_CSV)
    df["zone_minutes"] = df["zone_minutes"].map(json.loads)
    df["journal"] = df["journal"].map(
        lambda s: json.loads(s) if isinstance(s, str) else {}
    )

    def _row_json(row: pd.Series) -> dict:
        rec = row.to_dict()
        for opt in (
            "workout_strain", "workout_hr_mean", "workout_hr_max",
            "skin_temp_dev_c",
        ):
            v = rec.get(opt)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                rec[opt] = None
        return WhoopDaily.model_validate(rec).model_dump(mode="json")

    today_row = _row_json(df.iloc[-1])
    history = [_row_json(df.iloc[i]) for i in range(len(df) - 1)]

    resp = client.post(
        "/daily_recommendation",
        json={"today": today_row, "history": history, "profile": PROFILE},
    )
    assert resp.status_code == 200, resp.text
    rec = Recommendation.model_validate(resp.json()["recommendation"])
    assert rec.date == date.fromisoformat(today_row["date"])

    # And the plan endpoint on the same scenario's start date.
    resp = client.post(
        "/weekly_plan",
        json={
            "profile": PROFILE,
            "week_number": 1,
            "start_date": today_row["date"],
        },
    )
    assert resp.status_code == 200, resp.text
    plan = WeeklyPlan.model_validate(resp.json()["weekly_plan"])
    assert len(plan.days) == 7
