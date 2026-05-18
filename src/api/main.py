"""Sawe FastAPI service layer (Phase 6).

A thin HTTP wrapper over the deterministic in-memory recommender. It
exposes the Phase 5 orchestrator (:func:`recommend_daily_workout`) and
the Phase 4 planner (:func:`generate_weekly_plan`) behind REST and
nothing more: **no training logic lives here** (``CLAUDE.md``) — every
numeric/safety decision stays in the rule / metric / planner layers and
the domain objects are returned verbatim. No auth, persistence,
scheduling, or WHOOP integration in this phase.

Run locally::

    uvicorn src.api.main:app --reload

Then the interactive OpenAPI docs are at
http://127.0.0.1:8000/docs, the raw schema at
http://127.0.0.1:8000/openapi.json, and the liveness probe at
http://127.0.0.1:8000/healthz.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.agent import recommend_daily_workout
from src.api.deps import get_agent_version, get_runner_profile
from src.api.schemas import (
    AGENT_VERSION,
    DailyRecommendationRequest,
    DailyRecommendationResponse,
    DataFreshness,
    HealthResponse,
    Meta,
    WeeklyPlanRequest,
    WeeklyPlanResponse,
)
from src.ingest.loaders import load_whoop_history
from src.ingest.whoop_client import (
    WhoopAuthError,
    WhoopError,
    WhoopRateLimitError,
)
from src.models import RunnerProfile, WhoopDaily
from src.planner.weekly_plan import generate_weekly_plan

# Optional WhoopDaily fields: a DataFrame round-trip turns an absent value
# into float NaN, which the schema rejects — coerce those back to None.
_OPTIONAL_WHOOP_FIELDS = (
    "workout_strain",
    "workout_hr_mean",
    "workout_hr_max",
    "skin_temp_dev_c",
)

app = FastAPI(
    title="Sawe — Running Coach Agent",
    version=AGENT_VERSION,
    summary="Deterministic WHOOP-driven running-coach agent.",
    description=(
        "Ingests WHOOP daily data and returns a daily workout "
        "recommendation or an expanded weekly plan, per "
        "`docs/framework.md` §9/§12. This is a thin HTTP layer: all "
        "training logic, safety overrides, and red-flag medical "
        "escalations live in the rule / metric / planner modules and are "
        "returned verbatim — never paraphrased by the API."
    ),
    openapi_tags=[
        {"name": "meta", "description": "Liveness / service metadata."},
        {
            "name": "recommendations",
            "description": (
                "Phase 5 deterministic daily orchestrator "
                "(framework.md §9 decision tree)."
            ),
        },
        {
            "name": "plans",
            "description": (
                "Phase 4 weekly templates expanded onto calendar dates "
                "(framework.md §10/§12)."
            ),
        },
    ],
)

# Permissive CORS for local development only (any localhost / loopback
# port). Production origins would be configured explicitly elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def _pydantic_validation_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Surface a domain-model ``ValidationError`` as a clean 422.

    FastAPI already returns 422 for request-body parsing failures; this
    catches a :class:`pydantic.ValidationError` raised *inside* an
    endpoint (e.g. a model rebuilt from the history DataFrame) so it does
    not leak as an opaque 500. The detail is the pydantic error list,
    flattened to JSON-safe primitives.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "type": e.get("type"),
                    "loc": list(e.get("loc", ())),
                    "msg": e.get("msg"),
                }
                for e in exc.errors()
            ]
        },
    )


@app.exception_handler(ValueError)
async def _value_error_handler(
    request: Request, exc: ValueError
) -> JSONResponse:
    """A bad client/plan combination is a 422, not a server fault.

    The recommender raises :class:`ValueError` when the resolved weekly
    plan has no entry for ``today`` (a mis-dated request) — that is a
    client-data problem, so it must not surface as a 500. Registered
    after the more specific :class:`ValidationError` handler, which wins
    for validation errors via the exception MRO.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(WhoopAuthError)
async def _whoop_auth_handler(
    request: Request, exc: WhoopAuthError
) -> JSONResponse:
    """WHOOP credentials are missing/expired → 401, not a 500.

    The athlete must re-run ``auth_setup``; surface that verbatim rather
    than leaking it as an opaque server fault.
    """
    return JSONResponse(status_code=401, content={"detail": str(exc)})


@app.exception_handler(WhoopRateLimitError)
async def _whoop_rate_limit_handler(
    request: Request, exc: WhoopRateLimitError
) -> JSONResponse:
    """WHOOP rate limit not cleared after retries → propagate 429."""
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(WhoopError)
async def _whoop_error_handler(
    request: Request, exc: WhoopError
) -> JSONResponse:
    """Any other upstream WHOOP failure (network / API) → 502.

    Registered after the specific subclasses; Starlette resolves handlers
    by the exception MRO so :class:`WhoopAuthError` /
    :class:`WhoopRateLimitError` still win for their own types.
    """
    return JSONResponse(
        status_code=502,
        content={"detail": f"Upstream WHOOP error: {exc}"},
    )


def _meta(data_freshness: Optional[DataFreshness] = None) -> Meta:
    """Fresh per-response metadata (new UTC timestamp each call).

    ``data_freshness`` is supplied only by ``GET /me/today`` (the
    live-WHOOP endpoint); it stays ``None`` — serialized as ``null`` — for
    the request-body endpoints, whose input freshness the caller controls.
    """
    return Meta(data_freshness=data_freshness)


def _wall_clock_today() -> date:
    """Server-local calendar date (indirection so tests can pin it)."""
    return date.today()


def _data_freshness(latest_whoop_date: date) -> DataFreshness:
    """Compare the freshest WHOOP record against the wall clock.

    Pure transparency metadata: the recommender always runs on the most
    recent WHOOP row regardless — this only tells the caller how far that
    row trails today. No training logic, no behavioral change, here
    (``CLAUDE.md``: the API stays a thin layer over deterministic rules).
    """
    wall = _wall_clock_today()
    days_behind = (wall - latest_whoop_date).days
    is_stale = days_behind >= 1
    if is_stale:
        note = (
            "WHOOP has not yet synced today's recovery. The recommendation "
            f"below reflects {latest_whoop_date.isoformat()} data."
        )
    else:
        note = f"WHOOP data is current as of {latest_whoop_date.isoformat()}."
    return DataFreshness(
        wall_clock_today=wall,
        latest_whoop_date=latest_whoop_date,
        days_behind=days_behind,
        is_stale=is_stale,
        note=note,
    )


def _whoop_row_to_model(row: pd.Series) -> WhoopDaily:
    """Rebuild a :class:`WhoopDaily` from one loader DataFrame row.

    Mirrors the synthetic-CSV path: coerce NaN optionals back to ``None``
    so the schema validates. No reshaping — the row already carries the
    exact WhoopDaily fields.
    """
    rec = row.to_dict()
    for field in _OPTIONAL_WHOOP_FIELDS:
        value = rec.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            rec[field] = None
    return WhoopDaily.model_validate(rec)


@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness probe",
    description=(
        "Cheap, dependency-free health check. Returns the literal "
        "service status and the running agent version. Suitable for "
        "container / load-balancer liveness and readiness probes."
    ),
)
async def healthz(
    version: str = Depends(get_agent_version),
) -> HealthResponse:
    return HealthResponse(status="ok", version=version)


@app.post(
    "/daily_recommendation",
    response_model=DailyRecommendationResponse,
    tags=["recommendations"],
    summary="Daily workout recommendation",
    description=(
        "Run the Phase 5 deterministic orchestrator for one "
        "athlete-day. The body carries today's WHOOP record, the "
        "trailing history window (send the last ~90 days; framework.md "
        "§9 rolling baselines need up to 28), the runner profile, and an "
        "optional 1-based block week number. The endpoint rebuilds the "
        "history DataFrame, expands the planner week anchored on today's "
        "date, and returns the recommender's `Recommendation` verbatim "
        "under a `meta` block. Safety overrides and red-flag medical "
        "escalations come from the rule layer and are never paraphrased "
        "here."
    ),
)
async def daily_recommendation(
    req: DailyRecommendationRequest,
) -> DailyRecommendationResponse:
    # Trailing window → DataFrame, exactly the shape the Phase 5
    # recommender's own tests use (one WhoopDaily.model_dump() per row),
    # sorted by date so the rolling metrics see a chronological series.
    ordered = sorted(req.history, key=lambda w: w.date)
    history_df = pd.DataFrame([w.model_dump() for w in ordered])

    # Anchor the planner week on today's date so the resolved plan always
    # contains today's session (the recommender raises ValueError if it
    # does not — handled above as a 422).
    week_number = req.week_number or 1
    plan = generate_weekly_plan(
        req.profile, week_number, start_date=req.today.date
    )

    recommendation = recommend_daily_workout(
        req.today, history_df, req.profile, plan
    )
    return DailyRecommendationResponse(
        meta=_meta(), recommendation=recommendation
    )


@app.get(
    "/me/today",
    response_model=DailyRecommendationResponse,
    tags=["recommendations"],
    summary="Today's recommendation from live WHOOP data",
    description=(
        "Pull the configured athlete's WHOOP history (last 90 days, "
        "disk-cached 15 min; pass `?refresh=true` to bypass the cache and "
        "force a live re-fetch) and run the Phase 5 deterministic "
        "orchestrator for today. No request body: WHOOP auth is via "
        "environment credentials and the runner profile comes from "
        "`SAWE_*` env vars (documented defaults otherwise). The data "
        "layer is the only thing new here — the recommender, rules, "
        "metrics, and planner are unchanged and the `Recommendation` is "
        "returned verbatim; `meta.data_freshness` reports how far the "
        "latest WHOOP record trails the wall clock without altering it. "
        "Upstream WHOOP failures map to 401 (auth), 429 (rate limit), or "
        "502 (network/API)."
    ),
)
async def me_today(
    refresh: bool = Query(
        False,
        description=(
            "Bypass the 15-minute disk cache and force a fresh WHOOP "
            "fetch (the fresh result is written back to the cache)."
        ),
    ),
    profile: RunnerProfile = Depends(get_runner_profile),
) -> DailyRecommendationResponse:
    # Data layer only (WhoopClient + normalizer behind the cache); the
    # recommender wiring below is identical to /daily_recommendation —
    # no training logic is introduced in the API (CLAUDE.md).
    history = load_whoop_history(90, force_refresh=refresh)
    if history.empty:
        raise HTTPException(
            status_code=503,
            detail=(
                "No WHOOP data available for the configured account. "
                "Run `python -m src.ingest.auth_setup` and confirm the "
                "WHOOP_* environment variables are set."
            ),
        )

    ordered = history.sort_values("date").reset_index(drop=True)
    today = _whoop_row_to_model(ordered.iloc[-1])
    history_df = ordered.iloc[:-1].reset_index(drop=True)

    # Anchor the planner week on today's date so the resolved plan always
    # contains today's session (recommender raises ValueError otherwise →
    # handled above as a 422).
    plan = generate_weekly_plan(profile, 1, start_date=today.date)
    recommendation = recommend_daily_workout(
        today, history_df, profile, plan
    )
    # Pure metadata: how far today's WHOOP record trails the wall clock.
    # The recommendation above is computed identically with or without it.
    return DailyRecommendationResponse(
        meta=_meta(data_freshness=_data_freshness(today.date)),
        recommendation=recommendation,
    )


@app.post(
    "/weekly_plan",
    response_model=WeeklyPlanResponse,
    tags=["plans"],
    summary="Weekly training plan",
    description=(
        "Expand the Phase 4 template for a runner's (tier, goal) into a "
        "dated seven-day `WeeklyPlan` for the given 1-based block week, "
        "starting on `start_date`. Pure planning — no WHOOP data and no "
        "safety logic is applied here (those are per-day, via "
        "`/daily_recommendation`). Returned verbatim under a `meta` "
        "block."
    ),
)
async def weekly_plan(req: WeeklyPlanRequest) -> WeeklyPlanResponse:
    plan = generate_weekly_plan(
        req.profile, req.week_number, start_date=req.start_date
    )
    return WeeklyPlanResponse(meta=_meta(), weekly_plan=plan)
