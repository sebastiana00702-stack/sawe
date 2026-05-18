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

import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.agent import recommend_daily_workout
from src.api.deps import get_agent_version
from src.api.schemas import (
    AGENT_VERSION,
    DailyRecommendationRequest,
    DailyRecommendationResponse,
    HealthResponse,
    Meta,
    WeeklyPlanRequest,
    WeeklyPlanResponse,
)
from src.planner.weekly_plan import generate_weekly_plan

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


def _meta() -> Meta:
    """Fresh per-response metadata (new UTC timestamp each call)."""
    return Meta()


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
