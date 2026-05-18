"""Phase 6 request/response envelopes.

Thin transport models that wrap the *unchanged* Phase 1 domain models
(:class:`~src.models.WhoopDaily`, :class:`~src.models.RunnerProfile`,
:class:`~src.models.Recommendation`, :class:`~src.models.WeeklyPlan`).

Per ``CLAUDE.md`` the API is a thin wrapper: these schemas only carry
data in and out. They add **no training logic** and never reshape or
rename what the recommender / planner produce — the domain object is
embedded verbatim, accompanied only by a small :class:`Meta` block
(server timestamp + agent version).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.models import Recommendation, RunnerProfile, WeeklyPlan, WhoopDaily

#: Single source of truth for the agent / API version. Surfaced by
#: ``GET /healthz``, the OpenAPI ``info.version``, and every response Meta.
AGENT_VERSION = "0.5.0"


def _utc_now() -> datetime:
    """Timezone-aware UTC now (per-response timestamp factory)."""
    return datetime.now(timezone.utc)


class DataFreshness(BaseModel):
    """How current the WHOOP data behind a recommendation is.

    Pure caller-transparency metadata. The recommender always runs on the
    most recent WHOOP record regardless; this block only reports how far
    that record trails the wall clock so a caller can tell when WHOOP has
    not yet synced today's recovery. Per ``CLAUDE.md`` it carries **no
    training logic** and never changes the recommendation.
    """

    model_config = ConfigDict(extra="forbid")

    wall_clock_today: date = Field(
        description="Server-local calendar date the response was built on."
    )
    latest_whoop_date: date = Field(
        description=(
            "Date of the most recent WHOOP record the recommendation used."
        )
    )
    days_behind: int = Field(
        description="``wall_clock_today - latest_whoop_date`` in days."
    )
    is_stale: bool = Field(
        description=(
            "True when ``days_behind >= 1`` — today's recovery is not yet "
            "synced and the recommendation reflects an earlier day."
        )
    )
    note: str = Field(
        description="Human-readable explanation of the freshness state."
    )


class Meta(BaseModel):
    """Per-response server metadata (not part of the domain model)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="Server time the response was produced (UTC).",
    )
    agent_version: str = Field(
        default=AGENT_VERSION, description="Recommender/agent version."
    )
    data_freshness: Optional[DataFreshness] = Field(
        default=None,
        description=(
            "WHOOP data-staleness indicator. Populated only by "
            "`GET /me/today` (the live-WHOOP endpoint); `null` on the "
            "request-body endpoints, whose data the caller already owns."
        ),
    )


class DailyRecommendationRequest(BaseModel):
    """Inputs for ``POST /daily_recommendation``.

    ``history`` is the trailing window the framework.md §9 rolling
    baselines roll over (ACWR/monotony need up to 28 days — send the
    last ~90). ``week_number`` selects the 1-based block week the planner
    expands; when omitted the planner's first week is used. The planner
    week is anchored on ``today.date`` so the resolved plan always
    contains today's session.
    """

    model_config = ConfigDict(extra="forbid")

    today: WhoopDaily
    history: list[WhoopDaily] = Field(
        default_factory=list, description="Trailing days, ~last 90."
    )
    profile: RunnerProfile
    week_number: Optional[int] = Field(
        default=None, ge=1, description="1-based block week; default 1."
    )


class WeeklyPlanRequest(BaseModel):
    """Inputs for ``POST /weekly_plan``."""

    model_config = ConfigDict(extra="forbid")

    profile: RunnerProfile
    week_number: int = Field(ge=1, description="1-based block week.")
    start_date: date = Field(description="Calendar date of plan day 0.")


class DailyRecommendationResponse(BaseModel):
    """The recommender's :class:`Recommendation`, returned verbatim."""

    model_config = ConfigDict(extra="forbid")

    meta: Meta = Field(default_factory=Meta)
    recommendation: Recommendation


class WeeklyPlanResponse(BaseModel):
    """The planner's :class:`WeeklyPlan`, returned verbatim."""

    model_config = ConfigDict(extra="forbid")

    meta: Meta = Field(default_factory=Meta)
    weekly_plan: WeeklyPlan


class HealthResponse(BaseModel):
    """Liveness payload for ``GET /healthz``."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    version: str = AGENT_VERSION
