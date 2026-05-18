"""FastAPI dependency scaffold (Phase 6, extended in Phase 7).

Cross-cutting per-request concerns live here so endpoints stay thin. As
of Phase 7 it also resolves the single-athlete :class:`RunnerProfile` for
the ``GET /me/today`` endpoint from environment variables — the WHOOP
integration is single-user (auth is env-based, see CLAUDE.md), so the
profile is configuration, not request data, and carries documented
defaults so the endpoint works out of the box.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

from src.api.schemas import AGENT_VERSION
from src.models import RunnerProfile

# Defaults for an unconfigured install — a middle-of-the-road recreational
# runner. Override any of these via the SAWE_* environment variables.
_PROFILE_DEFAULTS = {
    "tier": "intermediate",
    "goal": "health",
    "age": "30",
    "sex": "M",
    "current_mpw": "25",
    "longest_recent_run_mi": "10",
}


def get_agent_version() -> str:
    """Inject the running agent / API version.

    Scaffold dependency — kept trivial on purpose. When real settings or
    a session layer arrive they slot in here without touching endpoints.
    """
    return AGENT_VERSION


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(f"SAWE_{name.upper()}")
    return value if value not in (None, "") else default


def get_runner_profile() -> RunnerProfile:
    """Resolve the configured athlete profile (env vars + defaults).

    The WHOOP integration is single-user and credential-driven, so the
    profile is install configuration rather than per-request input. Pure
    parsing/validation only — no training logic (that stays in the rule /
    planner layers per CLAUDE.md). Pydantic enforces the literal/range
    bounds, so a bad env value surfaces as a clean 422.
    """
    race_date = _env("target_race_date")
    return RunnerProfile(
        tier=_env("tier", _PROFILE_DEFAULTS["tier"]),
        goal=_env("goal", _PROFILE_DEFAULTS["goal"]),
        age=int(_env("age", _PROFILE_DEFAULTS["age"])),
        sex=_env("sex", _PROFILE_DEFAULTS["sex"]),
        hrmax_measured=(
            int(_env("hrmax_measured"))
            if _env("hrmax_measured")
            else None
        ),
        vdot=float(_env("vdot")) if _env("vdot") else None,
        current_mpw=float(
            _env("current_mpw", _PROFILE_DEFAULTS["current_mpw"])
        ),
        longest_recent_run_mi=float(
            _env(
                "longest_recent_run_mi",
                _PROFILE_DEFAULTS["longest_recent_run_mi"],
            )
        ),
        target_race_date=(
            date.fromisoformat(race_date) if race_date else None
        ),
    )
