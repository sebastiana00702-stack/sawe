"""FastAPI dependency scaffold (Phase 6).

Phase 6 deliberately ships **no** auth, persistence, scheduling, or WHOOP
integration, so there are no real per-request dependencies yet. This
module exists so the wiring is already in place: future cross-cutting
concerns (API keys, DB sessions, per-request config) get a single home
and the endpoints already pull from it via ``Depends``. For now it only
injects the agent version so that value has one source of truth.
"""

from __future__ import annotations

from src.api.schemas import AGENT_VERSION


def get_agent_version() -> str:
    """Inject the running agent / API version.

    Scaffold dependency — kept trivial on purpose. When real settings or
    a session layer arrive they slot in here without touching endpoints.
    """
    return AGENT_VERSION
