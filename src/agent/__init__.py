"""The Phase 5 recommendation engine (framework.md §9/§12).

The deterministic orchestrator: it sequences the Phase 2 metrics, the
Phase 3 §9/§11 safety gates, and the Phase 4 weekly templates into one
validated :class:`~src.models.Recommendation`. No training logic or LLM
lives here — only the framework.md §9 ordering and the §12 output shape.
Re-exported so callers (and the future Phase 6 FastAPI layer) can
``from src.agent import recommend_daily_workout``.
"""

from src.agent.rationale import build_rationale
from src.agent.recommender import recommend_daily_workout
from src.agent.state import (
    classify_readiness,
    classify_training_state,
    consecutive_red_streak,
    days_since_last_hard,
    trailing_streak,
)

__all__ = [
    "recommend_daily_workout",
    "classify_training_state",
    "classify_readiness",
    "days_since_last_hard",
    "consecutive_red_streak",
    "trailing_streak",
    "build_rationale",
]
