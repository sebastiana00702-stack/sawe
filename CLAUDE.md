# Sawe — Running Coach Agent

Named after Sebastian Sawe, Kenyan marathoner.

## Project
Rule-based Python agent that ingests WHOOP data and recommends daily running workouts.
Full spec: docs/framework.md (READ THIS FIRST for any training-logic question)

## Stack
- Python 3.11+
- pandas, numpy, pydantic, fastapi, sqlalchemy, apscheduler, pytest
- SQLite for dev

## Architecture
- src/models/      Pydantic schemas (WhoopDaily, RunnerProfile, Recommendation)
- src/metrics/     Rolling baselines, ACWR, monotony, z-scores
- src/rules/       Safety gates, downgrade logic, deload triggers
- src/planner/     Weekly templates per tier/goal
- src/agent/       Orchestrator: combines metrics + rules + plan → JSON
- src/api/         FastAPI endpoints
- tests/           Pytest, one file per module

## Rules
- Every safety threshold from framework.md §9 must be a named constant in src/rules/thresholds.py
- Every rule needs a unit test with the exact threshold boundary
- No training logic in the API layer
- LLM integration is a future layer — keep recommendation logic deterministic