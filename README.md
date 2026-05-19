# Sawe

**An adaptive running coach that turns WHOOP recovery data into evidence-based daily workout recommendations.**

The training logic comes from published exercise science: polarized intensity distribution (Seiler, 2014), VO₂max interval work (Helgerud, 2007), and Gabbett's acute:chronic workload limits (2016), with tapering research shaping the peak phases. A FastAPI service reads live WHOOP data and returns the day's workout as JSON.

Named after [Sebastian Sawe](https://en.wikipedia.org/wiki/Sebastian_Sawe), the 2025 Berlin Marathon champion (marathon best 2:02:05).

## Example output

The agent runs every safety gate and shows which thresholds it checked. A real recommendation from a recent green-recovery day:

```json
{
  "meta": {
    "data_freshness": {
      "latest_whoop_date": "2026-05-18",
      "is_stale": false,
      "note": "WHOOP data is current as of 2026-05-18."
    }
  },
  "recommendation": {
    "date": "2026-05-18",
    "training_state": "borderline",
    "readiness": "moderate",
    "recommendation": {
      "type": "easy",
      "duration_min": 30,
      "structure": "30 min easy, conversational [HR 65-79% max, RPE 4, E]",
      "intensity_target": {"hr_pct_max": [65, 79], "rpe": 4, "pace_label": "E"},
      "rationale": [
        "Recovery 89% (Green), primed, plan as scheduled",
        "Sleep 8.39 h (above 6 h floor)",
        "Last hard session 2 days ago",
        "Training state borderline; readiness moderate",
        "All safety gates clear."
      ],
      "downgrade_path": {
        "if_recovery_drops_below_50": "convert to easy Z1-Z2 aerobic",
        "if_sleep_below_6h": "downgrade any quality session / shorten to easy",
        "if_rhr_above_baseline_5bpm": "drop intensity — easy or rest, no quality",
        "if_two_consecutive_reds": "mandatory rest day"
      }
    }
  }
}
```

Every decision names the exact WHOOP number and rule behind it.

## Architecture

Rule-based, no LLM in the decision path. Every recommendation traces back to a numeric threshold.

```
src/
├── models/      Pydantic schemas (WhoopDaily, RunnerProfile, Recommendation)
├── metrics/     Rolling baselines, ACWR, monotony, HRV z-scores
├── rules/       Safety gates, threshold table, downgrade logic
├── planner/     Weekly templates per tier (beginner → competitive) and goal (5K → marathon)
├── agent/       Recommendation engine — combines metrics + rules + plan
├── api/         FastAPI service layer
└── ingest/      WHOOP v2 OAuth client + data normalizer
```

The data flow is one-directional:

```
WHOOP API → normalizer → DataFrame → metrics → safety gates → planner → recommendation JSON
```

Each layer can be tested on its own. The recommender only wires the layers together; the metrics and rules modules hold the actual logic.

## Stack

- Python 3.11+
- FastAPI, Pydantic, pandas, httpx
- WHOOP v2 REST API (OAuth 2.0 with refresh token rotation)
- 316 unit + integration tests (pytest)
- Local cache with 15-minute TTL

## What it actually does

Given a runner's profile (tier, goal, current mileage) and the last 90 days of their WHOOP data, the agent:

1. Computes rolling baselines for HRV, resting heart rate, and training load
2. Classifies today's readiness (high / moderate / low / very_low)
3. Classifies training state (fresh / functional / borderline / strained / overreached / detraining)
4. Evaluates every safety gate from the framework's §9 threshold table
5. Looks up the planned workout from the runner's weekly template
6. Applies downgrades if any safety gate trips
7. Returns a structured Recommendation with workout type, intensity targets, rationale, downgrade paths, and alternatives

Safety gates:

- Recovery <34% → rest or walk only
- HRV <1 SD below baseline for 3+ days → reduce load
- RHR >5 bpm above 7-day baseline → caution, no intensity
- ACWR >1.5 (Gabbett) → cap or cut load
- Sleep <6h → downgrade; <5h → mandatory rest
- 2 consecutive Red recoveries → mandatory rest day
- Monotony >2.0 (Foster) → force variety

## Tiers supported

Six runner profiles with different weekly volumes, longest runs, and quality tolerances:

| Tier | Years | Weekly mileage | Long run | Quality sessions |
|---|---|---|---|---|
| True Beginner | 0-3 mo | 0-10 (run/walk) | 20-30 min | 0/wk |
| Novice | 3-12 mo | 10-20 | 4-6 mi | 1 light/wk |
| Intermediate | 1-3 yr | 20-30 | 8-12 mi | 1 quality + 1 long |
| Advanced | 3+ yr | 30-50 | 14-18 mi | 2 quality + 1 long |
| Competitive | 5+ yr | 50-80+ | 18-22 mi | 2-3 quality |
| Masters/Returning | varies | varies | varies | step down one tier |

Each tier has weekly templates for general health, 5K, 10K, half marathon, and marathon training.

## Run locally

```bash
git clone https://github.com/sebastiana00702-stack/sawe
cd sawe
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest
```

To run against your own WHOOP data:

```bash
# 1. Register a WHOOP developer app at developer.whoop.com
#    Redirect URI: http://localhost:8080/callback
#    Scopes: read:recovery read:cycles read:sleep read:workout read:profile offline

# 2. Copy .env.example to .env and fill in WHOOP_CLIENT_ID, WHOOP_CLIENT_SECRET

# 3. Run the one-time OAuth flow
python -m src.ingest.auth_setup

# 4. Start the API
uvicorn src.api.main:app --reload

# 5. Visit http://localhost:8000/docs and hit GET /me/today
```

## What I learned building this

- Phase 7.4 stayed green through every test even though its fixtures were UUID-keyed and never matched WHOOP's integer cycle_ids, so the live API quietly lost data until I hand-wrote a regression test from a real response dump and finally saw it.
- Most OAuth tutorials skip refresh-token rotation even though it's standard for security-conscious providers (RFC 6749 §6); Sawe writes the rotated token back to .env atomically and pushes it into os.environ so the next client instance picks it up.
- Because a WHOOP cycle starts at wake rather than midnight, a late bedtime can map two distinct cycles onto one Python date, and the original normalizer silently overwrote one and dropped a day. It raises now. Silent data loss in training input is the worst class of bug.

## Roadmap

- Phase 8 (optional): LLM rationale layer that replaces templated strings with a natural coach voice while keeping all numeric decisions in rule code
- Scheduled job to push daily recommendations to email/Discord/Slack
- Phone-friendly web interface
- Expand from rule-based to data-driven baseline learning as the user accumulates 6+ months of WHOOP history

## Limitations and disclaimers

Sawe is not a medical device and does not provide medical advice. Recommendations are for general fitness purposes only. The user is responsible for consulting a qualified healthcare professional before starting, modifying, or stopping any exercise program. If recommendations consistently flag overreaching, illness suspicion, or persistent low recovery, see a clinician.

See PRIVACY.md for data handling.
