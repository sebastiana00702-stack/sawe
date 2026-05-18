"""Single source of truth for every safety threshold.

Per ``CLAUDE.md``: *every* safety threshold from ``docs/framework.md`` §9
(and the §11 hard ceilings / red flags) lives here as a named constant, and
every rule that consumes one has a unit test pinned to its exact boundary
(see ``tests/test_rules.py``).

Constants are grouped by signal and annotated with the framework section and
the *exact comparison* the rule applies, because strict vs. non-strict
inequalities are load-bearing here. The §9 reference pseudocode uses strict
``<`` / ``>`` for the override tree (``recovery_score < 34``, ``acwr > 1.5``,
``sleep_hours < 5`` …) and non-strict ``>=`` / ``<=`` for the
``readiness`` / ``training_state`` classifiers; :mod:`src.rules.gates`
mirrors those operators verbatim against the constants below.

Nothing in this module imports application code — it is pure data so the
gate logic, the planner, and the tests share one definition and can never
drift.
"""

from __future__ import annotations

# ==========================================================================
# Recovery score bands (framework.md §8 / §9 threshold table)
# WHOOP bands: Red 0-33, Yellow 34-66, Green 67-100.
# ==========================================================================

#: Top of the Red band — recovery ``<= 33`` is Red ("rest/walk only").
RECOVERY_RED_MAX = 33
#: Top of the Yellow band — ``34..66`` is Yellow ("moderate ceiling").
RECOVERY_YELLOW_MAX = 66
#: Bottom of the Green band — recovery ``>= 67`` is Green ("plan as-is").
#: Also the §9 readiness "high" recovery floor.
RECOVERY_GREEN_MIN = 67

#: §9 override-tree cutoff: ``recovery_score < 34`` -> rest/walk. Equal to
#: ``RECOVERY_RED_MAX + 1`` and kept explicit so the rule reads ``< 34``
#: exactly as the framework pseudocode does.
RECOVERY_REST_BELOW = 34
#: §9 moderate-downgrade cutoff: ``recovery_score < 67`` downgrades a quality
#: session. Equal to ``RECOVERY_GREEN_MIN``.
RECOVERY_MODERATE_BELOW = 67
#: §11 full-rest trigger: Recovery ``< 20`` forces full rest regardless of
#: anything else.
RECOVERY_FULL_REST_BELOW = 20

# ==========================================================================
# Readiness classifier cutoffs (framework.md §9 ``readiness()``)
# high   : rec >= 67 and sleep_h >= 7 and rhr_d <= 3
# moderate: rec >= 50 and sleep_h >= 6
# low     : rec >= 34
# very_low: otherwise
# ==========================================================================

READINESS_HIGH_RECOVERY = 67       # == RECOVERY_GREEN_MIN
READINESS_HIGH_SLEEP_H = 7.0
READINESS_HIGH_RHR_DELTA = 3.0
READINESS_MODERATE_RECOVERY = 50
READINESS_MODERATE_SLEEP_H = 6.0
READINESS_LOW_RECOVERY = 34        # == RECOVERY_REST_BELOW

# ==========================================================================
# Sleep (framework.md §8 / §9 / §11)
# ==========================================================================

#: §9: sleep ``< 5 h`` -> rest/walk. Also the §11 hard ceiling
#: "no interval session if sleep < 5 h".
SLEEP_REST_BELOW_H = 5.0
#: §9: sleep ``< 6 h`` -> downgrade a quality session.
SLEEP_DOWNGRADE_BELOW_H = 6.0
#: §11 full-rest trigger: sleep ``< 4 h`` forces full rest.
SLEEP_FULL_REST_BELOW_H = 4.0
#: §8: sleep performance ``< 0.85`` (85%) -> downgrade.
SLEEP_PERFORMANCE_DOWNGRADE_BELOW = 0.85

# ==========================================================================
# Resting heart rate deviation (framework.md §8 / §9 / §11)
# rhr_delta = today RHR - 28-day rolling mean (src.metrics.rhr_delta)
# ==========================================================================

#: §9: ``rhr_delta > 5`` -> caution, no intensity. Also the
#: overreaching/illness RHR component (paired with HRV or RR).
RHR_DELTA_CAUTION = 5.0
#: §9 override tree: ``rhr_delta > 7`` (with HRV crash) -> rest + illness flag.
RHR_DELTA_ILLNESS_REST = 7.0
#: §11 red flag: RHR persistently ``> +10 bpm`` above baseline for
#: ``> 7 days`` -> medical referral.
RHR_DELTA_RED_FLAG = 10.0
RHR_RED_FLAG_PERSIST_DAYS = 7
#: §9 readiness "high" requires ``rhr_delta <= 3``.
RHR_DELTA_HIGH_READINESS = 3.0
#: §11 race-pace ceiling: "unless RHR is at baseline" — at/below the 28-day
#: mean, i.e. ``rhr_delta <= 0``.
RHR_DELTA_AT_BASELINE = 0.0

# ==========================================================================
# Respiratory rate deviation (framework.md §8 / §9)
# resp_rate_delta = today RR - 28-day baseline (src.metrics.resp_rate_deviation)
# ==========================================================================

#: §9: RHR ``+5`` AND RR ``+2`` -> illness flag, rest. Sustained +2 br/min
#: is the §8 illness early-warning.
RESP_RATE_DELTA_ILLNESS = 2.0

# ==========================================================================
# HRV rMSSD z-score (framework.md §8 / §9 / §11)
# hrv_z = (today - 28-day mean) / 28-day SD (src.metrics.hrv_zscore)
# ==========================================================================

#: §9: HRV ``< baseline - 1 SD`` is the low-HRV signal. A single day is
#: noise; sustained for 3+ days is a load-reduction / deload signal.
HRV_Z_LOW = -1.0
#: §9 / §6: HRV low must persist ``>= 3`` consecutive days to act on.
HRV_LOW_PERSIST_DAYS = 3
#: §9: when HRV has been low for 3+ days, reduce load 20-30%. The agent
#: applies the conservative end (cut 30% -> load factor 0.70).
HRV_LOW_LOAD_CUT_MIN = 0.20
HRV_LOW_LOAD_CUT_MAX = 0.30
HRV_LOW_LOAD_CUT_APPLIED = 0.30
#: §11 red flag: HRV crash ``> 2 SD`` below baseline (z < -2) combined with
#: weight loss / amenorrhea / mood disturbance -> suspect RED-S, medical.
HRV_Z_RED_FLAG = -2.0

# ==========================================================================
# Acute:chronic workload ratio (framework.md §6 / §9 / §11)
# acwr = acute_load / chronic_load (src.metrics.acwr)
# ==========================================================================

#: §11 sweet spot lower bound (Gabbett 2016). Below this -> detraining.
ACWR_SWEET_LOW = 0.8
#: §11 sweet spot upper bound. ``0.8..1.3`` is "green".
ACWR_SWEET_HIGH = 1.3
#: §9 / §11 hard stop: ``acwr > 1.5`` -> hold/cut load (Maupin 2020).
ACWR_HARD_STOP = 1.5
#: §9 override tree: when ``acwr > 1.5`` the planned session's load is
#: halved.
ACWR_DOWNGRADE_FACTOR = 0.5

# ==========================================================================
# Foster monotony (framework.md §9 deload trigger)
# monotony = mean(day_strain) / SD(day_strain) over 7 d
# ==========================================================================

#: §9: monotony ``> 2.0`` over the past 7 days -> force variety / deload.
MONOTONY_HIGH = 2.0

# ==========================================================================
# Recovery / illness streaks (framework.md §6 / §9 / §11)
# ==========================================================================

#: §9 / §11: ``>= 2`` consecutive Red recoveries -> mandatory rest;
#: hard ceiling "no two consecutive Red recoveries with a hard workout".
RED_STREAK_MANDATORY_REST = 2
#: §11 hard ceiling: no race-pace work after ``>= 2`` consecutive Yellows
#: unless RHR is at baseline.
YELLOW_STREAK_RACEPACE_BLOCK = 2
#: §9 deload trigger: ``>= 5`` Yellow recoveries within a 7-day window.
YELLOW_DELOAD_COUNT = 5
YELLOW_DELOAD_WINDOW_DAYS = 7
#: §6 overreaching detector: sustained Yellow for ``>= 5`` days (the
#: OR-branch alongside "2 consecutive Reds").
OVERREACH_YELLOW_DAYS = 5

# ==========================================================================
# Overreaching / poor-adaptation detector (framework.md §6, also §9 deload)
# Fires only when ALL components hold.
# ==========================================================================

#: HRV 7-day mean < 28-day mean - 1 SD for ``>= 3`` consecutive days.
OVERREACH_HRV_PERSIST_DAYS = 3
#: RHR 7-day mean > 28-day mean + 5 bpm.
OVERREACH_RHR_DELTA = 5.0
#: 2 consecutive Reds OR sustained Yellow 5+ days (see above).
OVERREACH_RED_STREAK = 2
#: §6 action: prescribe a 5-7 day deload at 50% volume, no intensity.
OVERREACH_DELOAD_VOLUME_FACTOR = 0.5
OVERREACH_DELOAD_DAYS_MIN = 5
OVERREACH_DELOAD_DAYS_MAX = 7

# ==========================================================================
# Hard-session spacing (framework.md §9 / §11)
# ==========================================================================

#: §9: a VO2max session requires ``days_since_hard >= 2``; below that the
#: quality session is converted to easy+strides.
DAYS_SINCE_HARD_MIN = 2
#: §11: minimum 48 h between intervals/threshold/long-run.
HARD_SPACING_HOURS = 48
#: §11: 72 h after a race-effort.
RACE_SPACING_HOURS = 72
#: §11: same-site soreness > 72 h -> cross-train.
SAME_SITE_SORENESS_HOURS = 72

# ==========================================================================
# Deload & mileage progression (framework.md §9 / §10 / §11)
# ==========================================================================

#: §9 deload trigger: 3 consecutive completed build weeks.
BUILD_WEEKS_BEFORE_DELOAD = 3
#: §9 ``next_week_target_mpw``: when a deload is due, next week = 60%.
MPW_DELOAD_FACTOR = 0.60
#: §9: ``acwr_now > 1.3`` -> hold mileage (factor 1.0).
MPW_HOLD_ABOVE_ACWR = 1.3          # == ACWR_SWEET_HIGH
MPW_HOLD_FACTOR = 1.0
#: §9: ``acwr_now < 0.8`` -> ramp +10%.
MPW_RAMP_BELOW_ACWR = 0.8          # == ACWR_SWEET_LOW
MPW_RAMP_DETRAIN_FACTOR = 1.10
#: §9: otherwise the default +7% week-on-week.
MPW_RAMP_NORMAL_FACTOR = 1.07
#: §11 soft rule: cap the absolute weekly bump at 10% (Buist 2008 is weak;
#: ACWR is the hard gate). The +7%/+10% ramps already respect this.
MPW_SOFT_CAP_PCT = 0.10

# ==========================================================================
# Subjective journal red flags / ceilings (framework.md §5 / §11)
# WHOOP Journal soreness is a 0-5 self-report.
# ==========================================================================

#: §5 / §11 hard ceiling: no NEW sprint introduction if hamstring/calf
#: soreness journal score is ``>= 3`` out of 5.
SORENESS_SPRINT_BLOCK = 3

# ==========================================================================
# Overtraining persistence (framework.md §11, Meeusen 2013)
# ==========================================================================

#: §11: flag Non-Functional Overreaching when poor recovery markers persist
#: ``> 14 days`` despite reduced load.
NFOR_PERSIST_DAYS = 14
