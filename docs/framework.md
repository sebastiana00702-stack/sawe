# Adaptive Running-Coach Agent: Evidence-Based Training Framework for a WHOOP-Driven Python Project

**TL;DR**
- A deterministic, rule-based Python agent can safely translate WHOOP recovery, HRV (rMSSD), RHR, sleep, and strain (0–21 Borg-derived scale) into daily running prescriptions if it builds personal rolling baselines, computes 7-day acute / 28-day chronic load, and clamps recommendations with hard safety thresholds (Recovery <34% → no hard workout; Gabbett ACWR >1.5 → cap or cut load).
- The training science is convergent: 75–85% easy / 15–20% hard (Seiler polarized; Stöggl & Sperlich, Frontiers in Physiology 2014;5:33), 1–2 VO2max sessions/week using 4×4 min at 90–95% HRmax (Helgerud et al., Med Sci Sports Exerc 2007) or 30 s @ 100% vV̇O2max / 30 s @ 50% (Billat et al., Eur J Appl Physiol 2000;81:188–96), threshold work just below LT2 (Norwegian double-threshold; Bakken/Tjelta), 8 s hill sprints for neuromuscular work (Magness), and 2–3 week tapers with 40–60% volume reduction at maintained intensity (Mujika & Padilla, Med Sci Sports Exerc 2003;35:1182–7).
- The agent must NOT diagnose or replace medical advice; it must surface red-flag escalations (chest pain, syncope, persistent RHR ≥+7 bpm with HRV crash, suspected RED-S) and force rest when objective markers say so. Aerobic fitness is the strongest single modifiable mortality predictor (Mandsager et al., JAMA Network Open 2018;1(6):e183605, "no observed upper limit of benefit", N=122,007).

---

## 1. Feasibility of the Agent

### What WHOOP exposes
| Metric | Reliability | Use in agent |
|---|---|---|
| Recovery Score (0–100%) | High (composite, daily) | Primary readiness gate |
| HRV rMSSD (overnight, SWS-weighted) | High for **trends**, noisy day-to-day | Compare to 7-day rolling baseline |
| Resting heart rate | High | Compare to 7-day & 28-day baseline |
| Sleep performance %, need vs got, stages | Moderate–High | Daily intensity gate |
| Day strain & workout strain (0–21) | High intra-individual; **not** comparable across people | Load tracking |
| Heart rate during workouts | Bellenger et al. (Sensors 2021;21(10):3571, doi:10.3390/s21103571), an AIS-funded independent validation of WHOOP 2.0, reported HR bias ≤0.39±0.38% and limits of agreement ≤1.56%, the best wearable HR/HRV accuracy in their 6-device study | Zone distribution |
| Respiratory rate | Moderate | Illness early warning |
| Skin temperature deviation | Moderate (WHOOP 4.0+) | Illness/recovery flag |
| WHOOP Journal (subjective: soreness, stress, alcohol) | User-dependent | Optional inputs |

**Recovery Score composition** (WHOOP support docs, "WHOOP Recovery"): based on four metrics — Sleep, HRV, RHR, and respiratory rate. Bands: **Green 67–100%**, **Yellow 34–66%**, **Red 0–33%**. HRV is the dominant input. However, Dial et al. (Physiological Reports 2025, e70706, doi:10.14814/phy2.70706) recently warned that "Without explicit manufacturer transparency, end users cannot discern how metrics are calculated or weighted." The agent should therefore treat the score as a high-quality composite signal — not a transparent formula — and corroborate it with raw HRV, RHR, and sleep duration trends.

### Reliable vs. cautious metrics
- **Reliable for direction**: 7+ day rolling HRV, 7-day RHR mean, sleep duration deficit, training-load trends.
- **Cautious**: single-day HRV (high biological noise, autonomic transients), absolute strain comparisons across users, absolute HR-derived calorie estimates.

### What the agent must NOT claim
- Not a medical device; not diagnosing AFib, anemia, RED-S, or infection. Surface flags, never conclusions.
- Always provide a fallback ("see a clinician if X persists ≥Y days").
- Implementation: hard-code a `disclaimer` field on every recommendation; force `requires_medical_review=True` if red-flag thresholds tripped (see §11).

### Limitations of consumer wearables
- Optical PPG-derived HRV under-/over-estimates absolute rMSSD vs ECG, but personal trend tracking is what matters for readiness.
- Chest straps (e.g., Polar H10) remain gold standard for **intra-workout** HR. WHOOP's wrist/arm PPG is excellent at rest, more error during high-cadence running — flag drift between expected and observed HR at known pace.

---

## 2. Core Principles of Running Training

| Principle | Evidence | Implementable rule |
|---|---|---|
| Progressive overload / "10% rule" | Buist et al. (AJSM 2008;36(1):33–39, doi:10.1177/0363546507307505): RCT with 532 novice runners showed 13-week graded program built on the 10% rule produced no reduction in running-related injury rate (20.8% vs 20.3%) vs. a standard 8-week plan. Use as a soft guide. | Prefer ACWR ceiling (Gabbett) over naive 10% |
| SAID / specificity | Daniels' Running Formula | Match workout type to race demand |
| Recovery/supercompensation | Meeusen et al. (Med Sci Sports Exerc 2013;45(1):186–205, joint ECSS/ACSM consensus) | Mandatory rest day if 2 consecutive Reds |
| Easy/hard distribution (polarized) | Seiler & Kjerland (Scand J Med Sci Sports 2006); Stöggl & Sperlich (Frontiers in Physiology 2014;5:33, doi:10.3389/fphys.2014.00033, N=48 well-trained athletes, 9 wk: "polarized training has greater impact on key endurance variables than threshold, high intensity, or high volume training"); Muñoz et al. (IJSPP 2014) | Track weekly Z1/Z2/Z3 minutes, alert if Z2 ("grey zone") > 5–10% |
| Aerobic base | Seiler (IJSPP 2010) | ≥75% of weekly time at conversational HR |
| Periodization | Issurin block; Mujika tapering | Mesocycle blocks + weekly deload trigger |
| Deload every 3–4 weeks | Foster (Med Sci Sports Exerc 1998;30(7):1164–8) | Auto-deload when 3 builds complete OR monotony > 2 |
| Sleep & adaptation | Mah et al. (Sleep 2011;34(7):943–950): Stanford basketball players extending to ≥10 h/night improved 282-ft sprint times from 16.2→15.5 s and free-throw accuracy +9% | Downgrade hard sessions when sleep < 6 h |
| Mobility/warm-up | ACSM/NSCA | Always append `dynamic_warmup` and `cooldown` strings to plan output |

**Mobility / warm-up / cooldown reminders the agent surfaces (no strength programming):**
- Pre-run dynamic: 5–10 min easy jog + leg swings, walking lunges, A-skips, B-skips; 4× 60–80 m strides before any quality session.
- Post-run cooldown: 5–10 min easy jog/walk + 20–30 s holds of calf, hamstring, quad, hip-flexor each side.
- These are sidecar text fields on the plan output, not separate workouts.

---

## 3. General Health Running Plan

ACSM (Garber et al., Med Sci Sports Exerc 2011, position stand) and WHO 2020 PA guidelines: ≥150 min moderate or ≥75 min vigorous aerobic activity weekly, plus 2× resistance work. Mandsager et al. (JAMA Netw Open 2018) demonstrated cardiorespiratory fitness is inversely associated with mortality with no upper ceiling (N=122,007, median 8.4 yr follow-up).

**Zone 2 anchor**: 60–70% HRmax / 65–75% HRR / RPE 3–4 / talk-test = full sentences.

### Per-tier weekly structure (general health)
| Tier | Days/wk | Weekly min | Long run | Strides? |
|---|---|---|---|---|
| True Beginner (0–3 mo) | 3 | 60–90 (run/walk) | 20–30 min | No |
| Novice (3–12 mo) | 3–4 | 90–150 | 30–45 min | Optional (wk 6+) |
| Intermediate | 4–5 | 180–300 | 60–90 min | 1×/wk |
| Advanced | 5–6 | 300–500 | 90–120 min | 2×/wk |
| Competitive | 6–7 | 500–800+ | 120–150 min | Integrated |

**Adjustment rule**:
```python
if recovery < 34: workout = "rest_or_walk"
elif recovery < 67 and sleep_hours < 6: workout = "easy_short"
elif RHR_today > RHR_baseline_7d + 5: workout = "easy_short"
else: workout = plan[today]
```

### Runner tier definitions (used everywhere below)
| Tier | Years | mpw | Long run | 5K benchmark | Quality tolerance |
|---|---|---|---|---|---|
| **True Beginner** | 0–3 mo | 0–10 (run/walk) | 20–30 min | >35 min or n/a | 0/wk |
| **Novice** | 3–12 mo | 10–20 | 4–6 mi | 28–35 min | 1 light/wk |
| **Intermediate** | 1–3 yr | 20–30 | 8–12 mi | sub-25 | 1 quality + 1 long |
| **Advanced** | 3+ yr | 30–50 | 14–18 mi | sub-20 | 2 quality + 1 long |
| **Competitive** | 5+ yr | 50–80+ | 18–22 mi | sub-18 | 2–3 quality (incl. doubles) |
| **Masters/Returning** | varies | varies | varies | varies | Step down one tier |

---

## 4. VO2 Max Improvement Plan

**Why it matters**: Mandsager et al. (JAMA Netw Open 2018;1(6):e183605, doi:10.1001/jamanetworkopen.2018.3605) — extreme cardiorespiratory fitness was associated with the lowest risk-adjusted all-cause mortality with no upper limit of benefit (N=122,007, Cleveland Clinic).

**Trainability**: Bacon, Carter, Ogle & Joyner (PLoS ONE 2013;8(9):e73182, doi:10.1371/journal.pone.0073182) meta-analyzed 37 studies / 40 training groups / 334 participants and reported a mean VO₂max increase of **0.51 L·min⁻¹ (95% CI 0.43–0.60; standardized ES 0.86)**. The 9 studies producing the largest gains (**~0.85 L·min⁻¹, roughly 15–20%**) used **3–5 min intervals near VO₂max** (Hickson-style) — strongly supporting the Helgerud 4×4 design. Bacon et al.: "all subjects can show marked improvements in VO2max if training programs that include periods of high intensity (∼90% of VO2max) exercise are used."

**Best-evidence protocols**:
| Protocol | Source | Structure | Result |
|---|---|---|---|
| Helgerud 4×4 | Helgerud et al. (Med Sci Sports Exerc 2007;39(4):665–71) | 4 × 4 min at **90–95% HRmax**, 3 min jog at ~70% HRmax, 3×/wk, 8 wk | **VO₂max +7.2%**, SV +10% |
| Billat 30/30 | Billat et al. (Eur J Appl Physiol 2000;81(3):188–96) | 30 s at **100% vV̇O2max** / 30 s at 50% vV̇O2max, to exhaustion (~19 reps mean) | Sustained VO₂max for **~7 min 51 s** vs ~2 min 42 s with continuous strenuous work |
| Norwegian double-threshold | Bakken; Tjelta (IJSPP) | 2 × controlled threshold sessions/day, lactate 2.5–3.5 mmol/L, 2–4 d/wk in peak block | Higher weekly threshold volume at lower fatigue cost |

**Intensity targets**: 90–95% HRmax (Helgerud); 95–100% vV̇O2max (Billat); RPE 8–9; conversation impossible.
**Frequency**: 1–2 sessions/week for non-elites.
**Block duration**: 6–12 weeks; plateau by ~week 10 — rotate to specific endurance.

### Agent VO2max session decision rule
```python
def prescribe_vo2max(today_recovery, days_since_last_hard, weekly_vo2_count, sleep_hours):
    if today_recovery < 67:           return downgrade_to_easy()
    if days_since_last_hard < 2:      return downgrade_to_easy()
    if weekly_vo2_count >= 2:         return tempo_or_long_substitute()
    if sleep_hours < 6:               return downgrade_to_tempo()
    return prescribe_4x4_or_30_30()
```

---

## 5. Top-Speed and Speed-Endurance Plan

Differentiate (Magness, *Science of Running*; NSCA position):
- **Top speed / alactic** (≤10 s): hill sprints, flying 20–30 m, full recovery.
- **Speed endurance** (10–40 s): 150–400 m repeats, incomplete recovery.
- **Anaerobic capacity** (40–90 s): 300–600 m at >100% vV̇O2max.
- **VO₂max** (3–8 min): see §4.

**Magness hill-sprint protocol** (verbatim, X/Twitter Jul 2025 and "Sprint Training for Distance Runners" PDF): "Warm-up. Find a moderate hill. Sprint up it for ~8 seconds. Take a long recovery (2+ min). Repeat 4 to 8 times. Cool down."

**Progression sequence** (agent state machine for runners adding speed):
1. Strides 4–6 × 20 s @ ~90% (2 weeks) →
2. Hill sprints 4–6 × 8 s, walk-down recovery (3–4 weeks) →
3. Hill sprints 6–10 × 8 s (3–4 weeks) →
4. Flat short repeats 100–200 m with full recovery →
5. 300–400 m repeats with incomplete recovery.

**Frequency**: 1–2×/wk, integrated with easy days (strides at the end of an easy run). Never on tight or fatigued legs.

### Agent sprint logic
```python
if recovery >= 67 and not hamstring_soreness_flag and days_since_long_run >= 2:
    allow_sprint_session()
else:
    substitute("easy_with_strides" if recovery >= 50 else "easy_short")
```

---

## 6. Long-Distance Event Training

**Aerobic base**: 4–8 weeks of mostly easy running before introducing intensity; expand chronic load (28-day rolling) ≤10% week-on-week.

**Long-run proportion**: Daniels recommends long run ≤25–30% of weekly mileage when mileage > 40 mpw; capped at 2.5 hours absolute. Pfitzinger's *Advanced Marathoning* prescribes mid-week medium-long runs (11–15 mi) alongside the Sunday long run, with long runs at endurance pace 10–20% slower than marathon pace.

**Tempo / threshold**: Daniels T-pace = 83–88% VO₂max / 88–92% HRmax, sustained 20–40 min or cruise intervals 3–5 × 5–15 min with 1–3 min jog. Norwegian double-threshold: 2 controlled 2.5–3.5 mmol/L sessions/day, 2–4 days/wk in peak block.

**Marathon-pace work**: 8–16 mi at MP inserted in long runs (Pfitzinger progression long run).

**Fueling**: Jeukendrup (Sports Med 2014;44 Suppl 1:S25–33) — up to ~60 g/h glucose alone for events <2 h; **~90 g/h glucose:fructose 2:1** for events >2.5 h; higher (up to 120 g/h) only with gut training. Annotate plan, do not prescribe medically.

**Taper**: Mujika & Padilla (Med Sci Sports Exerc 2003;35:1182–7) — "maintaining training intensity, reducing the training volume (up to 60–90%) and slightly reducing training frequency (no more than 20%); optimal duration 4 to >28 d." Bosquet et al. (2007) meta-analysis: 41–60% volume reduction over 8–14 d produced the largest race-time improvement.

**ACWR monitoring**: Gabbett (Br J Sports Med 2016;50:273–80): sweet spot **0.8–1.3**. Maupin et al. (Open Access J Sports Med 2020;11:51–75, doi:10.2147/OAJSM.S231405) systematic review of 27 studies: ACWR ≥1.50 was associated with increased injury risk pre-season (OR=3.03) and in-season (OR=2.33).

### Agent long-run downgrade rule
```python
def long_run_decision(rec, sleep_h, RHR_delta, ACWR, planned_long_mi, last_week_long_mi):
    if rec < 34 or RHR_delta > 7:           return 0  # rest
    if rec < 50 or sleep_h < 6:             return planned_long_mi * 0.6
    if ACWR > 1.5:                           return min(planned_long_mi, last_week_long_mi)
    if rec < 67 and ACWR > 1.3:              return planned_long_mi * 0.8
    return planned_long_mi
```

### Detecting poor adaptation
Fire `overreaching_flag = True` if all of:
- HRV 7-day mean < HRV 28-day mean − 1 SD for 3+ consecutive days, AND
- RHR 7-day mean > RHR 28-day mean + 5 bpm, AND
- Two consecutive Red recoveries OR sustained Yellow for 5+ days.
Action: prescribe a 5–7 day deload (50% volume, no intensity).

---

## 7. Training Intensity Zones

**Heart-rate zones aligned to Daniels' 5 zones** (VDOT-derived):
| Zone | Daniels | %HRmax | %VO₂max | RPE | Talk test |
|---|---|---|---|---|---|
| E | Easy | 65–79 | 59–74 | 3–4 | Full sentences |
| M | Marathon | 80–89 | 75–84 | 5–6 | Phrases |
| T | Threshold | 88–92 | 83–88 | 7 | 3–4 word bursts |
| I | Interval | 95–100 | 95–100 | 8–9 | One word |
| R | Repetition | >100 | >100 | 9–10 | None |

**Seiler 3-zone polarized**: Z1 < LT1 (~first ventilatory threshold, ~75–80% HRmax) | Z2 between LT1–LT2 ("grey zone") | Z3 > LT2 (~92%+ HRmax). Target: **~80% Z1, ≤5% Z2, ~15–20% Z3** by session count.

**HRmax**: Use **Tanaka et al. (J Am Coll Cardiol 2001;37:153–6): HRmax ≈ 208 − 0.7 × age** (meta-analysis 351 studies / 18,712 subjects; SD ~10 bpm). The classical "220 − age" systematically over-estimates older adults.

**Karvonen HRR**: target = RHR + frac × (HRmax − RHR). More individualized than %HRmax; used by Pfitzinger.

**RPE**: Borg 6–20 (RPE × 10 ≈ HR in untrained adults) or modified CR-10.

**Pace zones**: Daniels VDOT from a recent race; recalibrate every 4–8 weeks. McMillan equivalent.

**Power (Stryd)**: Optional secondary signal; not core to a WHOOP-based agent.

**For a WHOOP-based agent**: primary = HR & HR-derived zones. Secondary = RPE prompted via WHOOP Journal. Pace via phone GPS optional. Power not assumed.

---

## 8. WHOOP-Specific Data Interpretation

| Metric | Interpretation | Agent trigger |
|---|---|---|
| Recovery 67–100% (Green) | Adapted, primed | Allow planned hard session |
| Recovery 34–66% (Yellow) | Maintenance only | Cap session at moderate intensity |
| Recovery 0–33% (Red) | Rest needed | Force easy/off |
| HRV rMSSD | Use 7-day rolling mean; flag if today < mean − 1 SD | Single day = noise, 3-day cluster = signal |
| RHR | 7-day rolling mean; flag if today > mean + 5 bpm | Combined with HRV drop → illness suspect |
| Sleep performance % & need vs got | <85% performance OR <6 h slept = downgrade | Hard floor: any quality session at <5 h |
| Sleep stages | Track SWS deficit over 7 d | Inform deload timing |
| Day strain (0–21) | Borg-derived, **logarithmic** ("more effort to move 16→17 than 4→5", WHOOP) | Use raw value as load input |
| Workout strain | TRIMP-like load | Sum for ACWR |
| HR drift | Pace held but HR rising >5% second half of run | Dehydration/fatigue flag |
| Pace at given HR | Improving = aerobic fitness ↑ | Reward signal for plan |
| Acute load (7-d strain sum) | Fatigue proxy | Numerator of ACWR |
| Chronic load (28-d, normalized weekly) | Fitness proxy | Denominator of ACWR |
| ACWR | Sweet spot 0.8–1.3; >1.5 = danger | Cap progression |
| Monotony (Foster) | mean(daily load) / SD(daily load); >2 = high risk | Force easy days |
| Strain (Foster) | weekly load × monotony | Track week-over-week |
| Respiratory rate | Baseline ±1 br/min; sustained +2 br/min = illness suspect | Force rest, prompt sick-check |
| Skin temp | WHOOP 4.0+; +0.5 °C deviation flag | Combine with RHR↑ + HRV↓ |
| Journal entries | Subjective soreness/stress/alcohol | Optional cap on intensity |

---

## 9. Agent Decision Logic (CORE)

### Data schema (Pydantic-style)
```python
from pydantic import BaseModel
from datetime import date
from typing import Optional, Literal

class WhoopDaily(BaseModel):
    date: date
    recovery_score: int            # 0-100
    hrv_rmssd: float               # ms
    rhr: int                       # bpm
    sleep_performance: float       # 0-1
    sleep_hours: float
    sleep_need_hours: float
    rem_min: int; sws_min: int; light_min: int
    day_strain: float              # 0-21
    workout_strain: Optional[float] = None
    workout_hr_mean: Optional[int] = None
    workout_hr_max: Optional[int] = None
    zone_minutes: dict             # {"Z1":..,"Z2":..,"Z3":..,"Z4":..,"Z5":..}
    respiratory_rate: float
    skin_temp_dev_c: Optional[float] = None
    journal: dict = {}

class RunnerProfile(BaseModel):
    tier: Literal["beginner","novice","intermediate","advanced","competitive"]
    goal: Literal["health","5K","10K","HM","marathon","speed"]
    age: int
    sex: Literal["M","F","other"]
    hrmax_measured: Optional[int] = None
    vdot: Optional[float] = None
    current_mpw: float
    longest_recent_run_mi: float
    target_race_date: Optional[date] = None
```

### Derived metrics
```python
def hrmax(p): return p.hrmax_measured or round(208 - 0.7 * p.age)

def hrv_baseline_7d(df): return df["hrv_rmssd"].rolling(7).mean()
def hrv_sd_28d(df):      return df["hrv_rmssd"].rolling(28).std()
def hrv_zscore(df):
    return (df["hrv_rmssd"] - df["hrv_rmssd"].rolling(28).mean()) / hrv_sd_28d(df)

def rhr_baseline_7d(df): return df["rhr"].rolling(7).mean()
def rhr_delta(df):       return df["rhr"] - df["rhr"].rolling(28).mean()

def acute_load(df):      return df["day_strain"].rolling(7).sum()
def chronic_load(df):    return df["day_strain"].rolling(28).sum() / 4   # weekly-equivalent
def acwr(df):            return acute_load(df) / chronic_load(df)

def monotony(df_week):
    m = df_week["day_strain"].mean()
    s = df_week["day_strain"].std()
    return m / s if s > 0 else float("inf")

def strain(df_week): return df_week["day_strain"].sum() * monotony(df_week)
```

### Classifications
```python
def training_state(acwr, hrv_z, rhr_d, red_streak):
    if red_streak >= 2 or (hrv_z < -1 and rhr_d > 5): return "overreached"
    if acwr > 1.5: return "strained"
    if acwr < 0.8: return "detraining"
    if 0.8 <= acwr <= 1.3: return "functional"
    return "borderline"

def readiness(rec, sleep_h, rhr_d):
    if rec >= 67 and sleep_h >= 7 and rhr_d <= 3: return "high"
    if rec >= 50 and sleep_h >= 6:                return "moderate"
    if rec >= 34:                                  return "low"
    return "very_low"
```

### Workout selection decision tree
```python
def choose_workout(today_plan, w: WhoopDaily, p: RunnerProfile, hist):
    # 1) Hard safety overrides
    if w.recovery_score < 34: return rest_or_walk()
    if w.sleep_hours < 5: return rest_or_walk()
    if hist.red_streak >= 2: return mandatory_rest_day()
    if hist.rhr_delta > 7 and hist.hrv_z < -1: return rest_and_flag_illness()
    if hist.acwr > 1.5: return downgrade(today_plan, factor=0.5)

    # 2) Moderate downgrades
    if w.recovery_score < 67 and today_plan.type in {"VO2max","threshold","long_run"}:
        return downgrade(today_plan)
    if w.sleep_hours < 6 and today_plan.type in {"VO2max","threshold","sprint"}:
        return downgrade(today_plan)

    # 3) Spacing
    if today_plan.type == "VO2max" and hist.days_since_hard < 2:
        return convert_to_easy_with_strides()

    # 4) Green light
    return today_plan
```

### Mileage progression (ACWR-aware)
```python
def next_week_target_mpw(current_mpw, acwr_now, deload_due):
    if deload_due:         return current_mpw * 0.6
    if acwr_now > 1.3:     return current_mpw         # hold
    if acwr_now < 0.8:     return current_mpw * 1.10
    return current_mpw * 1.07
```

### Deload triggers (fire any)
- 3 consecutive build weeks complete.
- monotony > 2.0 over the past 7 days (Foster 1998).
- 5+ Yellow recoveries in a 7-day window.
- HRV 7-day mean < 28-day mean − 1 SD AND RHR_delta > 5 for 3+ days.

### Concrete threshold table (single source of truth)
| Signal | Threshold | Action |
|---|---|---|
| Recovery score | <34 | rest/walk only |
| Recovery score | 34–66 | moderate ceiling |
| Recovery score | ≥67 | plan as-scheduled |
| HRV today | < baseline − 1 SD for 3+ days | reduce load 20–30% |
| RHR today | > 7-day mean + 5 bpm | caution, no intensity |
| RHR + RR | RHR +5 and RR +2 | illness flag, rest |
| Sleep last night | <6 h | downgrade |
| Sleep | <5 h | rest |
| ACWR | >1.5 | hold/cut |
| ACWR | 0.8–1.3 | green |
| Monotony | >2.0 | force variety/rest |
| 2 consecutive Reds | — | mandatory rest |
| Days since hard | <2 | no quality |

---

## 10. Example Plans

### True Beginner — General Health (Weeks 1–8, run-walk)
| Wk | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| 1 | Walk 30 | Off | 1 min run / 2 min walk × 8 | Off | Walk 30 | Off | 1:2 × 8 |
| 4 | Off | 3 min run / 2 min walk × 5 | Off | Walk 30 | Off | 3:2 × 5 | Walk 40 |
| 8 | Off | Run 20 cont. | Off | Walk 30 | Off | Run 25 | Walk 45 |
If recovery <34 any day: substitute walk 20.

### Novice — General Health (Weeks 1–6, building to 25-min continuous)
Three runs/wk, all conversational; one optional 4× strides week 5+; long run capped at 45 min by week 8. Downgrade rule: sleep <6 h → swap to walk + 10-min jog.

### Intermediate VO2max Focus (8 weeks)
| Day | Workout | Purpose | Target |
|---|---|---|---|
| Mon | 30–40 min easy + 4 strides | recovery + neuromuscular | Z1, RPE 3 |
| Tue | 4×4 min @ 90–95% HRmax, 3 min jog | VO₂max | I-pace |
| Wed | 30–45 min easy | recovery | Z1 |
| Thu | 20–30 min T-pace tempo | LT2 lift | T-pace |
| Fri | Off | — | — |
| Sat | 6×8 s hill sprints + 45 min easy | speed + endurance | R/Z1 |
| Sun | 75–90 min long run | aerobic | Z1–Z2 |
Embedded downgrades: Tue VO₂max → 30 min steady if Recovery <67; Sun long → cut 30% if sleep <6 h Friday.

### Advanced Speed-Endurance Block (4 weeks)
- Tue: 8×400 m @ 5K pace, 60 s jog
- Thu: 5×1 km @ 10K pace, 90 s jog
- Sat: hill sprints 8×8 s + 60 min easy
- Sun: 90–120 min long, last 20 min at MP

### 5K Plan (Intermediate, 8 weeks, peak 30 mpw)
Weeks 1–3 base; Weeks 4–6 VO₂max emphasis (4×4, 5×1 km @ I); Weeks 7–8 sharpen (200 s, 400 s @ R-pace); race week taper 50% volume, keep one 4×200 m.

### 10K Plan (Advanced, 12 weeks)
Tue threshold cruise intervals 4×8 min @ T; Thu VO₂max (4×4 or 6×1 km @ I-pace); Sat hill sprints + easy; Sun long 14–18 mi with 4–6 mi @ MP+10 s in weeks 7–10.

### Half-Marathon Plan (Intermediate, 12 weeks, peak 35 mpw, Pfitzinger-style)
Mid-week medium-long 8–12 mi (endurance pace). Sunday long 12–14 mi with progressive segments. Threshold 25–40 min cumulative at T. Two-week taper.

### Marathon Plan (Advanced, 18 weeks, peak 55 mpw)
Phases: Endurance (wk 1–6) → LT+Endurance (wk 7–12) → Race Prep (wk 13–16) → Taper (wk 17–18). Long runs to 20–22 mi with 12–16 mi at MP. Tuesday tempo/LT intervals; Thursday VO₂max or hills; Sunday long. Every 4th week is a deload (volume −25–35%, intensity preserved). Taper: wk −2 = 70%, wk −1 = 50%, race week = 30%.

Every plan: agent appends `dynamic_warmup_str` and `cooldown_str`.

---

## 11. Safety & Injury Prevention

- **Mileage progression**: ACWR sweet spot 0.8–1.3 (Gabbett 2016 BJSM; Maupin et al. OAJSM 2020). Cap absolute weekly bump at 10% as a soft rule; hard-stop at ACWR >1.5.
- **Hard-day spacing**: minimum 48 h between intervals/threshold/long-run, 72 h after a race-effort.
- **When to cross-train**: same-site soreness >72 h → swap to bike/swim/elliptical at equivalent strain.
- **Full rest**: any one of — Recovery <20, sleep <4 h, fever, sharp pain altering gait.
- **Red flags → medical referral** (force `requires_medical_review=True`):
  - Chest pain, syncope, palpitations during exercise
  - Resting HR persistently +10 bpm above baseline >7 days
  - HRV crash (>2 SD below baseline) with weight loss, amenorrhea, mood disturbance → suspect RED-S (Mountjoy et al., IOC consensus update, Br J Sports Med 2018;52:687–97)
  - Pain that worsens during a run
  - Persistent illness symptoms
- **Overtraining**: Meeusen et al. (2013) — distinguish Functional Overreaching (recovers in days, performance bounces), Non-Functional Overreaching (weeks), Overtraining Syndrome (months). Agent flags NFOR if poor recovery markers persist >14 d despite reduced load.
- **Hard ceilings** the agent should never override:
  - No two consecutive Red recoveries with a hard workout.
  - No interval session if sleep <5 h.
  - No new sprint introduction if hamstring/calf journal score ≥3/5.
  - No race-pace work after 2 consecutive Yellows unless RHR is at baseline.

---

## 12. Final Deliverable — Implementation-Ready Framework

### Suggested stack
- `pandas` — time-series of WHOOP daily data, rolling windows.
- `numpy` — z-scores, SD.
- `scikit-learn` — trend detection (linear regression slope on 28-day strain, anomaly detection on HRV).
- `pydantic` — data validation (schemas above).
- `FastAPI` — agent backend; REST endpoint `/daily_recommendation`.
- `SQLite` (dev) or `Postgres` + `SQLAlchemy` (prod) — persistence.
- `apscheduler` — nightly WHOOP pull and plan regeneration.

### Example daily recommendation JSON
```json
{
  "date": "2026-05-18",
  "athlete_id": "u_001",
  "training_state": "functional",
  "readiness": "moderate",
  "recommendation": {
    "type": "threshold_intervals",
    "duration_min": 50,
    "structure": "10 min warmup + 4x8 min @ T-pace (88-92% HRmax) w/ 2 min jog + 10 min cooldown",
    "intensity_target": {"hr_pct_max": [88, 92], "rpe": 7, "pace_label": "T"},
    "rationale": [
      "Recovery 58% (Yellow), within moderate band",
      "ACWR 1.12 in sweet spot",
      "Last hard session 3 days ago",
      "Sleep 6.8 h (above 6h floor)"
    ],
    "downgrade_path": {
      "if_recovery_drops_below_50": "convert to 40 min easy Z1-Z2",
      "if_sleep_below_6": "convert to 4x5 min @ T",
      "if_rhr_elevated_5": "rest day"
    },
    "alternatives": [
      {"type": "easy_run", "duration_min": 45},
      {"type": "cross_train", "modality": "bike", "duration_min": 60}
    ],
    "warmup": "5-10 min easy jog, leg swings 10/side, A-skips 2x20m, B-skips 2x20m, 4 strides 80m",
    "cooldown": "10 min easy jog, standing calf/hamstring/quad/hip-flexor 30s each side",
    "disclaimer": "Not medical advice. Stop if pain, chest discomfort, or dizziness."
  },
  "flags": []
}
```

### Example weekly plan JSON
```json
{
  "week_starting": "2026-05-18",
  "athlete_id": "u_001",
  "tier": "intermediate",
  "goal": "10K",
  "phase": "specific_prep",
  "planned_mpw": 32,
  "days": [
    {"date": "2026-05-18", "type": "easy", "duration_min": 40},
    {"date": "2026-05-19", "type": "vo2max", "structure": "4x4 min @ 90-95% HRmax"},
    {"date": "2026-05-20", "type": "easy", "duration_min": 35},
    {"date": "2026-05-21", "type": "threshold", "structure": "20 min @ T"},
    {"date": "2026-05-22", "type": "rest"},
    {"date": "2026-05-23", "type": "speed", "structure": "6x8s hill sprints + 45 min easy"},
    {"date": "2026-05-24", "type": "long_run", "duration_min": 90}
  ],
  "deload_due_in_weeks": 2,
  "acwr_target": [0.9, 1.2]
}
```

### Deterministic vs LLM-personalizable
| Layer | Deterministic (rule code) | LLM-personalizable (later) |
|---|---|---|
| What workout & intensity | YES | NO |
| Safety overrides | YES | NO |
| Mileage progression math | YES | NO |
| Tone, motivation, encouragement | NO | YES |
| Plain-English explanation of "why this workout" | NO | YES |
| Mapping qualitative journal entries to readiness flags | optional ML | YES |
| Surfacing red flags | YES (hard rules) | NO (must not be paraphrased away) |

### Limitations & disclaimers (must appear in product)
- Not a medical device, not diagnosing.
- Wearable HR/HRV is best for **trend**, not absolute clinical value.
- Algorithms assume the device is worn consistently; gaps degrade recommendations.
- VO₂max trainability is heterogeneous (Bacon 2013); no plan guarantees a specific % gain.
- RED-S, atrial fibrillation, and other clinical conditions REQUIRE clinician evaluation.

---

## Recommendations (build order for the CS student)
1. **Week 1–2**: WHOOP API ingestion → pandas DataFrame → compute rolling baselines and ACWR. Verify outputs against the WHOOP app.
2. **Week 3**: Implement the threshold table in §9 as a single `evaluate_safety_gates(WhoopDaily, history) -> list[Flag]` function. Unit test each rule.
3. **Week 4**: Implement runner-tier profiles and the 6 example weekly templates. Generate plans week-by-week.
4. **Week 5**: Wire FastAPI `/recommendation` endpoint that combines `(template + safety_gates + downgrade_rules) → JSON`.
5. **Week 6**: Add overreaching/illness detector and deload trigger; integrate WHOOP Journal inputs.
6. **Week 7+**: Optional LLM layer for natural-language rationale only — keep all numeric decisions in rule code.

**Benchmarks that should change recommendations**:
- If user has 4+ consecutive weeks at ACWR in 0.8–1.3 with no red flags → unlock next mileage tier.
- If 2 consecutive deload triggers fire within 6 weeks → reduce baseline weekly load by 15% and re-baseline.
- If user completes a tune-up race and pace-at-HR improves >3% → recalculate VDOT/HRmax.
- If recovery <50 for 5/7 days for two consecutive weeks → escalate to "consider medical/coach review."

## Caveats
- WHOOP's recovery-score weighting and HRV computation are not fully documented (Dial et al., Physiological Reports 2025, doi:10.14814/phy2.70706, critiques opacity). Validate internally against personal trends rather than treating the score as a transparent formula.
- ACWR's predictive validity has been questioned in some recent meta-analyses; treat it as one input among several, not the sole gate.
- The 10% rule has weak RCT support (Buist 2008); ACWR plus recovery markers is a better gating approach.
- Sleep extension evidence (Mah 2011) was in basketball players; transfer to recreational runners is plausible but not directly tested at the same magnitude.
- Norwegian double-threshold is elite-level; mid-pack recreational runners should default to 1 threshold + 1 VO2max per week.
- Helgerud's 4×4 was studied in moderately trained populations; well-trained runners may need 6×3 min or 5×5 min variants once they plateau.
- All recommendations assume the user is medically cleared to exercise. The agent must surface a one-time screening (e.g., PAR-Q+) before activation.

---

### Completion table
| Spec item | Covered |
|---|---|
| Feasibility (§1) | ✅ |
| Core principles (§2) | ✅ |
| General health plan (§3) | ✅ |
| VO2max plan (§4 — Helgerud, Billat, Bacon, Mandsager) | ✅ |
| Speed plan (§5 — Magness 8 s hills) | ✅ |
| Long-distance (§6 — Pfitzinger, Daniels, Norwegian, Mujika, Gabbett/Maupin) | ✅ |
| Intensity zones (§7 — Daniels, Seiler, Tanaka) | ✅ |
| WHOOP interpretation (§8) | ✅ |
| Agent decision logic w/ pseudocode + thresholds (§9) | ✅ |
| Example plans for 5+ tiers + 4 race distances (§10) | ✅ |
| Safety, RED-S, OTS (§11 — Meeusen, Mountjoy) | ✅ |
| Implementation framework with JSON + libs (§12) | ✅ |
| Citations throughout | ✅ |
| Rule-implementable recommendations | ✅ |