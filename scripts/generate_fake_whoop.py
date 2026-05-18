"""Generate 90 days of realistic synthetic WHOOP daily data.

Writes ``data/fake_whoop.csv`` with one row per day. The series is built
around personal baselines (framework.md §1: trends matter more than absolute
values) with a weekly training microcycle driving day strain, plus injected
stress events:

  * 3-4 isolated bad-recovery days (poor sleep + autonomic dip).
  * One 5-day illness period: elevated RHR, depressed HRV, raised
    respiratory rate and skin temperature (framework.md §8 illness signals).

``zone_minutes`` and ``journal`` are stored as JSON strings so the CSV
round-trips cleanly through :class:`WhoopDaily`. Every generated row is
validated against the model before the file is written; the script exits
non-zero if any row fails.

Deterministic: uses a fixed RNG seed so the dataset is reproducible.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python scripts/generate_fake_whoop.py` from anywhere in the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models import WhoopDaily  # noqa: E402

N_DAYS = 90
SEED = 20260518
OUT_PATH = REPO_ROOT / "data" / "fake_whoop.csv"

# Personal baselines for a reasonably fit recreational runner.
HRV_BASELINE = 68.0  # ms rMSSD
RHR_BASELINE = 48  # bpm
RESP_BASELINE = 14.5  # breaths/min

# Weekly microcycle (index 0 = Monday). Target day strain + workout label;
# strain follows the 0-21 Borg-derived scale (framework.md §8).
WEEK_CYCLE = [
    (8.0, "easy"),  # Mon easy
    (15.5, "vo2max"),  # Tue quality
    (7.0, "easy"),  # Wed recovery
    (13.0, "threshold"),  # Thu tempo
    (3.0, "rest"),  # Fri off
    (12.0, "speed"),  # Sat hills + easy
    (16.5, "long_run"),  # Sun long run
]


def _zone_split(workout: str, total_min: float, rng: np.random.Generator) -> dict:
    """Distribute training minutes across HR zones by workout type.

    Polarized: most time in Z1, hard days add Z3-Z5 (framework.md §7).
    """
    if workout == "rest" or total_min <= 0:
        return {"Z1": 0.0, "Z2": 0.0, "Z3": 0.0, "Z4": 0.0, "Z5": 0.0}

    if workout in {"easy"}:
        weights = np.array([0.80, 0.18, 0.02, 0.0, 0.0])
    elif workout == "long_run":
        weights = np.array([0.70, 0.25, 0.05, 0.0, 0.0])
    elif workout == "threshold":
        weights = np.array([0.45, 0.15, 0.35, 0.05, 0.0])
    elif workout == "vo2max":
        weights = np.array([0.40, 0.10, 0.20, 0.25, 0.05])
    elif workout == "speed":
        weights = np.array([0.55, 0.10, 0.10, 0.15, 0.10])
    else:
        weights = np.array([0.80, 0.18, 0.02, 0.0, 0.0])

    jitter = rng.normal(0, 0.02, size=5)
    weights = np.clip(weights + jitter, 0, None)
    weights = weights / weights.sum()
    minutes = np.round(weights * total_min, 1)
    return {f"Z{i + 1}": float(minutes[i]) for i in range(5)}


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    start = date(2026, 2, 17)  # 90 days ending ~ today (2026-05-18)

    # Bad-recovery days (isolated) and a 5-day illness block. Chosen indices
    # avoid overlapping each other.
    bad_days = {12, 31, 53, 74}
    illness_start = 60
    illness_days = set(range(illness_start, illness_start + 5))

    rows = []
    for i in range(N_DAYS):
        d = start + timedelta(days=i)
        dow = d.weekday()
        target_strain, workout = WEEK_CYCLE[dow]

        is_bad = i in bad_days
        is_ill = i in illness_days

        # --- Latent autonomic state -------------------------------------
        # Slow ~28-day fitness wave + day-to-day biological noise.
        wave = np.sin(2 * np.pi * i / 28.0)
        hrv = HRV_BASELINE + 6.0 * wave + rng.normal(0, 5.5)
        rhr = RHR_BASELINE - 1.5 * wave + rng.normal(0, 1.8)
        resp = RESP_BASELINE + rng.normal(0, 0.5)
        skin_temp = float(rng.normal(0, 0.12))

        # --- Sleep -------------------------------------------------------
        sleep_need = float(np.clip(rng.normal(8.0, 0.25), 7.0, 9.0))
        sleep_hours = float(np.clip(rng.normal(7.3, 0.85), 4.0, 9.5))

        if is_bad:
            hrv -= rng.uniform(18, 28)
            rhr += rng.uniform(5, 9)
            sleep_hours = float(np.clip(rng.normal(5.2, 0.5), 4.0, 6.0))
        if is_ill:
            # Sustained illness signature (framework.md §8).
            hrv -= rng.uniform(28, 42)
            rhr += rng.uniform(7, 12)
            resp += rng.uniform(2.0, 4.0)
            skin_temp += rng.uniform(0.5, 1.0)
            sleep_hours = float(np.clip(rng.normal(5.8, 0.6), 4.0, 7.0))

        hrv = float(np.clip(hrv, 15.0, 180.0))
        rhr_i = int(round(np.clip(rhr, 35, 95)))
        resp = float(np.clip(resp, 9.0, 28.0))

        sleep_perf = float(np.clip(sleep_hours / sleep_need + rng.normal(0, 0.04), 0, 1))
        total_sleep_min = sleep_hours * 60.0
        rem_min = int(round(total_sleep_min * rng.uniform(0.20, 0.24)))
        sws_min = int(round(total_sleep_min * rng.uniform(0.15, 0.20)))
        light_min = max(0, int(round(total_sleep_min)) - rem_min - sws_min)

        # --- Recovery score (composite of HRV, RHR, sleep) --------------
        hrv_z = (hrv - HRV_BASELINE) / 12.0
        rhr_delta = rhr_i - RHR_BASELINE
        latent = (
            62.0
            + 16.0 * hrv_z
            - 2.6 * rhr_delta
            + 7.0 * (sleep_hours - 7.0)
            + rng.normal(0, 5.0)
        )
        if is_bad:
            latent = min(latent, rng.uniform(18, 31))
        if is_ill:
            latent = min(latent, rng.uniform(8, 28))
        recovery = int(np.clip(round(latent), 1, 99))

        # --- Strain & workout block -------------------------------------
        if is_bad or is_ill:
            # Athlete backs off when wrecked; mostly easy/off.
            day_strain = float(np.clip(rng.normal(4.5, 1.5), 0, 21))
            workout = "rest" if (is_ill or rng.random() < 0.5) else "easy"
        else:
            day_strain = float(
                np.clip(target_strain + rng.normal(0, 1.3), 0, 21)
            )

        if workout == "rest":
            workout_strain = None
            workout_hr_mean = None
            workout_hr_max = None
            train_min = 0.0
        else:
            workout_strain = float(np.clip(day_strain - rng.uniform(1.5, 3.0), 0, 21))
            hrmax_est = 188  # ~ Tanaka for a 28 y/o; just for plausible HR
            if workout in {"vo2max", "speed"}:
                workout_hr_mean = int(rng.uniform(0.80, 0.86) * hrmax_est)
                workout_hr_max = int(rng.uniform(0.93, 0.99) * hrmax_est)
            elif workout in {"threshold"}:
                workout_hr_mean = int(rng.uniform(0.82, 0.88) * hrmax_est)
                workout_hr_max = int(rng.uniform(0.88, 0.93) * hrmax_est)
            else:
                workout_hr_mean = int(rng.uniform(0.68, 0.76) * hrmax_est)
                workout_hr_max = int(rng.uniform(0.78, 0.85) * hrmax_est)
            train_min = float(np.clip(rng.normal(55, 12), 25, 130))

        zone_minutes = _zone_split(workout, train_min, rng)

        # --- Journal (occasional subjective entries) --------------------
        journal: dict = {}
        if is_ill:
            journal = {"feeling_sick": True, "stress": int(rng.integers(3, 5))}
        elif is_bad:
            journal = {"soreness": int(rng.integers(2, 5)), "stress": int(rng.integers(2, 5))}
        elif rng.random() < 0.18:
            journal = {"soreness": int(rng.integers(0, 3))}
            if rng.random() < 0.4:
                journal["alcohol"] = True

        rows.append(
            {
                "date": d.isoformat(),
                "recovery_score": recovery,
                "hrv_rmssd": round(hrv, 1),
                "rhr": rhr_i,
                "sleep_performance": round(sleep_perf, 3),
                "sleep_hours": round(sleep_hours, 2),
                "sleep_need_hours": round(sleep_need, 2),
                "rem_min": rem_min,
                "sws_min": sws_min,
                "light_min": light_min,
                "day_strain": round(day_strain, 1),
                "workout_strain": (
                    None if workout_strain is None else round(workout_strain, 1)
                ),
                "workout_hr_mean": workout_hr_mean,
                "workout_hr_max": workout_hr_max,
                "zone_minutes": json.dumps(zone_minutes),
                "respiratory_rate": round(resp, 1),
                "skin_temp_dev_c": round(skin_temp, 2),
                "journal": json.dumps(journal),
            }
        )

    return pd.DataFrame(rows)


def _validate(df: pd.DataFrame) -> None:
    """Round-trip every row through WhoopDaily; raise on the first failure."""
    for idx, rec in enumerate(df.to_dict(orient="records")):
        rec = dict(rec)
        rec["zone_minutes"] = json.loads(rec["zone_minutes"])
        rec["journal"] = json.loads(rec["journal"])
        # CSV serialises missing optionals as NaN; convert back to None.
        for opt in ("workout_strain", "workout_hr_mean", "workout_hr_max",
                    "skin_temp_dev_c"):
            if rec[opt] is None or (
                isinstance(rec[opt], float) and pd.isna(rec[opt])
            ):
                rec[opt] = None
        try:
            WhoopDaily.model_validate(rec)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise SystemExit(
                f"Generated row {idx} ({rec['date']}) failed validation:\n{exc}"
            )


def main() -> None:
    df = generate()
    _validate(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    ill = df.iloc[60:65]
    print(f"Wrote {len(df)} rows to {OUT_PATH.relative_to(REPO_ROOT)}")
    print(
        "Recovery: "
        f"min={df.recovery_score.min()} "
        f"mean={df.recovery_score.mean():.1f} "
        f"max={df.recovery_score.max()}"
    )
    print(
        f"HRV rMSSD: {df.hrv_rmssd.min()}-{df.hrv_rmssd.max()} ms | "
        f"RHR: {df.rhr.min()}-{df.rhr.max()} bpm | "
        f"day_strain: {df.day_strain.min()}-{df.day_strain.max()}"
    )
    print(f"Bad-recovery days (recovery<34): {(df.recovery_score < 34).sum()}")
    print(
        "Illness window (days 60-64): "
        f"recovery {ill.recovery_score.tolist()}, "
        f"RHR {ill.rhr.tolist()}, "
        f"resp {ill.respiratory_rate.tolist()}"
    )
    print("All rows validated against WhoopDaily.")


if __name__ == "__main__":
    main()
