"""WHOOP daily data record.

One row per calendar day, mirroring the ``WhoopDaily`` schema in
``docs/framework.md`` §9. This is the raw ingestion model: it carries the
metrics the agent reasons over (recovery, HRV rMSSD, RHR, sleep, day strain)
plus optional workout/illness signals.

Field ranges follow the WHOOP semantics documented in framework.md §8:
recovery is a 0-100 percentage, day/workout strain is the 0-21 Borg-derived
scale, sleep performance is a 0-1 fraction. Out-of-range values are rejected
so downstream metric/rule code can trust the inputs.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# WHOOP exposes a 5-zone HR distribution; the agent only ever expects these.
_ALLOWED_ZONES = {"Z1", "Z2", "Z3", "Z4", "Z5"}


class WhoopDaily(BaseModel):
    """A single day of WHOOP-derived physiological data.

    Mirrors framework.md §9 exactly, with explicit physiological bounds added
    so invalid ingests fail fast rather than corrupting rolling baselines.
    """

    model_config = ConfigDict(extra="forbid")

    date: date

    # Primary readiness gate (framework.md §8): Red 0-33, Yellow 34-66,
    # Green 67-100.
    recovery_score: int = Field(ge=0, le=100, description="WHOOP recovery %, 0-100")

    # Overnight rMSSD in milliseconds. Physiologically strictly positive;
    # upper bound is generous to allow very fit autonomic profiles.
    hrv_rmssd: float = Field(gt=0, le=300, description="Overnight HRV rMSSD, ms")

    rhr: int = Field(ge=20, le=120, description="Resting heart rate, bpm")

    sleep_performance: float = Field(
        ge=0, le=1, description="Sleep performance fraction, 0-1"
    )
    sleep_hours: float = Field(ge=0, le=24, description="Hours actually slept")
    sleep_need_hours: float = Field(
        ge=0, le=24, description="WHOOP-computed sleep need, hours"
    )

    rem_min: int = Field(ge=0, description="REM sleep, minutes")
    sws_min: int = Field(ge=0, description="Slow-wave (deep) sleep, minutes")
    light_min: int = Field(ge=0, description="Light sleep, minutes")

    # 0-21 Borg-derived logarithmic scale (framework.md §8).
    day_strain: float = Field(ge=0, le=21, description="WHOOP day strain, 0-21")

    workout_strain: Optional[float] = Field(
        default=None, ge=0, le=21, description="Strain of the logged workout, 0-21"
    )
    workout_hr_mean: Optional[int] = Field(default=None, ge=20, le=250)
    workout_hr_max: Optional[int] = Field(default=None, ge=20, le=250)

    # Minutes spent in each HR zone, e.g. {"Z1": 30, "Z2": 5, ...}.
    zone_minutes: dict[str, float] = Field(
        description="Minutes per HR zone, keys Z1-Z5"
    )

    respiratory_rate: float = Field(
        gt=0, le=40, description="Overnight respiratory rate, breaths/min"
    )

    # WHOOP 4.0+ skin temperature deviation from personal baseline; may be
    # negative. None when the device does not report it.
    skin_temp_dev_c: Optional[float] = Field(default=None, ge=-5, le=5)

    # Free-form WHOOP Journal entries (soreness, stress, alcohol, ...).
    journal: dict = Field(default_factory=dict)

    @field_validator("zone_minutes")
    @classmethod
    def _validate_zone_minutes(cls, v: dict[str, float]) -> dict[str, float]:
        bad_keys = set(v) - _ALLOWED_ZONES
        if bad_keys:
            raise ValueError(
                f"zone_minutes has unexpected keys {sorted(bad_keys)}; "
                f"allowed keys are {sorted(_ALLOWED_ZONES)}"
            )
        negatives = {k: x for k, x in v.items() if x < 0}
        if negatives:
            raise ValueError(f"zone_minutes values must be >= 0; got {negatives}")
        return v
