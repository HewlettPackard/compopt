"""
compopt.physics.electricity
============================
Realistic grid electricity price model for data-centre cost optimisation.

Models US commercial Time-of-Use (TOU) electricity pricing based on typical
utility tariffs (e.g. PG&E E-19, SDG&E TOU-8, ERCOT real-time):

Pricing tiers
-------------
- **Off-peak** (nights / weekends)  : ~$0.06 – 0.08 / kWh
- **Mid-peak** (shoulder hours)     : ~$0.10 – 0.14 / kWh
- **On-peak** (weekday afternoons)  : ~$0.18 – 0.28 / kWh
- **Critical peak** (rare events)   : $0.40 – 0.75 / kWh

Additional effects
------------------
- Seasonal multiplier (summer > winter by ~25 %)
- Day-of-week factor (weekends cheaper)
- Real-time noise: a mean-reverting Ornstein–Uhlenbeck process (±15 %)
- Optional renewable surplus credit: solar production can drive prices
  negative in mid-day, as observed in California and Texas

Usage
-----
>>> from compopt.physics.electricity import GridElectricityPriceModel
>>> model = GridElectricityPriceModel(seed=42)
>>> model.reset(start_hour=8.0, day_of_year=180)
>>> price = model.step(dt_s=5.0)   # $/kWh at current simulated time
>>> model.hour_of_day              # float in [0, 24)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Seasonal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _seasonal_multiplier(day_of_year: float) -> float:
    """
    Returns a seasonal price multiplier in [0.85, 1.25].

    Peak in summer (~day 196, mid-July) and a secondary peak in winter
    (~day 355, late December) due to heating load, with a spring minimum.
    """
    # Summer peak (cosine with period 365 days, max at day 196)
    summer = 0.20 * math.cos(2 * math.pi * (day_of_year - 196) / 365)
    # Winter bump (secondary cosine, smaller amplitude)
    winter = 0.05 * math.cos(2 * math.pi * (day_of_year - 355) / 183)
    return 1.0 + summer + winter


# ──────────────────────────────────────────────────────────────────────────────
# TOU tariff definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TOUTariff:
    """
    Time-of-Use tariff definition for a single utility.

    Attributes
    ----------
    off_peak_rate : float
        Base off-peak rate, $/kWh.
    mid_peak_rate : float
        Mid-peak (shoulder) rate, $/kWh.
    on_peak_rate : float
        On-peak rate (weekday afternoon), $/kWh.
    critical_peak_rate : float
        Critical-peak-pricing (CPP) rate, $/kWh.
    on_peak_start_h : float
        Hour-of-day when on-peak period starts (weekdays).
    on_peak_end_h : float
        Hour-of-day when on-peak period ends (weekdays).
    mid_peak_morning_start_h : float
        Start of morning mid-peak shoulder period.
    mid_peak_morning_end_h : float
        End of morning mid-peak shoulder period.
    mid_peak_evening_start_h : float
        Start of evening mid-peak shoulder period.
    mid_peak_evening_end_h : float
        End of evening mid-peak shoulder period.
    critical_peak_prob : float
        Probability per episode that a CPP event is active (rare).
    """
    off_peak_rate: float = 0.065
    mid_peak_rate: float = 0.115
    on_peak_rate:  float = 0.220
    critical_peak_rate: float = 0.520

    on_peak_start_h: float = 14.0   # 2 pm
    on_peak_end_h:   float = 21.0   # 9 pm

    mid_peak_morning_start_h: float = 8.5
    mid_peak_morning_end_h:   float = 14.0
    mid_peak_evening_start_h: float = 21.0
    mid_peak_evening_end_h:   float = 23.0

    critical_peak_prob: float = 0.05  # 5 % of episodes

    # Preset tariffs ──────────────────────────────────────────────────────────

    @classmethod
    def pge_e19(cls) -> "TOUTariff":
        """Approximation of PG&E E-19 commercial TOU tariff (California)."""
        return cls(
            off_peak_rate=0.065,
            mid_peak_rate=0.120,
            on_peak_rate=0.245,
            critical_peak_rate=0.600,
            on_peak_start_h=14.0,
            on_peak_end_h=21.0,
        )

    @classmethod
    def ercot_rtm(cls) -> "TOUTariff":
        """Approximation of ERCOT real-time market average (Texas).
        Lower base rates but higher volatility; handled via noise model."""
        return cls(
            off_peak_rate=0.040,
            mid_peak_rate=0.075,
            on_peak_rate=0.180,
            critical_peak_rate=0.750,
            on_peak_start_h=15.0,
            on_peak_end_h=20.0,
        )

    @classmethod
    def sdge_tou8(cls) -> "TOUTariff":
        """SDG&E TOU-8 commercial tariff (San Diego, higher baseline)."""
        return cls(
            off_peak_rate=0.095,
            mid_peak_rate=0.155,
            on_peak_rate=0.310,
            critical_peak_rate=0.720,
            on_peak_start_h=16.0,
            on_peak_end_h=21.0,
        )

    @classmethod
    def default(cls) -> "TOUTariff":
        """Balanced default suitable for RL benchmarking."""
        return cls()


# ──────────────────────────────────────────────────────────────────────────────
# Main price model
# ──────────────────────────────────────────────────────────────────────────────

class GridElectricityPriceModel:
    """
    Simulated grid electricity price model for data-centre RL environments.

    Combines:
    - Time-of-Use pricing (peak / off-peak / shoulder)
    - Seasonal multiplier (summer / winter)
    - Day-of-week discount (weekends)
    - Ornstein–Uhlenbeck mean-reverting real-time noise (±15 %)
    - Optional renewable surplus credit (solar mid-day dip, California duck curve)
    - Rare Critical Peak Pricing (CPP) events

    Parameters
    ----------
    tariff : TOUTariff | None
        Tariff definition. Defaults to ``TOUTariff.default()``.
    noise_sigma : float
        Volatility of the OU noise process (std of increments).
    noise_theta : float
        Mean-reversion speed of the OU process (higher → faster reversion).
    renewable_credit : bool
        When True, a solar surplus credit is applied during mid-day hours
        on sunny days (25 % of days), reflecting California duck-curve dynamics.
    seed : int | None
        RNG seed for reproducibility.

    Examples
    --------
    >>> model = GridElectricityPriceModel(seed=0)
    >>> model.reset(start_hour=8.0, day_of_year=200)
    >>> for _ in range(12):                      # 1 minute at dt=5 s
    ...     price = model.step(dt_s=5.0)
    ...     print(f"{model.hour_of_day:.3f} h  →  ${price:.4f}/kWh")
    """

    def __init__(
        self,
        tariff: Optional[TOUTariff] = None,
        noise_sigma: float = 0.012,
        noise_theta: float = 0.08,
        renewable_credit: bool = True,
        seed: Optional[int] = None,
    ):
        self.tariff = tariff or TOUTariff.default()
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta
        self.renewable_credit = renewable_credit
        self._rng = np.random.default_rng(seed)

        # State initialised in reset()
        self._hour_of_day: float = 8.0
        self._day_of_year: float = 180.0
        self._day_of_week: int   = 1        # 0 = Mon … 6 = Sun
        self._ou_noise: float    = 0.0
        self._cpp_active: bool   = False
        self._solar_day: bool    = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def hour_of_day(self) -> float:
        """Current simulated hour of day in [0, 24)."""
        return self._hour_of_day

    @property
    def price(self) -> float:
        """Current electricity price in $/kWh."""
        return self._compute_price(self._hour_of_day)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        start_hour: Optional[float] = None,
        day_of_year: Optional[float] = None,
        day_of_week: Optional[int] = None,
    ) -> float:
        """
        Reset the price model to a new episode start.

        Parameters
        ----------
        start_hour : float, optional
            Hour of day at episode start in [0, 24). Random if None.
        day_of_year : float, optional
            Day of year in [0, 365). Random if None.
        day_of_week : int, optional
            Day of week 0=Mon…6=Sun. Random if None.

        Returns
        -------
        float
            Initial electricity price in $/kWh.
        """
        self._hour_of_day = float(
            start_hour if start_hour is not None
            else self._rng.uniform(0.0, 24.0)
        )
        self._day_of_year = float(
            day_of_year if day_of_year is not None
            else self._rng.uniform(0.0, 365.0)
        )
        self._day_of_week = int(
            day_of_week if day_of_week is not None
            else self._rng.integers(0, 7)
        )
        self._ou_noise  = 0.0
        self._cpp_active = self._rng.random() < self.tariff.critical_peak_prob
        self._solar_day  = self.renewable_credit and (self._rng.random() < 0.25)
        return self.price

    def step(self, dt_s: float) -> float:
        """
        Advance the clock by ``dt_s`` seconds and return the new price.

        The OU noise is updated, and the clock wraps at midnight (24 h).

        Parameters
        ----------
        dt_s : float
            Time increment in seconds.

        Returns
        -------
        float
            Updated electricity price in $/kWh.
        """
        # Advance OU noise: dX = -θ·X·dt + σ·√dt·N(0,1)
        dt_h = dt_s / 3600.0
        dW = self._rng.standard_normal()
        self._ou_noise = (
            self._ou_noise * (1.0 - self.noise_theta * dt_h)
            + self.noise_sigma * math.sqrt(dt_h) * dW
        )
        self._ou_noise = float(np.clip(self._ou_noise, -0.25, 0.25))

        # Advance clock (wrap midnight)
        self._hour_of_day = (self._hour_of_day + dt_h) % 24.0

        # New day? Update day-of-week / solar flag
        if self._hour_of_day < dt_h:          # crossed midnight
            self._day_of_week  = (self._day_of_week + 1) % 7
            self._day_of_year  = (self._day_of_year + 1) % 365
            self._cpp_active   = self._rng.random() < self.tariff.critical_peak_prob
            self._solar_day    = self.renewable_credit and (self._rng.random() < 0.25)

        return self.price

    def get_state(self) -> dict:
        """Return the current price model state for logging / info dicts."""
        return {
            "grid_price_per_kWh": self.price,
            "grid_hour_of_day":   self._hour_of_day,
            "grid_day_of_week":   self._day_of_week,
            "grid_cpp_active":    float(self._cpp_active),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tou_rate(self, hour: float, is_weekend: bool) -> float:
        """Return the base TOU rate for the given hour and weekend flag."""
        t = self.tariff
        if is_weekend:
            # Weekends: only off-peak and a mild mid-peak shoulder
            if t.mid_peak_morning_start_h <= hour < t.on_peak_end_h:
                return t.mid_peak_rate
            return t.off_peak_rate

        # Weekday schedule:
        if t.on_peak_start_h <= hour < t.on_peak_end_h:
            return t.on_peak_rate
        if (t.mid_peak_morning_start_h <= hour < t.mid_peak_morning_end_h or
                t.mid_peak_evening_start_h <= hour < t.mid_peak_evening_end_h):
            return t.mid_peak_rate
        return t.off_peak_rate

    def _solar_credit(self, hour: float) -> float:
        """
        California duck-curve renewable surplus credit ($/kWh, always ≤ 0).

        Strong solar generation from 10:00–15:00 depresses wholesale prices;
        on a 'solar day' the mid-day price dips by up to $0.06/kWh.
        """
        if not self._solar_day:
            return 0.0
        # Gaussian centred at noon with σ = 1.5 h, max depth = -0.06 $/kWh
        return -0.06 * math.exp(-0.5 * ((hour - 12.0) / 1.5) ** 2)

    def _compute_price(self, hour: float) -> float:
        """Full price calculation for the current state."""
        is_weekend = self._day_of_week >= 5   # Sat or Sun

        base = self._tou_rate(hour, is_weekend)

        # Critical peak override
        if self._cpp_active and self.tariff.on_peak_start_h <= hour < self.tariff.on_peak_end_h:
            base = self.tariff.critical_peak_rate

        # Seasonal multiplier
        seasonal = _seasonal_multiplier(self._day_of_year)

        # Renewable credit (mid-day solar surplus)
        solar = self._solar_credit(hour)

        # OU noise is a *fractional* perturbation
        price = max(0.001, base * seasonal * (1.0 + self._ou_noise) + solar)
        return round(price, 6)
