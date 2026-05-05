"""
compopt.physics.fluids
======================
Thermo-fluid building blocks: coolant loops, CDU, cooling towers, water models.

All classes use forward-Euler or RK4 time integration and support
**vectorised (batched) operation** via NumPy broadcasting for parallel
simulation of many identical plants (e.g. for RL vectorised envs).

Units
-----
Temperature: °C   |  Power: W   |  Flow: kg/s   |  Volume: m³
Resistance: K/W   |  Capacitance: J/K
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# CoolantLoop — lumped single-node coolant control volume
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CoolantLoop:
    """
    Lumped coolant control volume with inlet, advective outflow, and heat source.

    Energy balance:
        C_th · dT/dt = Q_in − ṁ · c_p · (T − T_in)

    Parameters
    ----------
    T_in_C      : Inlet temperature [°C]
    m_dot_kg_s  : Mass flow rate [kg/s]
    c_p_J_kgK   : Specific heat capacity [J/(kg·K)]
    V_m3        : Control volume [m³]
    rho_kg_m3   : Coolant density [kg/m³]
    """
    T_in_C:      float = 40.0
    m_dot_kg_s:  float = 0.25
    c_p_J_kgK:   float = 4000.0
    V_m3:        float = 1.5e-4
    rho_kg_m3:   float = 1000.0

    # State (auto-initialised)
    T_coolant_C: float = field(init=False)
    C_th_J_K:    float = field(init=False)

    def __post_init__(self):
        self.T_coolant_C = self.T_in_C
        self.C_th_J_K    = self.V_m3 * self.rho_kg_m3 * self.c_p_J_kgK

    def reset(self):
        """Reset coolant temperature to current inlet value."""
        self.T_coolant_C = self.T_in_C

    def step(self, dt: float, Q_in_W: float):
        """Semi-implicit advance by *dt* seconds (unconditionally stable)."""
        # dT/dt = (Q + mdot*cp*(Tin - T)) / C  →  implicit in T:
        # T_new = (T_old + dt/C*(Q + mdot*cp*Tin)) / (1 + dt/C*mdot*cp)
        m_cp = self.m_dot_kg_s * self.c_p_J_kgK
        a    = dt * m_cp / self.C_th_J_K
        new_T = (
            self.T_coolant_C + dt * (Q_in_W + m_cp * self.T_in_C) / self.C_th_J_K
        ) / (1.0 + a)
        # Clamp to prevent overflow and unrealistic temperatures
        self.T_coolant_C = float(np.clip(np.nan_to_num(new_T, nan=self.T_in_C, posinf=1e4, neginf=-1e4), -273.15, 1e4))

    @property
    def T_out_C(self) -> float:
        """Outlet temperature (= well-mixed tank temperature)."""
        return self.T_coolant_C


# ──────────────────────────────────────────────────────────────────────────────
# CDU — Coolant Distribution Unit
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CDU:
    """
    Coolant Distribution Unit model (rack-to-facility heat exchanger).

    The CDU transfers heat from the *hot* (rack-side) loop to the *cold*
    (facility-side) loop through a counter-flow heat exchanger characterised
    by an overall UA value.

    Energy balance (hot side, lumped):
        C_hot · dT_hot/dt = Q_rack_total − UA · (T_hot − T_cold) − ṁ_hot · c_p · (T_hot − T_hot,in)

    Facility side is assumed much larger; its temperature responds slower
    (modelled with a separate CoolantLoop or held constant).

    Parameters
    ----------
    UA_W_K          : Overall heat-transfer coefficient × area [W/K]
    T_facility_C    : Facility-side (cold) supply temperature [°C]
    pump_power_W_per_kgs : Pump power per unit flow [W/(kg/s)]
    """
    UA_W_K:              float = 5000.0    # scaled for smaller heat loads
    T_facility_C:        float = 30.0
    pump_power_W_per_kgs: float = 500.0    # pumping parasitic

    # Larger thermal masses for numerical stability
    hot_loop:  CoolantLoop = field(default_factory=lambda: CoolantLoop(
        T_in_C=40.0, m_dot_kg_s=2.0, V_m3=0.05, rho_kg_m3=1000.0))
    cold_loop: CoolantLoop = field(default_factory=lambda: CoolantLoop(
        T_in_C=30.0, m_dot_kg_s=4.0, V_m3=0.10, rho_kg_m3=1000.0))

    def reset(self):
        self.hot_loop.reset()
        self.cold_loop.reset()

    def step(self, dt: float, Q_rack_total_W: float):
        """Advance CDU dynamics by dt seconds."""
        T_h = self.hot_loop.T_coolant_C
        T_c = self.cold_loop.T_coolant_C
        Q_hx = self.UA_W_K * (T_h - T_c)
        # hot side sees rack heat in, loses to HX
        self.hot_loop.step(dt, Q_in_W=Q_rack_total_W - Q_hx)
        # cold side gains from HX, loses to facility
        self.cold_loop.T_in_C = self.T_facility_C
        self.cold_loop.step(dt, Q_in_W=Q_hx)

    @property
    def pump_power_W(self) -> float:
        return self.pump_power_W_per_kgs * (
            self.hot_loop.m_dot_kg_s + self.cold_loop.m_dot_kg_s)

    @property
    def T_supply_to_racks_C(self) -> float:
        return self.hot_loop.T_in_C

    @property
    def T_return_from_racks_C(self) -> float:
        return self.hot_loop.T_coolant_C


# ──────────────────────────────────────────────────────────────────────────────
# CoolingTower — evaporative / dry cooling rejection to ambient
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CoolingTower:
    """
    Simplified cooling tower / dry cooler model.

    Energy balance:
        C_basin · dT_basin/dt = Q_facility − η · ṁ_air · c_p,air · (T_basin − T_amb)

    Water consumption is proportional to evaporative heat rejection.

    Parameters
    ----------
    T_ambient_C      : Outdoor dry-bulb temperature [°C]
    T_wetbulb_C      : Outdoor wet-bulb temperature [°C]
    fan_UA_W_K       : Effective air-side UA for fan operation [W/K]
    basin_C_J_K      : Basin thermal capacitance [J/K]
    evap_fraction    : Fraction of heat rejected via evaporation (0–1)
    latent_heat_J_kg : Latent heat of vaporisation of water [J/kg]
    fan_power_W      : Parasitic fan power [W]
    """
    T_ambient_C:      float = 25.0
    T_wetbulb_C:      float = 18.0
    fan_UA_W_K:       float = 3000.0     # scaled down for smaller heat loads
    basin_C_J_K:      float = 2000000.0  # larger basin for stability (2 MJ/K)
    evap_fraction:    float = 0.75
    latent_heat_J_kg: float = 2.26e6
    fan_power_W:      float = 2000.0     # scaled down

    T_basin_C:     float = field(init=False)
    water_used_L:  float = field(init=False, default=0.0)

    def __post_init__(self):
        self.T_basin_C = self.T_ambient_C + 5.0

    def reset(self):
        self.T_basin_C    = self.T_ambient_C + 5.0
        self.water_used_L = 0.0

    def step(self, dt: float, Q_facility_W: float):
        """Advance tower dynamics."""
        # Effective rejection temperature is between ambient and wet-bulb
        T_eff = self.evap_fraction * self.T_wetbulb_C + \
                (1.0 - self.evap_fraction) * self.T_ambient_C
        Q_reject = self.fan_UA_W_K * (self.T_basin_C - T_eff)
        dT = dt * (Q_facility_W - Q_reject) / self.basin_C_J_K
        self.T_basin_C += dT

        # Water consumption: evaporative portion
        Q_evap = Q_reject * self.evap_fraction
        water_kg = max(0.0, Q_evap * dt / self.latent_heat_J_kg)
        self.water_used_L += water_kg  # ~1 kg ≈ 1 L for water

    @property
    def T_supply_C(self) -> float:
        """Temperature supplied to facility loop."""
        return self.T_basin_C


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised (batched) CoolantLoop for parallel environments
# ──────────────────────────────────────────────────────────────────────────────

class BatchCoolantLoop:
    """
    Vectorised coolant loop operating on (B,) shaped arrays,
    enabling parallel simulation of B environments simultaneously.

    All state is stored as NumPy arrays of shape (batch_size,).
    """

    def __init__(self, batch_size: int,
                 T_in_C: float = 40.0,
                 m_dot_kg_s: float = 0.25,
                 c_p_J_kgK: float = 4000.0,
                 V_m3: float = 1.5e-4,
                 rho_kg_m3: float = 1000.0):
        self.batch_size = batch_size
        self.c_p        = c_p_J_kgK
        self.C_th       = V_m3 * rho_kg_m3 * c_p_J_kgK

        self.T_in       = np.full(batch_size, T_in_C, dtype=np.float64)
        self.m_dot      = np.full(batch_size, m_dot_kg_s, dtype=np.float64)
        self.T_coolant  = np.full(batch_size, T_in_C, dtype=np.float64)

    def reset(self, mask: Optional[np.ndarray] = None):
        """Reset coolant temps. If *mask* given, only reset those indices."""
        if mask is None:
            self.T_coolant[:] = self.T_in
        else:
            self.T_coolant[mask] = self.T_in[mask]

    def step(self, dt: float, Q_in_W: np.ndarray):
        """Vectorised semi-implicit step (unconditionally stable)."""
        m_cp = self.m_dot * self.c_p
        a = dt * m_cp / self.C_th
        self.T_coolant = (
            self.T_coolant + dt * (Q_in_W + m_cp * self.T_in) / self.C_th
        ) / (1.0 + a)
