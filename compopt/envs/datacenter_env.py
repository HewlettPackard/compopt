"""
compopt.envs.datacenter_env
===========================
Gymnasium environment for full data-center control.

**Difficulty level: Hard** — joint cooling + scheduling optimisation.

Actions (multi-dimensional):
    [0]  rack coolant flow normalised [0, 1]  (applied to all racks)
    [1]  CDU hot-side pump flow normalised [0, 1]
    [2]  cooling tower fan speed normalised [0, 1]

Observation: 16-element data-center state vector (17 elements when
``grid_electricity=True``, with grid price appended as the last entry).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from compopt.physics.server import DataCenterModel, build_default_datacenter
from compopt.physics.electricity import GridElectricityPriceModel, TOUTariff
from compopt.rewards.functions import CompositeReward, datacenter_reward


class DataCenterEnv(gym.Env):
    """
    Full data-center Gymnasium environment.

    Co-optimises cooling (rack flow, CDU pump, tower fan) to minimise
    energy, water, cost while keeping GPUs below thermal limits.

    Grid-electricity mode
    ---------------------
    When ``grid_electricity=True`` the environment integrates a realistic
    :class:`~compopt.physics.electricity.GridElectricityPriceModel`.  The
    current tariff price ($/kWh) is appended as the 17th observation
    element and exposed in the ``info`` dict under ``"grid_price_per_kWh"``.
    Use ``grid_aware_datacenter_reward()`` to reward load-shifting behaviour.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 dc: Optional[DataCenterModel] = None,
                 dt: float               = 5.0,
                 episode_length_s: float = 3600.0,
                 reward_fn: Optional[CompositeReward] = None,
                 render_mode: Optional[str] = None,
                 n_racks: int            = 2,
                 servers_per_rack: int   = 4,
                 gpus_per_server: int    = 2,
                 gpu_preset: str         = "H100_SXM",
                 T_ambient_C: float      = 25.0,
                 workload_profile: str   = "sinusoidal",
                 workload_period_s: float = 300.0,
                 # Flow ranges
                 rack_flow_min: float    = 0.5,
                 rack_flow_max: float    = 4.0,
                 cdu_flow_min: float     = 1.0,
                 cdu_flow_max: float     = 8.0,
                 fan_power_min: float    = 5000.0,
                 fan_power_max: float    = 30000.0,
                 # Grid electricity
                 grid_electricity: bool  = False,
                 grid_tariff: Optional[TOUTariff] = None,
                 grid_start_hour: Optional[float] = None,
                 grid_day_of_year: Optional[float] = None,
                 grid_seed: Optional[int] = None):
        super().__init__()

        self.dt               = dt
        self.episode_length_s = episode_length_s
        self.reward_fn        = reward_fn or datacenter_reward()
        self.render_mode      = render_mode

        self._n_racks          = n_racks
        self._servers_per_rack = servers_per_rack
        self._gpus_per_server  = gpus_per_server
        self._gpu_preset       = gpu_preset
        self._T_ambient_C      = T_ambient_C
        self._workload_profile = workload_profile
        self._workload_period_s = workload_period_s

        self.rack_flow_min  = rack_flow_min
        self.rack_flow_max  = rack_flow_max
        self.cdu_flow_min   = cdu_flow_min
        self.cdu_flow_max   = cdu_flow_max
        self.fan_power_min  = fan_power_min
        self.fan_power_max  = fan_power_max

        # Grid electricity price model
        self.grid_electricity   = grid_electricity
        self._grid_start_hour   = grid_start_hour
        self._grid_day_of_year  = grid_day_of_year
        self._grid_price_model: Optional[GridElectricityPriceModel] = None
        if grid_electricity:
            self._grid_price_model = GridElectricityPriceModel(
                tariff=grid_tariff or TOUTariff.default(),
                seed=grid_seed,
            )

        self.dc = dc or build_default_datacenter(
            n_racks=n_racks,
            servers_per_rack=servers_per_rack,
            gpus_per_server=gpus_per_server,
            gpu_preset=gpu_preset,
            T_ambient_C=T_ambient_C,
            workload=workload_profile,
            workload_period_s=workload_period_s)

        # 3 continuous actions: rack_flow, cdu_pump, tower_fan
        self.action_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32))

        # Observation: 16 base elements + 1 grid-price element (if enabled)
        obs_dim = 17 if grid_electricity else 16
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -50.0, dtype=np.float32),
            high=np.full(obs_dim, 1e6, dtype=np.float32))

        self._elapsed_s  = 0.0
        self._prev_cost  = 0.0
        self._prev_water = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.dc = build_default_datacenter(
            n_racks=self._n_racks,
            servers_per_rack=self._servers_per_rack,
            gpus_per_server=self._gpus_per_server,
            gpu_preset=self._gpu_preset,
            T_ambient_C=self._T_ambient_C,
            workload=self._workload_profile,
            workload_period_s=self._workload_period_s)
        self._elapsed_s  = 0.0
        self._prev_cost  = 0.0
        self._prev_water = 0.0

        # Reset grid electricity price model
        if self._grid_price_model is not None:
            self._grid_price_model.reset(
                start_hour=self._grid_start_hour,
                day_of_year=self._grid_day_of_year,
            )

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector (16 or 17 elements)."""
        obs = np.nan_to_num(
            self.dc.get_observation(), nan=0.0, posinf=1e6, neginf=-1e6
        )
        obs = np.clip(obs, -1e6, 1e6).astype(np.float32)
        if self._grid_price_model is not None:
            price = np.float32(self._grid_price_model.price)
            obs = np.append(obs, price)
        return obs

    def step(self, action):
        """Advance one time-step with cooling actions for racks, CDU, and tower.

        Args:
            action: Array of shape ``(3,)`` in ``[0, 1]`` controlling
                rack flow, CDU set-point, and tower fan speed.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.clip(action, 0.0, 1.0)

        # Apply actions
        # [0] rack coolant flow (all racks)
        rack_flow = self.rack_flow_min + action[0] * (
            self.rack_flow_max - self.rack_flow_min)
        for rack in self.dc.racks:
            rack.rack_coolant.m_dot_kg_s = rack_flow

        # [1] CDU hot-side pump
        cdu_flow = self.cdu_flow_min + action[1] * (
            self.cdu_flow_max - self.cdu_flow_min)
        self.dc.cdu.hot_loop.m_dot_kg_s = cdu_flow

        # [2] Cooling tower fan
        fan_power = self.fan_power_min + action[2] * (
            self.fan_power_max - self.fan_power_min)
        self.dc.cooling_tower.fan_power_W = fan_power

        # Advance
        self.dc.step(self.dt)
        self._elapsed_s += self.dt

        # Advance grid electricity price model
        if self._grid_price_model is not None:
            self._grid_price_model.step(self.dt)

        obs = self._get_obs()

        step_cost  = self.dc.total_cost_dollar - self._prev_cost
        step_water = self.dc.cooling_tower.water_used_L - self._prev_water
        self._prev_cost  = self.dc.total_cost_dollar
        self._prev_water = self.dc.cooling_tower.water_used_L

        info = {
            "T_hotspot_C":       float(obs[0]),
            "T_gpu_hotspot_C":   float(obs[0]),
            "PUE":               float(obs[10]),
            "WUE_L_per_kWh":     float(obs[11]),
            "total_power_kW":    float(obs[8] + obs[9]),
            "IT_power_kW":       float(obs[8]),
            "step_cost_dollar":  step_cost,
            "water_used_L":      step_water,
            "flow_kg_s":         rack_flow,
            "m_dot_kg_s":        rack_flow,
        }

        # Inject grid electricity state into info dict
        if self._grid_price_model is not None:
            info.update(self._grid_price_model.get_state())
            # Compute electricity cost this step: P [kW] × price [$/kWh] × dt [h]
            dt_h = self.dt / 3600.0
            info["grid_electricity_cost_dollar"] = (
                info["total_power_kW"] * info["grid_price_per_kWh"] * dt_h
            )

        reward, breakdown = self.reward_fn(info)
        info["reward_breakdown"] = breakdown

        truncated = self._elapsed_s >= self.episode_length_s
        return obs, float(reward), False, truncated, info

    def render(self):
        if self.render_mode == "human":
            state = self.dc.get_full_state_for_llm()
            dc = state["datacenter"]
            grid_str = ""
            if self._grid_price_model is not None:
                p = self._grid_price_model
                grid_str = (f"  grid=${p.price:.4f}/kWh"
                            f"  h={p.hour_of_day:.1f}")
            print(f"t={dc['time_s']:.0f}s  PUE={dc['PUE']:.3f}  "
                  f"T_hot={dc['T_hotspot_C']:.1f}°C  "
                  f"P={dc['total_power_kW']:.1f}kW  "
                  f"water={dc['water_used_L']:.1f}L  "
                  f"cost=${dc['cost_dollar']:.4f}{grid_str}")
