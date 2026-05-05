"""
compopt.envs.rack_env
=====================
Gymnasium environment for rack-level liquid-cooling control.

**Difficulty level: Medium** — single rack with multiple GPUs.

Action: normalised rack coolant flow [0, 1].
Observation: 10-element vector (GPU0 sensors + rack telemetry).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from compopt.physics.server import RackModel, build_default_rack
from compopt.rewards.functions import CompositeReward, cooling_only_reward


class RackCoolingEnv(gym.Env):
    """
    Gymnasium env for rack-level liquid-cooling control.

    Compatible with the original ``rack_env.py`` interface but with
    modular rewards and richer info dicts.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 rack: Optional[RackModel] = None,
                 dt: float               = 1.0,
                 episode_length_s: float = 1800.0,
                 m_dot_min: float        = 0.5,
                 m_dot_max: float        = 4.0,
                 target_hotspot_C: float = 80.0,
                 reward_fn: Optional[CompositeReward] = None,
                 render_mode: Optional[str] = None,
                 n_servers: int          = 4,
                 gpus_per_server: int    = 1,
                 gpu_preset: str         = "H100_SXM",
                 workload_profile: str   = "sinusoidal",
                 workload_period_s: float = 300.0):
        super().__init__()

        self.dt               = dt
        self.episode_length_s = episode_length_s
        self.m_dot_min        = m_dot_min
        self.m_dot_max        = m_dot_max
        self.target_C         = target_hotspot_C
        self.reward_fn        = reward_fn or cooling_only_reward(target_hotspot_C)
        self.render_mode      = render_mode
        self._n_servers       = n_servers
        self._gpus_per_server = gpus_per_server
        self._gpu_preset      = gpu_preset
        self._workload_profile = workload_profile
        self._workload_period_s = workload_period_s

        self.rack = rack or build_default_rack(
            n_servers=n_servers, gpus_per_server=gpus_per_server,
            gpu_preset=gpu_preset, workload=workload_profile,
            workload_period_s=workload_period_s)

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32))

        self.observation_space = spaces.Box(
            low=np.zeros(10, dtype=np.float32),
            high=np.full(10, 500.0, dtype=np.float32))

        self._elapsed_s = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rack = build_default_rack(
            n_servers=self._n_servers,
            gpus_per_server=self._gpus_per_server,
            gpu_preset=self._gpu_preset,
            workload=self._workload_profile,
            workload_period_s=self._workload_period_s)
        self.rack.rack_coolant.m_dot_kg_s = 0.5 * (self.m_dot_min + self.m_dot_max)
        self._elapsed_s = 0.0
        obs = self.rack.get_rack_observation().astype(np.float32)
        return obs, {}

    def step(self, action):
        """Advance one time-step with the given rack coolant flow-rate action.

        Args:
            action: Array of shape ``(1,)`` in ``[0, 1]``, mapped to
                ``[m_dot_min, m_dot_max]``.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        a = float(np.clip(action[0], 0.0, 1.0))
        self.rack.rack_coolant.m_dot_kg_s = (
            self.m_dot_min + a * (self.m_dot_max - self.m_dot_min))
        self.rack.step(self.dt)
        self._elapsed_s += self.dt

        obs = self.rack.get_rack_observation().astype(np.float32)

        # Compute cooling power from pump work
        # Pump power scales with flow^3 (affinity laws)
        # P_pump = k * flow_rate^3
        P_IT_W = float(obs[9])  # GPU + system power
        flow_normalized = (self.rack.rack_coolant.m_dot_kg_s - self.m_dot_min) / (self.m_dot_max - self.m_dot_min)
        # Base pump power at max flow is ~5% of IT power
        # At lower flows, it scales with flow^3
        P_pump_base_W = P_IT_W * 0.05
        P_pump_W = P_pump_base_W * (flow_normalized ** 3 + 0.1)  # Add minimum for static losses
        P_total_facility_W = P_IT_W + P_pump_W
        
        # Compute PUE (Power Usage Effectiveness)
        pue = P_total_facility_W / P_IT_W if P_IT_W > 0 else 1.0
        
        # Compute WUE (Water Usage Effectiveness) - for liquid cooling
        # Assume negligible evaporative water loss in direct liquid cooling
        # WUE in L/kWh - for closed-loop liquid cooling this is very low
        wue = 0.01  # Minimal water usage for closed-loop system

        info = {
            "T_hotspot_C":   float(obs[1]),
            "T_gpu_hotspot_C": float(obs[1]),
            "T_hbm_C":      float(obs[2]),
            "flow_kg_s":    self.rack.rack_coolant.m_dot_kg_s,
            "m_dot_kg_s":   self.rack.rack_coolant.m_dot_kg_s,
            "P_total_W":    float(obs[9]),
            "P_IT_W":       P_IT_W,
            "P_cooling_W":  P_pump_W,
            "pue":          pue,
            "wue":          wue,
        }

        reward, breakdown = self.reward_fn(info)
        info["reward_breakdown"] = breakdown

        truncated = self._elapsed_s >= self.episode_length_s
        return obs, float(reward), False, truncated, info

    def render(self):
        if self.render_mode == "human":
            tele = self.rack.get_rack_telemetry()
            print(f"t={tele['time_s']:.0f}s  "
                  f"T_hot={tele['rack_T_hotspot_C']:.1f}°C  "
                  f"P={tele['rack_total_power_W']:.0f}W  "
                  f"flow={self.rack.rack_coolant.m_dot_kg_s:.2f} kg/s")
