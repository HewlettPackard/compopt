"""
compopt.envs.chip_env
=====================
Gymnasium environment for single-chip thermal control.

**Difficulty level: Easy** — ideal for getting started with RL cooling control.

Action: continuous flow rate normalised to [0, 1].
Observation: 7-element vector (GPU temps, coolant, power).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from compopt.physics.chip import GPUChipModel, make_gpu
from compopt.physics.fluids import CoolantLoop
from compopt.physics.workloads import make_sinusoidal_profile
from compopt.rewards.functions import CompositeReward, cooling_only_reward


class ChipThermalEnv(gym.Env):
    """
    Gymnasium env for single GPU chip thermal management.

    Action space : Box(1,) in [0, 1] — normalised coolant flow
    Obs space    : Box(7,) — [T_mean, T_hotspot, T_hbm, T_vrm, T_cool_in, T_cool_out, P_total]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 gpu_preset: str         = "H100_SXM",
                 workload: str           = "sinusoidal",
                 workload_period_s: float = 180.0,
                 dt: float               = 1.0,
                 episode_length_s: float = 600.0,
                 m_dot_min: float        = 0.05,
                 m_dot_max: float        = 0.5,
                 target_hotspot_C: float = 80.0,
                 reward_fn: Optional[CompositeReward] = None,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.gpu_preset      = gpu_preset
        self.workload        = workload
        self.workload_period_s = workload_period_s
        self.dt              = dt
        self.episode_length_s = episode_length_s
        self.m_dot_min       = m_dot_min
        self.m_dot_max       = m_dot_max
        self.target_C        = target_hotspot_C
        self.reward_fn       = reward_fn or cooling_only_reward(target_hotspot_C)
        self.render_mode     = render_mode

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32))

        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.full(7, 300.0, dtype=np.float32))

        self._chip: Optional[GPUChipModel] = None
        self._elapsed_s = 0.0

    def _make_chip(self) -> GPUChipModel:
        from compopt.physics.workloads import (
            make_sinusoidal_profile, make_step_profile,
            make_burst_profile, make_stochastic_profile)

        nx, ny = 4, 4
        period = self.workload_period_s
        profile_fns = {
            "sinusoidal": lambda: make_sinusoidal_profile(nx, ny, period_s=period),
            "step":       lambda: make_step_profile(nx, ny),
            "burst":      lambda: make_burst_profile(nx, ny),
            "stochastic": lambda: make_stochastic_profile(nx, ny),
        }
        profile = profile_fns.get(self.workload,
                                   profile_fns["sinusoidal"])()
        # Use smaller coolant capacity for faster thermal response and
        # larger temperature swings with flow rate changes
        coolant = CoolantLoop(T_in_C=40.0, m_dot_kg_s=0.25, c_p_J_kgK=800.0)
        return make_gpu(self.gpu_preset,
                        power_profile=profile,
                        coolant_loop=coolant)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._chip = self._make_chip()
        self._chip.coolant_loop.m_dot_kg_s = 0.5 * (self.m_dot_min + self.m_dot_max)
        self._elapsed_s = 0.0
        obs = self._chip.get_observation().astype(np.float32)
        return obs, {}

    def step(self, action):
        """Advance one time-step with the given coolant flow-rate action.

        Args:
            action: Array of shape ``(1,)`` in ``[0, 1]``, mapped to
                ``[m_dot_min, m_dot_max]``.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        a = float(np.clip(action[0], 0.0, 1.0))
        self._chip.coolant_loop.m_dot_kg_s = (
            self.m_dot_min + a * (self.m_dot_max - self.m_dot_min))
        self._chip.step(self.dt)
        self._elapsed_s += self.dt

        obs = np.nan_to_num(self._chip.get_observation(), nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.clip(obs, -1e6, 1e6).astype(np.float32)

        # Compute cooling power from pump work (simplified estimate)
        P_IT_W = float(obs[6])  # GPU power
        P_pump_W = P_IT_W * 0.03  # ~3% pump overhead for chip-level cooling
        P_total_facility_W = P_IT_W + P_pump_W
        
        # Compute PUE (Power Usage Effectiveness)
        pue = P_total_facility_W / P_IT_W if P_IT_W > 0 else 1.0
        
        # Compute WUE (Water Usage Effectiveness) - for liquid cooling
        wue = 0.01  # Minimal water usage for closed-loop chip cooling

        info = {
            "T_hotspot_C":   float(obs[1]),
            "T_gpu_hotspot_C": float(obs[1]),
            "T_hbm_C":      float(obs[2]),
            "flow_kg_s":    self._chip.coolant_loop.m_dot_kg_s,
            "m_dot_kg_s":   self._chip.coolant_loop.m_dot_kg_s,
            "P_total_W":    float(obs[6]),
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
        if self.render_mode == "human" and self._chip is not None:
            d = self._chip.get_sensors_dict()
            print(f"t={d['time_s']:.1f}s  T_hot={d['T_gpu_hotspot_C']:.1f}°C  "
                  f"P={d['P_total_W']:.0f}W  "
                  f"flow={self._chip.coolant_loop.m_dot_kg_s:.3f} kg/s")
