"""
compopt.utils.metrics
=====================
Evaluation metrics and logging utilities for CompOpt benchmarks.

Computes and records:
- Thermal safety metrics (violation rate, mean/max hotspot)
- Energy efficiency (PUE, kWh)
- Water efficiency (WUE, total litres)
- Cost ($)
- Scheduling metrics (throughput, utilisation, wait times)
- Reward statistics
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EpisodeMetrics:
    """Accumulates metrics over a single episode."""
    T_hotspot_history:  List[float] = field(default_factory=list)
    reward_history:     List[float] = field(default_factory=list)
    flow_history:       List[float] = field(default_factory=list)
    power_history:      List[float] = field(default_factory=list)
    pue_history:        List[float] = field(default_factory=list)
    wue_history:        List[float] = field(default_factory=list)
    cost_history:       List[float] = field(default_factory=list)
    util_history:       List[float] = field(default_factory=list)
    time_history:       List[float] = field(default_factory=list)

    n_thermal_violations: int = 0
    total_water_L:        float = 0.0
    total_cost_dollar:    float = 0.0
    total_jobs_completed: int = 0

    def record(self, info: Dict[str, float], reward: float, time_s: float = 0.0):
        """Record one step's info dict."""
        self.T_hotspot_history.append(info.get("T_hotspot_C",
                                               info.get("T_gpu_hotspot_C", 0.0)))
        self.reward_history.append(reward)
        self.flow_history.append(info.get("flow_kg_s",
                                          info.get("m_dot_kg_s", 0.0)))
        self.power_history.append(info.get("total_power_kW",
                                           info.get("P_total_W", 0.0) / 1000.0))
        self.pue_history.append(info.get("PUE", 1.0))
        self.wue_history.append(info.get("WUE_L_per_kWh", 0.0))
        self.cost_history.append(info.get("step_cost_dollar", 0.0))
        self.util_history.append(info.get("node_utilisation", 0.0))
        self.time_history.append(time_s)

        if info.get("T_hotspot_C", info.get("T_gpu_hotspot_C", 0.0)) > 83.0:
            self.n_thermal_violations += 1
        self.total_water_L    += info.get("water_used_L", 0.0)
        self.total_cost_dollar += info.get("step_cost_dollar", 0.0)
        self.total_jobs_completed += int(info.get("jobs_completed_step", 0))

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        T = np.array(self.T_hotspot_history) if self.T_hotspot_history else np.array([0.0])
        R = np.array(self.reward_history) if self.reward_history else np.array([0.0])
        n = max(1, len(self.T_hotspot_history))
        return {
            "episode_length":      n,
            "T_hotspot_mean":      float(np.mean(T)),
            "T_hotspot_max":       float(np.max(T)),
            "T_hotspot_std":       float(np.std(T)),
            "thermal_violation_rate": self.n_thermal_violations / n,
            "reward_total":        float(np.sum(R)),
            "reward_mean":         float(np.mean(R)),
            "PUE_mean":            float(np.mean(self.pue_history)) if self.pue_history else 1.0,
            "WUE_mean":            float(np.mean(self.wue_history)) if self.wue_history else 0.0,
            "total_water_L":       self.total_water_L,
            "total_cost_dollar":   self.total_cost_dollar,
            "total_energy_kWh":    float(np.sum(self.power_history)) * (1.0 / 3600.0) if self.power_history else 0.0,
            "node_util_mean":      float(np.mean(self.util_history)) if self.util_history else 0.0,
            "total_jobs_completed": self.total_jobs_completed,
        }


def evaluate_agent(env, agent, n_episodes: int = 5,
                   max_steps: int = 10000) -> Dict[str, float]:
    """
    Run an agent on an environment for multiple episodes and return
    aggregated metrics.

    Parameters
    ----------
    env       : gymnasium.Env
    agent     : object with .predict(obs) → (action, info) and .reset()
    n_episodes: number of evaluation episodes
    max_steps : max steps per episode

    Returns
    -------
    dict of averaged metrics across episodes
    """
    all_summaries = []
    for ep in range(n_episodes):
        metrics = EpisodeMetrics()
        agent.reset()
        obs, _ = env.reset()
        for step in range(max_steps):
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            metrics.record(info, reward, time_s=step)
            if terminated or truncated:
                break
        all_summaries.append(metrics.summary())

    # Average across episodes
    keys = all_summaries[0].keys()
    avg = {}
    for k in keys:
        vals = [s[k] for s in all_summaries]
        avg[k] = float(np.mean(vals))
        avg[f"{k}_std"] = float(np.std(vals))
    avg["n_episodes"] = n_episodes
    return avg


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation"):
    """Pretty-print a metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if not k.endswith("_std"):
            std_key = f"{k}_std"
            if std_key in metrics:
                print(f"  {k:30s}: {v:10.3f}  ± {metrics[std_key]:.3f}")
            else:
                print(f"  {k:30s}: {v:10.3f}")
    print(f"{'='*60}\n")
