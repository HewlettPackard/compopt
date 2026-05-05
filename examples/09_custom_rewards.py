#!/usr/bin/env python3
"""
CompOpt Example 09 — Custom Reward Function
============================================
Shows how to compose custom multi-objective reward functions from
CompOpt's modular reward components.

Run:
    python examples/09_custom_rewards.py
"""

import numpy as np
import compopt
from compopt.rewards.functions import (
    CompositeReward,
    ThermalPenalty,
    ThermalViolation,
    HBMPenalty,
    EnergyPenalty,
    CoolantFlowPenalty,
    PUEReward,
    CostPenalty,
    WaterPenalty,
)
from compopt.agents.baselines import PIDCoolingAgent
from compopt.utils.metrics import EpisodeMetrics, print_metrics

# ── Define custom reward functions ────────────────────────────────────────

# Reward 1: Safety-focused (heavy thermal penalty, strict violation)
safety_reward = CompositeReward([
    ThermalPenalty(weight=5.0, target_C=75.0),       # stricter target
    ThermalViolation(weight=500.0, limit_C=83.0),    # big violation penalty
    HBMPenalty(weight=2.0, limit_C=90.0),
    CoolantFlowPenalty(weight=0.001, m_dot_min=0.05),
])

# Reward 2: Efficiency-focused (energy + cost + PUE)
efficiency_reward = CompositeReward([
    ThermalPenalty(weight=0.5, target_C=80.0),        # mild thermal
    ThermalViolation(weight=50.0, limit_C=83.0),
    EnergyPenalty(weight=0.1, P_ref_kW=1.0),
    CoolantFlowPenalty(weight=0.05, m_dot_min=0.05),
])

# Reward 3: Balanced multi-objective
balanced_reward = CompositeReward([
    ThermalPenalty(weight=1.0, target_C=80.0),
    ThermalViolation(weight=100.0, limit_C=83.0),
    HBMPenalty(weight=0.5, limit_C=95.0),
    EnergyPenalty(weight=0.01, P_ref_kW=10.0),
    CoolantFlowPenalty(weight=0.01, m_dot_min=0.05),
])

# ── Evaluate each reward on the same environment ─────────────────────────
rewards = {
    "Safety":     safety_reward,
    "Efficiency": efficiency_reward,
    "Balanced":   balanced_reward,
}

agent = PIDCoolingAgent(target_C=80.0, Kp=0.05, Ki=0.001, Kd=0.01, dt=1.0)

results = {}
for name, reward_fn in rewards.items():
    env = compopt.make("ChipThermal-v0",
                      dt=1.0,
                      episode_length_s=600.0,
                      reward_fn=reward_fn)

    metrics = EpisodeMetrics()
    obs, _ = env.reset()
    agent.reset()

    for step in range(600):
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward, time_s=step)
        if terminated or truncated:
            break

    summary = metrics.summary()
    results[name] = summary
    env.close()

# ── Show reward breakdowns ───────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Custom Reward Function Comparison (same PID agent)")
print(f"{'='*70}")
print(f"{'Reward':15s} | {'Total Reward':>14s} | {'T_max (°C)':>10s} | {'Violations':>10s}")
print(f"{'-'*70}")
for name, s in results.items():
    print(f"{name:15s} | {s['reward_total']:14.1f} | "
          f"{s['T_hotspot_max']:10.1f} | {s['thermal_violation_rate']:10.3f}")
print(f"{'='*70}")

# ── Show per-step reward breakdown for last episode ──────────────────────
print("\nDetailed breakdown (last step of Balanced reward):")
env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=10.0,
                  reward_fn=balanced_reward)
obs, _ = env.reset()
agent.reset()
for _ in range(10):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

bd = info.get("reward_breakdown", {})
print(f"  Step reward: {reward:.4f}")
for comp_name, val in bd.items():
    print(f"    {comp_name:30s}: {val:+.4f}")
env.close()
