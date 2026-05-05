#!/usr/bin/env python3
"""
CompOpt Example 03 — Full Data-Center Environment
==================================================
Demonstrates the DataCenter-v0 environment with multi-dimensional
cooling control (rack flow + CDU pump + cooling tower fan).

Run:
    python examples/03_datacenter.py
"""

import numpy as np
import compopt
from compopt.agents.baselines import DataCenterRuleAgent, RandomAgent
from compopt.utils.metrics import EpisodeMetrics, print_metrics

# ── Create environment ────────────────────────────────────────────────────
env = compopt.make("DataCenter-v0",
                  n_racks=2,
                  servers_per_rack=4,
                  gpus_per_server=2,
                  gpu_preset="H100_SXM",
                  dt=5.0,
                  episode_length_s=3600.0)

print("DataCenter-v0 Environment")
print(f"  Configuration: 2 racks × 4 servers × 2 GPUs = 16 GPUs")
print(f"  Action space:  {env.action_space}")
print(f"  Obs space:     {env.observation_space}")

# ── Run rule-based agent ─────────────────────────────────────────────────
agent = DataCenterRuleAgent(target_C=80.0)
metrics = EpisodeMetrics()

obs, _ = env.reset()
agent.reset()

for step in range(720):  # 3600s / 5s = 720 steps
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    metrics.record(info, reward, time_s=step * 5.0)

    if step % 72 == 0:  # print every ~360s
        print(f"  t={step*5:5.0f}s  T_hot={info.get('T_hotspot_C', 0):.1f}°C  "
              f"PUE={info.get('PUE', 0):.3f}  "
              f"cost=${info.get('step_cost_dollar', 0):.5f}")
    if terminated or truncated:
        break

summary = metrics.summary()
print_metrics(summary, title="Rule-Based Agent on DataCenter-v0")

# ── Key outputs ───────────────────────────────────────────────────────────
print("Key performance indicators:")
print(f"  Mean PUE:        {summary['PUE_mean']:.3f}")
print(f"  Total energy:    {summary['total_energy_kWh']:.2f} kWh")
print(f"  Total water:     {summary['total_water_L']:.2f} L")
print(f"  Total cost:      ${summary['total_cost_dollar']:.4f}")
print(f"  Violation rate:  {summary['thermal_violation_rate']:.3%}")

env.close()
