#!/usr/bin/env python3
"""
CompOpt Example 01 — Chip-Level Thermal Control (Quick Start)
=============================================================
Demonstrates the simplest CompOpt environment: single-GPU chip thermal
management with a PID baseline agent.

Run:
    python examples/01_chip_quickstart.py
"""

import numpy as np
import compopt
from compopt.agents.baselines import PIDCoolingAgent, RuleBasedCoolingAgent
from compopt.utils.metrics import EpisodeMetrics, print_metrics

# ── 1. Create the environment ────────────────────────────────────────────
env = compopt.make("ChipThermal-v0",
                  gpu_preset="H100_SXM",
                  workload="sinusoidal",
                  dt=1.0,
                  episode_length_s=600.0)

print(f"Action space : {env.action_space}")
print(f"Obs space    : {env.observation_space}")

# ── 2. Run a PID controller ──────────────────────────────────────────────
agent = PIDCoolingAgent(target_C=80.0, Kp=0.05, Ki=0.001, Kd=0.01, dt=1.0)
metrics = EpisodeMetrics()

obs, info = env.reset()
agent.reset()

for step in range(600):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    metrics.record(info, reward, time_s=step)
    if step % 60 == 0:
        print(f"  Step {step:4d}  T_hot={info['T_hotspot_C']:.1f}°C  "
              f"flow={info['flow_kg_s']:.3f} kg/s  reward={reward:.3f}")
    if terminated or truncated:
        break

print_metrics(metrics.summary(), title="PID Agent on ChipThermal-v0")

# ── 3. Compare with rule-based ────────────────────────────────────────────
rule_agent = RuleBasedCoolingAgent(target_C=80.0, deadband=2.0)
rule_metrics = EpisodeMetrics()

obs, _ = env.reset()
rule_agent.reset()
for step in range(600):
    action, _ = rule_agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rule_metrics.record(info, reward, time_s=step)
    if terminated or truncated:
        break

print_metrics(rule_metrics.summary(), title="Rule-Based Agent on ChipThermal-v0")

env.close()
print("Done!")
