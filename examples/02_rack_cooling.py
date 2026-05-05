#!/usr/bin/env python3
"""
CompOpt Example 02 — Rack-Level Cooling Control
================================================
Demonstrates the rack-level environment with 4 servers, each containing
a single H100 GPU. Compares PID and rule-based agents.

Run:
    python examples/02_rack_cooling.py
"""

import numpy as np
import compopt
from compopt.agents.baselines import PIDCoolingAgent, RandomAgent
from compopt.utils.metrics import evaluate_agent, print_metrics

# ── Create environment ────────────────────────────────────────────────────
env = compopt.make("RackCooling-v0",
                  n_servers=4,
                  gpus_per_server=1,
                  gpu_preset="H100_SXM",
                  dt=1.0,
                  episode_length_s=1800.0)

print(f"Environment: RackCooling-v0")
print(f"  Servers:  4 × H100_SXM")
print(f"  Action:   {env.action_space}")
print(f"  Obs:      {env.observation_space}")

# ── Evaluate agents ───────────────────────────────────────────────────────
agents = {
    "Random": RandomAgent(env.action_space),
    "PID (Kp=0.05)": PIDCoolingAgent(
        target_C=80.0, Kp=0.05, Ki=0.001, Kd=0.01, dt=1.0),
    "PID (Kp=0.10)": PIDCoolingAgent(
        target_C=80.0, Kp=0.10, Ki=0.002, Kd=0.02, dt=1.0),
}

results = {}
for name, agent in agents.items():
    print(f"\nEvaluating: {name}")
    result = evaluate_agent(env, agent, n_episodes=3, max_steps=1800)
    results[name] = result
    print_metrics(result, title=name)

# ── Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Agent':25s} | {'Reward':>10s} | {'T_max':>8s} | {'Violations':>10s}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:25s} | {r['reward_total']:10.1f} | "
          f"{r['T_hotspot_max']:8.1f} | "
          f"{r['thermal_violation_rate']:10.3f}")
print("=" * 70)

env.close()
