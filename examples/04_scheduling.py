#!/usr/bin/env python3
"""
CompOpt Example 04 — Job Scheduling Environment
================================================
Demonstrates the Scheduling-v0 environment with discrete actions
for job queue prioritisation.

Run:
    python examples/04_scheduling.py
"""

import numpy as np
import compopt
from compopt.agents.baselines import FCFSSchedulingAgent, RandomAgent
from compopt.utils.metrics import EpisodeMetrics, print_metrics

# ── Create environment ────────────────────────────────────────────────────
env = compopt.make("Scheduling-v0",
                  dt=60.0,
                  episode_length_s=86400.0,  # 24 hours
                  n_jobs=100)

print("Scheduling-v0 Environment")
print(f"  Cluster: 64 nodes × 4 GPUs = 256 GPUs")
print(f"  Action space:  {env.action_space}")
print(f"  Obs space:     {env.observation_space}")
print()

# ── Run FCFS agent ────────────────────────────────────────────────────────
fcfs_agent = FCFSSchedulingAgent()
fcfs_metrics = EpisodeMetrics()

obs, _ = env.reset()
for step in range(1440):  # 86400 / 60 = 1440
    action, _ = fcfs_agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    fcfs_metrics.record(info, reward, time_s=step * 60.0)
    if step % 144 == 0:
        print(f"  t={step*60/3600:.1f}h  queue={info['queue_length']:.0f}  "
              f"running={info['n_running']:.0f}  "
              f"util={info['node_utilisation']:.2f}")
    if terminated or truncated:
        break

print_metrics(fcfs_metrics.summary(), title="FCFS Agent on Scheduling-v0")

# ── Compare with Random ──────────────────────────────────────────────────
random_agent = RandomAgent(env.action_space)
random_metrics = EpisodeMetrics()

obs, _ = env.reset()
for step in range(1440):
    action, _ = random_agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    random_metrics.record(info, reward, time_s=step * 60.0)
    if terminated or truncated:
        break

# ── Summary ───────────────────────────────────────────────────────────────
fcfs_s = fcfs_metrics.summary()
rand_s = random_metrics.summary()

print(f"\n{'='*60}")
print(f"{'Agent':20s} | {'Reward':>10s} | {'Util':>8s} | {'Jobs Done':>10s}")
print(f"{'-'*60}")
print(f"{'FCFS':20s} | {fcfs_s['reward_total']:10.1f} | "
      f"{fcfs_s['node_util_mean']:8.3f} | {fcfs_s['total_jobs_completed']:10.0f}")
print(f"{'Random Priority':20s} | {rand_s['reward_total']:10.1f} | "
      f"{rand_s['node_util_mean']:8.3f} | {rand_s['total_jobs_completed']:10.0f}")
print(f"{'='*60}")

env.close()
