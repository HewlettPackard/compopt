#!/usr/bin/env python3
"""
CompOpt Example 05 — Joint Scheduling + Cooling (Expert)
=========================================================
Demonstrates the JointDCFlat-v0 environment: simultaneous workload
scheduling and cooling control with a flattened action/observation space
suitable for standard RL algorithms.

Run:
    python examples/05_joint_datacenter.py
"""

import numpy as np
import compopt
from compopt.utils.metrics import EpisodeMetrics, print_metrics

# ── Create joint environment (flat mode) ─────────────────────────────────
env = compopt.make("JointDCFlat-v0",
                  n_racks=2,
                  servers_per_rack=4,
                  gpus_per_server=2,
                  gpu_preset="H100_SXM",
                  dt=10.0,
                  episode_length_s=7200.0,
                  n_jobs=200)

print("JointDCFlat-v0 Environment")
print(f"  Configuration: 2 racks × 4 servers × 2 GPUs + 200 jobs")
print(f"  Action space:  {env.action_space}")
print(f"  Obs space:     {env.observation_space}")
print()

# ── Simple proportional-integral heuristic ────────────────────────────────
class SimpleJointAgent:
    """Heuristic agent: proportional cooling + FCFS scheduling."""
    def __init__(self, target_C=80.0):
        self.target_C = target_C
        self._integral = 0.0

    def predict(self, obs, deterministic=False):
        # obs[:16] = thermal, obs[16:28] = scheduler
        T_hot = float(obs[0])  # T_hotspot_max
        error = T_hot - self.target_C
        self._integral += error * 0.01
        self._integral = np.clip(self._integral, -5.0, 5.0)

        intensity = np.clip(0.5 + 0.02 * error + self._integral, 0.0, 1.0)
        # action[0:3] = cooling, action[3] = scheduling (0 = FCFS)
        action = np.array([intensity, intensity * 0.8, intensity * 0.4, 0.0],
                          dtype=np.float32)
        return action, {}

    def reset(self):
        self._integral = 0.0

# ── Run ───────────────────────────────────────────────────────────────────
agent = SimpleJointAgent(target_C=80.0)
metrics = EpisodeMetrics()

obs, _ = env.reset()
agent.reset()

for step in range(720):  # 7200s / 10s
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    metrics.record(info, reward, time_s=step * 10.0)

    if step % 72 == 0:
        print(f"  t={step*10:5.0f}s  T_hot={info.get('T_hotspot_C', 0):.1f}°C  "
              f"PUE={info.get('PUE', 0):.3f}  "
              f"queue={info.get('queue_length', 0):.0f}  "
              f"util={info.get('node_utilisation', 0):.2f}")
    if terminated or truncated:
        break

summary = metrics.summary()
print_metrics(summary, title="Simple Heuristic on JointDCFlat-v0")

env.close()
