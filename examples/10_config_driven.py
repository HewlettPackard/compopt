#!/usr/bin/env python3
"""
CompOpt Example 10 — Config-Driven Simulation
==============================================
Demonstrates loading pre-built cluster configurations from JSON presets
and building data centers from configuration files.

Run:
    python examples/10_config_driven.py
"""

from compopt.configs.loader import list_presets, load_config, build_from_config
from compopt.utils.metrics import EpisodeMetrics, print_metrics
from compopt.agents.baselines import DataCenterRuleAgent
import compopt

# ── List available presets ────────────────────────────────────────────────
print("Available configuration presets:")
for p in list_presets():
    print(f"  • {p}")
print()

# ── Load a preset ─────────────────────────────────────────────────────────
config = load_config("small_ai_cluster")
print("Loaded preset: small_ai_cluster")
print(f"  Racks:           {config.get('system', {}).get('n_racks', 'N/A')}")
print(f"  Nodes per rack:  {config.get('system', {}).get('nodes_per_rack', 'N/A')}")
print(f"  GPUs per node:   {config.get('system', {}).get('gpus_per_node', 'N/A')}")
print(f"  GPU preset:      {config.get('system', {}).get('gpu_preset', 'N/A')}")
print()

# ── Build datacenter from config ─────────────────────────────────────────
built = build_from_config("small_ai_cluster")
dc = built["datacenter"]
scheduler = built["scheduler"]
jobs = built["jobs"]
print("Built data center:")
print(f"  Racks:   {len(dc.racks)}")
total_gpus = sum(len(s.gpus) for r in dc.racks for s in r.servers)
print(f"  GPUs:    {total_gpus}")
print(f"  IT power: {dc.it_power_W/1000:.1f} kW (at startup)")
print(f"  Jobs:    {len(jobs)}")
print()

# ── Run a quick simulation with the config-built DC ──────────────────────
env = compopt.make("DataCenter-v0",
                  dc=dc,
                  dt=5.0,
                  episode_length_s=1800.0)

agent = DataCenterRuleAgent(target_C=80.0)
metrics = EpisodeMetrics()

obs, _ = env.reset()
agent.reset()

for step in range(360):  # 1800s / 5s
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    metrics.record(info, reward, time_s=step * 5.0)
    if terminated or truncated:
        break

print_metrics(metrics.summary(),
              title="Rule-Based Agent on Config-Driven DataCenter")

env.close()
print("Done!")
