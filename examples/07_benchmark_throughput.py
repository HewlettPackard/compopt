#!/usr/bin/env python3
"""
CompOpt Example 07 — Vectorised Simulation Benchmark
=====================================================
Benchmarks the throughput of CompOpt's BatchSimulator to measure
simulation steps per second across all environment difficulty levels.

Run:
    python examples/07_benchmark_throughput.py
"""

import compopt
from compopt.utils.vec_env import benchmark_throughput

# ── Benchmark configurations ─────────────────────────────────────────────
configs = [
    {
        "name": "ChipThermal-v0",
        "factory": lambda: compopt.make("ChipThermal-v0", dt=1.0,
                                        episode_length_s=600.0),
        "n_envs": [1, 4, 8, 16, 32],
    },
    {
        "name": "RackCooling-v0",
        "factory": lambda: compopt.make("RackCooling-v0", dt=1.0,
                                        n_servers=4, gpus_per_server=1,
                                        episode_length_s=1800.0),
        "n_envs": [1, 4, 8, 16],
    },
    {
        "name": "DataCenter-v0",
        "factory": lambda: compopt.make("DataCenter-v0", dt=5.0,
                                        n_racks=2, servers_per_rack=4,
                                        gpus_per_server=2,
                                        episode_length_s=3600.0),
        "n_envs": [1, 4, 8],
    },
    {
        "name": "Scheduling-v0",
        "factory": lambda: compopt.make("Scheduling-v0", dt=60.0,
                                        n_jobs=100,
                                        episode_length_s=86400.0),
        "n_envs": [1, 4, 8, 16],
    },
]

N_STEPS = 500  # steps per benchmark run

# ── Run benchmarks ────────────────────────────────────────────────────────
print("=" * 75)
print(f"{'Environment':25s} | {'N_envs':>6s} | {'Steps/sec':>12s} | {'Wall (s)':>8s}")
print("-" * 75)

for cfg in configs:
    for n in cfg["n_envs"]:
        try:
            result = benchmark_throughput(cfg["factory"],
                                          n_envs=n, n_steps=N_STEPS)
            print(f"{cfg['name']:25s} | {n:6d} | "
                  f"{result['steps_per_second']:12,.1f} | "
                  f"{result['wall_time_s']:8.3f}")
        except Exception as e:
            print(f"{cfg['name']:25s} | {n:6d} | ERROR: {e}")

print("=" * 75)
print("\nBenchmark complete.")
