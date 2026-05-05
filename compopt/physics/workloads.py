"""
compopt.physics.workloads
=========================
Synthetic workload / power-profile generators for GPU thermal simulations.

Each factory function returns a ``PowerProfile`` compatible with ``GPUChipModel``.
Profiles can be composed and mixed for realistic datacenter scenarios.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional

from compopt.physics.chip import PowerProfile


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tile_names(nx: int, ny: int) -> List[str]:
    return [f"tile_{r}_{c}" for r in range(ny) for c in range(nx)]


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal workload
# ──────────────────────────────────────────────────────────────────────────────

def make_sinusoidal_profile(nx: int, ny: int,
                            base_gpu_W: float = 150.0,
                            peak_gpu_W: float = 600.0,
                            period_s: float = 120.0) -> PowerProfile:
    """Smooth sinusoidal GPU power oscillation between *base* and *peak*."""
    tiles = _tile_names(nx, ny)
    n     = len(tiles)
    w     = 1.0 / n

    def tile_fn(tw: float) -> Callable[[float], float]:
        def f(t: float) -> float:
            mean = 0.5 * (peak_gpu_W + base_gpu_W)
            amp  = 0.5 * (peak_gpu_W - base_gpu_W)
            return tw * (mean + amp * np.sin(2 * np.pi * t / period_s))
        return f

    func: Dict[str, Callable[[float], float]] = {
        name: tile_fn(w) for name in tiles}
    func["HBM"] = lambda t: 40.0 + 40.0 * (np.sin(2*np.pi*t/(period_s*2)) > 0)
    func["VRM"] = lambda t: 0.05 * (
        0.5*(peak_gpu_W+base_gpu_W) +
        0.5*(peak_gpu_W-base_gpu_W) * np.sin(2*np.pi*t/period_s)) + 30.0
    func["PCB"] = lambda t: 10.0
    return PowerProfile(func_per_node=func)


# ──────────────────────────────────────────────────────────────────────────────
# Step workload
# ──────────────────────────────────────────────────────────────────────────────

def make_step_profile(nx: int, ny: int,
                      idle_W: float = 100.0,
                      load_W: float = 650.0,
                      step_times: Optional[List[float]] = None) -> PowerProfile:
    """Alternating idle / full-load step workload."""
    if step_times is None:
        step_times = [0, 120, 240, 360, 480]
    tiles = _tile_names(nx, ny)
    w     = 1.0 / len(tiles)

    def tile_fn(tw: float) -> Callable[[float], float]:
        def f(t: float) -> float:
            idx = sum(1 for ts in step_times if t >= ts)
            return tw * (load_W if idx % 2 == 1 else idle_W)
        return f

    func: Dict[str, Callable[[float], float]] = {
        name: tile_fn(w) for name in tiles}
    func["HBM"] = lambda t: 80.0 if any(
        t >= ts for ts in step_times[1::2]) else 20.0
    func["VRM"] = lambda t: 0.05 * load_W + 30.0
    func["PCB"] = lambda t: 10.0
    return PowerProfile(func_per_node=func)


# ──────────────────────────────────────────────────────────────────────────────
# Burst workload
# ──────────────────────────────────────────────────────────────────────────────

def make_burst_profile(nx: int, ny: int,
                       base_W: float = 150.0,
                       burst_W: float = 700.0,
                       burst_start: float = 60.0,
                       burst_duration: float = 30.0) -> PowerProfile:
    """Short power burst followed by return to baseline."""
    tiles = _tile_names(nx, ny)
    w     = 1.0 / len(tiles)

    def tile_fn(tw: float) -> Callable[[float], float]:
        def f(t: float) -> float:
            in_burst = burst_start <= t < burst_start + burst_duration
            return tw * (burst_W if in_burst else base_W)
        return f

    func: Dict[str, Callable[[float], float]] = {
        name: tile_fn(w) for name in tiles}
    func["HBM"] = lambda t: (80.0 if burst_start <= t < burst_start + burst_duration
                              else 20.0)
    func["VRM"] = lambda t: 0.05 * burst_W + 30.0
    func["PCB"] = lambda t: 10.0
    return PowerProfile(func_per_node=func)


# ──────────────────────────────────────────────────────────────────────────────
# Trace-driven workload (replay real GPU power traces)
# ──────────────────────────────────────────────────────────────────────────────

def make_trace_profile(nx: int, ny: int,
                       times_s: np.ndarray,
                       power_W: np.ndarray,
                       hbm_frac: float = 0.12,
                       vrm_frac: float = 0.08) -> PowerProfile:
    """
    Replay a measured GPU power trace via linear interpolation.

    Parameters
    ----------
    times_s  : 1-D array of time stamps [s]
    power_W  : 1-D array of total GPU power [W] at each time stamp
    hbm_frac : fraction of total power dissipated by HBM
    vrm_frac : fraction of total power dissipated by VRM
    """
    tiles = _tile_names(nx, ny)
    n     = len(tiles)
    w     = (1.0 - hbm_frac - vrm_frac) / n

    def tile_fn(tw: float) -> Callable[[float], float]:
        def f(t: float) -> float:
            return tw * float(np.interp(t, times_s, power_W))
        return f

    func: Dict[str, Callable[[float], float]] = {
        name: tile_fn(w) for name in tiles}
    func["HBM"] = lambda t: hbm_frac * float(np.interp(t, times_s, power_W))
    func["VRM"] = lambda t: vrm_frac * float(np.interp(t, times_s, power_W)) + 15.0
    func["PCB"] = lambda t: 10.0
    return PowerProfile(func_per_node=func)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-tenant / mixed workload
# ──────────────────────────────────────────────────────────────────────────────

def make_mixed_profile(nx: int, ny: int,
                       profiles: List[PowerProfile],
                       weights: Optional[List[float]] = None) -> PowerProfile:
    """
    Weighted mixture of multiple power profiles (e.g. training + inference).

    Parameters
    ----------
    profiles : list of PowerProfile instances to mix
    weights  : optional weights (must sum to 1); default uniform
    """
    if weights is None:
        weights = [1.0 / len(profiles)] * len(profiles)
    tiles = _tile_names(nx, ny)
    all_nodes = tiles + ["HBM", "VRM", "PCB"]

    def mixed_fn(node_name: str) -> Callable[[float], float]:
        def f(t: float) -> float:
            total = 0.0
            for p, wt in zip(profiles, weights):
                fn = p.func_per_node.get(node_name)
                total += wt * (fn(t) if fn else 0.0)
            return total
        return f

    func = {name: mixed_fn(name) for name in all_nodes}
    return PowerProfile(func_per_node=func)


# ──────────────────────────────────────────────────────────────────────────────
# Stochastic workload (random arrivals)
# ──────────────────────────────────────────────────────────────────────────────

def make_stochastic_profile(nx: int, ny: int,
                            base_W: float = 100.0,
                            peak_W: float = 700.0,
                            mean_duration_s: float = 60.0,
                            mean_interval_s: float = 120.0,
                            seed: int = 42,
                            horizon_s: float = 3600.0) -> PowerProfile:
    """
    Generate random bursty workload with Poisson arrivals and exponential
    durations.  Pre-computes events up to *horizon_s*.
    """
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    while t < horizon_s:
        gap = rng.exponential(mean_interval_s)
        dur = rng.exponential(mean_duration_s)
        t  += gap
        events.append((t, t + dur, rng.uniform(base_W, peak_W)))

    tiles = _tile_names(nx, ny)
    n     = len(tiles)
    w     = 1.0 / n

    def tile_fn(tw: float) -> Callable[[float], float]:
        def f(t: float) -> float:
            pwr = base_W
            for (t0, t1, p) in events:
                if t0 <= t < t1:
                    pwr = max(pwr, p)
            return tw * pwr
        return f

    func: Dict[str, Callable[[float], float]] = {
        name: tile_fn(w) for name in tiles}
    func["HBM"] = lambda t: 60.0
    func["VRM"] = lambda t: 45.0
    func["PCB"] = lambda t: 10.0
    return PowerProfile(func_per_node=func)
