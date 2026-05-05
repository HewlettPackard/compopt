"""
compopt.physics.chip
====================
GPU chip-level thermal RC-network model.

Supports H100, B200, and custom GPU configurations via parameterised dataclasses.
The model represents the die as an nx × ny grid of tiles plus lumped HBM, VRM,
and PCB nodes, connected to a direct-to-chip cold-plate (CoolantLoop).

**Vectorised stepping** is supported via ``ChipModelVec`` for batched RL.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from compopt.physics.fluids import CoolantLoop


# ──────────────────────────────────────────────────────────────────────────────
# PowerProfile
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PowerProfile:
    """
    Time-varying power profile for thermal nodes.

    Attributes
    ----------
    func_per_node : dict mapping node name → callable(t) → power [W].
    """
    func_per_node: Dict[str, Callable[[float], float]]

    def power_vector(self, t: float, node_names: List[str]) -> np.ndarray:
        P = np.zeros(len(node_names))
        for i, name in enumerate(node_names):
            fn = self.func_per_node.get(name)
            P[i] = fn(t) if fn is not None else 0.0
        return P


# ──────────────────────────────────────────────────────────────────────────────
# GPU presets (H100, B200, generic)
# ──────────────────────────────────────────────────────────────────────────────
#
# Thermal resistance values are calibrated so that:
# - At TDP (700W for H100) with max coolant flow (0.5 kg/s), T_die ≈ 65-70°C
# - At TDP with min coolant flow (0.05 kg/s), T_die ≈ 80-85°C (near thermal limit)
# - This provides realistic operating range for control algorithms
#
# Note: R_tile_to_coolant is the resistance for ONE tile. With 16 tiles in parallel,
# effective R = R_tile / 16. For T_die = T_coolant + R_eff * P_total:
#   70°C = 40°C + R_eff * 600W  →  R_eff ≈ 0.05 K/W  →  R_tile ≈ 0.8 K/W

GPU_PRESETS = {
    "H100_SXM": dict(
        tdp_W=700, die_area_mm2=814, nx_tiles=4, ny_tiles=4,
        C_tile_J_K=8.0, C_hbm_J_K=80.0, C_vrm_J_K=40.0, C_pcb_J_K=200.0,
        R_lateral_tile_K_W=0.5, R_tile_to_pcb_K_W=0.3,
        R_tile_to_coolant_K_W=0.8, R_hbm_to_pcb_K_W=0.4,
        R_vrm_to_pcb_K_W=0.4, R_hbm_to_coolant_K_W=1.2,
        R_vrm_to_coolant_K_W=1.5, R_pcb_to_coolant_K_W=2.0,
        T_max_die_C=83.0, T_max_hbm_C=95.0, T_max_vrm_C=105.0,
    ),
    "B200": dict(
        tdp_W=1000, die_area_mm2=1000, nx_tiles=6, ny_tiles=6,
        C_tile_J_K=6.0, C_hbm_J_K=120.0, C_vrm_J_K=60.0, C_pcb_J_K=300.0,
        R_lateral_tile_K_W=0.4, R_tile_to_pcb_K_W=0.25,
        R_tile_to_coolant_K_W=0.5, R_hbm_to_pcb_K_W=0.35,
        R_vrm_to_pcb_K_W=0.35, R_hbm_to_coolant_K_W=1.0,
        R_vrm_to_coolant_K_W=1.2, R_pcb_to_coolant_K_W=1.5,
        T_max_die_C=85.0, T_max_hbm_C=100.0, T_max_vrm_C=110.0,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# H100ChipModel
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUChipModel:
    """
    GPU chip-level thermal RC-network.

    Node layout
    -----------
    - ``tile_{r}_{c}`` : die tile at row r, col c  (nx × ny grid)
    - ``HBM``          : lumped HBM stack
    - ``VRM``          : lumped voltage-regulator module
    - ``PCB``          : lumped carrier board / spreader

    Parameters match the ``GPU_PRESETS`` dictionaries for common GPU SKUs.
    """
    nx_tiles:       int
    ny_tiles:       int
    power_profile:  PowerProfile
    coolant_loop:   CoolantLoop

    # Capacitances [J/K]
    C_tile_J_K:   float = 8.0
    C_hbm_J_K:    float = 80.0
    C_vrm_J_K:    float = 40.0
    C_pcb_J_K:    float = 200.0

    # Resistances [K/W]
    R_lateral_tile_K_W:    float = 0.05
    R_tile_to_pcb_K_W:     float = 0.03
    R_tile_to_coolant_K_W: float = 0.022
    R_hbm_to_pcb_K_W:      float = 0.06
    R_vrm_to_pcb_K_W:      float = 0.06
    R_hbm_to_coolant_K_W:  float = 0.06
    R_vrm_to_coolant_K_W:  float = 0.08
    R_pcb_to_coolant_K_W:  float = 0.15

    # Thermal limits
    T_max_die_C: float = 83.0
    T_max_hbm_C: float = 95.0
    T_max_vrm_C: float = 105.0

    # Initial conditions
    T_init_die_C:    float = 35.0
    T_init_others_C: float = 35.0

    # Internal state ────────────────────────────────────────────────────────
    node_names:         List[str]   = field(init=False)
    C_J_K:              np.ndarray  = field(init=False)
    G_W_K:              np.ndarray  = field(init=False)
    T_C:                np.ndarray  = field(init=False)
    time_s:             float       = field(init=False, default=0.0)
    _g_cool_vec_W_K:    np.ndarray  = field(init=False)
    _Q_coolant_last_W:  float       = field(init=False, default=0.0)
    _reported_power_W:  float       = field(init=False, default=0.0)

    def __post_init__(self):
        tile_names = [f"tile_{r}_{c}"
                      for r in range(self.ny_tiles)
                      for c in range(self.nx_tiles)]
        self.node_names = tile_names + ["HBM", "VRM", "PCB"]
        n = len(self.node_names)
        self.C_J_K = np.zeros(n)
        self.T_C   = np.zeros(n)

        for i, name in enumerate(self.node_names):
            if name.startswith("tile_"):
                self.C_J_K[i] = self.C_tile_J_K
                self.T_C[i]   = self.T_init_die_C
            elif name == "HBM":
                self.C_J_K[i] = self.C_hbm_J_K
                self.T_C[i]   = self.T_init_others_C
            elif name == "VRM":
                self.C_J_K[i] = self.C_vrm_J_K
                self.T_C[i]   = self.T_init_others_C
            elif name == "PCB":
                self.C_J_K[i] = self.C_pcb_J_K
                self.T_C[i]   = self.T_init_others_C

        self.G_W_K           = np.zeros((n, n))
        self._build_G(tile_names)
        self._g_cool_vec_W_K = self._build_coolant_coupling(tile_names)

    # ── helpers ────────────────────────────────────────────────────────────

    def _idx(self, name: str) -> int:
        return self.node_names.index(name)

    def _build_G(self, tile_names: List[str]):
        G = self.G_W_K
        for r in range(self.ny_tiles):
            for c in range(self.nx_tiles):
                i = self._idx(f"tile_{r}_{c}")
                for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0 <= nr < self.ny_tiles and 0 <= nc < self.nx_tiles:
                        j = self._idx(f"tile_{nr}_{nc}")
                        g = 1.0 / self.R_lateral_tile_K_W
                        G[i,i] += g; G[j,j] += g
                        G[i,j] -= g; G[j,i] -= g

        idx_pcb = self._idx("PCB")
        for name in tile_names:
            i = self._idx(name)
            g = 1.0 / self.R_tile_to_pcb_K_W
            G[i,i] += g; G[idx_pcb,idx_pcb] += g
            G[i,idx_pcb] -= g; G[idx_pcb,i] -= g

        for node, R in [("HBM", self.R_hbm_to_pcb_K_W),
                         ("VRM", self.R_vrm_to_pcb_K_W)]:
            i = self._idx(node); g = 1.0 / R
            G[i,i] += g; G[idx_pcb,idx_pcb] += g
            G[i,idx_pcb] -= g; G[idx_pcb,i] -= g

    def _build_coolant_coupling(self, tile_names: List[str]) -> np.ndarray:
        n      = len(self.node_names)
        g_cool = np.zeros(n)
        g_tile = 1.0 / self.R_tile_to_coolant_K_W
        for name in tile_names:
            g_cool[self._idx(name)] += g_tile
        g_cool[self._idx("HBM")] += 1.0 / self.R_hbm_to_coolant_K_W
        g_cool[self._idx("VRM")] += 1.0 / self.R_vrm_to_coolant_K_W
        g_cool[self._idx("PCB")] += 1.0 / self.R_pcb_to_coolant_K_W
        return g_cool

    # ── dynamics ──────────────────────────────────────────────────────────

    def _rhs(self, t: float, T_C: np.ndarray) -> np.ndarray:
        P       = self.power_profile.power_vector(t, self.node_names)
        T_cool  = self.coolant_loop.T_coolant_C
        # Sanitize T_C to prevent invalid values in matmul/subtract
        T_C_safe = np.nan_to_num(T_C, nan=T_cool, posinf=1e4, neginf=-273.15)
        T_C_safe = np.clip(T_C_safe, -273.15, 1e4)
        solid_W = self.G_W_K @ T_C_safe
        cool_W  = self._g_cool_vec_W_K * (T_C_safe - T_cool)
        self._Q_coolant_last_W = float(np.sum(cool_W))
        return (P - solid_W - cool_W) / self.C_J_K

    def _max_rk4_dt(self) -> float:
        """Estimate max stable dt for RK4 from spectral radius of Jacobian.

        The Jacobian diagonal includes both solid-solid conductance (G) and
        coolant coupling (g_cool), divided by the thermal capacitance.
        """
        effective_diag = (np.diag(self.G_W_K) + self._g_cool_vec_W_K) / self.C_J_K
        max_eigenvalue = float(np.max(effective_diag))
        # RK4 stability boundary on the negative real axis is ~2.78
        return 2.78 / max(max_eigenvalue, 1e-12)

    def step(self, dt: float):
        """RK4 advance for solid nodes with automatic sub-stepping;
        semi-implicit for coolant."""
        max_sub = self._max_rk4_dt() * 0.9        # safety margin
        n_sub   = max(1, int(np.ceil(dt / max_sub)))
        h       = dt / n_sub

        for _ in range(n_sub):
            t, T = self.time_s, self.T_C
            k1 = self._rhs(t,          T            )
            k2 = self._rhs(t + 0.5*h,  T + 0.5*h*k1)
            k3 = self._rhs(t + 0.5*h,  T + 0.5*h*k2)
            k4 = self._rhs(t +     h,  T +     h*k3 )
            self.T_C     = T + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self.time_s += h
            self.coolant_loop.step(h, Q_in_W=self._Q_coolant_last_W)

        P_tot = float(np.sum(
            self.power_profile.power_vector(self.time_s, self.node_names)))
        alpha = 0.2
        self._reported_power_W = (
            P_tot if self._reported_power_W == 0.0
            else alpha * P_tot + (1 - alpha) * self._reported_power_W)

    def reset(self):
        for i, name in enumerate(self.node_names):
            self.T_C[i] = (self.T_init_die_C if name.startswith("tile_")
                           else self.T_init_others_C)
        self.time_s            = 0.0
        self._reported_power_W = 0.0
        self._Q_coolant_last_W = 0.0
        self.coolant_loop.reset()

    # ── outputs ───────────────────────────────────────────────────────────

    def get_observation(self) -> np.ndarray:
        """Compact 7-element RL observation vector."""
        T_tiles = np.array([self.T_C[i] for i, n in enumerate(self.node_names)
                            if n.startswith("tile_")])
        P_total = float(np.sum(
            self.power_profile.power_vector(self.time_s, self.node_names)))
        return np.array([
            float(np.mean(T_tiles)),
            float(np.max(T_tiles)),
            float(self.T_C[self._idx("HBM")]),
            float(self.T_C[self._idx("VRM")]),
            float(self.coolant_loop.T_in_C),
            float(self.coolant_loop.T_coolant_C),
            P_total,
        ], dtype=np.float64)

    def get_sensors_dict(self) -> Dict[str, float]:
        """Rich nvidia-smi-style telemetry dictionary."""
        T_tiles = np.array([self.T_C[i] for i, n in enumerate(self.node_names)
                            if n.startswith("tile_")])
        P_total = float(np.sum(
            self.power_profile.power_vector(self.time_s, self.node_names)))
        return {
            "time_s":                 self.time_s,
            "T_gpu_mean_C":           float(np.mean(T_tiles)),
            "T_gpu_hotspot_C":        float(np.max(T_tiles)),
            "T_hbm_C":               float(self.T_C[self._idx("HBM")]),
            "T_vrm_C":               float(self.T_C[self._idx("VRM")]),
            "T_pcb_C":               float(self.T_C[self._idx("PCB")]),
            "T_coolant_inlet_C":     float(self.coolant_loop.T_in_C),
            "T_coolant_outlet_C":    float(self.coolant_loop.T_coolant_C),
            "P_total_W":             P_total,
            "P_reported_smoothed_W": self._reported_power_W,
        }

    def is_throttled(self) -> bool:
        """Return True if any node exceeds its thermal limit."""
        obs = self.get_observation()
        return (obs[1] > self.T_max_die_C or
                obs[2] > self.T_max_hbm_C or
                obs[3] > self.T_max_vrm_C)

    @property
    def utilisation(self) -> float:
        """Estimated GPU utilisation (0–1) based on instantaneous power vs TDP."""
        obs = self.get_observation()
        tdp = GPU_PRESETS.get("H100_SXM", {}).get("tdp_W", 700)
        return min(1.0, max(0.0, obs[6] / tdp))


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

def make_gpu(preset: str = "H100_SXM",
             power_profile: Optional[PowerProfile] = None,
             coolant_loop: Optional[CoolantLoop] = None,
             **overrides) -> GPUChipModel:
    """
    Create a GPUChipModel from a named preset with optional overrides.

    >>> gpu = make_gpu("H100_SXM", T_init_die_C=40.0)
    """
    params = dict(GPU_PRESETS[preset])
    params.update(overrides)
    nx = int(params.pop("nx_tiles"))
    ny = int(params.pop("ny_tiles"))
    params.pop("tdp_W", None)
    params.pop("die_area_mm2", None)
    params.pop("T_max_die_C", None)
    params.pop("T_max_hbm_C", None)
    params.pop("T_max_vrm_C", None)

    if coolant_loop is None:
        coolant_loop = CoolantLoop(T_in_C=40.0, m_dot_kg_s=0.25)
    if power_profile is None:
        # default idle profile
        from compopt.physics.workloads import make_sinusoidal_profile
        power_profile = make_sinusoidal_profile(nx, ny)

    return GPUChipModel(
        nx_tiles=nx, ny_tiles=ny,
        power_profile=power_profile,
        coolant_loop=coolant_loop,
        T_max_die_C=GPU_PRESETS[preset].get("T_max_die_C", 83.0),
        T_max_hbm_C=GPU_PRESETS[preset].get("T_max_hbm_C", 95.0),
        T_max_vrm_C=GPU_PRESETS[preset].get("T_max_vrm_C", 105.0),
        **params,
    )
