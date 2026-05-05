"""
compopt.physics.server
======================
Server-level and rack-level thermal models.

A **ServerNode** wraps one or more GPU chips, CPUs, memory, and NIC power
draws into a single compute node that plugs into a rack's coolant manifold.

A **RackModel** aggregates multiple servers sharing a rack coolant manifold
and exposes observation / telemetry interfaces.

A **DataCenterModel** aggregates racks, CDUs, and a cooling tower into a
full facility-level plant.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from compopt.physics.fluids import CoolantLoop, CDU, CoolingTower
from compopt.physics.chip import GPUChipModel, PowerProfile, make_gpu
from compopt.physics.workloads import make_sinusoidal_profile


# ──────────────────────────────────────────────────────────────────────────────
# ServerNode
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ServerNode:
    """
    Single compute node containing GPUs, CPU, memory, NIC, and NVMe.

    RAPS-compatible power breakdown:
        P_total = sum(GPU_power) + P_cpu + P_mem + P_nic + P_nvme

    Parameters
    ----------
    gpus          : list of GPUChipModel in this node
    P_cpu_idle_W  : CPU idle power [W]
    P_cpu_max_W   : CPU peak power [W]
    P_mem_W       : Memory subsystem power [W]
    P_nic_W       : Network interface card power [W]
    P_nvme_W      : NVMe / disk power [W]
    cpu_util      : CPU utilisation fraction [0, 1] (can be time-varying)
    """
    gpus:          List[GPUChipModel]
    P_cpu_idle_W:  float = 90.0
    P_cpu_max_W:   float = 280.0
    P_mem_W:       float = 74.0
    P_nic_W:       float = 20.0
    P_nvme_W:      float = 30.0
    cpu_util:      float = 0.5

    node_id:       int = 0
    is_down:       bool = False

    def step(self, dt: float):
        if self.is_down:
            return
        for gpu in self.gpus:
            gpu.step(dt)

    def reset(self):
        for gpu in self.gpus:
            gpu.reset()
        self.is_down = False

    @property
    def P_cpu_W(self) -> float:
        return self.P_cpu_idle_W + self.cpu_util * (self.P_cpu_max_W - self.P_cpu_idle_W)

    @property
    def P_total_W(self) -> float:
        gpu_power = sum(
            float(np.sum(g.power_profile.power_vector(g.time_s, g.node_names)))
            for g in self.gpus)
        return gpu_power + self.P_cpu_W + self.P_mem_W + self.P_nic_W + self.P_nvme_W

    @property
    def T_gpu_hotspot_C(self) -> float:
        if not self.gpus:
            return 0.0
        return max(g.get_observation()[1] for g in self.gpus)

    def get_telemetry(self) -> Dict[str, float]:
        tele = {
            "node_id":       self.node_id,
            "is_down":       float(self.is_down),
            "P_total_W":     self.P_total_W,
            "P_cpu_W":       self.P_cpu_W,
            "cpu_util":      self.cpu_util,
            "T_gpu_hotspot_C": self.T_gpu_hotspot_C,
        }
        for i, gpu in enumerate(self.gpus):
            for k, v in gpu.get_sensors_dict().items():
                tele[f"gpu{i}_{k}"] = v
        return tele


# ──────────────────────────────────────────────────────────────────────────────
# RackModel
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RackModel:
    """
    Rack-level model: multiple ServerNodes sharing a rack coolant manifold.

    Attributes
    ----------
    servers       : list of ServerNode instances
    rack_coolant  : CoolantLoop for the rack manifold
    rack_id       : integer rack identifier
    P_switch_W    : per-switch power draw [W]
    n_switches    : number of network switches in this rack
    psu_efficiency: PSU conversion efficiency (0–1)
    """
    servers:        List[ServerNode]
    rack_coolant:   CoolantLoop
    rack_id:        int = 0
    P_switch_W:     float = 250.0
    n_switches:     int = 2
    psu_efficiency: float = 0.96

    time_s: float = field(init=False, default=0.0)

    def step(self, dt: float):
        # Push rack coolant temperature into each GPU's coolant inlet
        for server in self.servers:
            for gpu in server.gpus:
                gpu.coolant_loop.T_in_C = self.rack_coolant.T_coolant_C

        Q_rack_W = 0.0
        for server in self.servers:
            server.step(dt)
            for gpu in server.gpus:
                dT = gpu.coolant_loop.T_coolant_C - gpu.coolant_loop.T_in_C
                Q_rack_W += (gpu.coolant_loop.m_dot_kg_s *
                             gpu.coolant_loop.c_p_J_kgK * dT)

        self.rack_coolant.step(dt, Q_in_W=Q_rack_W)
        self.time_s += dt

    def reset(self):
        for server in self.servers:
            server.reset()
        self.rack_coolant.reset()
        self.time_s = 0.0

    @property
    def P_total_W(self) -> float:
        """Total rack electrical power including switches and PSU losses."""
        node_power = sum(s.P_total_W for s in self.servers)
        switch_power = self.P_switch_W * self.n_switches
        return (node_power + switch_power) / self.psu_efficiency

    @property
    def T_hotspot_C(self) -> float:
        """Maximum GPU hotspot across all servers in this rack."""
        return max((s.T_gpu_hotspot_C for s in self.servers), default=0.0)

    def get_rack_observation(self) -> np.ndarray:
        """10-element compact observation for RL (GPU0 of server0)."""
        gpu0_obs = self.servers[0].gpus[0].get_observation()  # shape (7,)
        return np.concatenate([
            gpu0_obs,
            np.array([self.rack_coolant.T_in_C,
                      self.rack_coolant.T_coolant_C,
                      self.P_total_W])
        ])

    def get_rack_telemetry(self) -> Dict[str, float]:
        return {
            "time_s":            self.time_s,
            "rack_id":           self.rack_id,
            "rack_T_inlet_C":    float(self.rack_coolant.T_in_C),
            "rack_T_outlet_C":   float(self.rack_coolant.T_coolant_C),
            "rack_total_power_W": self.P_total_W,
            "rack_T_hotspot_C":  self.T_hotspot_C,
            "n_active_nodes":    sum(1 for s in self.servers if not s.is_down),
        }

    def get_full_state_for_llm(self) -> Dict[str, Dict]:
        state: Dict[str, Dict] = {"rack": self.get_rack_telemetry()}
        for i, server in enumerate(self.servers):
            state[f"server_{i}"] = server.get_telemetry()
        return state


# ──────────────────────────────────────────────────────────────────────────────
# DataCenterModel
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCenterModel:
    """
    Full data-center plant: racks + CDU + cooling tower + ambient weather.

    This is the top-level physical model that the ``DataCenterEnv``
    Gymnasium environment wraps.

    Parameters
    ----------
    racks          : list of RackModel instances
    cdu            : CDU instance (rack-to-facility heat exchanger)
    cooling_tower  : CoolingTower instance (heat rejection to ambient)
    power_cost_dollar_per_kWh : electricity cost [$/kWh]
    water_cost_dollar_per_L   : water cost [$/L]
    ambient_T_fn   : optional callable(t) → ambient dry-bulb T [°C]
    """
    racks:          List[RackModel]
    cdu:            CDU
    cooling_tower:  CoolingTower
    power_cost_dollar_per_kWh: float = 0.094
    water_cost_dollar_per_L:   float = 0.004

    ambient_T_fn:  Optional[object] = None  # Callable[[float], float]
    time_s:        float = field(init=False, default=0.0)

    # Accumulators
    _energy_J:       float = field(init=False, default=0.0)
    _water_L:        float = field(init=False, default=0.0)
    _cost_dollar:    float = field(init=False, default=0.0)

    def step(self, dt: float):
        """Advance entire data center by *dt* seconds."""
        # Update ambient temperature if dynamic
        if self.ambient_T_fn is not None:
            T_amb = float(self.ambient_T_fn(self.time_s))
            self.cooling_tower.T_ambient_C = T_amb
            self.cooling_tower.T_wetbulb_C = T_amb - 7.0  # rough approximation

        # CDU supplies coolant to all racks
        T_supply = self.cdu.hot_loop.T_in_C
        for rack in self.racks:
            rack.rack_coolant.T_in_C = T_supply

        # Step all racks
        Q_total_rack = 0.0
        for rack in self.racks:
            rack.step(dt)
            Q_total_rack += rack.P_total_W  # approximate: all electrical → heat

        # CDU transfers rack heat to facility loop
        self.cdu.step(dt, Q_rack_total_W=Q_total_rack)

        # Cooling tower rejects facility heat to ambient
        Q_facility = self.cdu.cold_loop.m_dot_kg_s * \
                     self.cdu.cold_loop.c_p_J_kgK * \
                     (self.cdu.cold_loop.T_coolant_C - self.cdu.T_facility_C)
        self.cooling_tower.step(dt, Q_facility_W=max(0.0, Q_facility))

        # Feed tower basin temp back into CDU facility supply
        self.cdu.T_facility_C = self.cooling_tower.T_supply_C
        # Feed CDU hot-side outlet back as rack supply
        for rack in self.racks:
            rack.rack_coolant.T_in_C = self.cdu.hot_loop.T_in_C

        self.time_s += dt

        # Accumulate cost/energy/water
        P_total = self.total_power_W
        self._energy_J   += P_total * dt
        self._water_L    += (self.cooling_tower.water_used_L -
                             self._water_L)  # delta
        energy_kWh        = P_total * dt / 3.6e6
        self._cost_dollar += (energy_kWh * self.power_cost_dollar_per_kWh)

    def reset(self):
        for rack in self.racks:
            rack.reset()
        self.cdu.reset()
        self.cooling_tower.reset()
        self.time_s      = 0.0
        self._energy_J   = 0.0
        self._water_L    = 0.0
        self._cost_dollar = 0.0

    # ── properties ────────────────────────────────────────────────────────

    @property
    def total_power_W(self) -> float:
        it_power = sum(r.P_total_W for r in self.racks)
        cooling_power = self.cdu.pump_power_W + self.cooling_tower.fan_power_W
        return it_power + cooling_power

    @property
    def it_power_W(self) -> float:
        return sum(r.P_total_W for r in self.racks)

    @property
    def PUE(self) -> float:
        """Power Usage Effectiveness = total / IT."""
        it = self.it_power_W
        return self.total_power_W / it if it > 0 else 1.0

    @property
    def WUE_L_per_kWh(self) -> float:
        """Water Usage Effectiveness [L/kWh]."""
        energy_kWh = self._energy_J / 3.6e6
        return self.cooling_tower.water_used_L / energy_kWh if energy_kWh > 0 else 0.0

    @property
    def total_cost_dollar(self) -> float:
        water_cost = self.cooling_tower.water_used_L * self.water_cost_dollar_per_L
        return self._cost_dollar + water_cost

    @property
    def T_hotspot_C(self) -> float:
        return max((r.T_hotspot_C for r in self.racks), default=0.0)

    def get_observation(self) -> np.ndarray:
        """
        Compact data-center-level observation for RL.

        Shape: (16,)
        [0] T_hotspot_max       [1] T_hotspot_mean
        [2] rack_coolant_in     [3] rack_coolant_out
        [4] CDU_hot_T           [5] CDU_cold_T
        [6] tower_basin_T       [7] ambient_T
        [8] IT_power_kW         [9] cooling_power_kW
        [10] PUE                [11] WUE
        [12] total_cost_$       [13] water_used_L
        [14] n_active_nodes     [15] time_h
        """
        hotspots = [r.T_hotspot_C for r in self.racks]
        n_active = sum(
            sum(1 for s in r.servers if not s.is_down) for r in self.racks)
        return np.array([
            max(hotspots),
            np.mean(hotspots),
            self.cdu.T_supply_to_racks_C,
            self.cdu.T_return_from_racks_C,
            self.cdu.hot_loop.T_coolant_C,
            self.cdu.cold_loop.T_coolant_C,
            self.cooling_tower.T_basin_C,
            self.cooling_tower.T_ambient_C,
            self.it_power_W / 1000.0,
            (self.cdu.pump_power_W + self.cooling_tower.fan_power_W) / 1000.0,
            self.PUE,
            self.WUE_L_per_kWh,
            self.total_cost_dollar,
            self.cooling_tower.water_used_L,
            float(n_active),
            self.time_s / 3600.0,
        ], dtype=np.float64)

    def get_full_state_for_llm(self) -> Dict:
        """Rich nested dict for LLM context."""
        state = {
            "datacenter": {
                "time_s": self.time_s,
                "PUE": round(self.PUE, 3),
                "WUE_L_per_kWh": round(self.WUE_L_per_kWh, 3),
                "total_power_kW": round(self.total_power_W / 1000, 2),
                "IT_power_kW": round(self.it_power_W / 1000, 2),
                "cost_dollar": round(self.total_cost_dollar, 4),
                "water_used_L": round(self.cooling_tower.water_used_L, 2),
                "T_ambient_C": round(self.cooling_tower.T_ambient_C, 1),
                "T_hotspot_C": round(self.T_hotspot_C, 1),
            }
        }
        for i, rack in enumerate(self.racks):
            state[f"rack_{i}"] = rack.get_rack_telemetry()
        return state


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_default_rack(n_servers: int = 4,
                       gpus_per_server: int = 1,
                       gpu_preset: str = "H100_SXM",
                       T_in_C: float = 40.0,
                       rack_id: int = 0,
                       workload: str = "sinusoidal",
                       workload_period_s: float = 300.0) -> RackModel:
    """Build a rack of servers with configurable workloads.
    
    Parameters
    ----------
    workload : str
        One of 'sinusoidal', 'step', 'burst', 'stochastic'.
    workload_period_s : float
        Period for sinusoidal workloads (default 300s).
    """
    from compopt.physics.workloads import (
        make_sinusoidal_profile, make_step_profile,
        make_burst_profile, make_stochastic_profile)
    
    nx = 4; ny = 4
    servers = []
    for s_idx in range(n_servers):
        gpus = []
        for _ in range(gpus_per_server):
            # Select workload profile
            if workload == "step":
                profile = make_step_profile(nx, ny, idle_W=150.0, load_W=650.0)
            elif workload == "burst":
                profile = make_burst_profile(nx, ny, base_W=150.0, burst_W=700.0,
                                             burst_start=60.0 + s_idx * 30.0,
                                             burst_duration=40.0)
            elif workload == "stochastic":
                profile = make_stochastic_profile(nx, ny, seed=s_idx * 17 + rack_id * 31)
            else:  # default sinusoidal
                profile = make_sinusoidal_profile(nx, ny,
                                                  base_gpu_W=150.0,
                                                  peak_gpu_W=650.0,
                                                  period_s=workload_period_s)
            coolant = CoolantLoop(T_in_C=T_in_C, m_dot_kg_s=0.25)
            gpus.append(make_gpu(gpu_preset,
                                 power_profile=profile,
                                 coolant_loop=coolant))
        servers.append(ServerNode(gpus=gpus, node_id=s_idx + rack_id * 100))
    rack_coolant = CoolantLoop(T_in_C=T_in_C, m_dot_kg_s=2.0,
                               V_m3=1e-3, rho_kg_m3=1000.0)
    return RackModel(servers=servers, rack_coolant=rack_coolant, rack_id=rack_id)


def build_default_datacenter(n_racks: int = 2,
                             servers_per_rack: int = 4,
                             gpus_per_server: int = 2,
                             gpu_preset: str = "H100_SXM",
                             T_ambient_C: float = 25.0,
                             workload: str = "sinusoidal",
                             workload_period_s: float = 300.0) -> DataCenterModel:
    """Build a small data center with configurable workload parameters."""
    racks = [build_default_rack(n_servers=servers_per_rack,
                                gpus_per_server=gpus_per_server,
                                gpu_preset=gpu_preset,
                                rack_id=i,
                                workload=workload,
                                workload_period_s=workload_period_s)
             for i in range(n_racks)]
    cdu = CDU(T_facility_C=T_ambient_C + 5.0)
    tower = CoolingTower(T_ambient_C=T_ambient_C,
                         T_wetbulb_C=T_ambient_C - 7.0)
    return DataCenterModel(racks=racks, cdu=cdu, cooling_tower=tower)
