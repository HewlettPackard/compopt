"""
compopt.configs.loader
======================
Load system configuration presets from JSON files.

Provides ``load_config()`` to load a named preset and ``build_from_config()``
to instantiate a full DataCenterModel + Scheduler from a config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from compopt.physics.server import (
    DataCenterModel, RackModel, ServerNode, build_default_rack,
    build_default_datacenter)
from compopt.physics.fluids import CDU, CoolantLoop, CoolingTower
from compopt.physics.chip import make_gpu
from compopt.physics.workloads import make_sinusoidal_profile
from compopt.scheduling.scheduler import Scheduler, SchedulerConfig, SchedulingPolicy
from compopt.scheduling.jobs import generate_random_jobs


_CONFIG_DIR = Path(__file__).parent


def list_presets() -> list:
    """List available configuration presets."""
    return [p.stem for p in _CONFIG_DIR.glob("*.json")]


def load_config(name: str) -> dict:
    """
    Load a configuration preset by name.

    Parameters
    ----------
    name : preset name (without .json extension)

    Returns
    -------
    dict with keys: system, power, scheduler, cooling
    """
    path = _CONFIG_DIR / f"{name}.json"
    if not path.exists():
        available = list_presets()
        raise FileNotFoundError(
            f"Config '{name}' not found. Available: {available}")
    with open(path) as f:
        return json.load(f)


def build_from_config(name: str,
                      workload: str = "sinusoidal",
                      n_jobs: int = 100,
                      seed: int = 42) -> Dict:
    """
    Build a DataCenterModel + Scheduler from a named preset.

    Returns
    -------
    dict with keys:
        "datacenter" : DataCenterModel
        "scheduler"  : Scheduler
        "jobs"       : list of Job
        "config"     : raw config dict
    """
    cfg = load_config(name)
    sys_cfg  = cfg.get("system", {})
    pwr_cfg  = cfg.get("power", {})
    sch_cfg  = cfg.get("scheduler", {})
    cool_cfg = cfg.get("cooling", {})

    n_racks        = sys_cfg.get("NUM_CDUS", 1) * sys_cfg.get("RACKS_PER_CDU", 2)
    nodes_per_rack = sys_cfg.get("NODES_PER_RACK", 4)
    gpus_per_node  = sys_cfg.get("GPUS_PER_NODE", 2)
    gpu_preset     = sys_cfg.get("GPU_PRESET", "H100_SXM")

    # Build datacenter
    dc = build_default_datacenter(
        n_racks=n_racks,
        servers_per_rack=nodes_per_rack,
        gpus_per_server=gpus_per_node,
        gpu_preset=gpu_preset,
        T_ambient_C=cool_cfg.get("T_FACILITY_C", 30.0) - 5.0)

    # Override CDU/tower params
    dc.cdu.UA_W_K = cool_cfg.get("CDU_UA_W_K", 50000)
    dc.cdu.T_facility_C = cool_cfg.get("T_FACILITY_C", 30.0)
    dc.cooling_tower.fan_power_W = cool_cfg.get("TOWER_FAN_POWER_W", 15000)
    dc.power_cost_dollar_per_kWh = pwr_cfg.get("POWER_COST", 0.094)
    dc.water_cost_dollar_per_L = cool_cfg.get("WATER_COST_DOLLAR_PER_L", 0.004)

    # Update server power params
    for rack in dc.racks:
        for server in rack.servers:
            server.P_cpu_idle_W = pwr_cfg.get("POWER_CPU_IDLE", 90)
            server.P_cpu_max_W  = pwr_cfg.get("POWER_CPU_MAX", 280)
            server.P_mem_W      = pwr_cfg.get("POWER_MEM", 74)
            server.P_nic_W      = pwr_cfg.get("POWER_NIC", 20)
            server.P_nvme_W     = pwr_cfg.get("POWER_NVME", 30)

    # Build scheduler
    total_nodes = n_racks * nodes_per_rack
    scheduler_config = SchedulerConfig(
        total_nodes=total_nodes,
        gpus_per_node=gpus_per_node,
        max_nodes_per_job=sch_cfg.get("MAX_NODES_PER_JOB", total_nodes))
    scheduler = Scheduler(config=scheduler_config)

    # Generate jobs
    jobs = generate_random_jobs(
        n_jobs=n_jobs,
        max_nodes=scheduler_config.max_nodes_per_job,
        seed=sch_cfg.get("SEED", seed),
        mean_arrival_s=sch_cfg.get("JOB_ARRIVAL_TIME", 900))

    return {
        "datacenter": dc,
        "scheduler":  scheduler,
        "jobs":       jobs,
        "config":     cfg,
    }
