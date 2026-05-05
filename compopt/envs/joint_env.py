"""
compopt.envs.joint_env
======================
Gymnasium environment for **joint** scheduling + cooling optimisation.

**Difficulty level: Expert** — full co-optimisation of workload placement
and thermo-fluidic cooling in a data center.

Actions (Dict space):
    "cooling"    : Box(3,)  — [rack_flow, cdu_pump, tower_fan] normalised
    "scheduling" : Discrete — queue index to prioritise (0 = no-op)

Observation (Dict space):
    "thermal"    : Box(16,) — data-center thermal/energy state
    "scheduler"  : Box(12,) — scheduler metrics + queue summary
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional

from compopt.physics.server import DataCenterModel, build_default_datacenter
from compopt.scheduling.scheduler import Scheduler, SchedulerConfig, SchedulingPolicy
from compopt.scheduling.jobs import Job, generate_random_jobs
from compopt.rewards.functions import CompositeReward, joint_reward


class JointDataCenterEnv(gym.Env):
    """
    Joint scheduling + cooling Gymnasium environment.

    This is the most challenging benchmark level, requiring the agent to
    simultaneously:
    1. Schedule jobs onto nodes (affecting power/thermal load distribution)
    2. Control cooling actuators (minimising energy, water, cost)
    3. Maintain thermal safety and SLA compliance

    The environment supports both Dict and flattened observation/action modes.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 n_racks: int            = 2,
                 servers_per_rack: int   = 4,
                 gpus_per_server: int    = 2,
                 gpu_preset: str         = "H100_SXM",
                 T_ambient_C: float      = 25.0,
                 dt: float               = 10.0,
                 episode_length_s: float = 7200.0,
                 n_jobs: int             = 200,
                 max_visible_queue: int  = 10,
                 reward_fn: Optional[CompositeReward] = None,
                 render_mode: Optional[str] = None,
                 flatten: bool           = False,
                 seed: int               = 42,
                 # Cooling ranges
                 rack_flow_min: float    = 0.5,
                 rack_flow_max: float    = 4.0,
                 cdu_flow_min: float     = 1.0,
                 cdu_flow_max: float     = 8.0,
                 fan_power_min: float    = 5000.0,
                 fan_power_max: float    = 30000.0):
        super().__init__()

        self.dt               = dt
        self.episode_length_s = episode_length_s
        self.max_visible      = max_visible_queue
        self.reward_fn        = reward_fn or joint_reward()
        self.render_mode      = render_mode
        self.flatten          = flatten

        self._n_racks          = n_racks
        self._servers_per_rack = servers_per_rack
        self._gpus_per_server  = gpus_per_server
        self._gpu_preset       = gpu_preset
        self._T_ambient_C      = T_ambient_C
        self._n_jobs           = n_jobs
        self._seed             = seed

        self.rack_flow_min  = rack_flow_min
        self.rack_flow_max  = rack_flow_max
        self.cdu_flow_min   = cdu_flow_min
        self.cdu_flow_max   = cdu_flow_max
        self.fan_power_min  = fan_power_min
        self.fan_power_max  = fan_power_max

        total_nodes = n_racks * servers_per_rack
        self.scheduler_config = SchedulerConfig(
            total_nodes=total_nodes,
            gpus_per_node=gpus_per_server,
            policy=SchedulingPolicy.RL_AGENT)  # Use RL_AGENT for proper RL control
        self.scheduler = Scheduler(config=self.scheduler_config)
        self._jobs_template = generate_random_jobs(
            n_jobs, max_nodes=total_nodes, seed=seed)

        self.dc = build_default_datacenter(
            n_racks=n_racks,
            servers_per_rack=servers_per_rack,
            gpus_per_server=gpus_per_server,
            gpu_preset=gpu_preset,
            T_ambient_C=T_ambient_C)

        if flatten:
            # Flattened: 3 cooling + 1 scheduling = 4 continuous
            self.action_space = spaces.Box(
                low=np.zeros(4, dtype=np.float32),
                high=np.ones(4, dtype=np.float32))
            self.observation_space = spaces.Box(
                low=np.full(28, -1e3, dtype=np.float32),
                high=np.full(28, 1e6, dtype=np.float32))
        else:
            self.action_space = spaces.Dict({
                "cooling": spaces.Box(
                    low=np.zeros(3, dtype=np.float32),
                    high=np.ones(3, dtype=np.float32)),
                "scheduling": spaces.Discrete(max_visible_queue + 1),
            })
            self.observation_space = spaces.Dict({
                "thermal": spaces.Box(
                    low=np.full(16, -1e3, dtype=np.float32),
                    high=np.full(16, 1e6, dtype=np.float32)),
                "scheduler": spaces.Box(
                    low=np.full(12, -1.0, dtype=np.float32),
                    high=np.full(12, 1e6, dtype=np.float32)),
            })

        self._elapsed_s  = 0.0
        self._prev_cost  = 0.0
        self._prev_water = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.dc = build_default_datacenter(
            n_racks=self._n_racks,
            servers_per_rack=self._servers_per_rack,
            gpus_per_server=self._gpus_per_server,
            gpu_preset=self._gpu_preset,
            T_ambient_C=self._T_ambient_C)

        self.scheduler.reset()
        import copy
        jobs = copy.deepcopy(self._jobs_template)
        self.scheduler.submit_batch(jobs)

        self._elapsed_s  = 0.0
        self._prev_cost  = 0.0
        self._prev_water = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """Advance one time-step with joint cooling + scheduling actions.

        Args:
            action: Dict with ``"cooling"`` (shape 3) and ``"scheduling"``
                (int), or a flat array of length 4 when ``flatten=True``.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # Parse action
        if self.flatten:
            action = np.asarray(action, dtype=np.float32)
            cooling_action = np.clip(action[:3], 0.0, 1.0)
            sched_idx = int(np.clip(action[3] * self.max_visible, 0,
                                     self.max_visible))
        else:
            cooling_action = np.clip(action["cooling"], 0.0, 1.0)
            sched_idx = int(action["scheduling"])

        # ── Apply cooling actions ─────────────────────────────────────
        rack_flow = self.rack_flow_min + cooling_action[0] * (
            self.rack_flow_max - self.rack_flow_min)
        for rack in self.dc.racks:
            rack.rack_coolant.m_dot_kg_s = rack_flow

        cdu_flow = self.cdu_flow_min + cooling_action[1] * (
            self.cdu_flow_max - self.cdu_flow_min)
        self.dc.cdu.hot_loop.m_dot_kg_s = cdu_flow

        fan_power = self.fan_power_min + cooling_action[2] * (
            self.fan_power_max - self.fan_power_min)
        self.dc.cooling_tower.fan_power_W = fan_power

        # ── Apply scheduling action ───────────────────────────────────
        if 0 < sched_idx <= len(self.scheduler.queue):
            job = self.scheduler.queue.pop(sched_idx - 1)
            self.scheduler.queue.insert(0, job)

        # ── Couple scheduler ↔ physics ────────────────────────────────
        # Map running jobs to server power profiles
        self._update_server_power()

        prev_completed = len(self.scheduler.completed)
        self.scheduler.step(self.dt)

        # ── Advance physics ───────────────────────────────────────────
        self.dc.step(self.dt)
        self._elapsed_s += self.dt

        new_completed = len(self.scheduler.completed) - prev_completed

        obs = self._get_obs()

        step_cost  = self.dc.total_cost_dollar - self._prev_cost
        step_water = self.dc.cooling_tower.water_used_L - self._prev_water
        self._prev_cost  = self.dc.total_cost_dollar
        self._prev_water = self.dc.cooling_tower.water_used_L

        info = {
            "T_hotspot_C":         self.dc.T_hotspot_C,
            "T_gpu_hotspot_C":     self.dc.T_hotspot_C,
            "PUE":                 self.dc.PUE,
            "WUE_L_per_kWh":      self.dc.WUE_L_per_kWh,
            "total_power_kW":      self.dc.total_power_W / 1000.0,
            "IT_power_kW":         self.dc.it_power_W / 1000.0,
            "step_cost_dollar":    step_cost,
            "water_used_L":        step_water,
            "flow_kg_s":           rack_flow,
            "m_dot_kg_s":          rack_flow,
            "jobs_completed_step": float(new_completed),
            "queue_length":        float(len(self.scheduler.queue)),
            "n_running":           float(len(self.scheduler.running)),
            "node_utilisation":    self.scheduler.utilisation,
            "gpu_utilisation":     self.scheduler.gpu_utilisation,
            "n_sla_violations":    0.0,
        }

        reward, breakdown = self.reward_fn(info)
        info["reward_breakdown"] = breakdown

        truncated = self._elapsed_s >= self.episode_length_s
        return obs, float(reward), False, truncated, info

    def _update_server_power(self):
        """
        Couple scheduler node allocations to physical server GPU utilisation.

        Running jobs increase the CPU utilisation of their assigned nodes.
        """
        # Reset all utilisation
        all_servers = []
        for rack in self.dc.racks:
            for server in rack.servers:
                server.cpu_util = 0.1  # idle baseline
                all_servers.append(server)

        for job in self.scheduler.running:
            if job.assigned_nodes:
                t_rel = self.dc.time_s - (job.start_time_s or 0.0)
                cpu_u = job.cpu_util_at(t_rel)
                for nid in job.assigned_nodes:
                    if 0 <= nid < len(all_servers):
                        all_servers[nid].cpu_util = min(
                            1.0, all_servers[nid].cpu_util + cpu_u)

    def _get_obs(self):
        thermal_obs = np.nan_to_num(self.dc.get_observation(), nan=0.0, posinf=1e6, neginf=-1e6)
        thermal_obs = np.clip(thermal_obs, -1e6, 1e6).astype(np.float32)

        sched_obs = self.scheduler.get_observation()
        queue = self.scheduler.queue[:self.max_visible]
        if queue:
            extra = np.array([
                np.mean([j.nodes_required for j in queue]),
                np.mean([j.gpus_per_node for j in queue]),
                np.mean([j.wall_time_s for j in queue]) / 3600.0,
                max(j.priority for j in queue),
            ], dtype=np.float64)
        else:
            extra = np.zeros(4, dtype=np.float64)
        sched_full = np.concatenate([sched_obs, extra]).astype(np.float32)

        if self.flatten:
            return np.concatenate([thermal_obs, sched_full])
        else:
            return {"thermal": thermal_obs, "scheduler": sched_full}

    def render(self):
        if self.render_mode == "human":
            dc = self.dc.get_full_state_for_llm()["datacenter"]
            m  = self.scheduler.get_metrics()
            print(f"t={dc['time_s']:.0f}s  PUE={dc['PUE']:.3f}  "
                  f"T_hot={dc['T_hotspot_C']:.1f}°C  "
                  f"queue={m['queue_length']:.0f}  "
                  f"util={m['node_utilisation']:.2f}  "
                  f"cost=${dc['cost_dollar']:.4f}")
