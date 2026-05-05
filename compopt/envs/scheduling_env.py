"""
compopt.envs.scheduling_env
===========================
Gymnasium environment for job scheduling on a data-center cluster.

**Difficulty level: Medium** — resource allocation without thermal coupling.

- **Actions**: discrete or multi-discrete — which queued job to dispatch next,
  or continuous priority adjustment.
- **Observation**: scheduler metrics + queue summary.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from compopt.scheduling.scheduler import Scheduler, SchedulerConfig, SchedulingPolicy
from compopt.scheduling.jobs import Job, JobState, generate_random_jobs
from compopt.rewards.functions import CompositeReward, scheduling_reward


class SchedulingEnv(gym.Env):
    """
    Gymnasium env for data-center job scheduling.

    The agent decides which queued job to dispatch at each decision point.

    - **Action space**: ``Discrete(max_dispatchable)`` —
      Index into the visible queue; 0 = skip (don't dispatch).
    - **Observation**: ``Box(12,)`` — scheduler metrics + queue summary.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 scheduler_config: Optional[SchedulerConfig] = None,
                 jobs: Optional[List[Job]] = None,
                 n_jobs: int             = 100,
                 dt: float               = 60.0,
                 episode_length_s: float = 86400.0,
                 max_visible_queue: int  = 10,
                 reward_fn: Optional[CompositeReward] = None,
                 render_mode: Optional[str] = None,
                 seed: int               = 42):
        super().__init__()

        self.dt               = dt
        self.episode_length_s = episode_length_s
        self.max_visible      = max_visible_queue
        self.reward_fn        = reward_fn or scheduling_reward()
        self.render_mode      = render_mode
        self._n_jobs          = n_jobs
        self._seed            = seed

        self.scheduler_config = scheduler_config or SchedulerConfig(
            total_nodes=64, gpus_per_node=4, policy=SchedulingPolicy.RL_AGENT)

        self.scheduler = Scheduler(config=self.scheduler_config)
        self._jobs_template = jobs or generate_random_jobs(
            n_jobs, max_nodes=self.scheduler_config.total_nodes, seed=seed)

        # Action: index into visible queue (0 = no-op)
        self.action_space = spaces.Discrete(max_visible_queue + 1)

        # Observation: 8 scheduler metrics + 4 queue summary features = 12
        self.observation_space = spaces.Box(
            low=np.full(12, -1.0, dtype=np.float32),
            high=np.full(12, 1e6, dtype=np.float32))

        self._elapsed_s       = 0.0
        self._prev_completed  = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scheduler.reset()

        # Re-generate or re-submit jobs
        import copy
        jobs = copy.deepcopy(self._jobs_template)
        self.scheduler.submit_batch(jobs)
        self._elapsed_s      = 0.0
        self._prev_completed = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Advance one scheduling decision step.

        Args:
            action: Integer index into the visible queue (0 = skip).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = int(action)

        # If agent selects a queue index, move that job to front
        if 0 < action <= len(self.scheduler.queue):
            idx = action - 1
            job = self.scheduler.queue.pop(idx)
            self.scheduler.queue.insert(0, job)

        prev_completed = len(self.scheduler.completed)
        self.scheduler.step(self.dt)
        self._elapsed_s += self.dt
        new_completed = len(self.scheduler.completed) - prev_completed

        obs = self._get_obs()

        # Check SLA violations
        sla_violations = 0
        for job in self.scheduler.completed[-new_completed:] if new_completed > 0 else []:
            if job.deadline_s is not None:
                total_time = (job.end_time_s or 0) - job.submit_time_s
                if total_time > job.deadline_s:
                    sla_violations += 1

        info = {
            "jobs_completed_step": float(new_completed),
            "queue_length":       float(len(self.scheduler.queue)),
            "n_running":          float(len(self.scheduler.running)),
            "node_utilisation":   self.scheduler.utilisation,
            "gpu_utilisation":    self.scheduler.gpu_utilisation,
            "n_sla_violations":   float(sla_violations),
        }

        reward, breakdown = self.reward_fn(info)
        info["reward_breakdown"] = breakdown

        truncated = (self._elapsed_s >= self.episode_length_s or
                     (len(self.scheduler.queue) == 0 and
                      len(self.scheduler.running) == 0))

        return obs, float(reward), False, truncated, info

    def _get_obs(self) -> np.ndarray:
        sched_obs = self.scheduler.get_observation()  # shape (8,)
        # Queue summary: top-k job sizes + GPU demand
        queue = self.scheduler.queue[:self.max_visible]
        if queue:
            avg_nodes = np.mean([j.nodes_required for j in queue])
            avg_gpus  = np.mean([j.gpus_per_node for j in queue])
            avg_wall  = np.mean([j.wall_time_s for j in queue]) / 3600.0
            max_pri   = max(j.priority for j in queue)
        else:
            avg_nodes = avg_gpus = avg_wall = max_pri = 0.0

        extra = np.array([avg_nodes, avg_gpus, avg_wall, max_pri],
                         dtype=np.float64)
        return np.concatenate([sched_obs, extra]).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            m = self.scheduler.get_metrics()
            print(f"t={m['time_s']:.0f}s  queue={m['queue_length']:.0f}  "
                  f"running={m['n_running']:.0f}  "
                  f"util={m['node_utilisation']:.2f}  "
                  f"completed={m['n_completed']:.0f}")
