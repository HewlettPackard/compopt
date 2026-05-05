"""
compopt.scheduling.scheduler
============================
Job schedulers for data-center resource allocation.

Implements RAPS-compatible scheduling policies:
- **FCFS**  : First-Come First-Served
- **SJF**   : Shortest Job First
- **PRQ**   : Priority Queue
- **Backfill**: FCFS with backfill (smaller jobs fill gaps)
- **ThermalAware**: Temperature-aware scheduling (CompOpt extension)

The scheduler integrates with the physical ``DataCenterModel`` to
enable co-optimisation of job placement and cooling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from compopt.scheduling.jobs import Job, JobState, JobType


class SchedulingPolicy(Enum):
    FCFS          = "fcfs"
    SJF           = "sjf"
    PRIORITY      = "prq"
    BACKFILL      = "backfill"
    THERMAL_AWARE = "thermal_aware"
    RL_AGENT      = "rl_agent"  # Let RL agent control queue order (no sorting)


@dataclass
class SchedulerConfig:
    """
    Scheduler configuration (RAPS-compatible).

    Parameters
    ----------
    total_nodes        : total compute nodes in the cluster
    gpus_per_node      : GPUs per node
    policy             : scheduling policy
    trace_quanta_s     : time between power samples [s]
    power_budget_W     : optional cluster-wide power cap [W]
    max_nodes_per_job  : maximum nodes per job
    preemption_enabled : whether higher-priority jobs can preempt
    """
    total_nodes:        int            = 64
    gpus_per_node:      int            = 4
    policy:             SchedulingPolicy = SchedulingPolicy.FCFS
    trace_quanta_s:     float          = 15.0
    power_budget_W:     Optional[float] = None
    max_nodes_per_job:  int            = 64
    preemption_enabled: bool           = False


@dataclass
class Scheduler:
    """
    Data-center job scheduler with multiple policies.

    Maintains a job queue, running set, and completed set.
    At each ``step()``, the scheduler attempts to dispatch queued jobs
    onto available nodes according to the configured policy.

    The scheduler also tracks:
    - Cluster utilisation (node and GPU)
    - Queue wait times
    - Job throughput / makespan
    - Power-aware metrics (if power_budget is set)
    """
    config:       SchedulerConfig
    time_s:       float          = 0.0

    # Job lists
    queue:        List[Job]      = field(default_factory=list)
    running:      List[Job]      = field(default_factory=list)
    completed:    List[Job]      = field(default_factory=list)

    # Node allocation: node_id → job_id (None = free)
    _node_alloc:  Dict[int, Optional[int]] = field(default_factory=dict)

    # Metrics accumulators
    _total_wait_s:    float = 0.0
    _total_completed: int   = 0
    _total_energy_J:  float = 0.0

    def __post_init__(self):
        for i in range(self.config.total_nodes):
            self._node_alloc[i] = None

    def reset(self):
        """Reset scheduler state."""
        self.time_s    = 0.0
        self.queue     = []
        self.running   = []
        self.completed = []
        self._node_alloc = {i: None for i in range(self.config.total_nodes)}
        self._total_wait_s    = 0.0
        self._total_completed = 0
        self._total_energy_J  = 0.0

    def submit(self, job: Job):
        """Submit a job to the queue."""
        job.state = JobState.QUEUED
        self.queue.append(job)

    def submit_batch(self, jobs: List[Job]):
        """Submit multiple jobs."""
        for j in jobs:
            self.submit(j)

    @property
    def free_nodes(self) -> List[int]:
        return [nid for nid, jid in self._node_alloc.items() if jid is None]

    @property
    def n_free_nodes(self) -> int:
        return len(self.free_nodes)

    @property
    def utilisation(self) -> float:
        """Node utilisation fraction [0, 1]."""
        n_used = self.config.total_nodes - self.n_free_nodes
        return n_used / max(1, self.config.total_nodes)

    @property
    def gpu_utilisation(self) -> float:
        """Estimated GPU utilisation across running jobs."""
        total_gpus = self.config.total_nodes * self.config.gpus_per_node
        used_gpus = sum(
            j.nodes_required * j.gpus_per_node
            for j in self.running)
        return min(1.0, used_gpus / max(1, total_gpus))

    def step(self, dt: float,
             node_temperatures: Optional[Dict[int, float]] = None,
             node_powers: Optional[Dict[int, float]] = None):
        """
        Advance scheduler by *dt* seconds.

        1. Retire completed/failed jobs.
        2. Dispatch queued jobs according to policy.
        3. Update metrics.

        Parameters
        ----------
        node_temperatures : optional dict of node_id → T_hotspot [°C]
                            (used by THERMAL_AWARE policy)
        node_powers       : optional dict of node_id → power [W]
                            (used for power-capped scheduling)
        """
        self.time_s += dt

        # ── 1. Retire finished jobs ──────────────────────────────────────
        still_running = []
        for job in self.running:
            elapsed = self.time_s - (job.start_time_s or 0.0)
            if elapsed >= job.wall_time_s:
                self._finish_job(job, JobState.COMPLETED)
            else:
                still_running.append(job)
        self.running = still_running

        # ── 2. Sort queue by policy ──────────────────────────────────────
        self._sort_queue(node_temperatures)

        # ── 3. Dispatch ──────────────────────────────────────────────────
        # For RL_AGENT policy: strict queue order (no backfilling)
        # For other policies: allow backfilling (try all jobs)
        if self.config.policy == SchedulingPolicy.RL_AGENT:
            # Only try to dispatch the first job (strict queue priority)
            if self.queue:
                job = self.queue[0]
                if job.submit_time_s <= self.time_s:
                    can_schedule, nodes = self._try_allocate(
                        job, node_temperatures, node_powers)
                    if can_schedule:
                        self._start_job(job, nodes)
                        self.queue.pop(0)
        else:
            # Default behavior: backfilling (try all jobs in order)
            new_queue = []
            for job in self.queue:
                if job.submit_time_s > self.time_s:
                    new_queue.append(job)
                    continue

                can_schedule, nodes = self._try_allocate(
                    job, node_temperatures, node_powers)
                if can_schedule:
                    self._start_job(job, nodes)
                else:
                    new_queue.append(job)
            self.queue = new_queue

    def _sort_queue(self, node_temps: Optional[Dict[int, float]] = None):
        policy = self.config.policy
        if policy == SchedulingPolicy.RL_AGENT:
            # RL agent controls queue order - do not re-sort
            return
        elif policy == SchedulingPolicy.FCFS:
            self.queue.sort(key=lambda j: j.submit_time_s)
        elif policy == SchedulingPolicy.SJF:
            self.queue.sort(key=lambda j: j.wall_time_s)
        elif policy == SchedulingPolicy.PRIORITY:
            self.queue.sort(key=lambda j: -j.priority)
        elif policy == SchedulingPolicy.THERMAL_AWARE:
            # Prefer smaller jobs when cluster is hot
            self.queue.sort(key=lambda j: (
                -j.priority, j.nodes_required))
        else:
            self.queue.sort(key=lambda j: j.submit_time_s)

    def _try_allocate(self, job: Job,
                      node_temps: Optional[Dict[int, float]] = None,
                      node_powers: Optional[Dict[int, float]] = None
                      ) -> Tuple[bool, List[int]]:
        """
        Try to allocate *job.nodes_required* free nodes.
        Returns (success, list_of_node_ids).
        """
        free = self.free_nodes
        if len(free) < job.nodes_required:
            return False, []

        # Power budget check
        if self.config.power_budget_W is not None and node_powers:
            current_power = sum(node_powers.get(n, 0) for n in
                                [nid for nid, jid in self._node_alloc.items()
                                 if jid is not None])
            # Rough estimate of new job power
            est_power = job.nodes_required * 700.0 * float(
                job.gpu_trace if isinstance(job.gpu_trace, (int, float))
                else np.mean(job.gpu_trace))
            if current_power + est_power > self.config.power_budget_W:
                return False, []

        # Thermal-aware: prefer coolest nodes
        if (self.config.policy == SchedulingPolicy.THERMAL_AWARE
                and node_temps):
            free_with_temps = [(nid, node_temps.get(nid, 50.0))
                               for nid in free]
            free_with_temps.sort(key=lambda x: x[1])
            selected = [nid for nid, _ in
                        free_with_temps[:job.nodes_required]]
        else:
            selected = free[:job.nodes_required]

        return True, selected

    def _start_job(self, job: Job, nodes: List[int]):
        job.state          = JobState.RUNNING
        job.start_time_s   = self.time_s
        job.assigned_nodes = nodes
        for nid in nodes:
            self._node_alloc[nid] = job.job_id
        self.running.append(job)

    def _finish_job(self, job: Job, state: JobState):
        job.state      = state
        job.end_time_s = self.time_s
        if job.assigned_nodes:
            for nid in job.assigned_nodes:
                self._node_alloc[nid] = None
        self.completed.append(job)
        if job.start_time_s is not None:
            self._total_wait_s += job.start_time_s - job.submit_time_s
        self._total_completed += 1

    # ── Metrics ───────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, float]:
        """Return scheduler metrics as a flat dict."""
        avg_wait = (self._total_wait_s / max(1, self._total_completed))
        return {
            "time_s":              self.time_s,
            "queue_length":        len(self.queue),
            "n_running":           len(self.running),
            "n_completed":         self._total_completed,
            "node_utilisation":    self.utilisation,
            "gpu_utilisation":     self.gpu_utilisation,
            "avg_wait_time_s":     avg_wait,
            "n_free_nodes":        self.n_free_nodes,
        }

    def get_observation(self) -> np.ndarray:
        """
        Compact scheduler observation for RL (shape: 8).

        [0] node_utilisation  [1] gpu_utilisation
        [2] queue_length      [3] n_running
        [4] avg_wait_time_s   [5] n_free_nodes
        [6] time_h            [7] queue_gpu_demand
        """
        m = self.get_metrics()
        queue_gpu_demand = sum(
            j.nodes_required * j.gpus_per_node for j in self.queue)
        return np.array([
            m["node_utilisation"],
            m["gpu_utilisation"],
            m["queue_length"],
            m["n_running"],
            m["avg_wait_time_s"],
            m["n_free_nodes"],
            self.time_s / 3600.0,
            float(queue_gpu_demand),
        ], dtype=np.float64)
