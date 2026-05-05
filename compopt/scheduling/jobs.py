"""
compopt.scheduling.jobs
=======================
RAPS-compatible job model and synthetic job generators.

A **Job** is a compute workload submitted to a data center scheduler.
Each job specifies resource requirements (nodes, GPUs), wall-time,
power traces (CPU/GPU utilisation over time), and priority.

Adapted from Anonymous RAPS ``job_dict`` format with extensions for
AI/ML workload characterisation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


class JobState(Enum):
    """Job lifecycle states (RAPS-compatible)."""
    QUEUED     = "QUEUED"
    RUNNING    = "RUNNING"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"
    CANCELLED  = "CANCELLED"
    TIMEOUT    = "TIMEOUT"
    NODE_FAIL  = "NODE_FAIL"
    PREEMPTED  = "PREEMPTED"


class JobType(Enum):
    """Workload type labels for AI/ML job characterisation."""
    TRAINING        = "training"
    INFERENCE       = "inference"
    FINE_TUNING     = "fine_tuning"
    DATA_PIPELINE   = "data_pipeline"
    HPC_SIMULATION  = "hpc_simulation"
    BENCHMARK       = "benchmark"
    GENERIC         = "generic"


@dataclass
class Job:
    """
    A compute job to be scheduled on data-center nodes.

    RAPS-compatible fields
    ----------------------
    job_id            : unique identifier
    name              : human-readable name
    nodes_required    : number of compute nodes requested
    gpus_per_node     : GPUs per node (for GPU-aware scheduling)
    wall_time_s       : requested wall-clock time [s]
    cpu_trace         : CPU utilisation trace [0, n_cpus] or scalar
    gpu_trace         : GPU utilisation trace [0, n_gpus] or scalar
    submit_time_s     : time at which job enters the queue [s]
    priority          : scheduling priority (higher = more important)

    CompOpt extensions
    ------------------
    job_type          : AI/ML workload type label
    power_budget_W    : optional per-job power cap [W]
    flops_required    : total compute requirement [FLOP]
    memory_GB         : per-node memory requirement [GB]
    deadline_s        : optional hard deadline [s from submit]
    thermal_sensitive : if True, job performance degrades with throttling
    """
    job_id:           int
    name:             str            = "unnamed_job"
    nodes_required:   int            = 1
    gpus_per_node:    int            = 1
    wall_time_s:      float          = 3600.0
    cpu_trace:        Union[float, np.ndarray] = 0.5
    gpu_trace:        Union[float, np.ndarray] = 0.8
    submit_time_s:    float          = 0.0
    priority:         int            = 0

    job_type:         JobType        = JobType.GENERIC
    power_budget_W:   Optional[float] = None
    flops_required:   float          = 0.0
    memory_GB:        float          = 0.0
    deadline_s:       Optional[float] = None
    thermal_sensitive: bool          = False

    # Runtime state (set by scheduler)
    state:            JobState       = field(default=JobState.QUEUED)
    start_time_s:     Optional[float] = None
    end_time_s:       Optional[float] = None
    assigned_nodes:   Optional[List[int]] = None
    energy_consumed_J: float         = 0.0

    @property
    def elapsed_s(self) -> float:
        if self.start_time_s is None:
            return 0.0
        end = self.end_time_s if self.end_time_s is not None else 0.0
        return end - self.start_time_s

    @property
    def remaining_s(self) -> float:
        if self.start_time_s is None:
            return self.wall_time_s
        return max(0.0, self.wall_time_s - self.elapsed_s)

    @property
    def is_active(self) -> bool:
        return self.state == JobState.RUNNING

    def gpu_util_at(self, t_relative: float) -> float:
        """GPU utilisation at time *t_relative* seconds since job start."""
        if isinstance(self.gpu_trace, (int, float)):
            return float(self.gpu_trace)
        idx = int(t_relative / 15.0)  # RAPS default trace quanta = 15 s
        idx = min(idx, len(self.gpu_trace) - 1)
        return float(self.gpu_trace[max(0, idx)])

    def cpu_util_at(self, t_relative: float) -> float:
        """CPU utilisation at time *t_relative* seconds since job start."""
        if isinstance(self.cpu_trace, (int, float)):
            return float(self.cpu_trace)
        idx = int(t_relative / 15.0)
        idx = min(idx, len(self.cpu_trace) - 1)
        return float(self.cpu_trace[max(0, idx)])

    def to_raps_dict(self) -> dict:
        """Export as RAPS-compatible job_dict."""
        return {
            "nodes_required":  self.nodes_required,
            "name":            self.name,
            "cpu_trace":       self.cpu_trace,
            "gpu_trace":       self.gpu_trace,
            "wall_time":       self.wall_time_s,
            "end_state":       self.state.value,
            "requested_nodes": self.assigned_nodes,
            "submit_time":     self.submit_time_s,
            "id":              self.job_id,
            "priority":        self.priority,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Job Generators
# ──────────────────────────────────────────────────────────────────────────────

def generate_random_jobs(n_jobs: int,
                         max_nodes: int = 64,
                         max_gpus_per_node: int = 4,
                         min_wall_s: float = 300.0,
                         max_wall_s: float = 43200.0,
                         mean_arrival_s: float = 120.0,
                         seed: int = 42,
                         job_end_probs: Optional[dict] = None) -> List[Job]:
    """
    Generate *n_jobs* random compute jobs with Poisson arrivals
    (RAPS-compatible synthetic workload generator).

    Parameters
    ----------
    n_jobs       : number of jobs to create
    max_nodes    : maximum nodes any single job can request
    max_wall_s   : maximum wall-time [s]
    mean_arrival_s: mean inter-arrival time [s]
    seed         : random seed for reproducibility
    job_end_probs: dict of {state_name: probability}

    Returns
    -------
    List[Job] sorted by submit_time_s
    """
    if job_end_probs is None:
        job_end_probs = {
            "COMPLETED": 0.63, "FAILED": 0.13, "CANCELLED": 0.12,
            "TIMEOUT": 0.11, "NODE_FAIL": 0.01,
        }

    rng = np.random.default_rng(seed)
    jobs = []
    t_submit = 0.0

    type_choices = list(JobType)

    for i in range(n_jobs):
        t_submit += rng.exponential(mean_arrival_s)
        n_nodes   = rng.integers(1, max(2, max_nodes + 1))
        gpus_pn   = rng.choice([1, 2, 4, 8])
        wall_s    = rng.uniform(min_wall_s, max_wall_s)
        gpu_util  = rng.uniform(0.2, 1.0)
        cpu_util  = rng.uniform(0.1, 0.9)
        jtype     = rng.choice(type_choices)
        priority  = rng.integers(0, 10)

        jobs.append(Job(
            job_id=i,
            name=f"job_{i}_{jtype.value}",
            nodes_required=int(n_nodes),
            gpus_per_node=int(gpus_pn),
            wall_time_s=float(wall_s),
            cpu_trace=float(cpu_util),
            gpu_trace=float(gpu_util),
            submit_time_s=float(t_submit),
            priority=int(priority),
            job_type=jtype,
            thermal_sensitive=bool(rng.random() < 0.3),
        ))

    return jobs


def generate_ai_training_jobs(n_jobs: int,
                              seed: int = 42,
                              mean_arrival_s: float = 600.0) -> List[Job]:
    """
    Generate AI training workloads with high GPU utilisation and long durations.
    Simulates LLM training, vision model training, etc.
    """
    rng = np.random.default_rng(seed)
    jobs = []
    t_submit = 0.0

    for i in range(n_jobs):
        t_submit += rng.exponential(mean_arrival_s)
        n_nodes = rng.choice([8, 16, 32, 64, 128])
        wall_s  = rng.uniform(3600, 86400)  # 1h to 24h
        # AI training → high, sustained GPU util with some variance
        n_steps = int(wall_s / 15.0)
        gpu_trace = np.clip(rng.normal(0.92, 0.05, size=n_steps), 0.5, 1.0)

        jobs.append(Job(
            job_id=1000 + i,
            name=f"llm_train_{i}",
            nodes_required=int(n_nodes),
            gpus_per_node=4,
            wall_time_s=float(wall_s),
            cpu_trace=float(rng.uniform(0.3, 0.6)),
            gpu_trace=gpu_trace,
            submit_time_s=float(t_submit),
            priority=rng.integers(5, 10),
            job_type=JobType.TRAINING,
            thermal_sensitive=True,
            memory_GB=80.0,
        ))

    return jobs


def generate_inference_jobs(n_jobs: int,
                            seed: int = 42,
                            mean_arrival_s: float = 30.0) -> List[Job]:
    """
    Generate inference-serving workloads: short, bursty, high GPU util.
    """
    rng = np.random.default_rng(seed)
    jobs = []
    t_submit = 0.0

    for i in range(n_jobs):
        t_submit += rng.exponential(mean_arrival_s)
        wall_s = rng.uniform(10, 300)
        gpu_util = rng.uniform(0.6, 1.0)

        jobs.append(Job(
            job_id=5000 + i,
            name=f"inference_{i}",
            nodes_required=1,
            gpus_per_node=rng.choice([1, 2]),
            wall_time_s=float(wall_s),
            cpu_trace=0.2,
            gpu_trace=float(gpu_util),
            submit_time_s=float(t_submit),
            priority=rng.integers(7, 10),
            job_type=JobType.INFERENCE,
            power_budget_W=float(rng.uniform(300, 600)),
        ))

    return jobs
