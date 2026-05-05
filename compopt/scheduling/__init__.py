"""
compopt.scheduling
==================

Job scheduling and workload generation for data-center simulation.

Sub-modules
-----------
jobs
    RAPS-compatible job model and synthetic generators.
scheduler
    Multi-policy scheduler (FCFS, SJF, priority, backfill, thermal-aware).
"""

from compopt.scheduling.jobs import (
    Job, JobState, JobType,
    generate_random_jobs, generate_ai_training_jobs, generate_inference_jobs,
)
from compopt.scheduling.scheduler import (
    Scheduler, SchedulerConfig, SchedulingPolicy,
)

__all__ = [
    "Job", "JobState", "JobType",
    "generate_random_jobs", "generate_ai_training_jobs", "generate_inference_jobs",
    "Scheduler", "SchedulerConfig", "SchedulingPolicy",
]
