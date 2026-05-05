"""
CompOpt Test Suite — Scheduling
================================
Tests for job generation, scheduler policies, and scheduling env.
"""

import numpy as np
import pytest

from compopt.scheduling.jobs import (
    Job, JobState, JobType, generate_random_jobs,
    generate_ai_training_jobs, generate_inference_jobs,
)
from compopt.scheduling.scheduler import Scheduler, SchedulerConfig, SchedulingPolicy


class TestJob:
    def test_create(self):
        job = Job(job_id=0, nodes_required=2, wall_time_s=3600,
                  submit_time_s=0.0)
        assert job.state == JobState.QUEUED

    def test_random_jobs(self):
        jobs = generate_random_jobs(50, max_nodes=16, seed=42)
        assert len(jobs) == 50
        assert all(j.nodes_required >= 1 for j in jobs)
        assert all(j.wall_time_s > 0 for j in jobs)

    def test_ai_training_jobs(self):
        jobs = generate_ai_training_jobs(20, seed=42)
        assert len(jobs) == 20
        assert all(j.job_type == JobType.TRAINING for j in jobs)

    def test_inference_jobs(self):
        jobs = generate_inference_jobs(30, seed=42)
        assert len(jobs) == 30
        assert all(j.job_type == JobType.INFERENCE for j in jobs)


class TestScheduler:
    def test_create(self):
        config = SchedulerConfig(total_nodes=16, gpus_per_node=4)
        sched = Scheduler(config=config)
        assert sched.utilisation == 0.0

    def test_submit_and_step(self):
        config = SchedulerConfig(total_nodes=16, gpus_per_node=4,
                                  policy=SchedulingPolicy.FCFS)
        sched = Scheduler(config=config)
        jobs = generate_random_jobs(10, max_nodes=4, seed=42)
        sched.submit_batch(jobs)
        assert len(sched.queue) == 10

        for _ in range(100):
            sched.step(60.0)

        # Some jobs should have started
        assert len(sched.running) + len(sched.completed) > 0

    def test_observation_shape(self):
        config = SchedulerConfig(total_nodes=16, gpus_per_node=4)
        sched = Scheduler(config=config)
        obs = sched.get_observation()
        assert obs.shape == (8,)

    def test_reset(self):
        config = SchedulerConfig(total_nodes=16, gpus_per_node=4)
        sched = Scheduler(config=config)
        jobs = generate_random_jobs(10, max_nodes=4, seed=42)
        sched.submit_batch(jobs)
        sched.step(60.0)
        sched.reset()
        assert len(sched.queue) == 0
        assert len(sched.running) == 0
        assert len(sched.completed) == 0

    def test_different_policies(self):
        """All policies should run without error."""
        for policy in SchedulingPolicy:
            config = SchedulerConfig(total_nodes=16, gpus_per_node=4,
                                      policy=policy)
            sched = Scheduler(config=config)
            jobs = generate_random_jobs(10, max_nodes=4, seed=42)
            sched.submit_batch(jobs)
            for _ in range(20):
                sched.step(60.0)

    def test_utilisation_bounded(self):
        config = SchedulerConfig(total_nodes=16, gpus_per_node=4)
        sched = Scheduler(config=config)
        jobs = generate_random_jobs(50, max_nodes=4, seed=42)
        sched.submit_batch(jobs)
        for _ in range(50):
            sched.step(60.0)
        assert 0.0 <= sched.utilisation <= 1.0
