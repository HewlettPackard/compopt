"""
CompOpt Test Suite — Vectorized Environments & Config Loader
=============================================================
Tests for BatchSimulator and configuration loading.
"""

import numpy as np
import pytest

import compopt
from compopt.utils.vec_env import BatchSimulator, benchmark_throughput
from compopt.configs.loader import list_presets, load_config, build_from_config


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized envs
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchSimulator:
    def test_create(self):
        batch = BatchSimulator(
            lambda: compopt.make("ChipThermal-v0", dt=1.0,
                                 episode_length_s=60.0),
            n_envs=2)
        assert len(batch.envs) == 2
        batch.close()

    def test_reset(self):
        batch = BatchSimulator(
            lambda: compopt.make("ChipThermal-v0", dt=1.0,
                                 episode_length_s=60.0),
            n_envs=3)
        obs = batch.reset()
        assert obs.shape == (3, 7)
        batch.close()

    def test_step(self):
        batch = BatchSimulator(
            lambda: compopt.make("ChipThermal-v0", dt=1.0,
                                 episode_length_s=60.0),
            n_envs=2)
        batch.reset()
        actions = np.array([[0.5], [0.7]], dtype=np.float32)
        obs, rewards, terms, truncs, infos = batch.step(actions)
        assert obs.shape == (2, 7)
        assert rewards.shape == (2,)
        assert len(infos) == 2
        batch.close()

    def test_auto_reset_on_truncation(self):
        batch = BatchSimulator(
            lambda: compopt.make("ChipThermal-v0", dt=10.0,
                                 episode_length_s=20.0),
            n_envs=2)
        batch.reset()
        actions = np.array([[0.5], [0.5]], dtype=np.float32)
        # After 2 steps of dt=10 → truncation → auto-reset
        for _ in range(5):
            obs, rewards, terms, truncs, infos = batch.step(actions)
        # Should still have valid obs (auto-reset happened)
        assert obs.shape == (2, 7)
        assert np.all(np.isfinite(obs))
        batch.close()

    def test_benchmark_throughput(self):
        result = benchmark_throughput(
            lambda: compopt.make("ChipThermal-v0", dt=1.0,
                                 episode_length_s=60.0),
            n_envs=2, n_steps=10)
        assert "steps_per_second" in result
        assert result["total_steps"] == 20
        assert result["wall_time_s"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_list_presets(self):
        presets = list_presets()
        assert len(presets) >= 3
        assert "small_ai_cluster" in presets

    def test_load_config(self):
        cfg = load_config("small_ai_cluster")
        assert "system" in cfg
        assert "power" in cfg
        assert "scheduler" in cfg

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")

    def test_build_from_config(self):
        result = build_from_config("small_ai_cluster", n_jobs=10)
        assert "datacenter" in result
        assert "scheduler" in result
        assert "jobs" in result
        assert len(result["jobs"]) == 10
        dc = result["datacenter"]
        assert hasattr(dc, "racks")
        assert dc.PUE >= 1.0
