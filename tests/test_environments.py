"""
CompOpt Test Suite — Gymnasium Environments
============================================
Tests that all registered environments can be created, reset, and stepped.
"""

import numpy as np
import pytest

import compopt
from compopt.envs.registry import list_envs


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_envs(self):
        envs = list_envs()
        assert len(envs) >= 6
        ids = [e["id"] for e in envs]
        assert "ChipThermal-v0" in ids
        assert "RackCooling-v0" in ids
        assert "DataCenter-v0" in ids
        assert "Scheduling-v0" in ids
        assert "JointDC-v0" in ids
        assert "JointDCFlat-v0" in ids

    def test_make_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown env_id"):
            compopt.make("NonExistent-v0")


# ──────────────────────────────────────────────────────────────────────────────
# ChipThermal-v0
# ──────────────────────────────────────────────────────────────────────────────

class TestChipThermalEnv:
    def test_create(self):
        env = compopt.make("ChipThermal-v0")
        assert env is not None
        env.close()

    def test_reset(self):
        env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=60.0)
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert obs.dtype == np.float32
        env.close()

    def test_step(self):
        env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=60.0)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert "T_hotspot_C" in info
        env.close()

    def test_truncation(self):
        env = compopt.make("ChipThermal-v0", dt=10.0, episode_length_s=30.0)
        obs, _ = env.reset()
        truncated = False
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                break
        assert truncated
        env.close()

    def test_action_space_bounds(self):
        env = compopt.make("ChipThermal-v0")
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# RackCooling-v0
# ──────────────────────────────────────────────────────────────────────────────

class TestRackCoolingEnv:
    def test_create(self):
        env = compopt.make("RackCooling-v0", n_servers=2, gpus_per_server=1)
        assert env is not None
        env.close()

    def test_reset(self):
        env = compopt.make("RackCooling-v0", n_servers=2, gpus_per_server=1,
                          dt=1.0, episode_length_s=60.0)
        obs, info = env.reset()
        assert obs.shape == (10,)
        env.close()

    def test_step(self):
        env = compopt.make("RackCooling-v0", n_servers=2, gpus_per_server=1)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == (10,)
        assert "T_hotspot_C" in info
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# DataCenter-v0
# ──────────────────────────────────────────────────────────────────────────────

class TestDataCenterEnv:
    def test_create(self):
        env = compopt.make("DataCenter-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1)
        assert env is not None
        env.close()

    def test_action_dim(self):
        env = compopt.make("DataCenter-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1)
        assert env.action_space.shape == (3,)
        env.close()

    def test_obs_dim(self):
        env = compopt.make("DataCenter-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1)
        obs, _ = env.reset()
        assert obs.shape == (16,)
        env.close()

    def test_step_returns_valid(self):
        env = compopt.make("DataCenter-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          dt=5.0, episode_length_s=50.0)
        obs, _ = env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert "PUE" in info
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Scheduling-v0
# ──────────────────────────────────────────────────────────────────────────────

class TestSchedulingEnv:
    def test_create(self):
        env = compopt.make("Scheduling-v0", n_jobs=20, dt=60.0)
        assert env is not None
        env.close()

    def test_obs_dim(self):
        env = compopt.make("Scheduling-v0", n_jobs=20, dt=60.0)
        obs, _ = env.reset()
        assert obs.shape == (12,)
        env.close()

    def test_discrete_action(self):
        env = compopt.make("Scheduling-v0", n_jobs=20, dt=60.0)
        assert hasattr(env.action_space, 'n')  # Discrete
        env.close()

    def test_step(self):
        env = compopt.make("Scheduling-v0", n_jobs=20, dt=60.0,
                          episode_length_s=3600.0)
        obs, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert "queue_length" in info
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# JointDCFlat-v0
# ──────────────────────────────────────────────────────────────────────────────

class TestJointDCFlatEnv:
    def test_create(self):
        env = compopt.make("JointDCFlat-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          n_jobs=20)
        assert env is not None
        env.close()

    def test_obs_dim(self):
        env = compopt.make("JointDCFlat-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          n_jobs=20)
        obs, _ = env.reset()
        assert obs.shape == (28,)
        env.close()

    def test_action_dim(self):
        env = compopt.make("JointDCFlat-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          n_jobs=20)
        assert env.action_space.shape == (4,)
        env.close()

    def test_step(self):
        env = compopt.make("JointDCFlat-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          n_jobs=20, dt=10.0, episode_length_s=100.0)
        obs, _ = env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert "T_hotspot_C" in info or "T_gpu_hotspot_C" in info
        assert "queue_length" in info
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# JointDC-v0 (Dict spaces)
# ──────────────────────────────────────────────────────────────────────────────

class TestJointDCDictEnv:
    def test_create(self):
        env = compopt.make("JointDC-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          n_jobs=20)
        assert env is not None
        env.close()

    def test_dict_obs(self):
        env = compopt.make("JointDC-v0",
                          n_racks=1, servers_per_rack=2, gpus_per_server=1,
                          n_jobs=20)
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert "thermal" in obs
        assert "scheduler" in obs
        assert obs["thermal"].shape == (16,)
        assert obs["scheduler"].shape == (12,)
        env.close()
