"""
CompOpt Test Suite — Rewards & Agents
======================================
Tests for composable reward functions and baseline agents.
"""

import numpy as np
import pytest

from compopt.rewards.functions import (
    CompositeReward, ThermalPenalty, ThermalViolation, HBMPenalty,
    EnergyPenalty, CoolantFlowPenalty, PUEReward, CostPenalty,
    WaterPenalty, WUEReward, ThroughputReward, QueuePenalty,
    UtilisationReward, SLAViolation,
    cooling_only_reward, datacenter_reward, scheduling_reward, joint_reward,
)
from compopt.agents.baselines import (
    RandomAgent, ConstantAgent, RuleBasedCoolingAgent,
    PIDCoolingAgent, DataCenterRuleAgent, FCFSSchedulingAgent,
)
from compopt.utils.metrics import EpisodeMetrics, evaluate_agent
import compopt


# ──────────────────────────────────────────────────────────────────────────────
# Reward components
# ──────────────────────────────────────────────────────────────────────────────

class TestRewardComponents:
    def test_thermal_penalty_safe(self):
        comp = ThermalPenalty(weight=1.0, target_C=80.0)
        info = {"T_hotspot_C": 75.0}
        assert comp(info) == 0.0

    def test_thermal_penalty_over(self):
        comp = ThermalPenalty(weight=1.0, target_C=80.0)
        info = {"T_hotspot_C": 85.0}
        assert comp(info) < 0.0

    def test_thermal_violation(self):
        comp = ThermalViolation(weight=100.0, limit_C=83.0)
        assert comp({"T_hotspot_C": 80.0}) == 0.0
        assert comp({"T_hotspot_C": 85.0}) == -100.0

    def test_energy_penalty(self):
        comp = EnergyPenalty(weight=0.01, P_ref_kW=100.0)
        assert comp({"total_power_kW": 50.0}) < 0.0

    def test_pue_reward(self):
        comp = PUEReward(weight=1.0)
        assert comp({"PUE": 1.2}) > 0.0
        assert comp({"PUE": 2.5}) < 0.0

    def test_throughput_reward(self):
        comp = ThroughputReward(weight=1.0)
        assert comp({"jobs_completed_step": 5.0}) == 5.0

    def test_composite_reward(self):
        reward_fn = CompositeReward([
            ThermalPenalty(weight=1.0, target_C=80.0),
            EnergyPenalty(weight=0.01, P_ref_kW=100.0),
        ])
        total, breakdown = reward_fn({"T_hotspot_C": 85.0, "total_power_kW": 50.0})
        assert total < 0.0
        assert "thermal_penalty" in breakdown
        assert "energy_penalty" in breakdown


class TestRewardPresets:
    def test_cooling_only(self):
        r = cooling_only_reward(target_C=80.0)
        assert isinstance(r, CompositeReward)

    def test_datacenter(self):
        r = datacenter_reward()
        assert isinstance(r, CompositeReward)

    def test_scheduling(self):
        r = scheduling_reward()
        assert isinstance(r, CompositeReward)

    def test_joint(self):
        r = joint_reward()
        assert isinstance(r, CompositeReward)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline Agents
# ──────────────────────────────────────────────────────────────────────────────

class TestBaselineAgents:
    def test_random_agent(self):
        env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=10.0)
        agent = RandomAgent(env.action_space)
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        assert env.action_space.contains(action)
        env.close()

    def test_constant_agent(self):
        agent = ConstantAgent(action=[0.5])
        action, _ = agent.predict(np.zeros(7))
        assert np.isclose(action[0], 0.5)

    def test_pid_agent(self):
        agent = PIDCoolingAgent(target_C=80.0, Kp=0.05, Ki=0.001, Kd=0.01)
        # obs[1] = T_hotspot (above target)
        obs = np.array([75.0, 85.0, 70.0, 60.0, 40.0, 42.0, 500.0],
                       dtype=np.float32)
        action, _ = agent.predict(obs)
        assert 0.0 <= action[0] <= 1.0

    def test_rule_based_agent(self):
        agent = RuleBasedCoolingAgent(target_C=80.0, deadband=2.0)
        obs = np.array([75.0, 85.0, 70.0, 60.0, 40.0, 42.0, 500.0],
                       dtype=np.float32)
        action, _ = agent.predict(obs)
        assert 0.0 <= action[0] <= 1.0

    def test_dc_rule_agent(self):
        agent = DataCenterRuleAgent(target_C=80.0)
        obs = np.zeros(16, dtype=np.float32)
        obs[0] = 85.0  # T_hotspot
        action, _ = agent.predict(obs)
        assert action.shape == (3,)
        assert np.all(action >= 0.0) and np.all(action <= 1.0)

    def test_fcfs_agent(self):
        agent = FCFSSchedulingAgent()
        action, _ = agent.predict(np.zeros(12))
        assert action == 0


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_episode_metrics(self):
        m = EpisodeMetrics()
        info = {"T_hotspot_C": 78.0, "flow_kg_s": 0.3, "P_total_W": 500.0}
        m.record(info, reward=1.0, time_s=0.0)
        summary = m.summary()
        assert "T_hotspot_mean" in summary
        assert summary["episode_length"] == 1

    def test_evaluate_agent(self):
        env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=10.0)
        agent = PIDCoolingAgent(target_C=80.0)
        result = evaluate_agent(env, agent, n_episodes=2, max_steps=10)
        assert "reward_total" in result
        assert "n_episodes" in result
        env.close()
