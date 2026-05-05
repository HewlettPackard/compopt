"""
compopt.rewards
===============

Composable, modular reward functions for data-center RL environments.

Reward components can be mixed and matched via :class:`CompositeReward`,
or users can use one of the provided presets:

- :func:`cooling_only_reward` — rack/chip cooling
- :func:`datacenter_reward` — full DC thermal + energy + water + cost
- :func:`scheduling_reward` — job scheduling throughput + utilisation
- :func:`joint_reward` — combined scheduling + cooling
- :func:`grid_aware_datacenter_reward` — full DC with real-time grid tariff
"""

from compopt.rewards.functions import (
    RewardComponent, CompositeReward,
    ThermalPenalty, ThermalViolation, HBMPenalty,
    EnergyPenalty, CoolantFlowPenalty, PUEReward, CostPenalty,
    GridElectricityCostPenalty,
    WaterPenalty, WUEReward,
    ThroughputReward, QueuePenalty, UtilisationReward, SLAViolation,
    cooling_only_reward, datacenter_reward, scheduling_reward, joint_reward,
    grid_aware_datacenter_reward,
)

__all__ = [
    "RewardComponent", "CompositeReward",
    "ThermalPenalty", "ThermalViolation", "HBMPenalty",
    "EnergyPenalty", "CoolantFlowPenalty", "PUEReward", "CostPenalty",
    "GridElectricityCostPenalty",
    "WaterPenalty", "WUEReward",
    "ThroughputReward", "QueuePenalty", "UtilisationReward", "SLAViolation",
    "cooling_only_reward", "datacenter_reward", "scheduling_reward", "joint_reward",
    "grid_aware_datacenter_reward",
]
