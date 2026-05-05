"""
compopt.rewards.functions
=========================
Composable, modular reward functions for data-center RL environments.

Each reward component is a callable that takes an ``info`` dict
(produced by env.step) and returns a scalar reward contribution.

Users can compose custom multi-objective rewards using ``CompositeReward``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Base interface
# ──────────────────────────────────────────────────────────────────────────────

class RewardComponent:
    """Base class for a single reward term."""
    name: str = "base"
    weight: float = 1.0

    def __call__(self, info: Dict[str, float]) -> float:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Thermal rewards
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThermalPenalty(RewardComponent):
    """
    Quadratic penalty for GPU hotspot exceeding target.

    r = -weight · max(0, T_hotspot - target)²
    """
    name:   str   = "thermal_penalty"
    weight: float = 1.0
    target_C: float = 80.0

    def __call__(self, info: Dict[str, float]) -> float:
        T = info.get("T_hotspot_C", info.get("T_gpu_hotspot_C", 0.0))
        return -self.weight * max(0.0, T - self.target_C) ** 2


@dataclass
class ThermalViolation(RewardComponent):
    """
    Hard penalty for exceeding absolute thermal limit.

    r = -weight   if T_hotspot > limit, else 0
    """
    name:    str   = "thermal_violation"
    weight:  float = 100.0
    limit_C: float = 83.0

    def __call__(self, info: Dict[str, float]) -> float:
        T = info.get("T_hotspot_C", info.get("T_gpu_hotspot_C", 0.0))
        return -self.weight if T > self.limit_C else 0.0


@dataclass
class HBMPenalty(RewardComponent):
    """Penalty for HBM temperature exceeding limit."""
    name:    str   = "hbm_penalty"
    weight:  float = 0.5
    limit_C: float = 95.0

    def __call__(self, info: Dict[str, float]) -> float:
        T = info.get("T_hbm_C", 0.0)
        return -self.weight * max(0.0, T - self.limit_C) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Energy / cost rewards
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergyPenalty(RewardComponent):
    """
    Penalty proportional to total power consumption.

    r = -weight · P_total / P_ref
    """
    name:   str   = "energy_penalty"
    weight: float = 0.01
    P_ref_kW: float = 100.0

    def __call__(self, info: Dict[str, float]) -> float:
        P = info.get("total_power_kW", info.get("P_total_W", 0.0) / 1000.0)
        return -self.weight * P / self.P_ref_kW


@dataclass
class CoolantFlowPenalty(RewardComponent):
    """
    Penalty for excess coolant flow (pumping energy cost).

    r = -weight · (m_dot - m_dot_min)³    [cubic for realistic pump power]
    
    This makes high flow significantly more expensive, creating a real
    trade-off between thermal safety and energy cost that RL can optimize.
    
    For constrained RL, use weight ~0.01-0.1 (flow ranges 0.5-4.0 kg/s,
    so cubic ranges from 0 to 42.875, which is huge without scaling!)
    """
    name:       str   = "flow_penalty"
    weight:     float = 0.01
    m_dot_min:  float = 0.5

    def __call__(self, info: Dict[str, float]) -> float:
        flow = info.get("flow_kg_s", info.get("m_dot_kg_s", 0.5))
        # Cubic penalty (realistic pump power law)
        # Scale by 1/100 to keep magnitudes reasonable (max ~0.43 per step instead of 43)
        delta_flow = flow - self.m_dot_min
        return -self.weight * (delta_flow ** 3) / 100.0  # Scaled for numerical stability


@dataclass
class PUEReward(RewardComponent):
    """
    Reward inversely proportional to PUE (closer to 1.0 is better).

    r = weight · (2.0 - PUE)    (positive when PUE < 2)
    """
    name:   str   = "pue_reward"
    weight: float = 1.0

    def __call__(self, info: Dict[str, float]) -> float:
        pue = info.get("PUE", 1.5)
        return self.weight * (2.0 - pue)


@dataclass
class CostPenalty(RewardComponent):
    """
    Dollar-cost penalty per time step.

    r = -weight · cost_delta_$
    """
    name:   str   = "cost_penalty"
    weight: float = 10.0

    def __call__(self, info: Dict[str, float]) -> float:
        cost = info.get("step_cost_dollar", 0.0)
        return -self.weight * cost


@dataclass
class GridElectricityCostPenalty(RewardComponent):
    """
    Penalises electricity expenditure at the *current grid tariff price*.

    Unlike the static ``CostPenalty``, this component is *time-aware*: it
    multiplies actual power consumption by the real-time grid price, so the
    agent learns to **shift or reduce load during expensive peak hours** and
    to exploit cheap off-peak / renewable-surplus windows.

    Formula
    -------
    .. math::

        r = -w \\cdot P_{\\text{total}} \\;[\\text{kW}]
              \\times p_{\\text{grid}} \\;[\\$/{\\text{kWh}}]
              \\times \\Delta t \\;[\\text{h}]

    where :math:`\\Delta t = dt / 3600` converts the step duration from
    seconds to hours, giving a cost increment in **US dollars per step**.

    Parameters
    ----------
    weight : float
        Scalar multiplier. Default 50.0 makes grid-cost contributions
        comparable in magnitude to thermal penalties.
    dt_s : float
        Environment time-step in seconds (must match ``DataCenterEnv.dt``).
    P_ref_kW : float
        Reference power used for normalisation (default 100 kW).  The
        per-step cost is divided by ``P_ref_kW`` to keep reward magnitudes
        scale-invariant across cluster sizes.

    Info keys consumed
    ------------------
    - ``total_power_kW`` : total facility power [kW]
    - ``grid_price_per_kWh`` : current tariff price [$/kWh] (set by
      ``DataCenterEnv`` when ``grid_electricity`` is enabled)

    Example
    -------
    >>> comp = GridElectricityCostPenalty(weight=50.0, dt_s=5.0)
    >>> info = {"total_power_kW": 120.0, "grid_price_per_kWh": 0.22}
    >>> comp(info)
    -0.33   # −50 × (120/100) × 0.22 × (5/3600)
    """
    name:     str   = "grid_electricity_cost"
    weight:   float = 50.0
    dt_s:     float = 5.0
    P_ref_kW: float = 100.0

    def __call__(self, info: Dict[str, float]) -> float:
        P_kW   = info.get("total_power_kW", info.get("P_total_W", 0.0) / 1000.0)
        price  = info.get("grid_price_per_kWh", 0.12)   # fallback: $0.12/kWh
        dt_h   = self.dt_s / 3600.0
        cost_dollar = P_kW * price * dt_h               # $ spent this step
        return -self.weight * cost_dollar / self.P_ref_kW


# ──────────────────────────────────────────────────────────────────────────────
# Water rewards
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WaterPenalty(RewardComponent):
    """
    Penalty for water consumption per time step.

    r = -weight · water_L_per_step / ref_L
    """
    name:   str   = "water_penalty"
    weight: float = 0.5
    ref_L:  float = 10.0

    def __call__(self, info: Dict[str, float]) -> float:
        water = info.get("water_used_L", 0.0)
        return -self.weight * water / self.ref_L


@dataclass
class WUEReward(RewardComponent):
    """Reward for low Water Usage Effectiveness."""
    name:   str   = "wue_reward"
    weight: float = 0.5

    def __call__(self, info: Dict[str, float]) -> float:
        wue = info.get("WUE_L_per_kWh", 2.0)
        return -self.weight * wue


# ──────────────────────────────────────────────────────────────────────────────
# Scheduling rewards
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThroughputReward(RewardComponent):
    """Reward for job completions in this step."""
    name:   str   = "throughput"
    weight: float = 1.0

    def __call__(self, info: Dict[str, float]) -> float:
        return self.weight * info.get("jobs_completed_step", 0.0)


@dataclass
class QueuePenalty(RewardComponent):
    """Penalty for long queues (waiting jobs)."""
    name:   str   = "queue_penalty"
    weight: float = 0.1

    def __call__(self, info: Dict[str, float]) -> float:
        return -self.weight * info.get("queue_length", 0.0)


@dataclass
class UtilisationReward(RewardComponent):
    """Reward for high cluster utilisation."""
    name:   str   = "utilisation"
    weight: float = 1.0

    def __call__(self, info: Dict[str, float]) -> float:
        return self.weight * info.get("node_utilisation", 0.0)


@dataclass
class SLAViolation(RewardComponent):
    """Penalty for jobs exceeding their deadline."""
    name:   str   = "sla_violation"
    weight: float = 50.0

    def __call__(self, info: Dict[str, float]) -> float:
        return -self.weight * info.get("n_sla_violations", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Composite reward
# ──────────────────────────────────────────────────────────────────────────────

class CompositeReward:
    """
    Multi-objective reward that sums weighted reward components.

    Example
    -------
    >>> reward_fn = CompositeReward([
    ...     ThermalPenalty(weight=1.0, target_C=80.0),
    ...     CoolantFlowPenalty(weight=0.01),
    ...     EnergyPenalty(weight=0.005),
    ...     WaterPenalty(weight=0.5),
    ... ])
    >>> total, breakdown = reward_fn(info_dict)
    """

    def __init__(self, components: List[RewardComponent]):
        self.components = components

    def __call__(self, info: Dict[str, float]):
        """Return (total_reward, breakdown_dict)."""
        breakdown = {}
        total = 0.0
        for comp in self.components:
            r = comp(info)
            # Sanitize reward to prevent overflow when casting
            r = float(np.clip(np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6))
            breakdown[comp.name] = r
            total += r
        return total, breakdown

    def __repr__(self):
        names = [f"{c.name}(w={c.weight})" for c in self.components]
        return f"CompositeReward([{', '.join(names)}])"


# ──────────────────────────────────────────────────────────────────────────────
# Preset reward configurations
# ──────────────────────────────────────────────────────────────────────────────

def cooling_only_reward(target_C: float = 80.0) -> CompositeReward:
    """
    Reward for rack/chip cooling control with energy-performance trade-off.
    
    Designed to make RL valuable:
    - Strong HBM penalty (weight=10.0) for safety
    - Mild flow penalty (weight=0.01, cubic) for realistic pump costs
    - Creates trade-off: adaptive control can save energy vs always-max-flow
    
    Optimal policy should:
    - Use high flow during high workload periods
    - Reduce flow during low workload to save energy
    - React to temperature trends before violations occur
    """
    return CompositeReward([
        ThermalPenalty(weight=1.0, target_C=target_C),
        ThermalViolation(weight=100.0, limit_C=target_C + 3.0),
        HBMPenalty(weight=10.0, limit_C=85.0),
        CoolantFlowPenalty(weight=0.01),  # 10x weaker for smoother optimization
    ])


def constrained_cooling_reward(target_C: float = 80.0) -> CompositeReward:
    """
    Balanced cooling reward that prevents exploits.
    
    Uses ORIGINAL reward structure (what baselines were evaluated on):
    - Thermal penalties for GPU hotspot
    - HBM penalty with 90°C limit (85°C too strict, 95°C too loose)
    - Flow penalty weight=0.1 (balanced compromise)
    
    This matches what the adaptive baseline was scored against,
    so RL results will be comparable. HBM peaks at 90.3°C with max cooling,
    so agent must learn to keep it just under 90°C.
    """
    return CompositeReward([
        ThermalPenalty(weight=1.0, target_C=target_C),
        ThermalViolation(weight=100.0, limit_C=target_C + 3.0),
        HBMPenalty(weight=10.0, limit_C=90.0),  # Tight but achievable with good control
        CoolantFlowPenalty(weight=0.1),  # Balanced: strong enough to matter
    ])


def datacenter_reward() -> CompositeReward:
    """Full data-center reward with thermal + energy + water + cost."""
    return CompositeReward([
        ThermalPenalty(weight=1.0, target_C=80.0),
        ThermalViolation(weight=100.0),
        EnergyPenalty(weight=0.01),
        PUEReward(weight=0.5),
        WaterPenalty(weight=0.5),
        CostPenalty(weight=5.0),
    ])


def scheduling_reward() -> CompositeReward:
    """Reward for job scheduling only."""
    return CompositeReward([
        ThroughputReward(weight=1.0),
        QueuePenalty(weight=0.1),
        UtilisationReward(weight=0.5),
        SLAViolation(weight=50.0),
    ])


def joint_reward() -> CompositeReward:
    """Joint scheduling + cooling + cost reward."""
    return CompositeReward([
        ThermalPenalty(weight=1.0, target_C=80.0),
        ThermalViolation(weight=100.0),
        EnergyPenalty(weight=0.01),
        WaterPenalty(weight=0.3),
        CostPenalty(weight=5.0),
        ThroughputReward(weight=2.0),
        QueuePenalty(weight=0.1),
        UtilisationReward(weight=0.5),
    ])


def grid_aware_datacenter_reward(dt_s: float = 5.0) -> CompositeReward:
    """
    Data-centre reward that penalises **real-time grid electricity costs**.

    This preset is designed for use with ``DataCenterEnv(grid_electricity=True)``.
    The agent is incentivised to:

    - Keep GPUs below thermal limits (safety)
    - Minimise total energy consumption (efficiency)
    - **Shift expensive compute away from on-peak tariff windows**
    - **Exploit off-peak / renewable-surplus price dips**

    Compared to ``datacenter_reward()``:

    - Replaces the static ``CostPenalty`` with ``GridElectricityCostPenalty``
      (same physical energy, but cost is modulated by the live tariff price).
    - Retains ``EnergyPenalty`` as a baseline incentive to reduce absolute
      power, independent of price.
    - Slightly reduces ``WaterPenalty`` weight to leave room for the price
      signal to dominate.

    Parameters
    ----------
    dt_s : float
        Environment time-step in seconds. **Must match** the ``dt`` passed to
        ``DataCenterEnv`` so that per-step energy is computed correctly.

    Example
    -------
    >>> import compopt
    >>> from compopt.rewards import grid_aware_datacenter_reward
    >>> env = compopt.make(
    ...     "DataCenter-v0",
    ...     grid_electricity=True,
    ...     reward_fn=grid_aware_datacenter_reward(dt_s=5.0),
    ... )
    """
    return CompositeReward([
        ThermalPenalty(weight=1.0, target_C=80.0),
        ThermalViolation(weight=100.0),
        EnergyPenalty(weight=0.005),
        PUEReward(weight=0.5),
        WaterPenalty(weight=0.3),
        GridElectricityCostPenalty(weight=50.0, dt_s=dt_s, P_ref_kW=100.0),
    ])
