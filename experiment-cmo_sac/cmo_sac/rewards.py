"""
Multi-Objective Reward Decomposition for CMO-SAC.

This module provides reward decomposition into separate objectives
(PUE, WUE, throughput) and constraint costs (thermal, HBM, SLA).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""
    name: str
    weight: float = 1.0
    minimize: bool = True  # True for costs (PUE, WUE), False for benefits (throughput)
    scale: float = 1.0  # Normalization scale
    target: Optional[float] = None  # Optional target value


@dataclass
class RewardComponents:
    """Container for decomposed reward components."""
    objectives: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    scalarized: float = 0.0
    constraint_violations: Dict[str, bool] = field(default_factory=dict)


class MultiObjectiveReward:
    """
    Multi-objective reward decomposition for data center control.
    
    Decomposes the CompOpt reward into separate objectives:
    - PUE (Power Usage Effectiveness) - minimize
    - WUE (Water Usage Effectiveness) - minimize  
    - Throughput (FLOPS/job completion) - maximize
    
    And constraint costs:
    - Thermal violation (GPU temperature > limit)
    - HBM violation (memory temperature > limit)
    - SLA violation (job deadline missed)
    
    Args:
        objective_configs: List of objective configurations
        constraint_thresholds: Dict mapping constraint names to thresholds
        scalarization: Method for combining objectives ('weighted', 'chebyshev', 'epsilon')
    """
    
    # Default objective configurations
    DEFAULT_OBJECTIVES = [
        ObjectiveConfig("pue", weight=1.0, minimize=True, scale=0.5),  # PUE ~1.1-2.0
        ObjectiveConfig("wue", weight=1.0, minimize=True, scale=2.0),  # WUE ~0.5-5.0
        ObjectiveConfig("throughput", weight=1.0, minimize=False, scale=1e15),  # PFLOPS
    ]
    
    # Default constraint thresholds
    DEFAULT_CONSTRAINTS = {
        "thermal": 75.0,      # GPU junction temperature limit (°C) - lowered to enable violations
        "hbm": 85.0,          # HBM temperature limit (°C) - lowered to enable violations
        "sla": 0.95,          # Job completion rate threshold
        "coolant_max": 40.0,  # Max coolant temperature (°C) - lowered to enable violations
    }
    
    def __init__(
        self,
        objective_configs: Optional[List[ObjectiveConfig]] = None,
        constraint_thresholds: Optional[Dict[str, float]] = None,
        scalarization: str = "weighted",
    ):
        self.objectives = objective_configs or self.DEFAULT_OBJECTIVES
        self.constraints = constraint_thresholds or self.DEFAULT_CONSTRAINTS.copy()
        self.scalarization = scalarization
        
        # Build lookup dicts
        self._obj_by_name = {obj.name: obj for obj in self.objectives}
        
        # Reference point for Chebyshev scalarization (ideal point)
        self._reference_point: Optional[Dict[str, float]] = None
        
        # Running statistics for normalization
        self._obj_means: Dict[str, float] = {obj.name: 0.0 for obj in self.objectives}
        self._obj_stds: Dict[str, float] = {obj.name: 1.0 for obj in self.objectives}
        self._update_count = 0
        
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update objective weights for scalarization."""
        for name, weight in weights.items():
            if name in self._obj_by_name:
                self._obj_by_name[name].weight = weight
                
    def set_reference_point(self, reference: Dict[str, float]) -> None:
        """Set reference point for Chebyshev scalarization."""
        self._reference_point = reference
        
    def extract_objectives(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        next_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract individual objective values from observation/info.
        
        Args:
            obs: Current observation dict
            info: Info dict from environment step
            next_obs: Optional next observation for delta computation
            
        Returns:
            Dict mapping objective names to values
        """
        objectives = {}
        
        # PUE: Power Usage Effectiveness
        # Check both lowercase 'pue' and uppercase 'PUE' (for different envs)
        if "pue" in info:
            objectives["pue"] = info["pue"]
        elif "PUE" in info:
            objectives["pue"] = info["PUE"]
        elif "total_power" in info and "it_power" in info:
            it_power = info["it_power"]
            if it_power > 0:
                objectives["pue"] = info["total_power"] / it_power
            else:
                objectives["pue"] = 1.0
        else:
            # Estimate from observation
            objectives["pue"] = self._estimate_pue(obs)
            
        # WUE: Water Usage Effectiveness
        # Check both 'wue' and 'WUE_L_per_kWh' (for different envs)
        if "wue" in info:
            objectives["wue"] = info["wue"]
        elif "WUE_L_per_kWh" in info:
            objectives["wue"] = info["WUE_L_per_kWh"]
        elif "water_usage" in info and "it_power" in info:
            it_power = info["it_power"]
            if it_power > 0:
                objectives["wue"] = info["water_usage"] / (it_power / 1e6)  # L/MWh
            else:
                objectives["wue"] = 0.0
        else:
            objectives["wue"] = self._estimate_wue(obs, info)
            
        # Throughput
        if "throughput" in info:
            objectives["throughput"] = info["throughput"]
        elif "jobs_completed" in info:
            objectives["throughput"] = info["jobs_completed"]
        else:
            objectives["throughput"] = self._estimate_throughput(obs, info)
            
        return objectives
    
    def extract_constraints(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Extract constraint costs and violation flags.
        
        Args:
            obs: Current observation dict
            info: Info dict from environment step
            
        Returns:
            Tuple of (constraint_costs, violation_flags)
        """
        costs = {}
        violations = {}
        
        # Thermal constraint (GPU temperature)
        t_gpu = self._get_max_temperature(obs, info, "gpu")
        thermal_limit = self.constraints["thermal"]
        costs["thermal"] = max(0.0, t_gpu - thermal_limit)
        violations["thermal"] = t_gpu > thermal_limit
        
        # HBM constraint (memory temperature)
        t_hbm = self._get_max_temperature(obs, info, "hbm")
        hbm_limit = self.constraints["hbm"]
        costs["hbm"] = max(0.0, t_hbm - hbm_limit)
        violations["hbm"] = t_hbm > hbm_limit
        
        # SLA constraint (job completion rate or violation count)
        # Different environments provide SLA metrics in different ways:
        # - Some provide completion_rate or sla_compliance (0-1 scale)
        # - Some provide n_sla_violations (count of violations in step)
        if "n_sla_violations" in info:
            # Scheduling-v0 and JointDC-v0 provide violation count
            # Convert to a violation flag (any violation = constraint violated)
            n_violations = float(info["n_sla_violations"])
            # Normalize by jobs completed this step (or use raw count)
            jobs_completed = float(info.get("jobs_completed_step", 1.0))
            if jobs_completed > 0:
                violation_rate = n_violations / jobs_completed
            else:
                violation_rate = 0.0 if n_violations == 0 else 1.0
            
            sla_threshold = self.constraints["sla"]  # e.g., 0.95 = max 5% violations
            costs["sla"] = max(0.0, violation_rate - (1.0 - sla_threshold))
            violations["sla"] = violation_rate > (1.0 - sla_threshold)
        else:
            # Other environments provide completion_rate or compliance
            completion_rate = info.get("sla_compliance", info.get("job_completion_rate", 1.0))
            sla_threshold = self.constraints["sla"]
            costs["sla"] = max(0.0, sla_threshold - completion_rate)
            violations["sla"] = completion_rate < sla_threshold
        
        # Coolant temperature constraint
        # Handle both dict and array obs types
        if isinstance(obs, dict):
            t_coolant = info.get("coolant_supply_temp", obs.get("T_supply", 25.0))
        else:
            t_coolant = info.get("coolant_supply_temp", 25.0)
        coolant_limit = self.constraints["coolant_max"]
        costs["coolant"] = max(0.0, t_coolant - coolant_limit)
        violations["coolant"] = t_coolant > coolant_limit
        
        return costs, violations
    
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> RewardComponents:
        """
        Compute decomposed multi-objective reward.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            info: Info dict from step
            weights: Optional weight override for scalarization
            
        Returns:
            RewardComponents with objectives, constraints, and scalarized reward
        """
        # Extract components
        objectives = self.extract_objectives(obs, info, next_obs)
        constraints, violations = self.extract_constraints(next_obs, info)
        
        # Update running statistics
        self._update_statistics(objectives)
        
        # Normalize objectives
        normalized = self._normalize_objectives(objectives)
        
        # Scalarize
        if weights is not None:
            temp_weights = {obj.name: obj.weight for obj in self.objectives}
            self.set_weights(weights)
            
        scalarized = self._scalarize(normalized)
        
        if weights is not None:
            self.set_weights(temp_weights)
            
        return RewardComponents(
            objectives=objectives,
            constraints=constraints,
            scalarized=scalarized,
            constraint_violations=violations,
        )
    
    def _scalarize(self, normalized: Dict[str, float]) -> float:
        """Combine normalized objectives into scalar reward."""
        if self.scalarization == "weighted":
            return self._weighted_sum(normalized)
        elif self.scalarization == "chebyshev":
            return self._chebyshev(normalized)
        elif self.scalarization == "epsilon":
            return self._epsilon_constraint(normalized)
        else:
            raise ValueError(f"Unknown scalarization: {self.scalarization}")
            
    def _weighted_sum(self, normalized: Dict[str, float]) -> float:
        """Weighted linear scalarization."""
        reward = 0.0
        for obj in self.objectives:
            value = normalized.get(obj.name, 0.0)
            # Negate if minimizing (so SAC maximizes -cost)
            sign = -1.0 if obj.minimize else 1.0
            reward += obj.weight * sign * value
        return reward
    
    def _chebyshev(self, normalized: Dict[str, float]) -> float:
        """
        Chebyshev (min-max) scalarization.
        
        Minimizes the maximum weighted distance to the reference point.
        Better for exploring the Pareto front.
        """
        if self._reference_point is None:
            # Use ideal point (0 for costs, 1 for benefits)
            self._reference_point = {
                obj.name: 0.0 if obj.minimize else 1.0
                for obj in self.objectives
            }
            
        max_dist = 0.0
        for obj in self.objectives:
            value = normalized.get(obj.name, 0.0)
            ref = self._reference_point.get(obj.name, 0.0)
            
            if obj.minimize:
                dist = obj.weight * (value - ref)  # Want value < ref
            else:
                dist = obj.weight * (ref - value)  # Want value > ref
                
            max_dist = max(max_dist, dist)
            
        return -max_dist  # Negate so SAC maximizes
    
    def _epsilon_constraint(self, normalized: Dict[str, float]) -> float:
        """
        Epsilon-constraint scalarization.
        
        Optimizes the first objective while treating others as constraints.
        """
        # Primary objective is first in list
        primary = self.objectives[0]
        value = normalized.get(primary.name, 0.0)
        sign = -1.0 if primary.minimize else 1.0
        return sign * value
    
    def _normalize_objectives(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """Normalize objectives using configured scales and running stats."""
        normalized = {}
        for obj in self.objectives:
            raw = objectives.get(obj.name, 0.0)
            # Scale to roughly [0, 1]
            normalized[obj.name] = raw / obj.scale
        return normalized
    
    def _update_statistics(self, objectives: Dict[str, float]) -> None:
        """Update running mean/std for adaptive normalization."""
        self._update_count += 1
        alpha = 1.0 / self._update_count
        
        for name, value in objectives.items():
            old_mean = self._obj_means[name]
            self._obj_means[name] = old_mean + alpha * (value - old_mean)
            
            # Welford's algorithm for variance
            if self._update_count > 1:
                old_var = self._obj_stds[name] ** 2
                new_var = old_var + alpha * ((value - old_mean) * (value - self._obj_means[name]) - old_var)
                self._obj_stds[name] = max(np.sqrt(new_var), 1e-8)
                
    def _estimate_pue(self, obs: Dict[str, Any]) -> float:
        """Estimate PUE from observation when not in info."""
        # Simple estimate based on cooling power fraction
        if isinstance(obs, dict):
            cooling_power = obs.get("cooling_power", obs.get("P_cool", 0.0))
            it_power = obs.get("it_power", obs.get("P_it", 1.0))
        else:
            # Assume obs is a numpy array - use typical values
            return 1.3
            
        if it_power > 0:
            return 1.0 + cooling_power / it_power
        return 1.0
    
    def _estimate_wue(self, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Estimate WUE from observation/info."""
        # WUE = water consumption / IT energy
        # Typical range: 0.5 - 5.0 L/kWh
        tower_flow = info.get("tower_water_flow", 0.0)
        it_power = info.get("it_power", obs.get("P_it", 1.0)) if isinstance(obs, dict) else 1.0
        
        if it_power > 0:
            # Rough conversion: flow rate to L/kWh
            return tower_flow * 3600 / (it_power / 1000)  
        return 1.0
    
    def _estimate_throughput(self, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Estimate throughput from observation/info."""
        # Use GPU utilization as proxy for throughput
        if "utilization" in info:
            return info["utilization"]
        elif isinstance(obs, dict) and "gpu_utilization" in obs:
            return obs["gpu_utilization"]
        return 1.0
    
    def _get_max_temperature(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        temp_type: str = "gpu"
    ) -> float:
        """Get maximum temperature of specified type."""
        if temp_type == "gpu":
            # Check all possible GPU temperature keys (with and without _C suffix)
            keys = ["T_hotspot_C", "T_gpu_hotspot_C", "T_junction", "T_gpu", 
                   "gpu_temp", "max_gpu_temp", "T_hotspot", "T_gpu_junction"]
        else:  # hbm
            # Check all possible HBM temperature keys (with and without _C suffix)
            keys = ["T_hbm_C", "T_hbm", "hbm_temp", "max_hbm_temp", "T_HBM"]
            
        for key in keys:
            if key in info:
                val = info[key]
                return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
            if isinstance(obs, dict) and key in obs:
                val = obs[key]
                return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
                
        # Default safe temperature
        return 70.0 if temp_type == "gpu" else 80.0


class WeightSampler:
    """
    Samples scalarization weights for Pareto front exploration.
    
    Supports multiple sampling strategies:
    - uniform: Uniform random on simplex
    - dirichlet: Dirichlet distribution (controllable concentration)
    - grid: Deterministic grid on simplex
    - adaptive: Focuses on underexplored regions
    """
    
    def __init__(
        self,
        n_objectives: int,
        strategy: str = "dirichlet",
        concentration: float = 1.0,
        grid_resolution: int = 10,
    ):
        self.n_objectives = n_objectives
        self.strategy = strategy
        self.concentration = concentration
        self.grid_resolution = grid_resolution
        
        # For adaptive sampling
        self._weight_history: List[np.ndarray] = []
        self._reward_history: List[float] = []
        
        # Pre-compute grid if needed
        if strategy == "grid":
            self._grid = self._generate_grid()
            self._grid_idx = 0
            
    def sample(self) -> np.ndarray:
        """Sample a weight vector."""
        if self.strategy == "uniform":
            return self._sample_uniform()
        elif self.strategy == "dirichlet":
            return self._sample_dirichlet()
        elif self.strategy == "grid":
            return self._sample_grid()
        elif self.strategy == "adaptive":
            return self._sample_adaptive()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
    def _sample_uniform(self) -> np.ndarray:
        """Uniform sampling on simplex."""
        # Exponential trick for uniform simplex
        x = np.random.exponential(scale=1.0, size=self.n_objectives)
        return x / x.sum()
    
    def _sample_dirichlet(self) -> np.ndarray:
        """Dirichlet distribution sampling."""
        alpha = np.ones(self.n_objectives) * self.concentration
        return np.random.dirichlet(alpha)
    
    def _sample_grid(self) -> np.ndarray:
        """Cycle through pre-computed grid points."""
        weights = self._grid[self._grid_idx]
        self._grid_idx = (self._grid_idx + 1) % len(self._grid)
        return weights
    
    def _sample_adaptive(self) -> np.ndarray:
        """Adaptive sampling focusing on underexplored regions."""
        if len(self._weight_history) < 10:
            return self._sample_dirichlet()
            
        # Compute density estimate and sample from low-density regions
        history = np.array(self._weight_history[-100:])  # Recent history
        candidate = self._sample_dirichlet()
        
        # Simple rejection: prefer candidates far from recent samples
        for _ in range(10):
            dists = np.linalg.norm(history - candidate, axis=1)
            if dists.min() > 0.1:  # Sufficiently far
                break
            candidate = self._sample_dirichlet()
            
        return candidate
    
    def _generate_grid(self) -> List[np.ndarray]:
        """Generate evenly-spaced grid on simplex."""
        from itertools import product
        
        n = self.grid_resolution
        k = self.n_objectives
        
        # Generate all combinations that sum to n
        grid = []
        
        def generate_combinations(remaining, depth, current):
            if depth == k - 1:
                current.append(remaining)
                grid.append(np.array(current) / n)
                current.pop()
                return
                
            for i in range(remaining + 1):
                current.append(i)
                generate_combinations(remaining - i, depth + 1, current)
                current.pop()
                
        generate_combinations(n, 0, [])
        return grid
    
    def update(self, weights: np.ndarray, reward: float) -> None:
        """Update history for adaptive sampling."""
        self._weight_history.append(weights.copy())
        self._reward_history.append(reward)
        
        # Limit history size
        if len(self._weight_history) > 1000:
            self._weight_history = self._weight_history[-500:]
            self._reward_history = self._reward_history[-500:]
