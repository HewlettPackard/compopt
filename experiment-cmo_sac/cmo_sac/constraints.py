"""
Constraint Management and Lagrangian Dual Optimization for CMO-SAC.

This module implements:
- Constraint tracking and cost computation
- Lagrangian dual optimization for constraint satisfaction
- Primal-dual update rules
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ConstraintSpec:
    """Specification for a single constraint."""
    name: str
    threshold: float  # Constraint limit d_i
    cost_type: str = "max"  # 'max' (upper bound) or 'min' (lower bound)
    discount: float = 0.99  # Discount factor for constraint return
    initial_lambda: float = 0.1  # Initial Lagrange multiplier
    # FIXED: Reduced from 0.001 to 0.0001 (10x slower)
    # This gives agent more time to adapt before penalties increase
    lambda_lr: float = 0.0001  # Learning rate for dual update (conservative)
    lambda_max: float = 100.0  # Maximum multiplier value


class ConstraintManager:
    """
    Manages constraint tracking and cost computation for CMDP.
    
    Tracks constraint violations across episodes and computes
    discounted constraint returns for Lagrangian optimization.
    
    Args:
        constraints: List of constraint specifications
        buffer_size: Size of rolling buffer for constraint statistics
    """
    
    # Default constraints for data center control
    DEFAULT_CONSTRAINTS = [
        ConstraintSpec("thermal", threshold=75.0, cost_type="max"),
        ConstraintSpec("hbm", threshold=85.0, cost_type="max"),
        ConstraintSpec("sla", threshold=0.95, cost_type="min"),
        ConstraintSpec("coolant", threshold=40.0, cost_type="max"),
    ]
    
    def __init__(
        self,
        constraints: Optional[List[ConstraintSpec]] = None,
        buffer_size: int = 10000,
    ):
        self.constraints = constraints or self.DEFAULT_CONSTRAINTS
        self.buffer_size = buffer_size
        
        # Build lookup
        self._constraint_by_name = {c.name: c for c in self.constraints}
        
        # Buffers for tracking constraint costs
        self._cost_buffers: Dict[str, deque] = {
            c.name: deque(maxlen=buffer_size)
            for c in self.constraints
        }
        
        # Episode-level tracking
        self._episode_costs: Dict[str, List[float]] = {
            c.name: [] for c in self.constraints
        }
        
        # Running statistics
        self._total_violations: Dict[str, int] = {c.name: 0 for c in self.constraints}
        self._total_steps = 0
        
    def reset_episode(self) -> None:
        """Reset episode-level tracking."""
        self._episode_costs = {c.name: [] for c in self.constraints}
        
    def compute_costs(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute constraint costs for current state.
        
        Args:
            obs: Current observation
            info: Info dict from environment
            
        Returns:
            Dict mapping constraint names to cost values
        """
        costs = {}
        
        for constraint in self.constraints:
            # Get current value for this constraint
            value = self._get_constraint_value(constraint.name, obs, info)
            
            # Compute cost based on constraint type
            if constraint.cost_type == "max":
                # Upper bound: cost is positive when value > threshold
                cost = max(0.0, value - constraint.threshold)
            else:
                # Lower bound: cost is positive when value < threshold
                cost = max(0.0, constraint.threshold - value)
                
            costs[constraint.name] = cost
            
            # Track for statistics
            self._episode_costs[constraint.name].append(cost)
            self._cost_buffers[constraint.name].append(cost)
            
            if cost > 0:
                self._total_violations[constraint.name] += 1
                
        self._total_steps += 1
        return costs
    
    def compute_episode_returns(self) -> Dict[str, float]:
        """
        Compute discounted constraint returns for completed episode.
        
        Returns:
            Dict mapping constraint names to discounted returns G_c
        """
        returns = {}
        
        for constraint in self.constraints:
            costs = self._episode_costs[constraint.name]
            if not costs:
                returns[constraint.name] = 0.0
                continue
                
            # Compute discounted return
            G = 0.0
            for cost in reversed(costs):
                G = cost + constraint.discount * G
                
            returns[constraint.name] = G
            
        return returns
    
    def get_violation_rates(self) -> Dict[str, float]:
        """Get violation rate for each constraint."""
        if self._total_steps == 0:
            return {c.name: 0.0 for c in self.constraints}
            
        return {
            name: count / self._total_steps
            for name, count in self._total_violations.items()
        }
    
    def get_mean_costs(self) -> Dict[str, float]:
        """Get mean cost for each constraint from buffer."""
        return {
            name: np.mean(list(buffer)) if buffer else 0.0
            for name, buffer in self._cost_buffers.items()
        }
    
    def _get_constraint_value(
        self,
        name: str,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """Extract constraint value from observation/info."""
        if name == "thermal":
            # GPU junction temperature
            # Try all possible temperature keys (with and without _C suffix)
            for key in ["T_hotspot_C", "T_gpu_hotspot_C", "T_junction", "T_gpu", 
                       "gpu_temp", "max_gpu_temp", "T_hotspot", "T_gpu_junction"]:
                if key in info:
                    val = info[key]
                    return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
                if isinstance(obs, dict) and key in obs:
                    val = obs[key]
                    return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
            # Fallback: assume not violating if not found
            return 70.0  # Safe default below typical threshold
            
        elif name == "hbm":
            # HBM memory temperature
            # Try all possible HBM temperature keys (with and without _C suffix)
            for key in ["T_hbm_C", "T_hbm", "hbm_temp", "max_hbm_temp", "T_HBM"]:
                if key in info:
                    val = info[key]
                    return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
                if isinstance(obs, dict) and key in obs:
                    val = obs[key]
                    return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
            # Fallback: assume not violating if not found
            return 80.0  # Safe default below typical threshold
            
        elif name == "sla":
            # SLA compliance rate (for scheduling environments)
            # Different environments provide SLA metrics differently:
            # - Some provide sla_compliance or job_completion_rate (0-1 scale)
            # - Some provide n_sla_violations (count of violations)
            
            # First check for direct compliance metrics
            sla = info.get("sla_compliance", info.get("job_completion_rate", None))
            if sla is not None:
                return float(sla)
            
            # Check for violation count (JointDC-v0, Scheduling-v0)
            if "n_sla_violations" in info:
                n_violations = float(info["n_sla_violations"])
                jobs_completed = float(info.get("jobs_completed_step", 0.0))
                
                if jobs_completed > 0:
                    # Return compliance rate (1.0 - violation_rate)
                    compliance = 1.0 - (n_violations / jobs_completed)
                    return max(0.0, min(1.0, compliance))
                elif n_violations > 0:
                    # Violations occurred but no jobs completed = 0% compliance
                    return 0.0
                else:
                    # No violations and no completions = perfect compliance
                    return 1.0
            
            # For non-scheduling environments (RackCooling, ChipThermal), assume perfect compliance
            return 1.0
            
        elif name == "coolant":
            # Coolant supply temperature
            for key in ["coolant_temp", "T_coolant", "T_supply", "coolant_supply_temp"]:
                if key in info:
                    val = info[key]
                    return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
                if isinstance(obs, dict) and key in obs:
                    val = obs[key]
                    return float(np.max(val)) if isinstance(val, (np.ndarray, list)) else float(val)
            # Fallback: safe default temperature
            return 25.0
            
        return 0.0


class LagrangianDualOptimizer:
    """
    Lagrangian dual optimizer for constrained MDPs.
    
    Implements the dual gradient ascent update for Lagrange multipliers:
    λ_i ← max(0, λ_i + η_λ * (E[G_c_i] - d_i))
    
    The multipliers convert constraint costs into reward penalties:
    L(π, λ) = J(π) - Σ_i λ_i * (E[G_c_i] - d_i)
    
    Args:
        constraint_manager: ConstraintManager instance
        device: Torch device for tensors
    """
    
    def __init__(
        self,
        constraint_manager: ConstraintManager,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LagrangianDualOptimizer")
            
        self.constraint_manager = constraint_manager
        self.device = torch.device(device)
        
        # Initialize Lagrange multipliers
        n_constraints = len(constraint_manager.constraints)
        self._log_lambdas = nn.Parameter(
            torch.zeros(n_constraints, device=self.device)
        )
        
        # Store constraint specs for reference
        self._specs = constraint_manager.constraints
        
        # Optimizer for dual variables
        self._optimizer = torch.optim.Adam(
            [self._log_lambdas],
            lr=self._specs[0].lambda_lr if self._specs else 0.001,
        )
        
        # History for logging
        self._lambda_history: List[Dict[str, float]] = []
        self._constraint_return_history: List[Dict[str, float]] = []
        
    @property
    def lambdas(self) -> torch.Tensor:
        """Get current Lagrange multipliers (positive via exp)."""
        return torch.exp(self._log_lambdas).clamp(max=self._max_lambda)
    
    @property
    def _max_lambda(self) -> float:
        """Maximum lambda value from specs."""
        return max(c.lambda_max for c in self._specs) if self._specs else 100.0
    
    def get_lambdas_dict(self) -> Dict[str, float]:
        """Get multipliers as dict for logging."""
        lambdas = self.lambdas.detach().cpu().numpy()
        return {
            spec.name: float(lambdas[i])
            for i, spec in enumerate(self._specs)
        }
    
    def compute_constraint_penalty(
        self,
        costs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Lagrangian penalty term for the objective.
        
        Args:
            costs: Dict mapping constraint names to cost tensors (batch)
            
        Returns:
            Penalty term: Σ_i λ_i * c_i
        """
        lambdas = self.lambdas
        penalty = torch.zeros(1, device=self.device)
        
        for i, spec in enumerate(self._specs):
            if spec.name in costs:
                cost = costs[spec.name]
                if not isinstance(cost, torch.Tensor):
                    cost = torch.tensor(cost, device=self.device)
                penalty = penalty + lambdas[i] * cost.mean()
                
        return penalty
    
    def dual_step(
        self,
        constraint_returns: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Perform dual gradient ascent step.
        
        Updates λ_i based on constraint satisfaction:
        - If E[G_c_i] > d_i (constraint violated), increase λ_i
        - If E[G_c_i] < d_i (constraint satisfied), decrease λ_i
        
        Args:
            constraint_returns: Dict of discounted constraint returns
            
        Returns:
            Dict of dual gradients for logging
        """
        self._optimizer.zero_grad()
        
        # Compute dual objective: maximize λ * (G_c - d)
        # Equivalently, minimize -λ * (G_c - d)
        dual_loss = torch.zeros(1, device=self.device)
        gradients = {}
        
        lambdas = self.lambdas
        
        for i, spec in enumerate(self._specs):
            G_c = constraint_returns.get(spec.name, 0.0)
            d = spec.threshold if spec.cost_type == "max" else -spec.threshold
            
            # For max constraints: want G_c <= 0 (scaled threshold already in cost)
            # Gradient is positive when constraint violated
            constraint_violation = G_c  # Already measures violation amount
            
            # Dual loss: -λ * violation (we maximize, so minimize negative)
            dual_loss = dual_loss - lambdas[i] * constraint_violation
            gradients[spec.name] = constraint_violation
            
        dual_loss.backward()
        self._optimizer.step()
        
        # Log history
        self._lambda_history.append(self.get_lambdas_dict())
        self._constraint_return_history.append(constraint_returns.copy())
        
        return gradients
    
    def augmented_reward(
        self,
        reward: torch.Tensor,
        costs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Lagrangian-augmented reward.
        
        r_aug = r - Σ_i λ_i * c_i
        
        Args:
            reward: Original reward tensor
            costs: Dict of constraint cost tensors
            
        Returns:
            Augmented reward tensor
        """
        penalty = self.compute_constraint_penalty(costs)
        return reward - penalty
    
    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            "log_lambdas": self._log_lambdas.data.cpu().numpy(),
            "optimizer": self._optimizer.state_dict(),
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load state from checkpoint."""
        self._log_lambdas.data = torch.tensor(
            state["log_lambdas"],
            device=self.device,
        )
        self._optimizer.load_state_dict(state["optimizer"])


class ConstraintBuffer:
    """
    Specialized replay buffer that tracks constraint costs.
    
    Extends standard replay buffer with constraint cost storage
    for computing Lagrangian penalties during training.
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        n_objectives: int = 3,
        n_constraints: int = 4,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ConstraintBuffer")
            
        self.capacity = capacity
        self.device = torch.device(device)
        
        # Standard replay buffer storage
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Multi-objective rewards
        self.rewards = np.zeros((capacity, n_objectives), dtype=np.float32)
        self.scalarized_rewards = np.zeros(capacity, dtype=np.float32)
        
        # Constraint costs
        self.constraint_costs = np.zeros((capacity, n_constraints), dtype=np.float32)
        
        # Scalarization weights used for each sample
        self.weights = np.zeros((capacity, n_objectives), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,  # Multi-objective reward vector
        scalarized_reward: float,
        constraint_costs: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        weights: np.ndarray,
    ) -> None:
        """Add transition with multi-objective rewards and constraint costs."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        # Only add rewards up to the buffer's n_objectives
        if reward.shape[0] >= self.rewards.shape[1]:
            self.rewards[self.ptr] = reward[:self.rewards.shape[1]]
        else:
            self.rewards[self.ptr] = reward
        self.scalarized_rewards[self.ptr] = scalarized_reward
        # Only add constraint costs if we have constraints
        if self.constraint_costs.shape[1] > 0:
            self.constraint_costs[self.ptr] = constraint_costs
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        # Only add weights up to the buffer's n_objectives
        if weights.shape[0] >= self.weights.shape[1]:
            self.weights[self.ptr] = weights[:self.weights.shape[1]]
        else:
            self.weights[self.ptr] = weights
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": torch.tensor(
                self.observations[idxs], device=self.device
            ),
            "actions": torch.tensor(
                self.actions[idxs], device=self.device
            ),
            "rewards": torch.tensor(
                self.rewards[idxs], device=self.device
            ),
            "scalarized_rewards": torch.tensor(
                self.scalarized_rewards[idxs], device=self.device
            ),
            "constraint_costs": torch.tensor(
                self.constraint_costs[idxs], device=self.device
            ),
            "next_observations": torch.tensor(
                self.next_observations[idxs], device=self.device
            ),
            "dones": torch.tensor(
                self.dones[idxs], device=self.device
            ),
            "weights": torch.tensor(
                self.weights[idxs], device=self.device
            ),
        }
    
    def __len__(self) -> int:
        return self.size
