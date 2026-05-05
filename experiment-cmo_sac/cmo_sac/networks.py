"""
Neural Network Architectures for CMO-SAC.

This module implements:
- Multi-objective critic (vector Q-functions)
- Constraint-aware actor
- Weight-conditioned policy for Pareto front exploration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build MLP with specified architecture."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
        
    layers = []
    dims = [input_dim] + hidden_dims
    
    act_fn = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(act_fn.get(activation, nn.ReLU)())
        
    layers.append(nn.Linear(dims[-1], output_dim))
    
    if output_activation is not None:
        layers.append(act_fn.get(output_activation, nn.Identity)())
        
    return nn.Sequential(*layers)


class CMOCritic(nn.Module):
    """
    Multi-Objective Critic for CMO-SAC.
    
    Learns separate Q-functions for each objective:
    Q_i(s, a) estimates expected return for objective i.
    
    Also learns constraint cost Q-functions:
    Q_c_j(s, a) estimates expected constraint cost for constraint j.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        n_objectives: Number of objectives (PUE, WUE, throughput)
        n_constraints: Number of constraints (thermal, HBM, SLA, coolant)
        hidden_dims: Hidden layer dimensions
        n_critics: Number of critic ensembles (for min Q trick)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_objectives: int = 3,
        n_constraints: int = 4,
        hidden_dims: List[int] = [256, 256],
        n_critics: int = 2,
    ):
        super().__init__()
        
        self.n_objectives = n_objectives
        self.n_constraints = n_constraints
        self.n_critics = n_critics
        
        # Ensemble of critics for each objective
        self.objective_critics = nn.ModuleList([
            nn.ModuleList([
                mlp(obs_dim + action_dim, hidden_dims, 1)
                for _ in range(n_critics)
            ])
            for _ in range(n_objectives)
        ])
        
        # Ensemble of critics for each constraint cost
        self.constraint_critics = nn.ModuleList([
            nn.ModuleList([
                mlp(obs_dim + action_dim, hidden_dims, 1)
                for _ in range(n_critics)
            ])
            for _ in range(n_constraints)
        ])
        
        # Combined scalarized critic (optional, for efficiency)
        self.scalarized_critics = nn.ModuleList([
            mlp(obs_dim + action_dim + n_objectives, hidden_dims, 1)
            for _ in range(n_critics)
        ])
        
    def forward_objectives(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute Q-values for all objectives.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            action: Action tensor [batch, action_dim]
            
        Returns:
            q1_values: List of Q1 values for each objective
            q2_values: List of Q2 values for each objective (if n_critics >= 2)
        """
        x = torch.cat([obs, action], dim=-1)
        
        q1_values = []
        q2_values = []
        
        for obj_critics in self.objective_critics:
            q1 = obj_critics[0](x)
            q1_values.append(q1)
            
            if self.n_critics >= 2:
                q2 = obj_critics[1](x)
                q2_values.append(q2)
                
        return q1_values, q2_values
    
    def forward_constraints(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute Q-values for all constraint costs.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            action: Action tensor [batch, action_dim]
            
        Returns:
            q1_costs: List of Q1 cost values for each constraint
            q2_costs: List of Q2 cost values for each constraint
        """
        x = torch.cat([obs, action], dim=-1)
        
        q1_costs = []
        q2_costs = []
        
        for con_critics in self.constraint_critics:
            q1 = con_critics[0](x)
            q1_costs.append(q1)
            
            if self.n_critics >= 2:
                q2 = con_critics[1](x)
                q2_costs.append(q2)
                
        return q1_costs, q2_costs
    
    def forward_scalarized(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scalarized Q-value conditioned on weights.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            action: Action tensor [batch, action_dim]
            weights: Scalarization weights [batch, n_objectives]
            
        Returns:
            q1: First critic Q-value
            q2: Second critic Q-value
        """
        x = torch.cat([obs, action, weights], dim=-1)
        
        q1 = self.scalarized_critics[0](x)
        q2 = self.scalarized_critics[1](x) if self.n_critics >= 2 else q1
        
        return q1, q2
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass computing all Q-values.
        
        Returns dict with:
        - objective_q1, objective_q2: Stacked objective Q-values [batch, n_obj]
        - constraint_q1, constraint_q2: Stacked constraint Q-values [batch, n_con]
        - scalarized_q1, scalarized_q2: Weight-conditioned Q-values [batch, 1]
        """
        q1_obj, q2_obj = self.forward_objectives(obs, action)
        q1_con, q2_con = self.forward_constraints(obs, action)
        
        result = {
            "objective_q1": torch.cat(q1_obj, dim=-1),
            # Handle empty constraint lists (when n_constraints=0)
            "constraint_q1": torch.cat(q1_con, dim=-1) if q1_con else torch.tensor([], device=obs.device).reshape(obs.shape[0], 0),
        }
        
        if q2_obj:
            result["objective_q2"] = torch.cat(q2_obj, dim=-1)
        if q2_con:
            result["constraint_q2"] = torch.cat(q2_con, dim=-1)
        elif self.n_critics >= 2:
            # Empty constraint tensor for consistency
            result["constraint_q2"] = torch.tensor([], device=obs.device).reshape(obs.shape[0], 0)
            
        if weights is not None:
            q1_scalar, q2_scalar = self.forward_scalarized(obs, action, weights)
            result["scalarized_q1"] = q1_scalar
            result["scalarized_q2"] = q2_scalar
            
        return result


class CMOActor(nn.Module):
    """
    Weight-Conditioned Actor for CMO-SAC.
    
    The policy is conditioned on scalarization weights, allowing it to
    produce different behaviors for different objective trade-offs:
    
    π(a|s, w) = policy conditioned on state s and weight vector w
    
    This enables Pareto front exploration by varying weights during
    training and deployment.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        n_objectives: Number of objectives for weight conditioning
        hidden_dims: Hidden layer dimensions
        log_std_bounds: Bounds for log standard deviation
        action_scale: Scale for action output
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_objectives: int = 3,
        hidden_dims: List[int] = [256, 256],
        log_std_bounds: Tuple[float, float] = (-20, 2),
        action_scale: float = 1.0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.log_std_min, self.log_std_max = log_std_bounds
        
        # Weight embedding
        self.weight_embed = nn.Sequential(
            nn.Linear(n_objectives, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Main policy network (state + weight embedding)
        self.trunk = mlp(
            obs_dim + 64,  # state + weight embedding
            hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims,
            hidden_dims[-1],
            activation="relu",
        )
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(
        self,
        obs: torch.Tensor,
        weights: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action given state and weight vector.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            weights: Weight vector [batch, n_objectives]
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled or mean action [batch, action_dim]
            log_prob: Log probability of action [batch, 1]
        """
        # Embed weights
        w_embed = self.weight_embed(weights)
        
        # Concatenate state and weight embedding
        x = torch.cat([obs, w_embed], dim=-1)
        
        # Forward through trunk
        features = self.trunk(x)
        
        # Get mean and log_std
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Safety: Check for NaN/Inf and clamp to prevent distribution errors
        if torch.any(torch.isnan(mean)) or torch.any(torch.isinf(mean)):
            # Silently handle NaN/Inf in actor mean
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.any(torch.isnan(std)) or torch.any(torch.isinf(std)):
            # Silently handle NaN/Inf in actor std
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=0.01)
        
        # Ensure std is always positive and bounded
        std = torch.clamp(std, min=1e-4, max=10.0)
        
        # Create distribution
        dist = Normal(mean, std)
        
        if deterministic:
            action_pre_tanh = mean
        else:
            action_pre_tanh = dist.rsample()
            
        # Apply tanh squashing
        action = torch.tanh(action_pre_tanh) * self.action_scale
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(action_pre_tanh).sum(dim=-1, keepdim=True)
        # Tanh squashing correction
        log_prob = log_prob - torch.log(
            1 - action.pow(2) + 1e-6
        ).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(
        self,
        obs: torch.Tensor,
        weights: torch.Tensor,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get action as numpy array."""
        with torch.no_grad():
            action, _ = self.forward(obs, weights, deterministic)
        return action.cpu().numpy()


class UnconditionalActor(nn.Module):
    """
    Standard SAC actor without weight conditioning.
    
    Used for single-objective baseline or when weights are fixed.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        action_scale: float = 1.0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        self.trunk = mlp(
            obs_dim,
            hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims,
            hidden_dims[-1],
        )
        
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action given state."""
        features = self.trunk(obs)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        
        if deterministic:
            action_pre_tanh = mean
        else:
            action_pre_tanh = dist.rsample()
            
        action = torch.tanh(action_pre_tanh) * self.action_scale
        
        log_prob = dist.log_prob(action_pre_tanh).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(
            1 - action.pow(2) + 1e-6
        ).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class ConstraintPredictor(nn.Module):
    """
    One-step constraint cost predictor for safety filtering.
    
    Predicts whether taking action a in state s will violate constraints
    at the next time step. Used by the SafetyFilter to project unsafe
    actions onto the safe set.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        n_constraints: Number of constraints to predict
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_constraints: int = 4,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        self.n_constraints = n_constraints
        
        # Predict constraint costs at next state
        self.cost_predictor = mlp(
            obs_dim + action_dim,
            hidden_dims,
            n_constraints,
        )
        
        # Predict violation probability
        self.violation_predictor = mlp(
            obs_dim + action_dim,
            hidden_dims,
            n_constraints,
            output_activation="sigmoid" if TORCH_AVAILABLE else None,
        )
        
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict constraint costs and violation probabilities.
        
        Args:
            obs: Current observation [batch, obs_dim]
            action: Proposed action [batch, action_dim]
            
        Returns:
            costs: Predicted constraint costs [batch, n_constraints]
            probs: Predicted violation probabilities [batch, n_constraints]
        """
        x = torch.cat([obs, action], dim=-1)
        
        costs = F.relu(self.cost_predictor(x))  # Costs are non-negative
        probs = torch.sigmoid(self.violation_predictor(x))
        
        return costs, probs
    
    def predict_safe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Predict whether action is safe (no violations).
        
        Returns:
            is_safe: Boolean tensor [batch,]
        """
        _, probs = self.forward(obs, action)
        is_safe = (probs < threshold).all(dim=-1)
        return is_safe
