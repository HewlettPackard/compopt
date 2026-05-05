"""
CMO-SAC Agent: Constrained Multi-Objective Soft Actor-Critic.

This module implements the main CMO-SAC agent combining:
- Weight-conditioned actor for Pareto front exploration
- Multi-objective critic ensemble
- Lagrangian dual optimization for constraints
- Optional safety filter for deployment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .networks import CMOCritic, CMOActor, ConstraintPredictor
from .constraints import ConstraintManager, LagrangianDualOptimizer, ConstraintBuffer
from .rewards import MultiObjectiveReward, WeightSampler
from .pareto import ParetoFrontGenerator
from .safety import SafetyFilter


class CMOSACAgent:
    """
    Constrained Multi-Objective Soft Actor-Critic Agent.
    
    Key features:
    1. Multi-objective optimization with Pareto front exploration
    2. Lagrangian relaxation for constraint satisfaction
    3. Weight-conditioned policy for different trade-offs
    4. Safety filter for constraint-respecting deployment
    
    The agent learns to optimize:
        L(π, λ) = J(π) - Σ_i λ_i * (E[G_c_i] - d_i)
        
    where J(π) is the scalarized multi-objective return and
    λ_i are Lagrange multipliers for constraint i.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        n_objectives: Number of objectives (default: 3 - PUE, WUE, throughput)
        n_constraints: Number of constraints (default: 4 - thermal, HBM, SLA, coolant)
        hidden_dims: Hidden layer dimensions for networks
        actor_lr: Learning rate for actor
        critic_lr: Learning rate for critic
        alpha_lr: Learning rate for entropy coefficient
        discount: Discount factor
        tau: Soft update coefficient for target networks
        initial_alpha: Initial entropy coefficient
        auto_alpha: Whether to automatically tune alpha
        device: Torch device
        weight_strategy: Strategy for sampling scalarization weights
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_objectives: int = 3,
        n_constraints: int = 4,
        hidden_dims: List[int] = [256, 256],
        actor_lr: float = 5e-5,  # Reduced from 1e-4 (more conservative)
        critic_lr: float = 5e-5,  # Reduced from 1e-4 (more conservative)
        alpha_lr: float = 1e-5,   # Keep very slow
        discount: float = 0.99,
        tau: float = 0.005,
        initial_alpha: float = 0.01,  # Start very low
        auto_alpha: bool = False,  # DISABLE auto-tuning - use fixed alpha
        device: str = "auto",
        weight_strategy: str = "dirichlet",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CMOSACAgent")
            
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.n_constraints = n_constraints
        self.discount = discount
        self.tau = tau
        
        # Networks
        self.actor = CMOActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_objectives=n_objectives,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        self.critic = CMOCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_objectives=n_objectives,
            n_constraints=n_constraints,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        self.critic_target = CMOCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_objectives=n_objectives,
            n_constraints=n_constraints,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Constraint predictor for safety filter
        self.constraint_predictor = ConstraintPredictor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_constraints=n_constraints,
            hidden_dims=hidden_dims[:1],  # Smaller network
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.predictor_optimizer = optim.Adam(
            self.constraint_predictor.parameters(), lr=critic_lr
        )
        
        # Entropy coefficient (alpha)
        self.auto_alpha = auto_alpha
        # FIXED: Use -0.1 * action_dim (very conservative)
        # -0.5 was still too high, causing alpha to increase to 0.28
        # For 1D action: target = -0.1 allows exploitation much earlier
        self.target_entropy = -0.1 * action_dim  # Very conservative entropy target
        
        if auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(initial_alpha),
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(initial_alpha), device=self.device)
            
        # Constraint management
        self.constraint_manager = ConstraintManager()
        self.lagrangian = LagrangianDualOptimizer(
            self.constraint_manager,
            device=str(self.device),
        )
        
        # Multi-objective reward
        self.mo_reward = MultiObjectiveReward()
        
        # Weight sampler for Pareto exploration
        self.weight_sampler = WeightSampler(
            n_objectives=n_objectives,
            strategy=weight_strategy,
        )
        
        # Pareto front tracker
        self.pareto_generator = ParetoFrontGenerator(
            n_objectives=n_objectives,
            objective_names=["pue", "wue", "throughput"],
            minimize={"pue": True, "wue": True, "throughput": False},
        )
        
        # Safety filter (disabled by default during training)
        self.safety_filter = None
        
        # Current scalarization weights
        self._current_weights = np.ones(n_objectives) / n_objectives
        
        # Training statistics
        self._n_updates = 0
        
    @property
    def alpha(self) -> torch.Tensor:
        """Get current entropy coefficient."""
        return self.log_alpha.exp()
    
    def select_action(
        self,
        obs: np.ndarray,
        weights: Optional[np.ndarray] = None,
        deterministic: bool = False,
        use_safety_filter: bool = False,
    ) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            weights: Optional scalarization weights (uses current if None)
            deterministic: If True, return mean action
            use_safety_filter: If True, filter action for safety
            
        Returns:
            action: Selected action
        """
        if weights is None:
            weights = self._current_weights
            
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if weights_t.dim() == 1:
            weights_t = weights_t.unsqueeze(0)
            
        with torch.no_grad():
            action, _ = self.actor(obs_t, weights_t, deterministic=deterministic)
            action = action.squeeze(0).cpu().numpy()
            
        # Apply safety filter if enabled
        if use_safety_filter and self.safety_filter is not None:
            action = self.safety_filter.filter(obs, action)
            
        return action
    
    def sample_weights(self) -> np.ndarray:
        """Sample new scalarization weights."""
        self._current_weights = self.weight_sampler.sample()
        return self._current_weights
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set scalarization weights."""
        self._current_weights = weights
        
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Update agent with batch of transitions.
        
        Args:
            batch: Dict with keys:
                - observations: [batch, obs_dim]
                - actions: [batch, action_dim]
                - rewards: [batch, n_objectives] multi-objective rewards
                - scalarized_rewards: [batch,] scalarized rewards
                - constraint_costs: [batch, n_constraints]
                - next_observations: [batch, obs_dim]
                - dones: [batch,]
                - weights: [batch, n_objectives]
                
        Returns:
            Dict of training metrics
        """
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        scalarized_rewards = batch["scalarized_rewards"]
        constraint_costs = batch["constraint_costs"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]
        weights = batch["weights"]
        
        batch_size = obs.shape[0]
        
        # ========== Critic Update ==========
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor(next_obs, weights)
            
            # Get target Q-values
            target_qs = self.critic_target.forward(
                next_obs, next_actions, weights
            )
            
            # Use min Q for each objective (conservative)
            target_q_obj = torch.min(
                target_qs["objective_q1"],
                target_qs.get("objective_q2", target_qs["objective_q1"]),
            )
            target_q_con = torch.min(
                target_qs["constraint_q1"],
                target_qs.get("constraint_q2", target_qs["constraint_q1"]),
            )
            
            # Scalarized target with Lagrangian penalty
            lambdas = self.lagrangian.lambdas
            scalarized_target = torch.zeros(batch_size, 1, device=self.device)
            
            for i in range(self.n_objectives):
                obj_weight = weights[:, i:i+1]
                obj_sign = -1.0 if i < 2 else 1.0  # Minimize PUE/WUE, maximize throughput
                scalarized_target += obj_weight * obj_sign * target_q_obj[:, i:i+1]
                
            # Subtract constraint penalty
            for i in range(self.n_constraints):
                scalarized_target -= lambdas[i] * target_q_con[:, i:i+1]
                
            # Subtract entropy bonus
            scalarized_target -= self.alpha * next_log_probs
            
            # Compute targets
            not_done = 1.0 - dones.unsqueeze(-1)
            
            # Objective targets
            obj_targets = rewards + self.discount * not_done * target_q_obj
            
            # Constraint targets
            con_targets = constraint_costs + self.discount * not_done * target_q_con
            
            # Scalarized target
            scalar_target = scalarized_rewards.unsqueeze(-1) + \
                self.discount * not_done * scalarized_target
        
        # Clamp targets to prevent Inf values
        max_q_value = 1e6  # Reasonable upper bound for Q-values
        obj_targets = torch.clamp(obj_targets, -max_q_value, max_q_value)
        con_targets = torch.clamp(con_targets, -max_q_value, max_q_value)
        scalar_target = torch.clamp(scalar_target, -max_q_value, max_q_value)
        
        # Check for NaN/Inf in targets
        if torch.any(torch.isnan(obj_targets)) or torch.any(torch.isinf(obj_targets)):
            pass  # Silently skip update on NaN/Inf
            print(f"  rewards: {rewards.mean()}, target_q_obj: {target_q_obj.mean()}")
            return {
                "critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0, "alpha": self.alpha.item(),
            }
        
        # Current Q-values
        current_qs = self.critic.forward(obs, actions, weights)
        
        # Clamp current Q-values too
        if "objective_q1" in current_qs:
            current_qs["objective_q1"] = torch.clamp(current_qs["objective_q1"], -max_q_value, max_q_value)
        if "objective_q2" in current_qs:
            current_qs["objective_q2"] = torch.clamp(current_qs["objective_q2"], -max_q_value, max_q_value)
        if "constraint_q1" in current_qs:
            current_qs["constraint_q1"] = torch.clamp(current_qs["constraint_q1"], -max_q_value, max_q_value)
        if "constraint_q2" in current_qs:
            current_qs["constraint_q2"] = torch.clamp(current_qs["constraint_q2"], -max_q_value, max_q_value)
        if "scalarized_q1" in current_qs:
            current_qs["scalarized_q1"] = torch.clamp(current_qs["scalarized_q1"], -max_q_value, max_q_value)
        if "scalarized_q2" in current_qs:
            current_qs["scalarized_q2"] = torch.clamp(current_qs["scalarized_q2"], -max_q_value, max_q_value)
        
        # Objective critic loss
        obj_loss = F.mse_loss(current_qs["objective_q1"], obj_targets)
        if "objective_q2" in current_qs:
            obj_loss += F.mse_loss(current_qs["objective_q2"], obj_targets)
            
        # Constraint critic loss
        con_loss = F.mse_loss(current_qs["constraint_q1"], con_targets)
        if "constraint_q2" in current_qs:
            con_loss += F.mse_loss(current_qs["constraint_q2"], con_targets)
            
        # Scalarized critic loss
        scalar_loss = F.mse_loss(current_qs["scalarized_q1"], scalar_target)
        if "scalarized_q2" in current_qs:
            scalar_loss += F.mse_loss(current_qs["scalarized_q2"], scalar_target)
            
        critic_loss = obj_loss + con_loss + scalar_loss
        
        # Check for NaN in critic loss
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            pass  # Silently skip critic update
            print(f"  obj_loss: {obj_loss}, con_loss: {con_loss}, scalar_loss: {scalar_loss}")
            # Skip this update
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": self.alpha.item(),
            }
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients to prevent exploding gradients and NaN
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # ========== Actor Update ==========
        new_actions, log_probs = self.actor(obs, weights)
        
        actor_qs = self.critic.forward(obs, new_actions, weights)
        
        # Scalarized Q-value
        q_scalarized = actor_qs["scalarized_q1"]
        
        # Actor loss: maximize Q - alpha * log_prob
        actor_loss = (self.alpha.detach() * log_probs - q_scalarized).mean()
        
        # Check for NaN in actor loss
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            pass  # Silently skip actor update
            print(f"  log_probs: {log_probs.mean()}, q_scalarized: {q_scalarized.mean()}")
            # Skip actor update but continue with rest
            actor_loss = torch.tensor(0.0, device=self.device)
        else:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradients to prevent exploding gradients and NaN
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
        
        # ========== Alpha Update ==========
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
        # ========== Constraint Predictor Update ==========
        pred_costs, pred_probs = self.constraint_predictor(obs, actions)
        
        # Predict cost at next state
        predictor_loss = F.mse_loss(pred_costs, constraint_costs)
        # Binary classification for violations
        violations = (constraint_costs > 0).float()
        predictor_loss += F.binary_cross_entropy(pred_probs, violations)
        
        self.predictor_optimizer.zero_grad()
        predictor_loss.backward()
        self.predictor_optimizer.step()
        
        # ========== Target Network Update ==========
        self._soft_update(self.critic, self.critic_target)
        
        self._n_updates += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": float(alpha_loss) if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "alpha": self.alpha.item(),
            "predictor_loss": predictor_loss.item(),
            "objective_q_mean": current_qs["objective_q1"].mean().item(),
            "constraint_q_mean": current_qs["constraint_q1"].mean().item(),
        }
    
    def dual_update(self, constraint_returns: Dict[str, float]) -> Dict[str, float]:
        """
        Update Lagrange multipliers based on constraint returns.
        
        Args:
            constraint_returns: Dict of discounted constraint cost returns
            
        Returns:
            Dict of dual update metrics
        """
        gradients = self.lagrangian.dual_step(constraint_returns)
        
        return {
            "lambdas": self.lagrangian.get_lambdas_dict(),
            "dual_gradients": gradients,
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
    def enable_safety_filter(self, action_bounds: Optional[Tuple] = None) -> None:
        """Enable safety filter for deployment."""
        self.safety_filter = SafetyFilter(
            constraint_predictor=self.constraint_predictor,
            action_bounds=action_bounds,
        )
        
    def disable_safety_filter(self) -> None:
        """Disable safety filter."""
        self.safety_filter = None
        
    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "constraint_predictor": self.constraint_predictor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "predictor_optimizer": self.predictor_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "lagrangian": self.lagrangian.state_dict(),
            "n_updates": self._n_updates,
        }, path)
        
    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.constraint_predictor.load_state_dict(checkpoint["constraint_predictor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.predictor_optimizer.load_state_dict(checkpoint["predictor_optimizer"])
        self.log_alpha = checkpoint["log_alpha"]
        self.lagrangian.load_state_dict(checkpoint["lagrangian"])
        self._n_updates = checkpoint["n_updates"]
        
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "n_objectives": self.n_objectives,
            "n_constraints": self.n_constraints,
            "discount": self.discount,
            "tau": self.tau,
            "auto_alpha": self.auto_alpha,
            "device": str(self.device),
        }
