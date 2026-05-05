"""
Safety Filter for CMO-SAC.

This module implements a safety filter that projects unsafe actions
onto the safe set using convex optimization (QP solver).
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class SafetyFilter:
    """
    Safety filter that projects unsafe actions onto the safe set.
    
    Uses the one-step thermal predictor to forecast constraint violations.
    If a proposed action would violate constraints, it is projected onto
    the closest safe action using quadratic programming.
    
    The optimization problem:
        min_a' ||a' - a||^2
        s.t.  g_i(s, a') <= 0  for all constraints i
        
    where g_i is the predicted constraint violation.
    
    Args:
        constraint_predictor: Neural network predicting constraint costs
        action_bounds: Tuple of (low, high) action bounds
        violation_threshold: Probability threshold for violation
        solver: QP solver to use ('cvxpy' or 'gradient')
    """
    
    def __init__(
        self,
        constraint_predictor: Optional[nn.Module] = None,
        action_bounds: Tuple[np.ndarray, np.ndarray] = None,
        violation_threshold: float = 0.5,
        solver: str = "gradient",
        gradient_steps: int = 10,
        gradient_lr: float = 0.1,
    ):
        self.predictor = constraint_predictor
        self.violation_threshold = violation_threshold
        self.solver = solver
        self.gradient_steps = gradient_steps
        self.gradient_lr = gradient_lr
        
        if action_bounds is not None:
            self.action_low, self.action_high = action_bounds
        else:
            self.action_low = None
            self.action_high = None
            
        # Statistics
        self.n_filtered = 0
        self.n_total = 0
        
    def filter(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        return_info: bool = False,
    ) -> np.ndarray:
        """
        Filter action to ensure safety.
        
        Args:
            obs: Current observation [obs_dim] or [batch, obs_dim]
            action: Proposed action [action_dim] or [batch, action_dim]
            return_info: If True, return additional info dict
            
        Returns:
            safe_action: Filtered safe action
            info: (optional) Dict with filtering details
        """
        self.n_total += 1
        
        if self.predictor is None:
            # No predictor, return original action
            if return_info:
                return action, {"filtered": False}
            return action
            
        # Check if action is safe
        is_safe, violation_probs = self._check_safety(obs, action)
        
        if is_safe:
            if return_info:
                return action, {"filtered": False, "violation_probs": violation_probs}
            return action
            
        # Action is unsafe, project onto safe set
        self.n_filtered += 1
        
        if self.solver == "cvxpy" and CVXPY_AVAILABLE:
            safe_action = self._solve_qp(obs, action)
        else:
            safe_action = self._solve_gradient(obs, action)
            
        if return_info:
            return safe_action, {
                "filtered": True,
                "original_action": action,
                "violation_probs": violation_probs,
            }
        return safe_action
    
    def _check_safety(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """Check if action is safe using predictor."""
        if not TORCH_AVAILABLE:
            return True, np.zeros(4)
            
        # Convert to torch
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            _, probs = self.predictor(obs_t, action_t)
            probs = probs.squeeze(0).numpy()
            
        is_safe = (probs < self.violation_threshold).all()
        return is_safe, probs
    
    def _solve_gradient(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """
        Solve safety projection using gradient descent.
        
        Minimizes: ||a' - a||^2 + λ * Σ max(0, g_i(s, a'))
        """
        if not TORCH_AVAILABLE:
            return action
            
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        # Initialize with original action
        a_opt = action_t.clone().requires_grad_(True)
        
        # Gradient descent
        for _ in range(self.gradient_steps):
            # Compute constraint violations
            costs, probs = self.predictor(obs_t, a_opt)
            
            # Compute loss
            distance_loss = ((a_opt - action_t) ** 2).sum()
            constraint_loss = (probs * costs).sum()  # Weighted by probability
            
            loss = distance_loss + 10.0 * constraint_loss
            
            # Backward
            loss.backward()
            
            # Update
            with torch.no_grad():
                a_opt -= self.gradient_lr * a_opt.grad
                
                # Clip to bounds
                if self.action_low is not None:
                    a_opt.clamp_(
                        min=torch.tensor(self.action_low),
                        max=torch.tensor(self.action_high),
                    )
                else:
                    a_opt.clamp_(-1.0, 1.0)
                    
                a_opt.grad.zero_()
                a_opt.requires_grad_(True)
                
        return a_opt.detach().squeeze(0).numpy()
    
    def _solve_qp(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """
        Solve safety projection using CVXPY.
        
        Solves the QP:
            min ||a' - a||^2
            s.t. A @ a' <= b  (linearized constraints)
        """
        if not CVXPY_AVAILABLE:
            return self._solve_gradient(obs, action)
            
        action_dim = len(action)
        
        # Decision variable
        a_opt = cp.Variable(action_dim)
        
        # Objective: minimize distance to original action
        objective = cp.Minimize(cp.sum_squares(a_opt - action))
        
        # Constraints
        constraints = []
        
        # Action bounds
        if self.action_low is not None:
            constraints.append(a_opt >= self.action_low)
            constraints.append(a_opt <= self.action_high)
        else:
            constraints.append(a_opt >= -1.0)
            constraints.append(a_opt <= 1.0)
            
        # Linearized safety constraints
        # Get gradient of constraint function w.r.t. action
        if TORCH_AVAILABLE and self.predictor is not None:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32, requires_grad=True).unsqueeze(0)
            
            costs, _ = self.predictor(obs_t, action_t)
            
            for i in range(costs.shape[-1]):
                # Compute gradient of constraint i
                cost_i = costs[0, i]
                cost_i.backward(retain_graph=True)
                
                grad = action_t.grad.squeeze(0).numpy()
                action_t.grad.zero_()
                
                # Linearized constraint: g(a) + grad @ (a' - a) <= 0
                # => grad @ a' <= grad @ a - g(a)
                g_val = cost_i.item()
                if g_val > 0:  # Only add if currently violated
                    rhs = np.dot(grad, action) - g_val * 0.5  # Some margin
                    constraints.append(grad @ a_opt <= rhs)
                    
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP)
            if a_opt.value is not None:
                return a_opt.value
        except Exception:
            pass
            
        # Fallback to gradient if QP fails
        return self._solve_gradient(obs, action)
    
    def get_filter_rate(self) -> float:
        """Get fraction of actions that were filtered."""
        if self.n_total == 0:
            return 0.0
        return self.n_filtered / self.n_total
    
    def reset_stats(self) -> None:
        """Reset filtering statistics."""
        self.n_filtered = 0
        self.n_total = 0


class ThermalSafetyFilter(SafetyFilter):
    """
    Specialized safety filter using physics-based thermal prediction.
    
    Uses a differentiable RC-network thermal model (from PI-HAC) to
    predict temperature violations, providing more accurate safety
    guarantees than learned predictors.
    """
    
    def __init__(
        self,
        thermal_model=None,  # ThermalPredictor from PI-HAC
        thermal_limit: float = 83.0,
        action_bounds: Tuple[np.ndarray, np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(
            constraint_predictor=None,
            action_bounds=action_bounds,
            **kwargs,
        )
        self.thermal_model = thermal_model
        self.thermal_limit = thermal_limit
        
    def _check_safety(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """Check thermal safety using physics model."""
        if self.thermal_model is None:
            return True, np.zeros(1)
            
        if not TORCH_AVAILABLE:
            return True, np.zeros(1)
            
        # Extract current state from observation
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Predict next temperature using thermal model
            # Assumes thermal_model has predict_next method
            T_next = self.thermal_model.predict_next(obs_t, action_t)
            T_max = T_next.max().item()
            
        violation = max(0.0, T_max - self.thermal_limit)
        is_safe = violation == 0.0
        
        return is_safe, np.array([violation])
