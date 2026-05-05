"""
Pareto Front Generation for CMO-SAC.

This module implements utilities for generating and analyzing
Pareto fronts from multi-objective RL policies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ParetoPoint:
    """A single point on the Pareto front."""
    objectives: Dict[str, float]
    weights: np.ndarray
    policy_params: Optional[Dict] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def dominates(self, other: "ParetoPoint", minimize: Dict[str, bool] = None) -> bool:
        """
        Check if this point dominates another.
        
        Dominance: better or equal in all objectives, strictly better in at least one.
        """
        if minimize is None:
            minimize = {name: True for name in self.objectives}
            
        dominated_any = False
        for name, value in self.objectives.items():
            other_value = other.objectives.get(name, float('inf'))
            
            if minimize.get(name, True):
                # Minimizing: smaller is better
                if value > other_value:
                    return False
                if value < other_value:
                    dominated_any = True
            else:
                # Maximizing: larger is better
                if value < other_value:
                    return False
                if value > other_value:
                    dominated_any = True
                    
        return dominated_any


class ParetoFrontGenerator:
    """
    Generates and maintains a Pareto front of policies.
    
    Supports multiple strategies for exploring the Pareto front:
    - Weight sweep: Train policies with different scalarization weights
    - Evolutionary: Use genetic algorithms to evolve policy population
    - Envelope: Use Q-envelope algorithm for continuous Pareto discovery
    
    Args:
        n_objectives: Number of objectives
        objective_names: Names of objectives
        minimize: Dict mapping objective names to minimize flag
        front_size: Maximum number of points to maintain
    """
    
    def __init__(
        self,
        n_objectives: int = 3,
        objective_names: List[str] = None,
        minimize: Dict[str, bool] = None,
        front_size: int = 100,
    ):
        self.n_objectives = n_objectives
        self.objective_names = objective_names or [f"obj_{i}" for i in range(n_objectives)]
        self.minimize = minimize or {name: True for name in self.objective_names}
        self.front_size = front_size
        
        # Store all evaluated points
        self._all_points: List[ParetoPoint] = []
        
        # Current Pareto front (non-dominated points)
        self._pareto_front: List[ParetoPoint] = []
        
    def add_point(
        self,
        objectives: Dict[str, float],
        weights: np.ndarray,
        policy_params: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Add a new evaluated point and update Pareto front.
        
        Args:
            objectives: Dict of objective values
            weights: Scalarization weights used
            policy_params: Optional policy parameters (state dict)
            metrics: Additional metrics (episode length, etc.)
            
        Returns:
            True if point is added to Pareto front
        """
        point = ParetoPoint(
            objectives=objectives.copy(),
            weights=weights.copy(),
            policy_params=policy_params,
            metrics=metrics or {},
        )
        
        self._all_points.append(point)
        
        # Check if point is dominated by any front point
        for front_point in self._pareto_front:
            if front_point.dominates(point, self.minimize):
                return False
                
        # Point is not dominated, add to front
        # Remove any points dominated by the new point
        self._pareto_front = [
            p for p in self._pareto_front
            if not point.dominates(p, self.minimize)
        ]
        self._pareto_front.append(point)
        
        # Prune if too many points
        if len(self._pareto_front) > self.front_size:
            self._prune_front()
            
        return True
    
    def _prune_front(self) -> None:
        """Prune front to maintain size limit using crowding distance."""
        if len(self._pareto_front) <= self.front_size:
            return
            
        # Compute crowding distance
        distances = self._compute_crowding_distances()
        
        # Keep points with highest crowding distance
        sorted_indices = np.argsort(distances)[::-1]
        keep_indices = sorted_indices[:self.front_size]
        self._pareto_front = [self._pareto_front[i] for i in keep_indices]
        
    def _compute_crowding_distances(self) -> np.ndarray:
        """Compute crowding distance for each front point."""
        n_points = len(self._pareto_front)
        distances = np.zeros(n_points)
        
        for obj_name in self.objective_names:
            # Sort by this objective
            values = [p.objectives.get(obj_name, 0) for p in self._pareto_front]
            sorted_idx = np.argsort(values)
            
            # Boundary points get infinite distance
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            # Compute normalized distance for interior points
            obj_range = max(values) - min(values)
            if obj_range > 0:
                for i in range(1, n_points - 1):
                    idx = sorted_idx[i]
                    prev_idx = sorted_idx[i - 1]
                    next_idx = sorted_idx[i + 1]
                    distances[idx] += (values[next_idx] - values[prev_idx]) / obj_range
                    
        return distances
    
    def get_pareto_front(self) -> List[ParetoPoint]:
        """Get current Pareto front."""
        return self._pareto_front.copy()
    
    def get_front_objectives(self) -> Dict[str, np.ndarray]:
        """Get objective values as arrays for plotting."""
        result = {name: [] for name in self.objective_names}
        
        for point in self._pareto_front:
            for name in self.objective_names:
                result[name].append(point.objectives.get(name, 0))
                
        return {name: np.array(values) for name, values in result.items()}
    
    def get_hypervolume(self, reference_point: Dict[str, float] = None) -> float:
        """
        Compute hypervolume indicator of the Pareto front.
        
        The hypervolume is the volume of the space dominated by the
        Pareto front and bounded by the reference point.
        """
        if not self._pareto_front:
            return 0.0
            
        # Default reference point (worst case for each objective)
        if reference_point is None:
            reference_point = {}
            for name in self.objective_names:
                values = [p.objectives.get(name, 0) for p in self._pareto_front]
                if self.minimize.get(name, True):
                    reference_point[name] = max(values) * 1.1
                else:
                    reference_point[name] = min(values) * 0.9
                    
        # Simple 2D hypervolume calculation
        if self.n_objectives == 2:
            return self._hypervolume_2d(reference_point)
            
        # For higher dimensions, use Monte Carlo approximation
        return self._hypervolume_mc(reference_point)
    
    def _hypervolume_2d(self, reference: Dict[str, float]) -> float:
        """Compute exact 2D hypervolume."""
        if len(self.objective_names) != 2:
            return 0.0
            
        obj1, obj2 = self.objective_names[0], self.objective_names[1]
        
        # Get points sorted by first objective
        points = [(p.objectives[obj1], p.objectives[obj2]) for p in self._pareto_front]
        
        # Normalize based on minimize flags
        min1, min2 = self.minimize.get(obj1, True), self.minimize.get(obj2, True)
        ref1, ref2 = reference[obj1], reference[obj2]
        
        if not min1:
            points = [(-x, y) for x, y in points]
            ref1 = -ref1
        if not min2:
            points = [(x, -y) for x, y in points]
            ref2 = -ref2
            
        points.sort()
        
        # Sweep algorithm
        hv = 0.0
        prev_x = points[0][0]
        min_y = ref2
        
        for x, y in points:
            if y < min_y:
                hv += (x - prev_x) * (min_y - y)
                min_y = y
            prev_x = x
            
        # Add final rectangle
        hv += (ref1 - prev_x) * (min_y - points[-1][1])
        
        return max(0.0, hv)
    
    def _hypervolume_mc(
        self,
        reference: Dict[str, float],
        n_samples: int = 10000,
    ) -> float:
        """Monte Carlo approximation of hypervolume."""
        # Get bounds
        mins = {name: float('inf') for name in self.objective_names}
        maxs = {name: float('-inf') for name in self.objective_names}
        
        for point in self._pareto_front:
            for name, value in point.objectives.items():
                mins[name] = min(mins[name], value)
                maxs[name] = max(maxs[name], value)
                
        # Sample random points
        n_dominated = 0
        
        for _ in range(n_samples):
            sample = {
                name: np.random.uniform(mins[name], reference[name])
                for name in self.objective_names
            }
            
            # Check if dominated by any front point
            for point in self._pareto_front:
                dominated = True
                for name, value in point.objectives.items():
                    sample_val = sample[name]
                    if self.minimize.get(name, True):
                        if value > sample_val:
                            dominated = False
                            break
                    else:
                        if value < sample_val:
                            dominated = False
                            break
                            
                if dominated:
                    n_dominated += 1
                    break
                    
        # Compute volume
        total_volume = np.prod([
            reference[name] - mins[name]
            for name in self.objective_names
        ])
        
        return total_volume * n_dominated / n_samples
    
    def suggest_weights(self, strategy: str = "gap") -> np.ndarray:
        """
        Suggest next weights to explore based on current front.
        
        Args:
            strategy: 'gap' (largest gap), 'random', or 'uniform'
            
        Returns:
            Weight vector for next policy training
        """
        if strategy == "random" or len(self._pareto_front) < 2:
            # Random weights on simplex
            weights = np.random.exponential(1.0, self.n_objectives)
            return weights / weights.sum()
            
        if strategy == "uniform":
            # Uniform grid point
            n = len(self._pareto_front) + 1
            idx = np.random.randint(n)
            weights = np.zeros(self.n_objectives)
            weights[idx % self.n_objectives] = 1.0
            # Smooth towards uniform
            weights = 0.3 * weights + 0.7 * np.ones(self.n_objectives) / self.n_objectives
            return weights
            
        # Gap-based: find largest gap in Pareto front
        front_weights = np.array([p.weights for p in self._pareto_front])
        
        # Find pair with largest distance
        max_dist = 0
        best_midpoint = np.ones(self.n_objectives) / self.n_objectives
        
        for i in range(len(front_weights)):
            for j in range(i + 1, len(front_weights)):
                dist = np.linalg.norm(front_weights[i] - front_weights[j])
                if dist > max_dist:
                    max_dist = dist
                    best_midpoint = (front_weights[i] + front_weights[j]) / 2
                    
        # Ensure valid simplex point
        best_midpoint = np.clip(best_midpoint, 0.01, None)
        return best_midpoint / best_midpoint.sum()
    
    def save(self, path: str) -> None:
        """Save Pareto front to file."""
        data = {
            "n_objectives": self.n_objectives,
            "objective_names": self.objective_names,
            "minimize": self.minimize,
            "front": [
                {
                    "objectives": p.objectives,
                    "weights": p.weights.tolist(),
                    "metrics": p.metrics,
                }
                for p in self._pareto_front
            ],
        }
        
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
    def load(self, path: str) -> None:
        """Load Pareto front from file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
            
        self.n_objectives = data["n_objectives"]
        self.objective_names = data["objective_names"]
        self.minimize = data["minimize"]
        
        self._pareto_front = [
            ParetoPoint(
                objectives=p["objectives"],
                weights=np.array(p["weights"]),
                metrics=p.get("metrics", {}),
            )
            for p in data["front"]
        ]
