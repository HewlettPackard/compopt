"""
compopt.envs.wrappers
=====================
Environment wrappers for observation/reward normalization and stability.

These wrappers are automatically applied by compopt.make() to prevent
numerical instability issues with large observation values.
"""

import numpy as np
import gymnasium as gym
from typing import Optional


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations using running mean and std.
    
    This wrapper prevents numerical instability in neural networks by
    ensuring observations stay in a reasonable range. It's especially
    important for CompOpt environments where some observation dimensions
    (e.g., power, workload metrics) can have values in the thousands.
    
    The normalization uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        epsilon: float = 1e-8,
        clip_range: float = 10.0
    ):
        """
        Args:
            env: The environment to wrap
            epsilon: Small value to prevent division by zero
            clip_range: Maximum absolute value after normalization
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # Initialize running statistics
        obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(obs_shape, dtype=np.float64)
        self.running_var = np.ones(obs_shape, dtype=np.float64)
        self.count = epsilon
        
        # Track if we should update statistics (disable during evaluation)
        self.training = True
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        # Convert to float64 for numerical stability
        obs = np.asarray(obs, dtype=np.float64)
        
        # Update running statistics (Welford's online algorithm)
        if self.training:
            self.count += 1
            delta = obs - self.running_mean
            self.running_mean += delta / self.count
            delta2 = obs - self.running_mean
            self.running_var += (delta * delta2 - self.running_var) / self.count
        
        # Normalize: (obs - mean) / std
        std = np.sqrt(self.running_var + self.epsilon)
        normalized_obs = (obs - self.running_mean) / std
        
        # Clip to prevent extreme values
        normalized_obs = np.clip(normalized_obs, -self.clip_range, self.clip_range)
        
        return normalized_obs.astype(np.float32)
    
    def set_training(self, training: bool):
        """Set whether to update running statistics."""
        self.training = training
    
    def get_statistics(self):
        """Get current running statistics."""
        return {
            'mean': self.running_mean.copy(),
            'std': np.sqrt(self.running_var + self.epsilon),
            'count': self.count
        }


class NormalizeReward(gym.RewardWrapper):
    """
    Normalize rewards using running statistics.
    
    This can help with training stability, but is less critical than
    observation normalization. Use with caution as it changes the reward scale.
    """
    
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip_range: tuple = (-10.0, 10.0),
        gamma: float = 0.99
    ):
        """
        Args:
            env: The environment to wrap
            epsilon: Small value to prevent division by zero
            clip_range: (min, max) range for clipped rewards
            gamma: Discount factor for return calculation
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.clip_min, self.clip_max = clip_range
        self.gamma = gamma
        
        # Running statistics for returns
        self.running_return = 0.0
        self.return_var = 1.0
        self.count = epsilon
        
        self.training = True
    
    def reward(self, reward: float) -> float:
        """Normalize reward using running return statistics."""
        reward = float(reward)
        
        if self.training:
            # Update return statistics
            self.running_return = self.gamma * self.running_return + reward
            self.count += 1
            
            # Update variance estimate
            mean_return = self.running_return / (1 - self.gamma ** self.count)
            delta = reward - mean_return
            self.return_var = 0.99 * self.return_var + 0.01 * delta ** 2
        
        # Normalize
        std = np.sqrt(self.return_var + self.epsilon)
        normalized_reward = reward / (std + self.epsilon)
        
        # Clip
        normalized_reward = np.clip(normalized_reward, self.clip_min, self.clip_max)
        
        return float(normalized_reward)
    
    def set_training(self, training: bool):
        """Set whether to update running statistics."""
        self.training = training


def apply_wrappers(
    env: gym.Env,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    obs_clip: float = 10.0,
    reward_clip: tuple = (-10.0, 10.0)
) -> gym.Env:
    """
    Apply normalization wrappers to an environment.
    
    Args:
        env: Environment to wrap
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        obs_clip: Clipping range for normalized observations
        reward_clip: Clipping range for normalized rewards
    
    Returns:
        Wrapped environment
    """
    if normalize_obs:
        env = NormalizeObservation(env, epsilon=1e-8, clip_range=obs_clip)
    
    if normalize_reward:
        env = NormalizeReward(env, epsilon=1e-8, clip_range=reward_clip)
    
    return env
