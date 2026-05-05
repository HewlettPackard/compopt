"""
Observation normalization wrapper for CompOpt environments
"""

import numpy as np
import gymnasium as gym


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations using running mean and std.
    This helps prevent numerical instability when observations have large values.
    """
    
    def __init__(self, env, epsilon=1e-8, clip=10.0):
        """
        Args:
            env: The environment to wrap
            epsilon: Small value to prevent division by zero
            clip: Maximum absolute value after normalization
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        
        # Initialize running statistics
        obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(obs_shape, dtype=np.float32)
        self.running_var = np.ones(obs_shape, dtype=np.float32)
        self.count = epsilon
    
    def observation(self, obs):
        """Normalize observation"""
        # Update running statistics
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
        
        # Normalize
        std = np.sqrt(self.running_var + self.epsilon)
        normalized_obs = (obs - self.running_mean) / std
        
        # Clip to prevent extreme values
        normalized_obs = np.clip(normalized_obs, -self.clip, self.clip)
        
        return normalized_obs.astype(np.float32)


class NormalizeReward(gym.RewardWrapper):
    """
    Normalize rewards using running statistics.
    Less aggressive than observation normalization.
    """
    
    def __init__(self, env, epsilon=1e-8, clip_range=(-10.0, 10.0), gamma=0.99):
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
        
        # Running statistics
        self.running_return = 0.0
        self.return_var = 1.0
        self.count = epsilon
    
    def reward(self, reward):
        """Normalize reward"""
        # Update return statistics
        self.running_return = self.gamma * self.running_return + reward
        self.count += 1
        
        delta = reward - self.running_return / (1 - self.gamma ** self.count)
        self.return_var = 0.99 * self.return_var + 0.01 * delta ** 2
        
        # Normalize
        std = np.sqrt(self.return_var + self.epsilon)
        normalized_reward = reward / (std + self.epsilon)
        
        # Clip
        normalized_reward = np.clip(normalized_reward, self.clip_min, self.clip_max)
        
        return float(normalized_reward)


def make_normalized_env(env_id, normalize_obs=True, normalize_reward=False, **kwargs):
    """
    Create an environment with normalization wrappers.
    
    Args:
        env_id: Environment ID
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        **kwargs: Additional arguments passed to env creation
    
    Returns:
        Wrapped environment
    """
    import compopt
    
    env = compopt.make(env_id, **kwargs)
    
    if normalize_obs:
        env = NormalizeObservation(env, epsilon=1e-8, clip=10.0)
    
    if normalize_reward:
        env = NormalizeReward(env, epsilon=1e-8, clip_range=(-10.0, 10.0))
    
    return env
