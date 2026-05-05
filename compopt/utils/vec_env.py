"""
compopt.utils.vec_env
=====================
Vectorised environment wrappers for parallel simulation.

``BatchSimulator`` runs N independent CompOpt environments in parallel
using NumPy vectorisation (no multiprocessing overhead), enabling
high-throughput RL data collection.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import copy


class BatchSimulator:
    """
    Run *N* copies of a CompOpt environment in lock-step.

    All environments share the same configuration but have independent
    state.  Environments that terminate/truncate are auto-reset.

    This is lighter than ``gymnasium.vector.AsyncVectorEnv`` because it
    avoids process-spawning overhead — suitable for fast pure-Python sims.

    Parameters
    ----------
    env_factory : callable() → gym.Env
    n_envs      : number of parallel environments
    """

    def __init__(self, env_factory, n_envs: int = 8):
        self.n_envs = n_envs
        self.envs   = [env_factory() for _ in range(n_envs)]
        # Pre-allocate obs buffer
        sample_obs, _ = self.envs[0].reset()
        if isinstance(sample_obs, np.ndarray):
            self.obs_shape = sample_obs.shape
            self.obs_buf   = np.zeros((n_envs,) + self.obs_shape,
                                       dtype=np.float32)
        else:
            self.obs_shape = None
            self.obs_buf   = None

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns stacked observations."""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            obs_list.append(obs)
        if self.obs_buf is not None:
            self.obs_buf[:] = np.array(obs_list, dtype=np.float32)
            return self.obs_buf.copy()
        return obs_list

    def step(self, actions: np.ndarray):
        """
        Step all environments with a batch of actions.

        Parameters
        ----------
        actions : (n_envs, action_dim) or (n_envs,) array

        Returns
        -------
        obs_batch    : (n_envs, obs_dim)
        rewards      : (n_envs,)
        terminateds  : (n_envs,)
        truncateds   : (n_envs,)
        infos        : list of dicts
        """
        rewards     = np.zeros(self.n_envs, dtype=np.float32)
        terminateds = np.zeros(self.n_envs, dtype=bool)
        truncateds  = np.zeros(self.n_envs, dtype=bool)
        infos       = []
        obs_list    = []

        for i, env in enumerate(self.envs):
            a = actions[i] if actions.ndim > 1 else actions[i:i+1]
            obs, rew, term, trunc, info = env.step(a)

            if term or trunc:
                obs, _ = env.reset()

            # Sanitize reward to prevent overflow when casting
            rewards[i]     = float(np.clip(np.nan_to_num(rew, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6))
            terminateds[i] = term
            truncateds[i]  = trunc
            infos.append(info)
            obs_list.append(obs)

        if self.obs_buf is not None:
            self.obs_buf[:] = np.array(obs_list, dtype=np.float32)
            return self.obs_buf.copy(), rewards, terminateds, truncateds, infos
        return obs_list, rewards, terminateds, truncateds, infos

    def close(self):
        for env in self.envs:
            env.close()


def benchmark_throughput(env_factory, n_envs: int = 8,
                         n_steps: int = 1000) -> Dict[str, float]:
    """
    Benchmark simulation throughput (steps/sec) for vectorised envs.

    Returns dict with 'total_steps', 'wall_time_s', 'steps_per_second'.
    """
    import time
    batch = BatchSimulator(env_factory, n_envs=n_envs)
    batch.reset()
    action_dim = batch.envs[0].action_space.shape
    if action_dim:
        actions = np.random.uniform(0, 1,
                                    size=(n_envs,) + action_dim).astype(np.float32)
    else:
        actions = np.zeros((n_envs, 1), dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        batch.step(actions)
    t1 = time.perf_counter()

    total = n_envs * n_steps
    wall  = t1 - t0
    batch.close()
    return {
        "n_envs":           n_envs,
        "n_steps":          n_steps,
        "total_steps":      total,
        "wall_time_s":      round(wall, 3),
        "steps_per_second": round(total / wall, 1),
    }
