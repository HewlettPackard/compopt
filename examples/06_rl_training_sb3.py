#!/usr/bin/env python3
"""
CompOpt Example 06 — RL Training with Stable-Baselines3
========================================================
Trains a PPO agent on the ChipThermal-v0 and RackCooling-v0 environments
using Stable-Baselines3.

Requires: pip install stable-baselines3

Run:
    python examples/06_rl_training_sb3.py
"""

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("stable-baselines3 not installed. Install with:")
    print("  pip install 'compopt[rl]'")
    print()

import compopt
from compopt.agents.baselines import PIDCoolingAgent
from compopt.utils.metrics import evaluate_agent, print_metrics

if not HAS_SB3:
    print("Falling back to baseline evaluation only.\n")

    env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=600.0)
    agent = PIDCoolingAgent(target_C=80.0, Kp=0.05, Ki=0.001, Kd=0.01, dt=1.0)
    result = evaluate_agent(env, agent, n_episodes=3, max_steps=600)
    print_metrics(result, title="PID Baseline (no SB3)")
    env.close()
    exit(0)


def make_env(env_id, **kwargs):
    """Factory for SB3 vectorized envs."""
    def _init():
        return compopt.make(env_id, **kwargs)
    return _init


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Part 1: Train PPO on ChipThermal-v0                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("=" * 60)
print("Training PPO on ChipThermal-v0")
print("=" * 60)

vec_env = DummyVecEnv([make_env("ChipThermal-v0", dt=1.0,
                                 episode_length_s=600.0)
                       for _ in range(4)])

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1,
)

model.learn(total_timesteps=20_000)

# Evaluate trained agent
eval_env = compopt.make("ChipThermal-v0", dt=1.0, episode_length_s=600.0)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"\nPPO on ChipThermal-v0: reward = {mean_reward:.1f} ± {std_reward:.1f}")

# Compare with PID baseline
pid_agent = PIDCoolingAgent(target_C=80.0, Kp=0.05, Ki=0.001, Kd=0.01, dt=1.0)
pid_result = evaluate_agent(eval_env, pid_agent, n_episodes=5, max_steps=600)
print(f"PID Baseline:          reward = {pid_result['reward_total']:.1f} "
      f"± {pid_result['reward_total_std']:.1f}")

eval_env.close()
vec_env.close()

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ Part 2: Train PPO on RackCooling-v0                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("Training PPO on RackCooling-v0")
print("=" * 60)

vec_env = DummyVecEnv([make_env("RackCooling-v0", dt=1.0,
                                 episode_length_s=1800.0,
                                 n_servers=4, gpus_per_server=1)
                       for _ in range(4)])

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
)

model.learn(total_timesteps=50_000)

eval_env = compopt.make("RackCooling-v0", dt=1.0, episode_length_s=1800.0)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"\nPPO on RackCooling-v0: reward = {mean_reward:.1f} ± {std_reward:.1f}")

eval_env.close()
vec_env.close()

print("\nDone! Models can be saved with model.save('compopt_ppo_rack')")
