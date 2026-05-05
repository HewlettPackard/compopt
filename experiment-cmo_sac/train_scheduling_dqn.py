#!/usr/bin/env python3
"""
Train DQN on Scheduling-v0 (discrete action space).

Since CMO-SAC only supports continuous actions, we use Stable-Baselines3's DQN
for the discrete scheduling problem.

Usage:
    python train_scheduling_dqn.py --total-timesteps 400000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from datetime import datetime

import compopt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def make_env(normalize_obs=True, normalize_reward=True):
    """Create Scheduling-v0 environment."""
    return compopt.make(
        "Scheduling-v0",
        dt=60.0,
        episode_length_s=86400.0,  # 24 hours
        n_jobs=100,
        normalize_obs=normalize_obs,
        normalize_reward=normalize_reward
    )


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Scheduling-v0")
    parser.add_argument("--total-timesteps", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    args = parser.parse_args()

    print("=" * 80)
    print("Training DQN on Scheduling-v0")
    print("=" * 80)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"dqn_Scheduling-v0_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()

    # Create environments
    print("Creating environments...")
    env = Monitor(make_env(normalize_obs=True, normalize_reward=True))
    eval_env = Monitor(make_env(normalize_obs=True, normalize_reward=True))
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create DQN agent
    print("Creating DQN agent...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=10000,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=args.gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=args.target_update_interval,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=args.seed,
    )
    
    print(f"Network architecture: {model.policy}")
    print()

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoints_dir),
        log_path=str(checkpoints_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=str(checkpoints_dir),
        name_prefix="model",
    )

    # Train
    print("Starting training...")
    print("-" * 80)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        progress_bar=True,
    )

    # Save final model
    final_model_path = checkpoints_dir / "final_model"
    model.save(final_model_path)
    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best model saved to: {checkpoints_dir / 'best_model.zip'}")
    print(f"Final model saved to: {final_model_path}.zip")
    print()
    
    # Evaluate final model
    print("Evaluating final model...")
    obs, _ = eval_env.reset()
    episode_rewards = []
    
    for ep in range(10):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
    
    print(f"Final evaluation ({len(episode_rewards)} episodes):")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}, Max: {np.max(episode_rewards):.2f}")
    print()


if __name__ == "__main__":
    main()
