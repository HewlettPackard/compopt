#!/usr/bin/env python3
"""
Train DQN on Scheduling-v0 with improved hyperparameters.

Based on analysis showing the original training got stuck, this version uses:
- Longer exploration (30% of training instead of 10%)
- Higher learning rate during exploration
- Larger network
- More training steps

Usage:
    python train_scheduling_dqn_improved.py --total-timesteps 1000000
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
    parser = argparse.ArgumentParser(description="Train DQN on Scheduling-v0 (Improved)")
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    print("=" * 80)
    print("Training DQN on Scheduling-v0 (IMPROVED HYPERPARAMETERS)")
    print("=" * 80)
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Seed: {args.seed}")
    print()
    
    print("Improvements over default:")
    print("  ✓ Longer exploration (30% vs 10%)")
    print("  ✓ Higher final epsilon (0.1 vs 0.05)")
    print("  ✓ Larger network (128-128 vs 64-64)")
    print("  ✓ Higher learning rate (3e-4 vs 1e-4)")
    print("  ✓ Larger replay buffer (500k vs 100k)")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"dqn_improved_Scheduling-v0_{timestamp}"
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

    # Create DQN agent with IMPROVED hyperparameters
    print("Creating DQN agent with improved hyperparameters...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  # HIGHER than default (was 1e-4)
        buffer_size=500_000,  # LARGER than default (was 100k)
        learning_starts=10000,  
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,  # LONGER exploration (was 0.1)
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,  # HIGHER final eps (was 0.05)
        policy_kwargs=dict(net_arch=[128, 128]),  # LARGER network (was [64, 64])
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
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save less frequently
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
    
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Final evaluation ({len(episode_rewards)} episodes):")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}, Max: {np.max(episode_rewards):.2f}")
    
    # Compare to baseline
    if mean_reward > -500:
        print(f"\n✅ SUCCESS: Beats FCFS baseline (-500)!")
        print(f"   Improvement: {mean_reward - (-500):+.1f}")
    else:
        print(f"\n⚠️  Still below FCFS baseline (-500)")
        print(f"   Gap: {mean_reward - (-500):.1f}")
    
    print()


if __name__ == "__main__":
    main()
