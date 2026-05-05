#!/usr/bin/env python3
"""
Train CMO-SAC on ChipThermal-v0 (easiest environment).

This is a simpler problem than RackCooling:
- Single GPU chip thermal control
- Clear objective: keep temperature near target
- No complex multi-component dynamics

Run: python train_chip.py
Monitor: python train_chip.py --monitor
Evaluate: python train_chip.py --evaluate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

import compopt
from cmo_sac import CMOSACAgent, CMOSACTrainer


def main():
    parser = argparse.ArgumentParser(description="Train CMO-SAC on ChipThermal-v0")
    parser.add_argument("--monitor", action="store_true", help="Monitor training progress")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    results_dir = Path("./results/cmo_sac_chip/ChipThermal-v0_seed0")
    
    # ====================================================================
    # MONITOR MODE
    # ====================================================================
    if args.monitor:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print("=" * 80)
        print("ChipThermal-v0 Training Progress")
        print("=" * 80)
        print()
        
        # Check for training logs
        train_log = results_dir / "training_log.csv"
        eval_log = results_dir / "eval_log.csv"
        
        if not train_log.exists():
            print("❌ No training logs found. Training may not have started.")
            print(f"   Looking for: {train_log}")
            return
        
        # Load logs
        df_train = pd.read_csv(train_log)
        
        print(f"✅ Training progress: {len(df_train)} updates")
        print(f"   Latest timestep: {df_train['timestep'].max():,}")
        print()
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Reward
        ax = axes[0, 0]
        if 'reward' in df_train.columns:
            ax.plot(df_train['timestep'], df_train['reward'], alpha=0.7)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Episode Reward')
            ax.set_title('Training Reward', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Losses
        ax = axes[0, 1]
        if 'actor_loss' in df_train.columns:
            ax.plot(df_train['timestep'], df_train['actor_loss'], label='Actor', alpha=0.7)
            ax.plot(df_train['timestep'], df_train['critic_loss'], label='Critic', alpha=0.7)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Evaluation (if available)
        ax = axes[1, 0]
        if eval_log.exists():
            df_eval = pd.read_csv(eval_log)
            ax.errorbar(df_eval['timestep'], df_eval['mean_reward'], 
                       yerr=df_eval['std_reward'], capsize=5, marker='o')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Evaluation Reward')
            ax.set_title('Evaluation Performance', fontweight='bold')
            ax.grid(True, alpha=0.3)
            print(f"✅ Evaluations: {len(df_eval)} checkpoints")
            print(f"   Best reward: {df_eval['mean_reward'].max():.1f}")
        else:
            ax.text(0.5, 0.5, 'No evaluation data yet', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Evaluation Performance', fontweight='bold')
        
        # Plot 4: Lambda (constraint penalties)
        ax = axes[1, 1]
        lambda_cols = [c for c in df_train.columns if c.startswith('lambda_')]
        if lambda_cols:
            for col in lambda_cols[:4]:  # Plot first 4 lambdas
                ax.plot(df_train['timestep'], df_train[col], label=col, alpha=0.7)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Lambda (Constraint Penalty)')
            ax.set_title('Lagrangian Multipliers', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No lambda data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Lagrangian Multipliers', fontweight='bold')
        
        plt.tight_layout()
        output_file = "chip_training_progress.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print()
        print(f"✅ Plot saved: {output_file}")
        print()
        print("To run again: python train_chip.py --monitor")
        print("=" * 80)
        return
    
    # ====================================================================
    # EVALUATE MODE
    # ====================================================================
    if args.evaluate:
        print("=" * 80)
        print("Evaluating Best Model")
        print("=" * 80)
        print()
        
        best_model = results_dir / "checkpoints" / "best_model.pth"
        if not best_model.exists():
            print(f"❌ No trained model found: {best_model}")
            return
        
        # Create environment
        env = compopt.make("ChipThermal-v0", normalize_obs=True, normalize_reward=False)
        
        # Create agent and load weights
        agent = CMOSACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            n_objectives=3,
            n_constraints=2,
            hidden_dims=[256, 256],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        agent.load(str(best_model))
        print(f"✅ Loaded model from {best_model}")
        print()
        
        # Evaluate
        episode_rewards = []
        for ep in range(20):
            obs, _ = env.reset(seed=1000 + ep)
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (done or truncated):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            print(f"Episode {ep+1:2d}: {total_reward:10.1f} ({steps} steps)")
        
        print()
        print("=" * 80)
        print(f"Mean reward: {np.mean(episode_rewards):10.1f} ± {np.std(episode_rewards):6.1f}")
        print("=" * 80)
        return
    
    # ====================================================================
    # TRAINING MODE
    # ====================================================================
    print("=" * 80)
    print("Training CMO-SAC on ChipThermal-v0")
    print("=" * 80)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Seed: {args.seed}")
    print(f"Output: {results_dir}")
    print("=" * 80)
    print()
    
    # Create environment
    env = compopt.make("ChipThermal-v0", normalize_obs=True, normalize_reward=True)
    print(f"✓ Environment: {env}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create agent
    agent = CMOSACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        n_objectives=3,  # ChipThermal has multiple objectives
        n_constraints=2,  # Temperature constraints
        hidden_dims=[256, 256],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"✓ Agent created (device: {agent.device})")
    print()
    
    # Create trainer
    trainer = CMOSACTrainer(
        agent=agent,
        env=env,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        eval_freq=5_000,
        n_eval_episodes=10,
        checkpoint_dir=results_dir / "checkpoints",
        log_freq=1000,
    )
    print(f"✓ Trainer created")
    print()
    
    # Train
    print("Starting training...")
    print("-" * 80)
    print("To monitor progress in another terminal:")
    print(f"  python train_chip.py --monitor")
    print("-" * 80)
    print()
    
    trainer.train(total_timesteps=args.timesteps)
    
    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print()
    print("To evaluate:")
    print(f"  python train_chip.py --evaluate")
    print()
    print("To view progress:")
    print(f"  python train_chip.py --monitor")


if __name__ == "__main__":
    main()
