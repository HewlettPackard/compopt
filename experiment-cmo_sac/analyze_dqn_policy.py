#!/usr/bin/env python3
"""
Analyze DQN policy behavior on Scheduling-v0.

Checks if the policy is:
- Always selecting the same action (degenerate)
- Exploring different actions
- Actually learning or stuck

Usage:
    python analyze_dqn_policy.py <checkpoint_path>
"""

import sys
from pathlib import Path
import numpy as np
import compopt
from stable_baselines3 import DQN


def analyze_policy(model_path, n_episodes=10):
    """Analyze DQN policy behavior."""
    print("=" * 80)
    print("DQN Policy Analysis - Scheduling-v0")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()
    
    # Load model
    model = DQN.load(model_path)
    print("✓ Model loaded")
    
    # Create environment
    env = compopt.make("Scheduling-v0", normalize_obs=True, normalize_reward=False)
    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print()
    
    # Collect data
    all_actions = []
    all_rewards = []
    all_metrics = {'util': [], 'jobs': [], 'queue': []}
    
    print(f"Running {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_actions = []
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert to int for hashability
            obs, reward, done, truncated, info = env.step(action)
            
            episode_actions.append(action)
            total_reward += reward
            steps += 1
            
            # Collect metrics
            all_metrics['util'].append(info.get('node_utilisation', 0))
            all_metrics['jobs'].append(info.get('jobs_completed_step', 0))
            all_metrics['queue'].append(info.get('queue_length', 0))
        
        all_actions.extend(episode_actions)
        all_rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.2f} ({steps} steps, {len(set(episode_actions))} unique actions)")
    
    print()
    print("=" * 80)
    print("Analysis Results")
    print("=" * 80)
    
    # Action distribution
    unique_actions = np.unique(all_actions)
    action_counts = {int(a): all_actions.count(a) for a in unique_actions}
    
    print(f"\nAction Distribution:")
    print(f"  Total actions taken: {len(all_actions)}")
    print(f"  Unique actions used: {len(unique_actions)} / {env.action_space.n}")
    print()
    
    # Sort by frequency
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    print("  Most frequent actions:")
    for action, count in sorted_actions[:5]:
        pct = 100 * count / len(all_actions)
        print(f"    Action {action}: {count:5d} times ({pct:5.1f}%)")
    
    # Check if degenerate
    print()
    if len(unique_actions) == 1:
        print("❌ DEGENERATE POLICY: Always selecting action", unique_actions[0])
        print("   This means the agent hasn't learned anything useful.")
    elif len(unique_actions) < 3:
        print("⚠️  WARNING: Very limited action diversity")
        print(f"   Only using {len(unique_actions)} out of {env.action_space.n} possible actions")
    elif sorted_actions[0][1] > 0.9 * len(all_actions):
        dominant_action, dominant_count = sorted_actions[0]
        print(f"⚠️  WARNING: One action dominates ({dominant_action}: {100*dominant_count/len(all_actions):.1f}%)")
    else:
        print("✓ Policy explores multiple actions")
    
    # Performance metrics
    print()
    print(f"Performance:")
    print(f"  Mean reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Min reward:  {np.min(all_rewards):.2f}")
    print(f"  Max reward:  {np.max(all_rewards):.2f}")
    
    # Scheduling metrics
    print()
    print(f"Scheduling Metrics:")
    print(f"  Avg utilization:  {np.mean(all_metrics['util']):.3f}")
    print(f"  Total jobs done:  {sum(all_metrics['jobs']):.0f}")
    print(f"  Avg queue length: {np.mean(all_metrics['queue']):.1f}")
    
    # Baseline comparison
    print()
    print("Comparison to baselines:")
    # Note: These baselines are for the FIXED Scheduling-v0 (with RL_AGENT policy)
    fcfs_baseline = -13447.0  # FCFS with fixed environment
    random_baseline = -12751.0  # Random policy with fixed environment
    mean_reward = np.mean(all_rewards)
    
    if mean_reward > fcfs_baseline:
        improvement = ((fcfs_baseline - mean_reward) / abs(fcfs_baseline)) * 100
        print(f"  ✅ BEATS FCFS baseline ({fcfs_baseline:.0f})")
        print(f"     Improvement: {-improvement:.1f}%")
    else:
        gap = mean_reward - fcfs_baseline
        pct = (gap / abs(fcfs_baseline)) * 100
        print(f"  ❌ Worse than FCFS baseline ({fcfs_baseline:.0f})")
        print(f"     Gap: {gap:.0f} ({pct:+.1f}%)")
    
    if mean_reward > random_baseline:
        improvement = ((random_baseline - mean_reward) / abs(random_baseline)) * 100
        print(f"  ✅ Beats random baseline ({random_baseline:.0f})")
        print(f"     Improvement: {-improvement:.1f}%")
    else:
        print(f"  ⚠️  Worse than random ({random_baseline:.0f})")
    
    print()
    print("=" * 80)
    
    env.close()
    return {
        'unique_actions': len(unique_actions),
        'action_distribution': action_counts,
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_utilization': float(np.mean(all_metrics['util'])),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dqn_policy.py <checkpoint_path>")
        print("\nExample:")
        print("  python analyze_dqn_policy.py results/dqn_Scheduling-v0_*/checkpoints/best_model.zip")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    analyze_policy(model_path)
