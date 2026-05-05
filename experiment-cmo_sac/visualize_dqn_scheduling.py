#!/usr/bin/env python3
"""
Visualize DQN training progress for Scheduling-v0 (simplified, 2x3 layout).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(log_dir):
    """Load data from tensorboard event files."""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    data = {}
    available_scalars = ea.Tags()['scalars']
    
    for tag in available_scalars:
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
        except Exception as e:
            print(f"Warning: Could not load {tag}: {e}")
    
    return data


def load_evaluation_data(checkpoints_dir):
    """Load evaluation data from evaluations.npz."""
    eval_file = Path(checkpoints_dir) / "evaluations.npz"
    
    if not eval_file.exists():
        return None
    
    try:
        data = np.load(eval_file)
        return {
            'timesteps': data['timesteps'],
            'results': data['results'],
            'ep_lengths': data['ep_lengths'],
        }
    except Exception as e:
        print(f"Warning: Could not load evaluations: {e}")
        return None


def plot_dqn_training(checkpoint_dir, output_file):
    """Create comprehensive DQN training visualization."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find tensorboard log directory
    tb_dir = checkpoint_dir.parent / "tensorboard"
    if not tb_dir.exists():
        print(f"Error: Tensorboard directory not found: {tb_dir}")
        return
    
    dqn_dirs = list(tb_dir.glob("DQN_*"))
    if not dqn_dirs:
        print(f"Error: No DQN tensorboard logs found in {tb_dir}")
        return
    
    tb_log_dir = dqn_dirs[0]
    print(f"Loading tensorboard data from: {tb_log_dir}")
    
    # Load data
    tb_data = load_tensorboard_data(tb_log_dir)
    eval_data = load_evaluation_data(checkpoint_dir)
    
    # Create figure - 2x3 layout
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("DQN Training Progress - Scheduling-v0", fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Evaluation Reward with baselines
    ax1 = fig.add_subplot(gs[0, 0])
    if eval_data is not None:
        timesteps = eval_data['timesteps']
        rewards = eval_data['results']
        
        # If rewards are multidimensional (multiple episodes per eval), take mean
        if len(rewards.shape) > 1:
            rewards_mean = rewards.mean(axis=1)
        else:
            rewards_mean = rewards
        
        ax1.plot(timesteps / 1000, rewards_mean, 'b-', linewidth=2, label='DQN')
        ax1.set_xlabel('Timesteps (×1000)')
        ax1.set_ylabel('Evaluation Reward (normalized)')
        ax1.set_title('Evaluation Performance', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Add note about normalization
        ax1.text(0.02, 0.02, 'Note: Rewards are normalized during training\nSee stats box for baseline comparison',
                transform=ax1.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    else:
        ax1.text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
        ax1.set_title('Evaluation Performance', fontweight='bold')
    
    # Plot 2: Training Loss
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train/loss' in tb_data:
        steps = tb_data['train/loss']['steps'] / 1000
        loss = tb_data['train/loss']['values']
        ax2.plot(steps, loss, 'g-', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Timesteps (×1000)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss (TD Error)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No loss data', ha='center', va='center')
        ax2.set_title('Training Loss', fontweight='bold')
    
    # Plot 3: Exploration Rate
    ax3 = fig.add_subplot(gs[0, 2])
    if 'rollout/exploration_rate' in tb_data:
        steps = tb_data['rollout/exploration_rate']['steps'] / 1000
        eps = tb_data['rollout/exploration_rate']['values']
        ax3.plot(steps, eps, 'orange', linewidth=2)
        ax3.set_xlabel('Timesteps (×1000)')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate (ε-greedy)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
    else:
        ax3.text(0.5, 0.5, 'No exploration data', ha='center', va='center')
        ax3.set_title('Exploration Rate', fontweight='bold')
    
    # Plot 4: Reward Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if eval_data is not None and len(eval_data['results']) > 0:
        rewards = eval_data['results']
        if len(rewards.shape) > 1:
            rewards_flat = rewards.mean(axis=1)
        else:
            rewards_flat = rewards
        
        ax4.hist(rewards_flat, bins=min(20, len(rewards_flat)), color='skyblue', edgecolor='black', alpha=0.7)
        mean_val = np.mean(rewards_flat)
        ax4.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Eval Reward Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No reward data', ha='center', va='center')
        ax4.set_title('Reward Distribution', fontweight='bold')
    
    # Plot 5: Training Statistics (text box)
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    stats_text = "Training Statistics\n" + "="*50 + "\n"
    
    if eval_data is not None:
        total_steps = int(eval_data['timesteps'][-1]) if len(eval_data['timesteps']) > 0 else 0
        n_evals = len(eval_data['timesteps'])
        
        def to_scalar(val):
            if isinstance(val, np.ndarray):
                return val.item() if val.size == 1 else val.mean()
            return float(val)
        
        latest_reward = to_scalar(eval_data['results'][-1]) if len(eval_data['results']) > 0 else 0
        best_reward = to_scalar(np.max(eval_data['results'])) if len(eval_data['results']) > 0 else 0
        mean_reward = to_scalar(np.mean(eval_data['results']))
        std_reward = to_scalar(np.std(eval_data['results']))
        
        stats_text += f"Total Timesteps: {total_steps:,}\n"
        stats_text += f"Evaluations: {n_evals}\n\n"
        stats_text += f"Latest Reward: {latest_reward:.2f}\n"
        stats_text += f"Best Reward: {best_reward:.2f}\n"
        stats_text += f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n"
        stats_text += f"\nBaselines (FIXED environment):\n"
        stats_text += f"  FCFS: -13,447\n"
        stats_text += f"  Random: -12,751 (5.2% better)\n"
    
    if 'rollout/exploration_rate' in tb_data:
        final_eps = tb_data['rollout/exploration_rate']['values'][-1]
        stats_text += f"\nFinal Exploration: {final_eps:.4f}\n"
    
    if 'train/loss' in tb_data:
        final_loss = tb_data['train/loss']['values'][-1]
        stats_text += f"Final Loss: {final_loss:.2e}\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_dqn_scheduling.py <checkpoint_dir> <output_image>")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    plot_dqn_training(checkpoint_dir, output_file)
