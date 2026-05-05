#!/usr/bin/env python3
"""
Visualize DQN training progress for Scheduling-v0.

Shows scheduling-specific metrics instead of thermal metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_sb3_evaluations(log_dir):
    """Load SB3 evaluation results from evaluations.npz."""
    eval_file = Path(log_dir) / "evaluations.npz"
    if not eval_file.exists():
        return None
    
    data = np.load(eval_file)
    return {
        'timesteps': data['timesteps'],
        'results': data['results'],
        'ep_lengths': data['ep_lengths']
    }

def load_sb3_monitor(log_dir):
    """Load SB3 monitor logs (episode-level data)."""
    # SB3 creates monitor files in the log directory
    monitor_files = list(Path(log_dir).parent.glob("*.monitor.csv"))
    
    if not monitor_files:
        return None
    
    # Read all monitor files and concatenate
    dfs = []
    for f in monitor_files:
        try:
            df = pd.read_csv(f, skiprows=1)  # Skip metadata line
            dfs.append(df)
        except:
            continue
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def visualize_scheduling_progress(log_dir, output_file):
    """Create visualization for Scheduling-v0 training."""
    
    log_dir = Path(log_dir)
    print(f"Loading logs from: {log_dir}")
    
    # Load data
    eval_data = load_sb3_evaluations(log_dir)
    monitor_data = load_sb3_monitor(log_dir)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('DQN Training Progress - Scheduling-v0', fontsize=16, fontweight='bold')
    
    # Plot 1: Evaluation Reward (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if eval_data is not None:
        timesteps = eval_data['timesteps']
        mean_rewards = np.mean(eval_data['results'], axis=1)
        std_rewards = np.std(eval_data['results'], axis=1)
        
        ax1.plot(timesteps / 1000, mean_rewards, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(timesteps / 1000, 
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.3, color='blue', label='±1 std')
        ax1.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax1.set_ylabel('Evaluation Reward', fontsize=11)
        ax1.set_title('Evaluation Performance', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Add latest value
        latest_reward = mean_rewards[-1]
        latest_std = std_rewards[-1]
        ax1.text(0.02, 0.98, f'Latest: {latest_reward:.2f} ± {latest_std:.2f}',
                transform=ax1.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No evaluation data yet', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Evaluation Performance', fontsize=12, fontweight='bold')
    
    # Plot 2: Episode Length (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    if eval_data is not None:
        mean_lengths = np.mean(eval_data['ep_lengths'], axis=1)
        ax2.plot(timesteps / 1000, mean_lengths, 'g-', linewidth=2)
        ax2.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax2.set_ylabel('Episode Length', fontsize=11)
        ax2.set_title('Episode Duration', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('Episode Duration', fontsize=12, fontweight='bold')
    
    # Plot 3: Training Reward (rolling mean) (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if monitor_data is not None and 'r' in monitor_data.columns:
        # Rolling average of episode rewards
        window = min(100, len(monitor_data) // 10)
        if window > 0:
            rolling_reward = monitor_data['r'].rolling(window=window, min_periods=1).mean()
            ax3.plot(rolling_reward.index, rolling_reward.values, 'purple', linewidth=2)
            ax3.set_xlabel('Episode', fontsize=11)
            ax3.set_ylabel('Reward (rolling avg)', fontsize=11)
            ax3.set_title(f'Training Reward (window={window})', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Training Reward', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No training data', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Training Reward', fontsize=12, fontweight='bold')
    
    # Plot 4: DQN Loss (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    # Try to load tensorboard logs or progress.csv
    progress_file = log_dir.parent / "progress.csv"
    if progress_file.exists():
        try:
            progress = pd.read_csv(progress_file)
            if 'train/loss' in progress.columns:
                ax4.semilogy(progress['time/total_timesteps'] / 1000, 
                           progress['train/loss'], 'r-', linewidth=1.5)
                ax4.set_xlabel('Timesteps (×1000)', fontsize=11)
                ax4.set_ylabel('Q-Network Loss (log)', fontsize=11)
                ax4.set_title('Training Loss', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                        transform=ax4.transAxes)
                ax4.set_title('Training Loss', fontsize=12, fontweight='bold')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error loading\nloss data', ha='center', va='center',
                    transform=ax4.transAxes)
            ax4.set_title('Training Loss', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title('Training Loss', fontsize=12, fontweight='bold')
    
    # Plot 5: Exploration Rate (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    if progress_file.exists():
        try:
            progress = pd.read_csv(progress_file)
            if 'rollout/exploration_rate' in progress.columns:
                ax5.plot(progress['time/total_timesteps'] / 1000,
                        progress['rollout/exploration_rate'], 'orange', linewidth=2)
                ax5.set_xlabel('Timesteps (×1000)', fontsize=11)
                ax5.set_ylabel('Exploration Rate', fontsize=11)
                ax5.set_title('Epsilon (Exploration)', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
                ax5.set_ylim([0, 1.1])
            else:
                ax5.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax5.transAxes)
                ax5.set_title('Exploration Rate', fontsize=12, fontweight='bold')
        except:
            ax5.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax5.transAxes)
            ax5.set_title('Exploration Rate', fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax5.transAxes)
        ax5.set_title('Exploration Rate', fontsize=12, fontweight='bold')
    
    # Plot 6: Reward Distribution (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    if eval_data is not None and len(eval_data['results']) > 0:
        # Latest evaluation episode rewards
        latest_eval = eval_data['results'][-1]
        ax6.hist(latest_eval, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax6.axvline(np.mean(latest_eval), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(latest_eval):.2f}')
        ax6.set_xlabel('Reward', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Latest Eval Reward Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'No evaluation data', ha='center', va='center',
                transform=ax6.transAxes)
        ax6.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    
    # Plot 7: Training Statistics (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis('off')
    
    # Compile statistics
    stats_text = "Training Statistics\n" + "=" * 40 + "\n\n"
    
    if eval_data is not None:
        n_evals = len(eval_data['timesteps'])
        total_steps = eval_data['timesteps'][-1] if n_evals > 0 else 0
        latest_reward = np.mean(eval_data['results'][-1]) if n_evals > 0 else 0
        best_reward = np.max([np.mean(r) for r in eval_data['results']]) if n_evals > 0 else 0
        
        stats_text += f"Total Timesteps: {total_steps:,}\n"
        stats_text += f"Evaluations: {n_evals}\n\n"
        stats_text += f"Latest Reward: {latest_reward:.2f}\n"
        stats_text += f"Best Reward: {best_reward:.2f}\n\n"
    
    if monitor_data is not None and len(monitor_data) > 0:
        stats_text += f"Training Episodes: {len(monitor_data)}\n"
        if 'r' in monitor_data.columns:
            stats_text += f"Mean Episode Reward: {monitor_data['r'].mean():.2f}\n"
            stats_text += f"Std Episode Reward: {monitor_data['r'].std():.2f}\n"
    
    if progress_file.exists():
        try:
            progress = pd.read_csv(progress_file)
            if len(progress) > 0:
                latest = progress.iloc[-1]
                if 'rollout/exploration_rate' in progress.columns:
                    stats_text += f"\nExploration Rate: {latest['rollout/exploration_rate']:.4f}\n"
                if 'train/learning_rate' in progress.columns:
                    stats_text += f"Learning Rate: {latest['train/learning_rate']:.2e}\n"
                if 'train/n_updates' in progress.columns:
                    stats_text += f"Network Updates: {int(latest['train/n_updates']):,}\n"
        except:
            pass
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 8: Notes (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    notes = "Scheduling-v0 Notes\n" + "=" * 40 + "\n\n"
    notes += "• Discrete action space (DQN)\n"
    notes += "• 64 nodes × 4 GPUs cluster\n"
    notes += "• 100 jobs per episode\n"
    notes += "• 24-hour simulation\n\n"
    notes += "Metrics to track:\n"
    notes += "• Reward (higher = better)\n"
    notes += "• Episode length (varies)\n"
    notes += "• Exploration decay\n\n"
    notes += "Expected:\n"
    notes += "• FCFS baseline: ~-500\n"
    notes += "• Random: ~-2000\n"
    notes += "• DQN target: beat FCFS\n"
    
    ax8.text(0.05, 0.95, notes, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    # Save
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_scheduling.py <log_dir> <output_file>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    visualize_scheduling_progress(log_dir, output_file)
