#!/usr/bin/env python3
"""
Visualize training progress from CMO-SAC logs.

This script can be run at any time during or after training to generate
progress plots showing:
- Reward curves
- Loss curves (actor, critic)
- Evaluation metrics (PUE, WUE, violations)
- Lagrangian multipliers

Usage:
    # For new training runs (results inside experiment-cmo_sac/)
    python visualize_training.py --log-dir experiment-cmo_sac/results/cmo_sac/single_training/RackCooling-v0_seed0/cmo_sac_RackCooling-v0_*/checkpoints
    
    # For old training runs (results in root results/)
    python visualize_training.py --log-dir results/cmo_sac/single_training/RackCooling-v0_seed0/cmo_sac_RackCooling-v0_*/checkpoints
    
    # With custom output
    python visualize_training.py --log-dir <path> --output my_progress.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from glob import glob

sns.set_style('whitegrid')


def load_logs(log_dir: Path):
    """Load all CSV logs from a training run (CMO-SAC or SB3 format)."""
    logs = {}
    log_type = "unknown"
    
    # Try CMO-SAC format first
    training_csv = log_dir / "training_log.csv"
    if training_csv.exists():
        logs['training'] = pd.read_csv(training_csv)
        log_type = "cmo_sac"
        print(f"✓ Loaded CMO-SAC training log: {len(logs['training'])} steps")
    
    eval_csv = log_dir / "eval_log.csv"
    if eval_csv.exists():
        logs['eval'] = pd.read_csv(eval_csv)
        print(f"✓ Loaded CMO-SAC eval log: {len(logs['eval'])} evaluations")
    
    loss_csv = log_dir / "losses.csv"
    if loss_csv.exists():
        logs['losses'] = pd.read_csv(loss_csv)
        print(f"✓ Loaded CMO-SAC loss log: {len(logs['losses'])} updates")
    
    # Try SB3 format (evaluations.npz and monitor CSVs)
    eval_npz = log_dir / "evaluations.npz"
    if eval_npz.exists():
        import numpy as np
        data = np.load(eval_npz)
        if 'results' in data and 'timesteps' in data:
            results = data['results']
            timesteps = data['timesteps']
            # Convert to DataFrame
            eval_data = []
            for t, r in zip(timesteps, results):
                eval_data.append({
                    'timestep': t,
                    'mean_reward': np.mean(r),
                    'std_reward': np.std(r),
                })
            logs['eval'] = pd.DataFrame(eval_data)
            log_type = "sb3"
            print(f"✓ Loaded SB3 evaluation log: {len(logs['eval'])} evaluations")
    
    # Look for monitor CSVs (SB3 training logs)
    monitor_files = list(log_dir.glob("*.monitor.csv")) + list((log_dir.parent).glob("*.monitor.csv"))
    if monitor_files and log_type == "sb3":
        # Combine all monitor files
        dfs = []
        for f in monitor_files[:5]:  # Limit to first 5 files
            try:
                df = pd.read_csv(f, skiprows=1)
                if 'r' in df.columns and 't' in df.columns:
                    dfs.append(df)
            except:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True).sort_values('t')
            # Estimate timesteps (cumulative episode lengths)
            combined['timestep'] = combined['l'].cumsum()
            combined['reward'] = combined['r']
            logs['training'] = combined[['timestep', 'reward']]
            print(f"✓ Loaded SB3 training log: {len(logs['training'])} episodes")
    
    logs['type'] = log_type
    return logs


def plot_training_progress(logs, output_path="training_progress.png"):
    """Create comprehensive training progress visualization."""
    
    log_type = logs.get('type', 'unknown')
    title = 'CMO-SAC Training Progress' if log_type == 'cmo_sac' else 'SAC Training Progress' if log_type == 'sb3' else 'Training Progress'
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    has_training = 'training' in logs and not logs['training'].empty
    has_eval = 'eval' in logs and not logs['eval'].empty
    has_losses = 'losses' in logs and not logs['losses'].empty
    
    # Plot 1: Training Rewards
    ax = axes[0, 0]
    if has_training:
        df = logs['training']
        ax.plot(df['timestep'] / 1000, df['reward'], alpha=0.3, color='blue')
        # Smoothed curve
        window = min(50, len(df) // 10)
        if window > 1:
            smoothed = df['reward'].rolling(window=window, center=True).mean()
            ax.plot(df['timestep'] / 1000, smoothed, linewidth=2, color='darkblue', label='Smoothed')
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title('Training Reward', fontsize=13, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No training data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation Rewards
    ax = axes[0, 1]
    if has_eval:
        df = logs['eval']
        ax.errorbar(df['timestep'] / 1000, df['mean_reward'], yerr=df['std_reward'],
                   fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('Mean Reward', fontsize=11)
        ax.set_title('Evaluation Reward', fontsize=13, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Actor & Critic Loss
    ax = axes[0, 2]
    if has_losses:
        df = logs['losses']
        # Subsample if too many points
        if len(df) > 1000:
            df = df.iloc[::len(df)//1000]
        ax.plot(df['timestep'] / 1000, df['actor_loss'], label='Actor Loss', alpha=0.7, linewidth=1.5)
        ax.plot(df['timestep'] / 1000, df['critic_loss'], label='Critic Loss', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Losses', fontsize=13, fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: PUE (Power Usage Effectiveness)
    ax = axes[1, 0]
    if has_training and 'pue' in logs['training'].columns:
        df = logs['training']
        ax.plot(df['timestep'] / 1000, df['pue'], alpha=0.3, color='orange')
        window = min(50, len(df) // 10)
        if window > 1:
            smoothed = df['pue'].rolling(window=window, center=True).mean()
            ax.plot(df['timestep'] / 1000, smoothed, linewidth=2, color='darkorange', label='Smoothed')
        # Auto-scale y-axis to show variation (add 5% padding)
        pue_min, pue_max = df['pue'].min(), df['pue'].max()
        pue_range = pue_max - pue_min
        if pue_range > 0:
            padding = max(0.001, pue_range * 0.1)  # 10% padding or minimum 0.001
            ax.set_ylim(pue_min - padding, pue_max + padding)
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('PUE', fontsize=11)
        ax.set_title('Power Usage Effectiveness', fontsize=13, fontweight='bold')
        if window > 1:
            ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No PUE data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: WUE (Water Usage Effectiveness)
    ax = axes[1, 1]
    if has_training and 'wue' in logs['training'].columns:
        df = logs['training']
        ax.plot(df['timestep'] / 1000, df['wue'], alpha=0.3, color='cyan')
        window = min(50, len(df) // 10)
        if window > 1:
            smoothed = df['wue'].rolling(window=window, center=True).mean()
            ax.plot(df['timestep'] / 1000, smoothed, linewidth=2, color='darkcyan', label='Smoothed')
        # Auto-scale y-axis to show variation (add 5% padding)
        wue_min, wue_max = df['wue'].min(), df['wue'].max()
        wue_range = wue_max - wue_min
        if wue_range > 0:
            padding = max(0.001, wue_range * 0.1)  # 10% padding or minimum 0.001
            ax.set_ylim(wue_min - padding, wue_max + padding)
        else:
            # If WUE is constant, show a narrow range around the value
            ax.set_ylim(wue_min - 0.005, wue_max + 0.005)
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('WUE (L/kWh)', fontsize=11)
        ax.set_title('Water Usage Effectiveness', fontsize=13, fontweight='bold')
        if window > 1:
            ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No WUE data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Constraint Violations
    ax = axes[1, 2]
    if has_training:
        df = logs['training']
        has_data = False
        for col, label in [('thermal_violations', 'Thermal'),
                          ('hbm_violations', 'HBM'),
                          ('sla_violations', 'SLA'),
                          ('coolant_violations', 'Coolant')]:
            if col in df.columns:
                window = min(50, len(df) // 10)
                if window > 1:
                    smoothed = df[col].rolling(window=window, center=True).mean() * 100
                    ax.plot(df['timestep'] / 1000, smoothed, label=label, linewidth=2, alpha=0.8)
                    has_data = True
        # Auto-scale y-axis to show variation
        if has_data:
            ax.set_ylim(bottom=0)  # Start from 0
            # Let matplotlib auto-scale the top
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('Violation Rate (%)', fontsize=11)
        ax.set_title('Constraint Violations', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No violation data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Alpha (Temperature Parameter)
    ax = axes[2, 0]
    if has_losses:
        df = logs['losses']
        if len(df) > 1000:
            df = df.iloc[::len(df)//1000]
        ax.plot(df['timestep'] / 1000, df['alpha'], linewidth=2, color='purple')
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('Alpha', fontsize=11)
        ax.set_title('Entropy Temperature (Alpha)', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No alpha data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Lagrangian Multipliers
    ax = axes[2, 1]
    if has_losses:
        df = logs['losses']
        if len(df) > 1000:
            df = df.iloc[::len(df)//1000]
        for col, label in [('lambda_thermal', 'Thermal'),
                          ('lambda_hbm', 'HBM'),
                          ('lambda_sla', 'SLA'),
                          ('lambda_coolant', 'Coolant')]:
            if col in df.columns:
                ax.plot(df['timestep'] / 1000, df[col], label=label, linewidth=2, alpha=0.8)
        ax.set_xlabel('Timesteps (×1000)', fontsize=11)
        ax.set_ylabel('Lambda Value', fontsize=11)
        ax.set_title('Lagrangian Multipliers', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No lambda data', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Training Statistics Summary
    ax = axes[2, 2]
    ax.axis('off')
    
    summary_text = "Training Statistics\n" + "="*35 + "\n\n"
    
    if has_training:
        df = logs['training']
        summary_text += f"Total Steps: {df['timestep'].max():,}\n"
        summary_text += f"Episodes: {df['episode'].max():,}\n"
        summary_text += f"Training Time: {df['elapsed_time_hrs'].max():.2f} hrs\n"
        summary_text += f"Final FPS: {df['fps'].iloc[-1]:.0f}\n\n"
        
        summary_text += f"Final Metrics:\n"
        summary_text += f"  Reward: {df['reward'].tail(100).mean():.4f}\n"
        if 'pue' in df.columns:
            summary_text += f"  PUE: {df['pue'].tail(100).mean():.3f}\n"
        if 'wue' in df.columns:
            summary_text += f"  WUE: {df['wue'].tail(100).mean():.3f}\n"
    
    if has_eval:
        df = logs['eval']
        if len(df) > 0:
            summary_text += f"\nEvaluation (latest):\n"
            last_eval = df.iloc[-1]
            summary_text += f"  Mean Reward: {last_eval['mean_reward']:.4f}\n"
            summary_text += f"  Std Reward: {last_eval['std_reward']:.4f}\n"
            if 'pue_mean' in last_eval:
                summary_text += f"  PUE: {last_eval['pue_mean']:.3f}\n"
            if 'wue_mean' in last_eval:
                summary_text += f"  WUE: {last_eval['wue_mean']:.3f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CMO-SAC training progress")
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Directory containing training logs (supports glob patterns)')
    parser.add_argument('--output', type=str, default='training_progress.png',
                       help='Output file path for visualization')
    args = parser.parse_args()
    
    # Handle glob patterns
    log_dirs = glob(args.log_dir, recursive=True)
    if not log_dirs:
        log_dirs = [args.log_dir]
    
    # Use first matching directory
    log_dir = Path(log_dirs[0])
    
    print(f"="*70)
    print(f"Visualizing Training Progress")
    print(f"="*70)
    print(f"Log directory: {log_dir}")
    print()
    
    if not log_dir.exists():
        print(f"❌ Error: Directory not found: {log_dir}")
        return
    
    # Load logs
    logs = load_logs(log_dir)
    
    if not logs:
        print("❌ Error: No log files found!")
        print(f"   Expected files: training_log.csv, eval_log.csv, losses.csv")
        return
    
    # Generate visualization
    print(f"\nGenerating visualization...")
    plot_training_progress(logs, output_path=args.output)
    
    print(f"\n{'='*70}")
    print(f"✓ Done! Open {args.output} to view training progress")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
