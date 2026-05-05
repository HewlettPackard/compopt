"""
Enhanced logging system for CMO-SAC training.

Tracks and saves:
- Training losses (actor, critic, alpha)
- Evaluation metrics (rewards, PUE, WUE, violations)
- Pareto front progress
- System metrics (FPS, memory)
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
import time
import numpy as np


class TrainingLogger:
    """
    Comprehensive training logger that saves metrics to multiple formats.
    
    Creates:
    - training_log.csv: Timestep-by-timestep training metrics
    - eval_log.csv: Evaluation results
    - losses.csv: Actor/critic losses over time
    - summary.json: Final summary statistics
    """
    
    def __init__(
        self,
        log_dir: Path,
        log_freq: int = 1000,
        window_size: int = 100,
    ):
        """
        Args:
            log_dir: Directory to save logs
            log_freq: How often to write to disk (in timesteps)
            window_size: Window for moving averages
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_freq = log_freq
        self.window_size = window_size
        
        # Buffers for metrics
        self.training_buffer = []
        self.eval_buffer = []
        self.loss_buffer = []
        
        # Windowed metrics for smoothing
        self.reward_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)
        
        # Tracking
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Initialize CSV files
        self._init_csv_files()
        
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Training log
        self.training_csv = self.log_dir / "training_log.csv"
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'episode', 'reward', 'length', 'fps',
                'pue', 'wue', 'throughput',
                'thermal_violations', 'hbm_violations', 'sla_violations', 'coolant_violations',
                'elapsed_time_hrs'
            ])
        
        # Evaluation log
        self.eval_csv = self.log_dir / "eval_log.csv"
        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'mean_reward', 'std_reward', 'mean_length',
                'pue_mean', 'wue_mean', 'throughput_mean',
                'thermal_violation_rate', 'hbm_violation_rate',
                'sla_violation_rate', 'coolant_violation_rate'
            ])
        
        # Loss log
        self.loss_csv = self.log_dir / "losses.csv"
        with open(self.loss_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'actor_loss', 'critic_loss', 'alpha', 'alpha_loss',
                'obj_critic_loss', 'con_critic_loss', 'lambda_thermal',
                'lambda_hbm', 'lambda_sla', 'lambda_coolant'
            ])
    
    def log_training_step(
        self,
        timestep: int,
        episode: int,
        reward: float,
        length: int,
        objectives: Dict[str, float],
        violations: Dict[str, float],
    ):
        """Log a training step."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        fps = timestep / elapsed if elapsed > 0 else 0
        
        self.reward_window.append(reward)
        
        row = {
            'timestep': timestep,
            'episode': episode,
            'reward': reward,
            'length': length,
            'fps': fps,
            'pue': objectives.get('pue', 0),
            'wue': objectives.get('wue', 0),
            'throughput': objectives.get('throughput', 0),
            'thermal_violations': violations.get('thermal', 0),
            'hbm_violations': violations.get('hbm', 0),
            'sla_violations': violations.get('sla', 0),
            'coolant_violations': violations.get('coolant', 0),
            'elapsed_time_hrs': elapsed / 3600,
        }
        
        self.training_buffer.append(row)
        
        # Write to disk periodically
        if len(self.training_buffer) >= self.log_freq // 100:  # ~10 entries per log_freq
            self._flush_training_buffer()
    
    def log_update(
        self,
        timestep: int,
        actor_loss: float,
        critic_loss: float,
        alpha: float,
        alpha_loss: float = 0.0,
        obj_critic_loss: float = 0.0,
        con_critic_loss: float = 0.0,
        lambdas: Optional[Dict[str, float]] = None,
    ):
        """Log a training update (losses)."""
        self.loss_window.append(critic_loss)
        
        row = {
            'timestep': timestep,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha': alpha,
            'alpha_loss': alpha_loss,
            'obj_critic_loss': obj_critic_loss,
            'con_critic_loss': con_critic_loss,
            'lambda_thermal': lambdas.get('thermal', 0) if lambdas else 0,
            'lambda_hbm': lambdas.get('hbm', 0) if lambdas else 0,
            'lambda_sla': lambdas.get('sla', 0) if lambdas else 0,
            'lambda_coolant': lambdas.get('coolant', 0) if lambdas else 0,
        }
        
        self.loss_buffer.append(row)
        
        # Write periodically
        if len(self.loss_buffer) >= 10:
            self._flush_loss_buffer()
    
    def log_evaluation(
        self,
        timestep: int,
        metrics: Dict[str, float],
    ):
        """Log evaluation results."""
        row = {
            'timestep': timestep,
            'mean_reward': metrics.get('eval/mean_reward', 0),
            'std_reward': metrics.get('eval/std_reward', 0),
            'mean_length': metrics.get('eval/mean_length', 0),
            'pue_mean': metrics.get('eval/pue_mean', 0),
            'wue_mean': metrics.get('eval/wue_mean', 0),
            'throughput_mean': metrics.get('eval/throughput_mean', 0),
            'thermal_violation_rate': metrics.get('eval/thermal_violation_rate', 0),
            'hbm_violation_rate': metrics.get('eval/hbm_violation_rate', 0),
            'sla_violation_rate': metrics.get('eval/sla_violation_rate', 0),
            'coolant_violation_rate': metrics.get('eval/coolant_violation_rate', 0),
        }
        
        self.eval_buffer.append(row)
        self._flush_eval_buffer()
        
        # Also print to console
        print(f"\n{'='*70}")
        print(f"EVALUATION @ {timestep:,} steps")
        print(f"{'='*70}")
        print(f"  Mean Reward: {row['mean_reward']:.4f} ± {row['std_reward']:.4f}")
        print(f"  Mean Length: {row['mean_length']:.1f}")
        print(f"  PUE: {row['pue_mean']:.3f} | WUE: {row['wue_mean']:.3f} | Throughput: {row['throughput_mean']:.2f}")
        print(f"  Violations: Thermal={row['thermal_violation_rate']*100:.1f}% | HBM={row['hbm_violation_rate']*100:.1f}%")
        print(f"{'='*70}\n")
    
    def _flush_training_buffer(self):
        """Write training buffer to CSV."""
        if not self.training_buffer:
            return
        
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.training_buffer[0].keys())
            writer.writerows(self.training_buffer)
        
        self.training_buffer = []
    
    def _flush_eval_buffer(self):
        """Write eval buffer to CSV."""
        if not self.eval_buffer:
            return
        
        with open(self.eval_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.eval_buffer[0].keys())
            writer.writerows(self.eval_buffer)
        
        self.eval_buffer = []
    
    def _flush_loss_buffer(self):
        """Write loss buffer to CSV."""
        if not self.loss_buffer:
            return
        
        with open(self.loss_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.loss_buffer[0].keys())
            writer.writerows(self.loss_buffer)
        
        self.loss_buffer = []
    
    def save_summary(self, final_metrics: Dict[str, Any]):
        """Save final summary statistics."""
        # Flush all buffers
        self._flush_training_buffer()
        self._flush_eval_buffer()
        self._flush_loss_buffer()
        
        total_time = time.time() - self.start_time
        
        summary = {
            'final_metrics': final_metrics,
            'training_time_seconds': total_time,
            'training_time_hours': total_time / 3600,
            'average_reward': float(np.mean(list(self.reward_window))) if self.reward_window else 0,
            'average_loss': float(np.mean(list(self.loss_window))) if self.loss_window else 0,
            'log_dir': str(self.log_dir),
        }
        
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training summary saved to {summary_path}")
    
    def print_progress(self, timestep: int, episode: int):
        """Print progress update to console."""
        if len(self.reward_window) == 0:
            return
        
        elapsed = time.time() - self.start_time
        fps = timestep / elapsed if elapsed > 0 else 0
        
        avg_reward = np.mean(list(self.reward_window))
        
        # Only print every N seconds to avoid spam
        current_time = time.time()
        if current_time - self.last_log_time > 10:  # Every 10 seconds
            print(f"[{timestep:7d}] Episode {episode:4d} | "
                  f"Reward: {avg_reward:7.4f} | FPS: {fps:5.0f}")
            self.last_log_time = current_time
