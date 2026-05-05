"""
CMO-SAC Trainer: Training Pipeline for Constrained Multi-Objective SAC.

This module implements the complete training loop including:
- Environment interaction with multi-objective reward decomposition
- Constraint buffer management
- Pareto front exploration via weight sampling
- Dual variable updates for constraint satisfaction
- Comprehensive logging and checkpointing
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from collections import deque

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .cmo_sac_agent import CMOSACAgent
from .constraints import ConstraintBuffer, ConstraintManager
from .rewards import MultiObjectiveReward, RewardComponents
from .pareto import ParetoFrontGenerator
from .logger import TrainingLogger


class CMOSACTrainer:
    """
    Training pipeline for CMO-SAC agent.
    
    Handles environment interaction, buffer management, training updates,
    and Pareto front exploration.
    
    Args:
        agent: CMOSACAgent instance
        env: Gymnasium environment
        eval_env: Optional separate environment for evaluation
        buffer_size: Replay buffer capacity
        batch_size: Training batch size
        learning_starts: Timesteps before training begins
        train_freq: How often to update (in timesteps)
        gradient_steps: Number of gradient steps per update
        weight_sample_freq: How often to sample new weights (in episodes)
        dual_update_freq: How often to update Lagrange multipliers (in episodes)
        log_freq: Logging frequency (in timesteps)
        eval_freq: Evaluation frequency (in timesteps)
        n_eval_episodes: Number of episodes per evaluation
        checkpoint_freq: Checkpoint frequency (in timesteps)
        checkpoint_dir: Directory for checkpoints
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
    """
    
    def __init__(
        self,
        agent: CMOSACAgent,
        env: Any,
        eval_env: Optional[Any] = None,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 10_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        weight_sample_freq: int = 1,
        dual_update_freq: int = 5,
        log_freq: int = 1000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        checkpoint_freq: int = 50_000,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "cmo-sac",
        wandb_run_name: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CMOSACTrainer")
            
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or env
        
        # Training parameters
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.weight_sample_freq = weight_sample_freq
        self.dual_update_freq = dual_update_freq
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize replay buffer
        obs_dim = agent.obs_dim
        action_dim = agent.action_dim
        
        self.buffer = ConstraintBuffer(
            capacity=buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_objectives=agent.n_objectives,
            n_constraints=agent.n_constraints,
            device=str(agent.device),
        )
        
        # Multi-objective reward calculator
        self.mo_reward = MultiObjectiveReward()
        
        # Constraint manager
        self.constraint_manager = ConstraintManager()
        
        # Initialize comprehensive logger
        self.logger = TrainingLogger(
            log_dir=self.checkpoint_dir,
            log_freq=log_freq,
        )
        
        # Statistics
        self._episode_rewards = deque(maxlen=100)
        self._episode_lengths = deque(maxlen=100)
        self._episode_objectives: Dict[str, deque] = {
            "pue": deque(maxlen=100),
            "wue": deque(maxlen=100),
            "throughput": deque(maxlen=100),
        }
        self._episode_violations: Dict[str, deque] = {
            "thermal": deque(maxlen=100),
            "hbm": deque(maxlen=100),
            "sla": deque(maxlen=100),
            "coolant": deque(maxlen=100),
        }
        
        # Timing
        self._start_time = None
        self._n_episodes = 0
        self._total_timesteps = 0
        
        # WandB setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=agent.get_config(),
            )
            
    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the agent for specified timesteps.
        
        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback function called every step
            
        Returns:
            Dict of training statistics
        """
        self._start_time = time.time()
        
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_objectives = {"pue": [], "wue": [], "throughput": []}
        episode_violations = {"thermal": 0, "hbm": 0, "sla": 0, "coolant": 0}
        
        # Sample initial weights
        weights = self.agent.sample_weights()
        self.constraint_manager.reset_episode()
        
        for timestep in range(1, total_timesteps + 1):
            self._total_timesteps = timestep
            
            # Select action
            if timestep < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(obs, weights, deterministic=False)
                
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Compute multi-objective reward decomposition
            reward_components = self.mo_reward.compute_reward(
                obs={"obs": obs} if isinstance(obs, np.ndarray) else obs,
                action=action,
                next_obs={"obs": next_obs} if isinstance(next_obs, np.ndarray) else next_obs,
                info=info,
                weights={f"obj_{i}": weights[i] for i in range(len(weights))},
            )
            
            # Compute constraint costs
            constraint_costs = self.constraint_manager.compute_costs(
                {"obs": next_obs} if isinstance(next_obs, np.ndarray) else next_obs,
                info,
            )
            
            # Store transition
            # Only include objectives up to agent's n_objectives
            obj_rewards_full = np.array([
                reward_components.objectives.get("pue", 0.0),
                reward_components.objectives.get("wue", 0.0),
                reward_components.objectives.get("throughput", 0.0),
            ])
            obj_rewards = obj_rewards_full[:self.agent.n_objectives]
            
            # Clip objective rewards to prevent extreme values
            obj_rewards = np.clip(obj_rewards, -1000.0, 1000.0)
            
            # Clip scalarized reward
            scalarized_reward_clipped = np.clip(reward_components.scalarized, -1000.0, 1000.0)
            
            # Only include constraint costs if agent has constraints
            if self.agent.n_constraints > 0:
                con_costs = np.array([
                    constraint_costs.get("thermal", 0.0),
                    constraint_costs.get("hbm", 0.0),
                    constraint_costs.get("sla", 0.0),
                    constraint_costs.get("coolant", 0.0),
                ])[:self.agent.n_constraints]  # Slice to match expected size
                # Clip constraint costs
                con_costs = np.clip(con_costs, -100.0, 100.0)
            else:
                con_costs = np.array([])
            
            # Weights to match n_objectives
            weights_sliced = weights[:self.agent.n_objectives] if len(weights) > self.agent.n_objectives else weights
            
            self.buffer.add(
                obs=obs if isinstance(obs, np.ndarray) else np.array(obs),
                action=action,
                reward=obj_rewards,
                scalarized_reward=scalarized_reward_clipped,
                constraint_costs=con_costs,
                next_obs=next_obs if isinstance(next_obs, np.ndarray) else np.array(next_obs),
                done=done,
                weights=weights_sliced,
            )
            
            # Track episode statistics
            episode_reward += reward
            episode_length += 1
            
            for obj_name in ["pue", "wue", "throughput"]:
                if obj_name in reward_components.objectives:
                    episode_objectives[obj_name].append(
                        reward_components.objectives[obj_name]
                    )
                    
            for con_name, violated in reward_components.constraint_violations.items():
                if violated and con_name in episode_violations:
                    episode_violations[con_name] += 1
                    
            # Training update
            if timestep >= self.learning_starts and timestep % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    batch = self.buffer.sample(self.batch_size)
                    train_metrics = self.agent.update(batch)
                    
                    # Log training losses
                    if train_metrics and 'critic_loss' in train_metrics:
                        lambdas = self.agent.lagrangian.get_lambdas_dict()
                        self.logger.log_update(
                            timestep=timestep,
                            actor_loss=train_metrics.get('actor_loss', 0),
                            critic_loss=train_metrics.get('critic_loss', 0),
                            alpha=train_metrics.get('alpha', 0),
                            alpha_loss=train_metrics.get('alpha_loss', 0),
                            obj_critic_loss=train_metrics.get('obj_critic_loss', 0),
                            con_critic_loss=train_metrics.get('con_critic_loss', 0),
                            lambdas=lambdas,
                        )
                    
            # Handle episode end
            if done:
                self._n_episodes += 1
                self._episode_rewards.append(episode_reward)
                self._episode_lengths.append(episode_length)
                
                # Compute average objectives and violations for this episode
                ep_objectives = {}
                for obj_name, values in episode_objectives.items():
                    if values:
                        avg_val = np.mean(values)
                        self._episode_objectives[obj_name].append(avg_val)
                        ep_objectives[obj_name] = float(avg_val)
                        
                ep_violations = {}
                for con_name, count in episode_violations.items():
                    rate = count / max(episode_length, 1)
                    self._episode_violations[con_name].append(rate)
                    ep_violations[con_name] = float(rate)
                
                # Log this episode
                self.logger.log_training_step(
                    timestep=timestep,
                    episode=self._n_episodes,
                    reward=float(episode_reward),
                    length=episode_length,
                    objectives=ep_objectives,
                    violations=ep_violations,
                )
                
                # Print progress periodically
                self.logger.print_progress(timestep, self._n_episodes)
                        
                # Dual update (Lagrangian multipliers)
                if self._n_episodes % self.dual_update_freq == 0:
                    constraint_returns = self.constraint_manager.compute_episode_returns()
                    self.agent.dual_update(constraint_returns)
                    
                # Sample new weights
                if self._n_episodes % self.weight_sample_freq == 0:
                    weights = self.agent.sample_weights()
                    
                # Reset for next episode
                obs, info = self.env.reset()
                self.constraint_manager.reset_episode()
                episode_reward = 0.0
                episode_length = 0
                episode_objectives = {"pue": [], "wue": [], "throughput": []}
                episode_violations = {"thermal": 0, "hbm": 0, "sla": 0, "coolant": 0}
            else:
                obs = next_obs
                
            # Logging
            if timestep % self.log_freq == 0:
                self._log_metrics(timestep)
                
            # Evaluation
            if timestep % self.eval_freq == 0:
                eval_metrics = self._evaluate()
                self.logger.log_evaluation(timestep, eval_metrics)
                
            # Checkpointing
            if timestep % self.checkpoint_freq == 0:
                self._save_checkpoint(timestep)
                
            # Callback
            if callback is not None:
                callback(self, timestep)
                
        # Final evaluation and save
        final_metrics = self._evaluate()
        self._save_checkpoint(total_timesteps)
        
        # Save Pareto front
        pareto_path = self.checkpoint_dir / "pareto_front.json"
        self.agent.pareto_generator.save(str(pareto_path))
        
        # Save training summary
        self.logger.save_summary(final_metrics)
        
        return {
            "final_eval": final_metrics,
            "total_episodes": self._n_episodes,
            "training_time": time.time() - self._start_time,
        }
    
    def _evaluate(
        self,
        n_episodes: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate agent over multiple episodes.
        
        Args:
            n_episodes: Number of evaluation episodes
            weights: Optional fixed weights for evaluation
            
        Returns:
            Dict of evaluation metrics
        """
        n_episodes = n_episodes or self.n_eval_episodes
        
        episode_rewards = []
        episode_lengths = []
        objective_means = {"pue": [], "wue": [], "throughput": []}
        violation_rates = {"thermal": [], "hbm": [], "sla": [], "coolant": []}
        
        eval_weights = weights if weights is not None else \
            np.ones(self.agent.n_objectives) / self.agent.n_objectives
            
        for _ in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            ep_objectives = {"pue": [], "wue": [], "throughput": []}
            ep_violations = {"thermal": 0, "hbm": 0, "sla": 0, "coolant": 0}
            
            while not done:
                action = self.agent.select_action(
                    obs, eval_weights, deterministic=True
                )
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                ep_length += 1
                
                # Track objectives from info
                for obj in ["pue", "wue", "throughput"]:
                    if obj in info:
                        ep_objectives[obj].append(info[obj])
                        
                # Track violations from info
                for con in ["thermal", "hbm", "sla", "coolant"]:
                    if f"{con}_violation" in info and info[f"{con}_violation"]:
                        ep_violations[con] += 1
                        
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            
            for obj, values in ep_objectives.items():
                if values:
                    objective_means[obj].append(np.mean(values))
                    
            for con, count in ep_violations.items():
                violation_rates[con].append(count / max(ep_length, 1))
                
        # Add to Pareto front if evaluating with specific weights
        if weights is not None:
            objectives = {
                obj: np.mean(values) if values else 0.0
                for obj, values in objective_means.items()
            }
            self.agent.pareto_generator.add_point(
                objectives=objectives,
                weights=weights,
                metrics={
                    "mean_reward": np.mean(episode_rewards),
                    "mean_length": np.mean(episode_lengths),
                },
            )
            
        return {
            "eval/mean_reward": np.mean(episode_rewards),
            "eval/std_reward": np.std(episode_rewards),
            "eval/mean_length": np.mean(episode_lengths),
            **{f"eval/{obj}_mean": np.mean(values) if values else 0.0
               for obj, values in objective_means.items()},
            **{f"eval/{con}_violation_rate": np.mean(values) if values else 0.0
               for con, values in violation_rates.items()},
        }
    
    def explore_pareto_front(
        self,
        n_policies: int = 20,
        timesteps_per_policy: int = 50_000,
    ) -> ParetoFrontGenerator:
        """
        Systematically explore Pareto front by training with different weights.
        
        Args:
            n_policies: Number of policies to train
            timesteps_per_policy: Training timesteps for each policy
            
        Returns:
            ParetoFrontGenerator with discovered Pareto front
        """
        for i in range(n_policies):
            # Get suggested weights based on current front
            weights = self.agent.pareto_generator.suggest_weights(strategy="gap")
            
            print(f"\nTraining policy {i+1}/{n_policies}")
            print(f"Weights: PUE={weights[0]:.3f}, WUE={weights[1]:.3f}, "
                  f"Throughput={weights[2]:.3f}")
            
            # Set fixed weights
            self.agent.set_weights(weights)
            
            # Train with these weights
            obs, info = self.env.reset()
            self.constraint_manager.reset_episode()
            
            for _ in range(timesteps_per_policy):
                action = self.agent.select_action(obs, weights, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Simplified buffer addition for Pareto exploration
                reward_components = self.mo_reward.compute_reward(
                    {"obs": obs}, action, {"obs": next_obs}, info
                )
                constraint_costs = self.constraint_manager.compute_costs(
                    {"obs": next_obs}, info
                )
                
                # Store transition
                obj_rewards_full = np.array([
                    reward_components.objectives.get("pue", 0.0),
                    reward_components.objectives.get("wue", 0.0),
                    reward_components.objectives.get("throughput", 0.0),
                ])
                obj_rewards = obj_rewards_full[:self.agent.n_objectives]
                
                # Clip objective rewards to prevent extreme values
                obj_rewards = np.clip(obj_rewards, -1000.0, 1000.0)
                
                # Clip scalarized reward
                scalarized_reward_clipped = np.clip(reward_components.scalarized, -1000.0, 1000.0)
                
                # Only include constraint costs if agent has constraints
                if self.agent.n_constraints > 0:
                    con_costs = np.array([
                        constraint_costs.get("thermal", 0.0),
                        constraint_costs.get("hbm", 0.0),
                        constraint_costs.get("sla", 0.0),
                        constraint_costs.get("coolant", 0.0),
                    ])[:self.agent.n_constraints]  # Slice to match expected size
                    # Clip constraint costs
                    con_costs = np.clip(con_costs, -100.0, 100.0)
                else:
                    con_costs = np.array([])
                
                # Weights to match n_objectives
                weights_sliced = weights[:self.agent.n_objectives] if len(weights) > self.agent.n_objectives else weights
                
                self.buffer.add(
                    obs=obs, action=action, reward=obj_rewards,
                    scalarized_reward=scalarized_reward_clipped,
                    constraint_costs=con_costs, next_obs=next_obs,
                    done=done, weights=weights_sliced,
                )
                
                # Training update
                if len(self.buffer) >= self.batch_size:
                    batch = self.buffer.sample(self.batch_size)
                    self.agent.update(batch)
                    
                if done:
                    obs, info = self.env.reset()
                    self.constraint_manager.reset_episode()
                else:
                    obs = next_obs
                    
            # Evaluate and add to Pareto front
            eval_metrics = self._evaluate(n_episodes=5, weights=weights)
            print(f"Evaluation: reward={eval_metrics['eval/mean_reward']:.2f}, "
                  f"PUE={eval_metrics.get('eval/pue_mean', 0):.3f}")
                  
        return self.agent.pareto_generator
    
    def _log_metrics(self, timestep: int) -> None:
        """Log training metrics."""
        elapsed_time = time.time() - self._start_time
        fps = timestep / elapsed_time if elapsed_time > 0 else 0
        
        metrics = {
            "train/timestep": timestep,
            "train/episodes": self._n_episodes,
            "train/fps": fps,
            "train/mean_reward": np.mean(self._episode_rewards) if self._episode_rewards else 0,
            "train/mean_length": np.mean(self._episode_lengths) if self._episode_lengths else 0,
        }
        
        # Add objective metrics
        for obj_name, values in self._episode_objectives.items():
            if values:
                metrics[f"train/{obj_name}_mean"] = np.mean(values)
                
        # Add violation metrics
        for con_name, values in self._episode_violations.items():
            if values:
                metrics[f"train/{con_name}_violation_rate"] = np.mean(values)
                
        # Add Lagrangian metrics
        lambdas = self.agent.lagrangian.get_lambdas_dict()
        for name, value in lambdas.items():
            metrics[f"lagrangian/{name}"] = value
            
        # Add Pareto front metrics
        hypervolume = self.agent.pareto_generator.get_hypervolume()
        metrics["pareto/hypervolume"] = hypervolume
        metrics["pareto/front_size"] = len(self.agent.pareto_generator.get_pareto_front())
        
        if self.use_wandb:
            wandb.log(metrics, step=timestep)
        else:
            print(f"[{timestep}] reward={metrics['train/mean_reward']:.2f}, "
                  f"eps={self._n_episodes}, fps={fps:.0f}")
            
    def _log_eval_metrics(self, metrics: Dict[str, float], timestep: int) -> None:
        """Log evaluation metrics."""
        if self.use_wandb:
            wandb.log(metrics, step=timestep)
        else:
            print(f"[Eval {timestep}] reward={metrics['eval/mean_reward']:.2f}, "
                  f"length={metrics['eval/mean_length']:.0f}")
            
    def _save_checkpoint(self, timestep: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestep}.pt"
        self.agent.save(str(checkpoint_path))
        
        # Also save Pareto front
        pareto_path = self.checkpoint_dir / f"pareto_{timestep}.json"
        self.agent.pareto_generator.save(str(pareto_path))
        
        print(f"Saved checkpoint to {checkpoint_path}")


def create_trainer(
    env_id: str = "JointDCFlat-v0",
    seed: int = 0,
    **kwargs,
) -> CMOSACTrainer:
    """
    Convenience function to create trainer with default settings.
    
    Args:
        env_id: Gymnasium environment ID
        seed: Random seed
        **kwargs: Additional arguments for trainer
        
    Returns:
        Configured CMOSACTrainer
    """
    if not GYM_AVAILABLE:
        raise ImportError("Gymnasium is required")
        
    # Create environments
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    
    # Set seeds
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    agent = CMOSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **{k: v for k, v in kwargs.items() 
           if k in CMOSACAgent.__init__.__code__.co_varnames},
    )
    
    # Create trainer
    trainer = CMOSACTrainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        **{k: v for k, v in kwargs.items() 
           if k in CMOSACTrainer.__init__.__code__.co_varnames},
    )
    
    return trainer
