#!/usr/bin/env python3
"""
CMO-SAC Experiment Runner.

Run experiments for Constrained Multi-Objective Soft Actor-Critic
on CompOpt data center cooling environments.

Usage:
    # Single training run
    python run_cmo_sac.py --mode train --env JointDCFlat-v0 --total-timesteps 500000
    
    # Pareto front exploration
    python run_cmo_sac.py --mode pareto --env JointDCFlat-v0 --n-policies 20
    
    # Ablation study
    python run_cmo_sac.py --mode ablation
    
    # Multi-seed experiment
    python run_cmo_sac.py --mode multi-seed --seeds 0 1 2 3 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: Gymnasium not available")

# Add parent directory to path for compopt
sys.path.insert(0, str(Path(__file__).parent.parent))

import compopt

from cmo_sac import (
    CMOSACAgent,
    CMOSACTrainer,
    ParetoFrontGenerator,
    MultiObjectiveReward,
)


def make_env(env_id, normalize_obs=True, normalize_reward=True):
    """
    Create environment with observation and reward normalization.
    
    Both are important for stable training:
    - normalize_obs: Scales observations to reasonable range
    - normalize_reward: Normalizes returns for stable value learning
    """
    return compopt.make(env_id, normalize_obs=normalize_obs, normalize_reward=normalize_reward)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CMO-SAC experiments for CompOpt benchmark"
    )
    
    # Mode
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "pareto", "ablation", "multi-seed", "evaluate", "sanity"],
        help="Experiment mode"
    )
    
    # Environment
    parser.add_argument(
        "--env", type=str, default="JointDCFlat-v0",
        choices=[
            "ChipThermal-v0", "RackCooling-v0", "DataCenter-v0",
            "Scheduling-v0", "JointDC-v0", "JointDCFlat-v0"
        ],
        help="CompOpt environment"
    )
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (1=no parallelization)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=1e-5, 
                        help="Learning rate for entropy temperature (alpha)")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    
    # CMO-SAC specific
    parser.add_argument("--n-objectives", type=int, default=3)
    parser.add_argument("--n-constraints", type=int, default=4)
    parser.add_argument("--weight-strategy", type=str, default="dirichlet",
                        choices=["dirichlet", "uniform", "grid", "adaptive"])
    parser.add_argument("--dual-update-freq", type=int, default=5)
    
    # Pareto exploration
    parser.add_argument("--n-policies", type=int, default=20)
    parser.add_argument("--timesteps-per-policy", type=int, default=50_000)
    
    # Multi-seed
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    
    # Logging
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cmo-sac-compopt")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()


def run_sanity_check(args):
    """Quick sanity check to verify everything works."""
    print("=" * 60)
    print("CMO-SAC Sanity Check")
    print("=" * 60)
    
    # Check imports
    print("\n1. Checking imports...")
    try:
        from cmo_sac.rewards import MultiObjectiveReward, WeightSampler
        from cmo_sac.constraints import ConstraintManager, ConstraintBuffer
        from cmo_sac.networks import CMOCritic, CMOActor
        from cmo_sac.pareto import ParetoFrontGenerator
        from cmo_sac.safety import SafetyFilter
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
        
    # Check PyTorch
    print("\n2. Checking PyTorch...")
    if TORCH_AVAILABLE:
        print(f"   ✓ PyTorch {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    else:
        print("   ✗ PyTorch not available")
        return False
        
    # Check Gymnasium
    print("\n3. Checking Gymnasium...")
    if GYM_AVAILABLE:
        print(f"   ✓ Gymnasium available")
    else:
        print("   ✗ Gymnasium not available")
        return False
        
    # Check CompOpt
    print("\n4. Checking CompOpt...")
    try:
        import compopt
        env = compopt.make("ChipThermal-v0")
        obs, info = env.reset()
        print(f"   ✓ CompOpt loaded")
        print(f"   ✓ Observation shape: {obs.shape}")
        print(f"   ✓ Action space: {env.action_space}")
    except Exception as e:
        print(f"   ✗ CompOpt error: {e}")
        return False
        
    # Test agent creation
    print("\n5. Testing agent creation...")
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = CMOSACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=[64, 64],  # Small for testing
        )
        print(f"   ✓ Agent created on {agent.device}")
        
        # Test action selection
        action = agent.select_action(obs)
        print(f"   ✓ Action selected: {action.shape}")
        
    except Exception as e:
        print(f"   ✗ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Test short training
    print("\n6. Testing short training loop...")
    try:
        from cmo_sac.constraints import ConstraintBuffer
        
        buffer = ConstraintBuffer(
            capacity=1000,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(agent.device),
        )
        
        # Collect some transitions
        for _ in range(100):
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            
            buffer.add(
                obs=obs,
                action=action,
                reward=np.array([1.5, 2.0, 0.8]),  # Mock objectives
                scalarized_reward=reward,
                constraint_costs=np.array([0.0, 0.0, 0.0, 0.0]),
                next_obs=next_obs,
                done=term or trunc,
                weights=np.array([0.33, 0.33, 0.34]),
            )
            
            if term or trunc:
                obs, info = env.reset()
            else:
                obs = next_obs
                
        print(f"   ✓ Buffer filled: {len(buffer)} transitions")
        
        # Test training update
        batch = buffer.sample(32)
        metrics = agent.update(batch)
        print(f"   ✓ Training update: critic_loss={metrics['critic_loss']:.4f}")
        
    except Exception as e:
        print(f"   ✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n" + "=" * 60)
    print("Sanity check PASSED ✓")
    print("=" * 60)
    return True


def run_single_training(args):
    """Run single training experiment."""
    print(f"\nTraining CMO-SAC on {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Device: {args.device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"cmo_sac_{args.env}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    # Create environment
    if args.n_envs > 1:
        # Use vectorized environments for parallel rollouts
        from compopt.utils.vec_env import BatchSimulator
        print(f"Using {args.n_envs} parallel environments")
        env_factory = lambda: make_env(args.env, normalize_obs=True)
        env = BatchSimulator(env_factory, n_envs=args.n_envs)
        eval_env = make_env(args.env, normalize_obs=True)  # Single env for eval
        
        # Get dimensions from first env
        temp_env = env_factory()
        obs_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        temp_env.close()
    else:
        # Single environment (standard mode)
        env = make_env(args.env, normalize_obs=True)
        eval_env = make_env(args.env, normalize_obs=True)
        obs_dim = env.observation_space.shape[0]
        
        # Handle both continuous (Box) and discrete (Discrete) action spaces
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
            action_dim = env.action_space.shape[0]  # Continuous (Box)
        elif hasattr(env.action_space, 'n'):
            action_dim = env.action_space.n  # Discrete
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action space type: {type(env.action_space).__name__}")
    
    # Create agent
    agent = CMOSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=args.n_objectives,
        n_constraints=args.n_constraints,
        hidden_dims=args.hidden_dims,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        initial_alpha=0.01,  # Fixed low alpha
        auto_alpha=False,     # Disable auto-tuning
        discount=args.discount,
        tau=args.tau,
        device=args.device,
        weight_strategy=args.weight_strategy,
    )
    
    # Create trainer
    trainer = CMOSACTrainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        dual_update_freq=args.dual_update_freq,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=str(output_dir / "checkpoints"),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"cmo_sac_{args.env}",
    )
    
    # Train
    results = trainer.train(total_timesteps=args.total_timesteps)
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        # Convert numpy types to Python types
        results_serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_serializable[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                results_serializable[k] = float(v)
            elif isinstance(v, dict):
                results_serializable[k] = {
                    kk: float(vv) if isinstance(vv, (np.float32, np.float64)) else vv
                    for kk, vv in v.items()
                }
            else:
                results_serializable[k] = v
        json.dump(results_serializable, f, indent=2)
        
    print(f"\nResults saved to {output_dir}")
    return results


def run_pareto_exploration(args):
    """Run Pareto front exploration experiment."""
    print(f"\nExploring Pareto front on {args.env}")
    print(f"Number of policies: {args.n_policies}")
    print(f"Timesteps per policy: {args.timesteps_per_policy}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"pareto_{args.env}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = make_env(args.env, normalize_obs=True)
    eval_env = make_env(args.env, normalize_obs=True)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    agent = CMOSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        device=args.device,
    )
    
    # Create trainer
    trainer = CMOSACTrainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    
    # Explore Pareto front
    pareto = trainer.explore_pareto_front(
        n_policies=args.n_policies,
        timesteps_per_policy=args.timesteps_per_policy,
    )
    
    # Save Pareto front
    pareto.save(str(output_dir / "pareto_front.json"))
    
    # Print summary
    front = pareto.get_pareto_front()
    print(f"\nPareto front contains {len(front)} points")
    print(f"Hypervolume: {pareto.get_hypervolume():.4f}")
    
    # Save objectives
    objectives = pareto.get_front_objectives()
    np.savez(
        output_dir / "pareto_objectives.npz",
        **objectives,
    )
    
    print(f"\nResults saved to {output_dir}")
    return pareto


def run_multi_seed(args):
    """Run multi-seed experiment for statistical significance."""
    print(f"\nMulti-seed experiment on {args.env}")
    print(f"Seeds: {args.seeds}")
    
    all_results = []
    
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")
        
        # Set seeds
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"cmo_sac_{args.env}_seed{seed}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        env = make_env(args.env, normalize_obs=True)
        eval_env = make_env(args.env, normalize_obs=True)
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Create agent
        agent = CMOSACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=args.hidden_dims,
            device=args.device,
        )
        
        # Create trainer
        trainer = CMOSACTrainer(
            agent=agent,
            env=env,
            eval_env=eval_env,
            checkpoint_dir=str(output_dir / "checkpoints"),
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=f"cmo_sac_{args.env}_seed{seed}",
        )
        
        # Train
        results = trainer.train(total_timesteps=args.total_timesteps)
        results["seed"] = seed
        all_results.append(results)
        
        # Save individual results
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }, f, indent=2)
            
    # Aggregate results
    aggregate_dir = Path(args.output_dir) / f"cmo_sac_{args.env}_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    final_rewards = [r["final_eval"]["eval/mean_reward"] for r in all_results]
    
    summary = {
        "env": args.env,
        "n_seeds": len(args.seeds),
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "min_reward": float(np.min(final_rewards)),
        "max_reward": float(np.max(final_rewards)),
        "seeds": args.seeds,
        "all_rewards": final_rewards,
    }
    
    with open(aggregate_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n{'='*60}")
    print("Multi-seed Summary")
    print(f"{'='*60}")
    print(f"Mean reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Min/Max: {summary['min_reward']:.2f} / {summary['max_reward']:.2f}")
    
    return summary


def run_ablation(args):
    """Run ablation study on CMO-SAC components."""
    print("\nRunning CMO-SAC Ablation Study")
    
    ablations = {
        "full": {
            "description": "Full CMO-SAC",
            "n_objectives": 3,
            "n_constraints": 4,
            "weight_strategy": "dirichlet",
        },
        "no_constraints": {
            "description": "Without constraint handling",
            "n_objectives": 3,
            "n_constraints": 0,
            "weight_strategy": "dirichlet",
        },
        "single_objective": {
            "description": "Single scalarized objective",
            "n_objectives": 1,
            "n_constraints": 4,
            "weight_strategy": "uniform",
        },
        "fixed_weights": {
            "description": "Fixed equal weights",
            "n_objectives": 3,
            "n_constraints": 4,
            "weight_strategy": "uniform",
        },
    }
    
    results = {}
    
    for ablation_name, ablation_config in ablations.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"Description: {ablation_config['description']}")
        print(f"{'='*60}")
        
        # Create environment
        env = make_env(args.env, normalize_obs=True)
        eval_env = make_env(args.env, normalize_obs=True)
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Create agent with ablation config
        agent = CMOSACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_objectives=ablation_config["n_objectives"],
            n_constraints=ablation_config["n_constraints"],
            hidden_dims=args.hidden_dims,
            device=args.device,
            weight_strategy=ablation_config["weight_strategy"],
        )
        
        # Create trainer
        output_dir = Path(args.output_dir) / f"ablation_{ablation_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = CMOSACTrainer(
            agent=agent,
            env=env,
            eval_env=eval_env,
            checkpoint_dir=str(output_dir / "checkpoints"),
        )
        
        # Short training for ablation
        ablation_timesteps = min(args.total_timesteps, 100_000)
        result = trainer.train(total_timesteps=ablation_timesteps)
        
        results[ablation_name] = {
            "config": ablation_config,
            "final_reward": result["final_eval"]["eval/mean_reward"],
        }
        
    # Save ablation results
    ablation_dir = Path(args.output_dir) / "ablation_summary"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    with open(ablation_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Print summary
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"{name:20s}: {result['final_reward']:.2f}")
        
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == "sanity":
        success = run_sanity_check(args)
        sys.exit(0 if success else 1)
        
    elif args.mode == "train":
        run_single_training(args)
        
    elif args.mode == "pareto":
        run_pareto_exploration(args)
        
    elif args.mode == "multi-seed":
        run_multi_seed(args)
        
    elif args.mode == "ablation":
        run_ablation(args)
        
    elif args.mode == "evaluate":
        # TODO: Implement evaluation from checkpoint
        print("Evaluation mode not yet implemented")
        

if __name__ == "__main__":
    main()
