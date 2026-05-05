#!/usr/bin/env python3
"""
Test suite for CMO-SAC implementation.

Verifies all components work correctly without requiring
full training runs.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not available")
        return False
        
    try:
        from cmo_sac.rewards import MultiObjectiveReward, WeightSampler, ObjectiveConfig
        print("  ✓ rewards module")
    except ImportError as e:
        print(f"  ✗ rewards module: {e}")
        return False
        
    try:
        from cmo_sac.constraints import ConstraintManager, LagrangianDualOptimizer, ConstraintBuffer
        print("  ✓ constraints module")
    except ImportError as e:
        print(f"  ✗ constraints module: {e}")
        return False
        
    try:
        from cmo_sac.networks import CMOCritic, CMOActor, ConstraintPredictor
        print("  ✓ networks module")
    except ImportError as e:
        print(f"  ✗ networks module: {e}")
        return False
        
    try:
        from cmo_sac.pareto import ParetoFrontGenerator, ParetoPoint
        print("  ✓ pareto module")
    except ImportError as e:
        print(f"  ✗ pareto module: {e}")
        return False
        
    try:
        from cmo_sac.safety import SafetyFilter
        print("  ✓ safety module")
    except ImportError as e:
        print(f"  ✗ safety module: {e}")
        return False
        
    try:
        from cmo_sac.cmo_sac_agent import CMOSACAgent
        print("  ✓ cmo_sac_agent module")
    except ImportError as e:
        print(f"  ✗ cmo_sac_agent module: {e}")
        return False
        
    try:
        from cmo_sac.trainer import CMOSACTrainer
        print("  ✓ trainer module")
    except ImportError as e:
        print(f"  ✗ trainer module: {e}")
        return False
        
    return True


def test_multi_objective_reward():
    """Test multi-objective reward decomposition."""
    print("\nTesting MultiObjectiveReward...")
    
    from cmo_sac.rewards import MultiObjectiveReward, WeightSampler
    
    # Create reward calculator
    mo_reward = MultiObjectiveReward()
    
    # Mock observation and info
    obs = {"T_junction": np.array([75.0, 78.0, 72.0])}
    info = {
        "pue": 1.35,
        "wue": 2.1,
        "throughput": 0.85,
        "sla_compliance": 0.98,
    }
    
    # Extract objectives
    objectives = mo_reward.extract_objectives(obs, info)
    print(f"  Objectives: {objectives}")
    assert "pue" in objectives
    assert "wue" in objectives
    assert "throughput" in objectives
    print("  ✓ objective extraction")
    
    # Extract constraints
    constraints, violations = mo_reward.extract_constraints(obs, info)
    print(f"  Constraints: {constraints}")
    print(f"  Violations: {violations}")
    print("  ✓ constraint extraction")
    
    # Test weight sampler
    sampler = WeightSampler(n_objectives=3, strategy="dirichlet")
    weights = sampler.sample()
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)
    print(f"  Sampled weights: {weights}")
    print("  ✓ weight sampling")
    
    return True


def test_constraint_manager():
    """Test constraint management and Lagrangian optimization."""
    print("\nTesting ConstraintManager and LagrangianDualOptimizer...")
    
    import torch
    from cmo_sac.constraints import ConstraintManager, LagrangianDualOptimizer
    
    # Create constraint manager
    constraint_mgr = ConstraintManager()
    
    # Mock observation and info
    obs = {"T_junction": 80.0}
    info = {"sla_compliance": 0.92}  # Below threshold
    
    # Compute costs
    costs = constraint_mgr.compute_costs(obs, info)
    print(f"  Constraint costs: {costs}")
    print("  ✓ cost computation")
    
    # Test Lagrangian optimizer
    lagrangian = LagrangianDualOptimizer(constraint_mgr, device="cpu")
    
    # Initial lambdas
    lambdas = lagrangian.get_lambdas_dict()
    print(f"  Initial lambdas: {lambdas}")
    
    # Simulate constraint returns
    constraint_returns = {
        "thermal": 0.5,  # Some violation
        "hbm": 0.0,
        "sla": 0.1,
        "coolant": 0.0,
    }
    
    # Dual step
    gradients = lagrangian.dual_step(constraint_returns)
    print(f"  Dual gradients: {gradients}")
    
    # Check lambdas updated
    new_lambdas = lagrangian.get_lambdas_dict()
    print(f"  Updated lambdas: {new_lambdas}")
    print("  ✓ Lagrangian dual optimization")
    
    return True


def test_networks():
    """Test neural network architectures."""
    print("\nTesting neural networks...")
    
    import torch
    from cmo_sac.networks import CMOCritic, CMOActor, ConstraintPredictor
    
    obs_dim = 20
    action_dim = 4
    n_objectives = 3
    n_constraints = 4
    batch_size = 32
    
    # Test CMOCritic
    critic = CMOCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        n_constraints=n_constraints,
        hidden_dims=[64, 64],
    )
    
    obs = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)
    weights = torch.softmax(torch.randn(batch_size, n_objectives), dim=-1)
    
    output = critic.forward(obs, action, weights)
    print(f"  Critic output keys: {output.keys()}")
    assert output["objective_q1"].shape == (batch_size, n_objectives)
    assert output["constraint_q1"].shape == (batch_size, n_constraints)
    assert output["scalarized_q1"].shape == (batch_size, 1)
    print("  ✓ CMOCritic")
    
    # Test CMOActor
    actor = CMOActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dims=[64, 64],
    )
    
    action, log_prob = actor.forward(obs, weights)
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size, 1)
    print(f"  Actor output: action={action.shape}, log_prob={log_prob.shape}")
    print("  ✓ CMOActor")
    
    # Test ConstraintPredictor
    predictor = ConstraintPredictor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_constraints=n_constraints,
        hidden_dims=[64],
    )
    
    costs, probs = predictor.forward(obs, action)
    assert costs.shape == (batch_size, n_constraints)
    assert probs.shape == (batch_size, n_constraints)
    print(f"  Predictor output: costs={costs.shape}, probs={probs.shape}")
    print("  ✓ ConstraintPredictor")
    
    return True


def test_pareto_front():
    """Test Pareto front utilities."""
    print("\nTesting ParetoFrontGenerator...")
    
    from cmo_sac.pareto import ParetoFrontGenerator, ParetoPoint
    
    pareto = ParetoFrontGenerator(
        n_objectives=3,
        objective_names=["pue", "wue", "throughput"],
        minimize={"pue": True, "wue": True, "throughput": False},
    )
    
    # Add some points
    points = [
        {"pue": 1.2, "wue": 2.0, "throughput": 0.9},
        {"pue": 1.3, "wue": 1.5, "throughput": 0.85},
        {"pue": 1.4, "wue": 1.2, "throughput": 0.8},
        {"pue": 1.25, "wue": 1.8, "throughput": 0.88},  # Dominated
    ]
    
    for i, obj in enumerate(points):
        weights = np.random.dirichlet([1, 1, 1])
        added = pareto.add_point(obj, weights)
        print(f"  Point {i}: {'added' if added else 'dominated'}")
        
    front = pareto.get_pareto_front()
    print(f"  Pareto front size: {len(front)}")
    
    # Test hypervolume
    hv = pareto.get_hypervolume()
    print(f"  Hypervolume: {hv:.4f}")
    
    # Test weight suggestion
    suggested = pareto.suggest_weights(strategy="gap")
    print(f"  Suggested weights: {suggested}")
    assert np.isclose(suggested.sum(), 1.0)
    print("  ✓ Pareto front utilities")
    
    return True


def test_agent():
    """Test CMO-SAC agent."""
    print("\nTesting CMOSACAgent...")
    
    import torch
    from cmo_sac.cmo_sac_agent import CMOSACAgent
    
    obs_dim = 20
    action_dim = 4
    
    agent = CMOSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        device="cpu",
    )
    
    print(f"  Device: {agent.device}")
    
    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.select_action(obs)
    assert action.shape == (action_dim,)
    print(f"  Action: {action}")
    print("  ✓ action selection")
    
    # Test with different weights
    weights = np.array([0.5, 0.3, 0.2])
    action_weighted = agent.select_action(obs, weights=weights)
    print(f"  Action (weighted): {action_weighted}")
    print("  ✓ weight-conditioned action")
    
    # Test deterministic action
    action_det = agent.select_action(obs, deterministic=True)
    print(f"  Action (deterministic): {action_det}")
    print("  ✓ deterministic action")
    
    return True


def test_buffer():
    """Test constraint-aware replay buffer."""
    print("\nTesting ConstraintBuffer...")
    
    import torch
    from cmo_sac.constraints import ConstraintBuffer
    
    buffer = ConstraintBuffer(
        capacity=1000,
        obs_dim=20,
        action_dim=4,
        n_objectives=3,
        n_constraints=4,
        device="cpu",
    )
    
    # Add transitions
    for _ in range(100):
        buffer.add(
            obs=np.random.randn(20).astype(np.float32),
            action=np.random.randn(4).astype(np.float32),
            reward=np.random.randn(3).astype(np.float32),
            scalarized_reward=np.random.randn(),
            constraint_costs=np.abs(np.random.randn(4)).astype(np.float32),
            next_obs=np.random.randn(20).astype(np.float32),
            done=np.random.rand() < 0.1,
            weights=np.array([0.33, 0.33, 0.34]),
        )
        
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample batch
    batch = buffer.sample(32)
    print(f"  Batch keys: {batch.keys()}")
    assert batch["observations"].shape == (32, 20)
    assert batch["rewards"].shape == (32, 3)
    assert batch["constraint_costs"].shape == (32, 4)
    print("  ✓ buffer operations")
    
    return True


def test_training_update():
    """Test a single training update."""
    print("\nTesting training update...")
    
    import torch
    from cmo_sac.cmo_sac_agent import CMOSACAgent
    from cmo_sac.constraints import ConstraintBuffer
    
    obs_dim = 20
    action_dim = 4
    batch_size = 32
    
    # Create agent
    agent = CMOSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        device="cpu",
    )
    
    # Create and fill buffer
    buffer = ConstraintBuffer(
        capacity=1000,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device="cpu",
    )
    
    for _ in range(100):
        buffer.add(
            obs=np.random.randn(obs_dim).astype(np.float32),
            action=np.random.randn(action_dim).astype(np.float32),
            reward=np.random.randn(3).astype(np.float32),
            scalarized_reward=np.random.randn(),
            constraint_costs=np.abs(np.random.randn(4)).astype(np.float32),
            next_obs=np.random.randn(obs_dim).astype(np.float32),
            done=np.random.rand() < 0.1,
            weights=np.array([0.33, 0.33, 0.34]),
        )
        
    # Sample batch and update
    batch = buffer.sample(batch_size)
    metrics = agent.update(batch)
    
    print(f"  Metrics: {metrics}")
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    print("  ✓ training update")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CMO-SAC Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Multi-Objective Reward", test_multi_objective_reward),
        ("Constraint Manager", test_constraint_manager),
        ("Neural Networks", test_networks),
        ("Pareto Front", test_pareto_front),
        ("Agent", test_agent),
        ("Buffer", test_buffer),
        ("Training Update", test_training_update),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
        
    n_passed = sum(1 for _, s in results if s)
    print(f"\n{n_passed}/{len(results)} tests passed")
    
    return all(s for _, s in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
