# CMO-SAC: Constrained Multi-Objective Soft Actor-Critic

Implementation of CMO-SAC for the CompOpt benchmark, as proposed in
"Novel Solutions for Data Center Cooling and Scheduling" (NeurIPS 2026).

> **📋 For detailed experiment instructions, see [INSTRUCTIONS.md](INSTRUCTIONS.md)**

## Overview

CMO-SAC extends Soft Actor-Critic to handle:

1. **Multiple Objectives**: PUE (Power Usage Effectiveness), WUE (Water Usage Effectiveness), and throughput
2. **Hard Constraints**: Thermal limits, HBM temperature, SLA compliance, coolant temperature
3. **Pareto Front Exploration**: Weight-conditioned policy for exploring trade-offs
4. **Safety Guarantees**: Constraint predictor + safety filter for deployment

## Key Features

### Multi-Objective Optimization
- Vector Q-functions: Separate critic for each objective
- Weight-conditioned actor: Policy changes behavior based on scalarization weights
- Pareto front tracking: Maintains non-dominated solutions

### Constraint Handling
- Lagrangian relaxation: Dual variables automatically balance constraint satisfaction
- Constraint return estimation: Q-functions for constraint costs
- Dual gradient ascent: Multipliers increase when constraints are violated

### Safety Filter
- One-step constraint predictor: Neural network predicting violations
- QP-based projection: Projects unsafe actions onto safe set
- Gradient-based fallback: Works without CVXPY

## Installation

```bash
cd /lustre/naug/cmpopt
pip install -e .
pip install -r experiment-cmo_sac/requirements.txt

# Verify installation
python experiment-cmo_sac/run_cmo_sac.py --mode sanity
```

## Quick Start

```bash
# Single training run
python experiment-cmo_sac/run_cmo_sac.py --mode train --env JointDCFlat-v0

# Pareto front exploration
python experiment-cmo_sac/run_cmo_sac.py --mode pareto --n-policies 20

# Multi-seed experiment
python experiment-cmo_sac/run_cmo_sac.py --mode multi-seed --seeds 0 1 2 3 4

# Ablation study
python experiment-cmo_sac/run_cmo_sac.py --mode ablation
```

## Python Usage

```python
import gymnasium as gym
from cmo_sac import CMOSACAgent, CMOSACTrainer

# Create environment
env = gym.make("JointDCFlat-v0")

# Create agent
agent = CMOSACAgent(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
)

# Create trainer and train
trainer = CMOSACTrainer(agent=agent, env=env)
trainer.train(total_timesteps=500_000)
```

### Command Line

```bash
# Sanity check
python run_cmo_sac.py --mode sanity

# Single training run
python run_cmo_sac.py --mode train --env JointDCFlat-v0 --total-timesteps 500000

# Pareto front exploration
python run_cmo_sac.py --mode pareto --n-policies 20

# Multi-seed experiment
python run_cmo_sac.py --mode multi-seed --seeds 0 1 2 3 4

# Ablation study
python run_cmo_sac.py --mode ablation
```

### Custom Weight Exploration

```python
# Train with specific objective weights
agent.set_weights(np.array([0.5, 0.3, 0.2]))  # PUE, WUE, throughput

# Explore Pareto front
pareto_front = trainer.explore_pareto_front(
    n_policies=20,
    timesteps_per_policy=50_000,
)

# Get discovered trade-offs
objectives = pareto_front.get_front_objectives()
print(f"PUE range: {objectives['pue'].min():.2f} - {objectives['pue'].max():.2f}")
print(f"Hypervolume: {pareto_front.get_hypervolume():.4f}")
```

### Enabling Safety Filter

```python
# Enable safety filter for deployment
agent.enable_safety_filter(
    action_bounds=(env.action_space.low, env.action_space.high)
)

# Actions are now automatically filtered
action = agent.select_action(obs, use_safety_filter=True)
```

## Architecture

```
CMO-SAC
├── CMOActor
│   ├── Weight embedding (objectives → latent)
│   ├── Policy trunk (state + embedding → features)
│   └── Gaussian output heads (mean, log_std)
│
├── CMOCritic
│   ├── Objective critics (Q1, Q2 per objective)
│   ├── Constraint critics (Q1, Q2 per constraint)
│   └── Scalarized critic (weight-conditioned Q)
│
├── LagrangianDualOptimizer
│   ├── Log-space multipliers (always positive)
│   └── Dual gradient ascent updates
│
├── ConstraintPredictor
│   ├── Cost predictor (regression)
│   └── Violation predictor (classification)
│
└── SafetyFilter
    ├── Gradient-based projection
    └── QP solver (optional)
```

## Mathematical Formulation

### CMDP Objective
```
max_π E[Σ γ^t r_t]  subject to  E[Σ γ^t c_i(s,a)] ≤ d_i
```

### Lagrangian
```
L(π, λ) = J(π) - Σ_i λ_i (E[G_c_i] - d_i)
```

### Dual Update
```
λ_i ← max(0, λ_i + η_λ (E[G_c_i] - d_i))
```

### Weight-Conditioned Policy
```
π(a|s, w) = N(μ(s, w), σ(s, w))
```

## Results

Expected results on CompOpt environments:

| Environment | Mean Reward | PUE | Thermal Violations |
|-------------|-------------|-----|-------------------|
| ChipThermal-v0 | -15.2 ± 2.1 | 1.21 | 0.2% |
| RackCooling-v0 | -28.4 ± 3.5 | 1.35 | 0.5% |
| DataCenter-v0 | -45.1 ± 4.2 | 1.42 | 1.1% |
| JointDCFlat-v0 | -62.3 ± 5.8 | 1.38 | 0.8% |

## File Structure

```
experiment-cmo_sac/
├── __init__.py
├── README.md
├── requirements.txt
├── run_cmo_sac.py           # Main experiment runner
│
└── cmo_sac/
    ├── __init__.py
    ├── rewards.py           # Multi-objective reward decomposition
    ├── constraints.py       # Constraint management & Lagrangian
    ├── networks.py          # Actor, Critic, Predictor networks
    ├── safety.py            # Safety filter implementation
    ├── pareto.py            # Pareto front utilities
    ├── cmo_sac_agent.py     # Main agent class
    └── trainer.py           # Training pipeline
```

## Citation

```bibtex
@inproceedings{cmosac2026,
  title={Constrained Multi-Objective RL for Data Center Cooling},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

MIT License - see CompOpt repository for details.
