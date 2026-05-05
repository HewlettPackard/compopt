# RL Training

CompOpt environments are fully compatible with standard RL libraries.
This guide covers training with Stable Baselines3 (SB3), evaluation,
and vectorised training.

## Stable Baselines3

### Quick Start

```python
from stable_baselines3 import PPO
import compopt

env = compopt.make("RackCooling-v0")

model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10)

model.learn(total_timesteps=100_000)
model.save("ppo_rack_cooling")
```

### Training on DataCenter-v0

```python
from stable_baselines3 import SAC

env = compopt.make("DataCenter-v0", dt=5.0)
model = SAC("MlpPolicy", env, verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000)
model.learn(total_timesteps=500_000)
```

### Training on JointDCFlat-v0

```python
from stable_baselines3 import PPO

# Must use the flat variant for SB3
env = compopt.make("JointDCFlat-v0",
                   n_racks=2,
                   n_jobs=200,
                   episode_length_s=7200)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

## Evaluation

Use the built-in evaluation utilities:

```python
from compopt.utils import evaluate_agent, print_metrics
from compopt.agents import PIDCoolingAgent

env = compopt.make("RackCooling-v0")
agent = PIDCoolingAgent(target_C=80.0)

metrics = evaluate_agent(env, agent, n_episodes=5)
print_metrics(metrics, title="PID Baseline")
```

### Compare Multiple Agents

```python
from compopt.agents import RandomAgent, RuleBasedCoolingAgent, PIDCoolingAgent
from compopt.utils import evaluate_agent
from compopt.utils.plotting import plot_comparison

env = compopt.make("RackCooling-v0")
results = {}

for name, agent in [
    ("Random", RandomAgent(env.action_space)),
    ("Rule-Based", RuleBasedCoolingAgent(target_C=80.0)),
    ("PID", PIDCoolingAgent(target_C=80.0)),
]:
    results[name] = evaluate_agent(env, agent, n_episodes=5)

plot_comparison(results, title="Agent Comparison on RackCooling-v0")
```

## Vectorised Training

Use `BatchSimulator` for high-throughput data collection:

```python
from compopt.utils import BatchSimulator, benchmark_throughput
import compopt

throughput = benchmark_throughput(
    lambda: compopt.make("ChipThermal-v0"),
    n_envs=16,
    n_steps=5000,
)
print(f"Throughput: {throughput['steps_per_second']:.0f} steps/s")
```

### Custom Training Loop

```python
import numpy as np
from compopt.utils import BatchSimulator

batch = BatchSimulator(
    lambda: compopt.make("ChipThermal-v0"),
    n_envs=8,
)

obs = batch.reset()  # shape: (8, 7)

for step in range(10000):
    actions = np.random.uniform(0, 1, size=(8, 1)).astype(np.float32)
    obs, rewards, terms, truncs, infos = batch.step(actions)
    # Feed to your RL algorithm...
```

## Baseline Agents

CompOpt ships with several baseline agents for benchmarking:

| Agent                      | Type       | Environments              |
|----------------------------|------------|---------------------------|
| `RandomAgent`              | Random     | All                       |
| `ConstantAgent`            | Constant   | All                       |
| `RuleBasedCoolingAgent`    | Bang-bang  | ChipThermal, RackCooling  |
| `PIDCoolingAgent`          | PID        | ChipThermal, RackCooling  |
| `DataCenterRuleAgent`      | Heuristic  | DataCenter                |
| `FCFSSchedulingAgent`      | FCFS       | Scheduling                |

## Tips for RL Training

1. **Start simple**: Train on `ChipThermal-v0` first, then scale up.
2. **Normalise observations**: Use SB3's `VecNormalize` wrapper.
3. **Custom rewards**: Tune reward weights for your objective.
4. **Curriculum learning**: Start with short episodes, increase over time.
5. **Monitor PUE/WUE**: These are the key real-world metrics.
