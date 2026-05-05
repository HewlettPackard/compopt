# Getting Started

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/compopt-benchmark/compopt.git
cd compopt
pip install -e ".[dev]"
```

### Core only (minimal dependencies)

```bash
pip install -e .
```

### With RL support

```bash
pip install -e ".[rl]"
```

### With LLM agent support

```bash
pip install -e ".[llm]"
```

### Everything

```bash
pip install -e ".[all]"
```

## Your First Simulation

### Chip-Level Thermal Control

The simplest entry point — control coolant flow to a single GPU chip:

```python
import compopt

env = compopt.make("ChipThermal-v0")
obs, info = env.reset()

for step in range(600):
    action = env.action_space.sample()  # random flow
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 100 == 0:
        print(f"Step {step}: T_hotspot={obs[1]:.1f}°C, reward={reward:.2f}")
```

### Rack-Level Cooling

Scale up to 4 GPUs sharing a rack coolant manifold:

```python
env = compopt.make("RackCooling-v0")
obs, info = env.reset()

for _ in range(1800):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Full Data Center

Control rack flow, CDU pump, and cooling tower simultaneously:

```python
env = compopt.make("DataCenter-v0")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (16,)
print(f"Action shape: {env.action_space.shape}")  # (3,)
```

## Available Environments

| Environment ID     | Difficulty | Action        | Obs Dim | Description                        |
|--------------------|------------|---------------|--------:|------------------------------------|
| `ChipThermal-v0`  | Easy       | Box(1)        |       7 | Single GPU chip thermal control    |
| `RackCooling-v0`  | Medium     | Box(1)        |      10 | Rack-level liquid cooling          |
| `Scheduling-v0`   | Medium     | Discrete(11)  |      12 | Job scheduling only                |
| `DataCenter-v0`   | Hard       | Box(3)        |      16 | Full datacenter cooling            |
| `JointDC-v0`      | Expert     | Dict          |    Dict | Joint scheduling + cooling (Dict)  |
| `JointDCFlat-v0`  | Expert     | Box(4)        |      28 | Joint scheduling + cooling (Flat)  |

## Next Steps

- {doc}`physics` — Understand the thermal-fluid models
- {doc}`environments` — Detailed environment API
- {doc}`rl_training` — Train RL agents
- {doc}`llm_agents` — Use LLM-based controllers
