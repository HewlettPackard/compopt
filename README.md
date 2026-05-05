# CompOpt — AI-Enabled Data Center Simulation Benchmark

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CompOpt** is an end-to-end, pure-Python simulator for AI data-center control research.
It provides Gymnasium-compatible environments for **workload scheduling**, **compute budgeting**, and **energy / water / cost-efficient liquid cooling** — all drivable by reinforcement learning (RL), classical controllers, or agentic AI (LLM + RAG).

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-fidelity physics** | Chip → Server → Rack → CDU → Cooling Tower thermal RC-network with RK4 integration |
| **GPU models** | Pre-calibrated H100 SXM (700 W) and B200 (1000 W) presets |
| **Gymnasium environments** | 6 registered envs from Easy (single chip) to Expert (joint scheduling + cooling) |
| **RAPS-compatible scheduling** | FCFS, SJF, Priority, Backfill, and ThermalAware policies |
| **Composable rewards** | 12+ reward components with 4 presets (cooling, datacenter, scheduling, joint) |
| **Baseline agents** | Random, Constant, Rule-Based, PID, DataCenter-Rule, FCFS |
| **LLM / Agentic AI** | vLLM + LangChain RAG agents with system prompts and JSON parsing |
| **Vectorised simulation** | `BatchSimulator` runs N envs in parallel without multiprocessing |
| **Config-driven** | JSON presets for Frontier-like, small AI cluster, and large B200 clusters |

---

## Installation

```bash
# Core (numpy + gymnasium only)
pip install -e .

# With plotting
pip install -e ".[plotting]"

# With Stable-Baselines3 RL
pip install -e ".[rl]"

# With LLM/agentic AI support
pip install -e ".[llm]"

# Everything + dev tools
pip install -e ".[dev]"
```

---

## Quick Start

```python
import compopt

# List available environments
for env_info in compopt.list_envs():
    print(f"  {env_info['id']:20s}  {env_info['description']}")

# Create and interact with an environment
env = compopt.make("ChipThermal-v0")
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

---

## Environments

CompOpt provides **six Gymnasium environments** at increasing difficulty levels:

| Environment | Difficulty | Action | Observation | Description |
|-------------|-----------|--------|-------------|-------------|
| `ChipThermal-v0` | ⭐ Easy | Box(1) — coolant flow | Box(7) — GPU temps + power | Single GPU chip thermal control |
| `RackCooling-v0` | ⭐⭐ Medium | Box(1) — rack flow | Box(10) — rack telemetry | Rack with multiple GPU servers |
| `Scheduling-v0` | ⭐⭐ Medium | Discrete(11) — queue priority | Box(12) — scheduler metrics | Job scheduling (no thermal) |
| `DataCenter-v0` | ⭐⭐⭐ Hard | Box(3) — flow/pump/fan | Box(16) — DC state | Full datacenter cooling |
| `JointDC-v0` | ⭐⭐⭐⭐ Expert | Dict{cooling+scheduling} | Dict{thermal+scheduler} | Joint optimisation (Dict spaces) |
| `JointDCFlat-v0` | ⭐⭐⭐⭐ Expert | Box(4) — cooling+sched | Box(28) — flat state | Joint optimisation (flat spaces) |

### Customisation

All environments accept keyword arguments:

```python
env = compopt.make("DataCenter-v0",
                  n_racks=4,
                  servers_per_rack=8,
                  gpus_per_server=4,
                  gpu_preset="B200",
                  dt=5.0,
                  episode_length_s=7200.0,
                  T_ambient_C=30.0)
```

---

## Physics Architecture

```
┌───────────────────────────────────────────────────────┐
│                    DataCenterModel                     │
│  ┌──────────┐  ┌──────────┐        ┌──────────┐      │
│  │  Rack 0   │  │  Rack 1   │  ...  │  Rack N   │      │
│  │ ┌──────┐ │  │ ┌──────┐ │        │ ┌──────┐ │      │
│  │ │Server│ │  │ │Server│ │        │ │Server│ │      │
│  │ │┌────┐│ │  │ │┌────┐│ │        │ │┌────┐│ │      │
│  │ ││GPU ││ │  │ ││GPU ││ │        │ ││GPU ││ │      │
│  │ │└────┘│ │  │ │└────┘│ │        │ │└────┘│ │      │
│  │ └──────┘ │  │ └──────┘ │        │ └──────┘ │      │
│  └──────────┘  └──────────┘        └──────────┘      │
│       ↕              ↕                   ↕            │
│  ┌──────────────────────────────────────────────┐     │
│  │            CDU (Heat Exchanger)               │     │
│  │  Hot Loop ←→ Counter-flow HX ←→ Cold Loop    │     │
│  └──────────────────────────────────────────────┘     │
│       ↕                                               │
│  ┌──────────────────────────────────────────────┐     │
│  │          Cooling Tower / Dry Cooler           │     │
│  │  Evaporative + Sensible → Ambient             │     │
│  └──────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────┘
```

**GPU thermal model**: nx × ny tile grid + HBM + VRM + PCB nodes, connected by lateral + vertical thermal resistances, coupled to a direct-to-chip cold plate. Solved with 4th-order Runge-Kutta.

---

## Reward System

CompOpt uses modular, composable rewards:

```python
from compopt.rewards.functions import (
    CompositeReward, ThermalPenalty, ThermalViolation,
    EnergyPenalty, WaterPenalty, PUEReward,
)

custom_reward = CompositeReward([
    ThermalPenalty(weight=2.0, target_C=78.0),
    ThermalViolation(weight=200.0, limit_C=83.0),
    EnergyPenalty(weight=0.01),
    WaterPenalty(weight=0.5),
    PUEReward(weight=1.0),
])

env = compopt.make("DataCenter-v0", reward_fn=custom_reward)
```

**Built-in presets**: `cooling_only_reward()`, `datacenter_reward()`, `scheduling_reward()`, `joint_reward()`

---

## RL Training

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import compopt

vec_env = DummyVecEnv([lambda: compopt.make("RackCooling-v0") for _ in range(4)])
model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, verbose=1)
model.learn(total_timesteps=100_000)
```

### Vectorised Simulation (no multiprocessing)

```python
from compopt.utils.vec_env import BatchSimulator
import numpy as np

batch = BatchSimulator(
    lambda: compopt.make("ChipThermal-v0"),
    n_envs=32,
)
obs = batch.reset()
actions = np.random.uniform(0, 1, size=(32, 1)).astype(np.float32)
obs, rewards, terms, truncs, infos = batch.step(actions)
```

---

## LLM / Agentic AI

CompOpt supports LLM-driven control via **vLLM + LangChain RAG**:

```python
from compopt.agents.llm_agent import LLMCoolingAgent

agent = LLMCoolingAgent(
    vllm_base_url="http://localhost:8000/v1",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    docs_path="docs/compopt_reference.txt",
)

env = compopt.make("RackCooling-v0")
obs, _ = env.reset()
for step in range(200):
    action, info = agent.predict(obs)
    obs, reward, _, truncated, _ = env.step(action)
```

---

## Baseline Agents

```python
from compopt.agents.baselines import (
    RandomAgent, ConstantAgent, RuleBasedCoolingAgent,
    PIDCoolingAgent, DataCenterRuleAgent, FCFSSchedulingAgent,
)
from compopt.utils.metrics import evaluate_agent, print_metrics

env = compopt.make("RackCooling-v0")
agents = {
    "Random": RandomAgent(env.action_space),
    "PID": PIDCoolingAgent(target_C=80.0, Kp=0.05),
    "Rule": RuleBasedCoolingAgent(target_C=80.0),
}

for name, agent in agents.items():
    result = evaluate_agent(env, agent, n_episodes=5)
    print_metrics(result, title=name)
```

---

## Configuration Presets

Load pre-built cluster configs:

```python
from compopt.configs.loader import list_presets, build_from_config

print(list_presets())  # ['frontier_like', 'small_ai_cluster', 'large_b200_cluster']

result = build_from_config("small_ai_cluster", n_jobs=100)
dc = result["datacenter"]      # DataCenterModel
scheduler = result["scheduler"] # Scheduler with RAPS-compatible policies
jobs = result["jobs"]           # Pre-generated synthetic workload
```

---

## Examples

See the `examples/` directory for runnable demonstrations:

| Script | Description |
|--------|-------------|
| `01_chip_quickstart.py` | Single-chip PID control |
| `02_rack_cooling.py` | Rack-level agent comparison |
| `03_datacenter.py` | Full datacenter simulation |
| `04_scheduling.py` | Job scheduling with FCFS vs random |
| `05_joint_datacenter.py` | Joint scheduling + cooling |
| `06_rl_training_sb3.py` | PPO training with Stable-Baselines3 |
| `07_benchmark_throughput.py` | Vectorised simulation speed test |
| `08_llm_agent.py` | LLM-based cooling control |
| `09_custom_rewards.py` | Custom reward function design |
| `10_config_driven.py` | Config-driven datacenter setup |

Run any example:
```bash
python examples/01_chip_quickstart.py
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Project Structure

```
compopt/
├── __init__.py              # Package entry (compopt.make, compopt.list_envs)
├── physics/
│   ├── fluids.py            # CoolantLoop, CDU, CoolingTower, BatchCoolantLoop
│   ├── chip.py              # GPUChipModel, PowerProfile, GPU_PRESETS, make_gpu()
│   ├── workloads.py         # Synthetic power profiles (sinusoidal, step, burst, stochastic)
│   └── server.py            # ServerNode, RackModel, DataCenterModel
├── scheduling/
│   ├── jobs.py              # Job dataclass, generators (random, training, inference)
│   └── scheduler.py         # Scheduler with FCFS/SJF/Priority/Backfill/ThermalAware
├── envs/
│   ├── chip_env.py          # ChipThermal-v0
│   ├── rack_env.py          # RackCooling-v0
│   ├── datacenter_env.py    # DataCenter-v0
│   ├── scheduling_env.py    # Scheduling-v0
│   ├── joint_env.py         # JointDC-v0, JointDCFlat-v0
│   └── registry.py          # Environment registration & factory
├── rewards/
│   └── functions.py         # 12+ composable reward components + presets
├── agents/
│   ├── baselines.py         # Random, Constant, RuleBased, PID, DC-Rule, FCFS
│   └── llm_agent.py         # LLM Cooling & Joint agents (vLLM + LangChain)
├── utils/
│   ├── vec_env.py           # BatchSimulator, benchmark_throughput
│   ├── metrics.py           # EpisodeMetrics, evaluate_agent, print_metrics
│   └── plotting.py          # Thermal, datacenter, and comparison plots
└── configs/
    ├── frontier_like.json   # Frontier-like cluster
    ├── small_ai_cluster.json
    ├── large_b200_cluster.json
    └── loader.py            # Config loading & datacenter builder
```

---

## Key Metrics Tracked

- **Thermal**: hotspot temperature, HBM/VRM temperatures, violation rate
- **Energy**: PUE (Power Usage Effectiveness), total kWh, cooling overhead
- **Water**: WUE (Water Usage Effectiveness), total litres consumed
- **Cost**: electricity + water costs in $/hour
- **Scheduling**: throughput (jobs/hour), node utilisation, queue length, SLA violations

---

## Key Equations

**Solid-node energy balance** (per node *i*):

```
C_i · dT_i/dt = P_i(t) − Σ_j G_ij(T_i − T_j) − g_cool_i(T_i − T_cool)
```

**Coolant energy balance**:

```
C_cool · dT_cool/dt = Q_cool_total − ṁ · c_p · (T_cool − T_in)
```

**CDU counter-flow heat exchanger**:

```
Q_hx = UA · LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
```

**Cooling tower**:

```
C_basin · dT_basin/dt = Q_facility − UA_fan · (T_basin − T_eff)
T_eff = f_evap · T_wetbulb + (1 − f_evap) · T_ambient
```

---

## Citation

If you use CompOpt in your research, please cite:

```bibtex
@inproceedings{compopt2026,
    title     = {{CompOpt}: An End-to-End AI Data Center Simulation Benchmark
                 for Reinforcement Learning and Agentic AI},
    author    = {{CompOpt Contributors} and {Oak Ridge National Laboratory}},
    booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
    year      = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- **Oak Ridge National Laboratory** — RAPS framework, ExaDigiT project
- **NVIDIA** — H100/B200 GPU specifications and thermal design guidelines
