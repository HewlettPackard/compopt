# Environments

CompOpt provides six Gymnasium-compatible environments at four difficulty
levels.  All follow the standard `gymnasium.Env` API.

## Environment Hierarchy

```
Easy:    ChipThermal-v0     →  Single GPU chip
Medium:  RackCooling-v0     →  Rack of 4 GPUs
         Scheduling-v0      →  Job scheduler (no thermal coupling)
Hard:    DataCenter-v0      →  Full DC: racks + CDU + cooling tower
Expert:  JointDC-v0         →  Joint scheduling + cooling (Dict)
         JointDCFlat-v0     →  Joint scheduling + cooling (Flat)
```

## ChipThermal-v0 (Easy)

Control coolant flow to a single GPU die.

| Property           | Value                                                              |
|--------------------|--------------------------------------------------------------------|
| **Action**         | `Box(1,)` in [0, 1] — normalised coolant mass-flow rate          |
| **Observation**    | `Box(7,)` — [T_mean, T_hotspot, T_hbm, T_vrm, T_in, T_out, P]  |
| **Reward**         | Quadratic thermal penalty + flow penalty                          |
| **Episode**        | 600 s (default), truncation only                                  |

```python
import compopt
env = compopt.make("ChipThermal-v0", gpu_preset="B200", dt=0.5)
```

### Configurable Parameters

- `gpu_preset` — `"H100_SXM"` or `"B200"`
- `workload` — `"sinusoidal"`, `"step"`, `"burst"`, `"stochastic"`
- `dt` — timestep [s]
- `episode_length_s` — episode duration [s]
- `m_dot_min`, `m_dot_max` — flow rate bounds [kg/s]
- `target_hotspot_C` — thermal target [°C]
- `reward_fn` — custom `CompositeReward` instance

## RackCooling-v0 (Medium)

Control rack-level coolant flow for a rack of multiple GPUs.

| Property           | Value                                                            |
|--------------------|------------------------------------------------------------------|
| **Action**         | `Box(1,)` in [0, 1] — normalised rack coolant flow             |
| **Observation**    | `Box(10,)` — GPU0 sensors + rack telemetry                     |
| **Reward**         | Thermal penalty + flow penalty                                  |
| **Episode**        | 1800 s (default)                                                |

```python
env = compopt.make("RackCooling-v0", n_servers=8, gpus_per_server=2)
```

## Scheduling-v0 (Medium)

Discrete job scheduling on a cluster (no thermal coupling).

| Property           | Value                                                            |
|--------------------|------------------------------------------------------------------|
| **Action**         | `Discrete(11)` — queue index to prioritise (0 = no-op)         |
| **Observation**    | `Box(12,)` — scheduler metrics + queue summary                 |
| **Reward**         | Throughput + utilisation - queue penalty - SLA violations        |
| **Episode**        | 86400 s (24 h) or until all jobs complete                       |

```python
env = compopt.make("Scheduling-v0", n_jobs=200, dt=60.0)
```

## DataCenter-v0 (Hard)

Multi-dimensional cooling control for a full data center.

| Property           | Value                                                            |
|--------------------|------------------------------------------------------------------|
| **Action**         | `Box(3,)` — [rack_flow, cdu_pump, tower_fan] normalised        |
| **Observation**    | `Box(16,)` — thermal + energy + cost + water state              |
| **Reward**         | Thermal + energy + PUE + water + cost                           |
| **Episode**        | 3600 s (default)                                                |

### Observation Vector (16 elements)

| Index | Field                | Units  |
|------:|----------------------|--------|
|     0 | T_hotspot_max        | °C     |
|     1 | T_hotspot_mean       | °C     |
|     2 | rack_coolant_in      | °C     |
|     3 | rack_coolant_out     | °C     |
|     4 | CDU_hot_T            | °C     |
|     5 | CDU_cold_T           | °C     |
|     6 | tower_basin_T        | °C     |
|     7 | ambient_T            | °C     |
|     8 | IT_power             | kW     |
|     9 | cooling_power        | kW     |
|    10 | PUE                  | —      |
|    11 | WUE                  | L/kWh  |
|    12 | total_cost           | $      |
|    13 | water_used           | L      |
|    14 | n_active_nodes       | —      |
|    15 | time                 | h      |

When `grid_electricity=True`, a 17th element is appended:

| Index | Field                    | Units  |
|------:|--------------------------|--------|
|    16 | grid_price               | $/kWh  |

```python
env = compopt.make("DataCenter-v0", n_racks=4, T_ambient_C=30.0)
```

### Grid Electricity Mode

`DataCenter-v0` can be configured to simulate realistic time-varying grid
electricity prices, enabling RL agents to learn **load-shifting** and
**demand-response** strategies.

```python
import compopt
from compopt.rewards import grid_aware_datacenter_reward

env = compopt.make(
    "DataCenter-v0",
    grid_electricity=True,          # enable grid price model
    grid_seed=42,                   # reproducible episodes
    dt=5.0,
    reward_fn=grid_aware_datacenter_reward(dt_s=5.0),
)
obs, info = env.reset()
# obs.shape == (17,)  ← 16 thermal/energy + grid_price
```

The grid price is exposed in the `info` dict at every step:

| Key                             | Description                          |
|---------------------------------|--------------------------------------|
| `grid_price_per_kWh`            | Current tariff price [$/kWh]         |
| `grid_electricity_cost_dollar`  | Electricity spend this step [$]      |
| `grid_hour_of_day`              | Simulated hour of day [0–24)         |
| `grid_day_of_week`              | Day of week (0=Mon … 6=Sun)          |
| `grid_cpp_active`               | 1.0 if a Critical Peak Pricing event |

#### Tariff Presets

Three utility tariff profiles are built in:

```python
from compopt.physics.electricity import TOUTariff

# PG&E E-19 (California commercial)
env = compopt.make("DataCenter-v0", grid_electricity=True,
                   grid_tariff=TOUTariff.pge_e19())

# ERCOT real-time market (Texas, lower base rate, higher peaks)
env = compopt.make("DataCenter-v0", grid_electricity=True,
                   grid_tariff=TOUTariff.ercot_rtm())

# SDG&E TOU-8 (San Diego, highest baseline)
env = compopt.make("DataCenter-v0", grid_electricity=True,
                   grid_tariff=TOUTariff.sdge_tou8())
```

#### Price Model Details

The `GridElectricityPriceModel` combines several realistic price signals:

- **Time-of-Use tiers** — off-peak (~$0.065/kWh nights/weekends),
  mid-peak (~$0.115/kWh shoulders), on-peak (~$0.22/kWh weekday afternoons)
- **Seasonal multiplier** — summer prices ~25 % higher than spring;
  secondary winter bump due to heating demand
- **Ornstein–Uhlenbeck noise** — mean-reverting real-time price fluctuations
  (±15 %), mimicking wholesale spot-market behaviour
- **Renewable surplus credit** — on 25 % of days a solar mid-day dip lowers
  prices by up to $0.06/kWh (California duck-curve effect)
- **Critical Peak Pricing (CPP)** — rare events (5 % of episodes) where
  on-peak prices spike to $0.52–$0.75/kWh

#### Example: Inspecting price variation

```python
from compopt.physics.electricity import GridElectricityPriceModel, TOUTariff

model = GridElectricityPriceModel(tariff=TOUTariff.pge_e19(), seed=0)
model.reset(start_hour=0.0, day_of_year=180)  # midnight, mid-summer

prices = []
for _ in range(288):          # 24 h at 5-minute resolution
    prices.append(model.step(dt_s=300.0))

import matplotlib.pyplot as plt
plt.plot([i * 5 / 60 for i in range(288)], prices)
plt.xlabel("Hour of day"); plt.ylabel("$/kWh"); plt.title("Grid price (PG&E E-19)")
plt.show()
```

## JointDC-v0 / JointDCFlat-v0 (Expert)

Joint scheduling + cooling optimisation — the most challenging benchmark.

**JointDC-v0** uses Dict action/observation spaces:
- `action["cooling"]` : `Box(3,)` — cooling actuators
- `action["scheduling"]` : `Discrete(11)` — queue prioritisation
- `obs["thermal"]` : `Box(16,)` — datacenter state
- `obs["scheduler"]` : `Box(12,)` — scheduler metrics

**JointDCFlat-v0** flattens everything for SB3 compatibility:
- Action: `Box(4,)` — 3 cooling + 1 scheduling (continuous)
- Obs: `Box(28,)` — 16 thermal + 12 scheduler

```python
# For SB3 (needs flat spaces):
env = compopt.make("JointDCFlat-v0", n_racks=2, n_jobs=200)

# For custom RL loops (Dict spaces):
env = compopt.make("JointDC-v0", n_racks=2, n_jobs=200)
```

## Custom Environments

Register your own environments:

```python
from compopt.envs import register

register(
    "MyCustomDC-v0",
    entry_point="my_module:MyCustomEnv",
    description="Custom datacenter environment",
)
```
