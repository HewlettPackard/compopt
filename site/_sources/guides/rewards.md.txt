# Reward System

CompOpt uses a modular, composable reward system. Individual reward
components can be mixed and matched to create custom multi-objective
reward functions.

## Architecture

```python
from compopt.rewards import CompositeReward, ThermalPenalty, EnergyPenalty

reward_fn = CompositeReward([
    ThermalPenalty(weight=1.0, target_C=80.0),
    EnergyPenalty(weight=0.01, P_ref_kW=100.0),
])

total, breakdown = reward_fn(info_dict)
# breakdown = {"thermal_penalty": -4.5, "energy_penalty": -0.03}
```

## Reward Components

### Thermal

| Component           | Formula                                              | Default Weight |
|---------------------|------------------------------------------------------|---------------:|
| `ThermalPenalty`    | $-w \cdot \max(0, T - T_{\text{target}})^2$         |            1.0 |
| `ThermalViolation`  | $-w$ if $T > T_{\text{limit}}$, else 0              |          100.0 |
| `HBMPenalty`        | $-w \cdot \max(0, T_{\text{HBM}} - T_{\text{lim}})^2$ |         0.5 |

### Energy & Cost

| Component                    | Formula                                                                                   | Default Weight |
|------------------------------|-------------------------------------------------------------------------------------------|---------------:|
| `EnergyPenalty`              | $-w \cdot P / P_{\text{ref}}$                                                             |           0.01 |
| `CoolantFlowPenalty`         | $-w \cdot (\dot{m} - \dot{m}_{\min})^3 / 100$                                            |           0.01 |
| `PUEReward`                  | $w \cdot (2.0 - \text{PUE})$                                                              |            1.0 |
| `CostPenalty`                | $-w \cdot \Delta\text{cost}$                                                              |           10.0 |
| `GridElectricityCostPenalty` | $-w \cdot P_{\text{kW}} \cdot p_{\$/\text{kWh}} \cdot \Delta t_h / P_{\text{ref}}$       |           50.0 |

### Water

| Component      | Formula                                      | Default Weight |
|----------------|----------------------------------------------|---------------:|
| `WaterPenalty` | $-w \cdot \text{water}_L / \text{ref}_L$     |            0.5 |
| `WUEReward`    | $-w \cdot \text{WUE}$                        |            0.5 |

### Scheduling

| Component           | Formula                             | Default Weight |
|---------------------|-------------------------------------|---------------:|
| `ThroughputReward`  | $w \cdot n_{\text{completed}}$      |            1.0 |
| `QueuePenalty`      | $-w \cdot \text{queue\_length}$     |            0.1 |
| `UtilisationReward` | $w \cdot \text{utilisation}$        |            1.0 |
| `SLAViolation`      | $-w \cdot n_{\text{violations}}$    |           50.0 |

## Presets

CompOpt provides five preset reward configurations:

```python
from compopt.rewards import (
    cooling_only_reward,             # chip/rack cooling
    datacenter_reward,               # full DC
    scheduling_reward,               # scheduling only
    joint_reward,                    # combined
    grid_aware_datacenter_reward,    # full DC with real-time grid tariff
)

# Use with any environment:
env = compopt.make("DataCenter-v0", reward_fn=datacenter_reward())
```

### Preset Compositions

**`cooling_only_reward`**: ThermalPenalty + ThermalViolation + FlowPenalty

**`datacenter_reward`**: ThermalPenalty + ThermalViolation + EnergyPenalty + PUEReward + WaterPenalty + CostPenalty

**`scheduling_reward`**: ThroughputReward + QueuePenalty + UtilisationReward + SLAViolation

**`joint_reward`**: All of the above combined

**`grid_aware_datacenter_reward`**: ThermalPenalty + ThermalViolation + EnergyPenalty + PUEReward + WaterPenalty + **GridElectricityCostPenalty**

## Grid Electricity Cost Reward

`GridElectricityCostPenalty` is a time-aware reward component that multiplies
actual power consumption by the *current* grid tariff price, incentivising the
agent to shift load away from expensive on-peak windows.

```python
from compopt.rewards import GridElectricityCostPenalty

comp = GridElectricityCostPenalty(
    weight=50.0,      # scalar multiplier
    dt_s=5.0,         # must match env dt
    P_ref_kW=100.0,   # reference power for normalisation
)
# Returns: −weight × P_kW × price_$/kWh × (dt_s/3600) / P_ref_kW
```

Use this component together with `DataCenterEnv(grid_electricity=True)`.
See the [Grid Electricity Mode](environments.md#grid-electricity-mode) section
for the matching environment feature.

## Custom Rewards

Create your own reward component by subclassing `RewardComponent`:

```python
from compopt.rewards import RewardComponent, CompositeReward
from dataclasses import dataclass

@dataclass
class MyCustomReward(RewardComponent):
    name: str = "my_reward"
    weight: float = 1.0

    def __call__(self, info):
        # Your custom logic
        return self.weight * info.get("my_metric", 0.0)

reward_fn = CompositeReward([
    MyCustomReward(weight=2.0),
    ThermalPenalty(weight=1.0),
])
```
