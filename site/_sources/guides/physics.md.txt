# Physics Engine

CompOpt's physics engine models heat transfer and fluid dynamics from the
individual GPU die all the way up to a full data center with cooling towers.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   DataCenterModel                    │
│  ┌────────────┐  ┌────────────┐         ┌─────────┐ │
│  │ RackModel 0│  │ RackModel 1│  ...    │   CDU   │ │
│  │┌──────────┐│  │┌──────────┐│         │ hot_loop│ │
│  ││ServerNode││  ││ServerNode││         │cold_loop│ │
│  ││┌────────┐││  ││┌────────┐││         └────┬────┘ │
│  │││GPU Chip│││  │││GPU Chip│││              │      │
│  │││RC-net  │││  │││RC-net  │││     ┌────────┴────┐ │
│  ││└────────┘││  ││└────────┘││     │CoolingTower │ │
│  │└──────────┘│  │└──────────┘│     │  evaporative│ │
│  └────────────┘  └────────────┘     └─────────────┘ │
└──────────────────────────────────────────────────────┘
```

## GPU Chip Model (`GPUChipModel`)

Each GPU is modelled as an RC thermal network:

- **Die tiles** (nx × ny grid) — small thermal masses with lateral conduction
- **HBM** — lumped high-bandwidth memory stack
- **VRM** — lumped voltage regulator module
- **PCB** — lumped carrier board / heat spreader

### Governing Equation

For each thermal node *i*:

$$
C_i \frac{dT_i}{dt} = P_i(t) - \sum_j G_{ij}(T_i - T_j) - g_{\text{cool},i}(T_i - T_{\text{coolant}})
$$

where:
- $C_i$ = thermal capacitance [J/K]
- $P_i(t)$ = time-varying power dissipation [W]
- $G_{ij}$ = conductance between nodes $i$ and $j$ [W/K]
- $g_{\text{cool},i}$ = coolant coupling conductance [W/K]

### Numerical Integration

The solid nodes use **4th-order Runge-Kutta (RK4)** with **automatic sub-stepping**.
The maximum stable timestep is estimated from the Jacobian's spectral radius:

$$
\Delta t_{\max} = \frac{2.78}{\max_i \frac{G_{ii} + g_{\text{cool},i}}{C_i}}
$$

If the user-requested `dt` exceeds this, the solver automatically subdivides
into stable sub-steps.

### GPU Presets

| Preset      | TDP   | Die Area  | Tiles | T_max_die |
|-------------|------:|----------:|------:|----------:|
| `H100_SXM`  | 700 W | 814 mm²  | 4×4   | 83°C      |
| `B200`       | 1000 W| 1000 mm² | 6×6   | 85°C      |

```python
from compopt.physics import make_gpu

gpu = make_gpu("H100_SXM")
gpu.step(dt=1.0)
print(gpu.get_sensors_dict())
```

## Coolant Loop (`CoolantLoop`)

A lumped control volume representing direct-to-chip liquid cooling:

$$
C_{\text{th}} \frac{dT}{dt} = Q_{\text{in}} - \dot{m} c_p (T - T_{\text{in}})
$$

**Symbol Definitions:**

| Symbol | Description | Units |
|--------|-------------|-------|
| $C_{\text{th}}$ | Thermal capacitance of the coolant volume | J/K |
| $T$ | Coolant temperature (well-mixed tank model) | °C |
| $Q_{\text{in}}$ | Heat input from the chip/component | W |
| $\dot{m}$ | Coolant mass flow rate | kg/s |
| $c_p$ | Specific heat capacity of coolant (e.g., water ≈ 4186) | J/(kg·K) |
| $T_{\text{in}}$ | Inlet coolant temperature (from CDU or chiller) | °C |

Uses a **semi-implicit** integration scheme (unconditionally stable):

$$
T^{n+1} = \frac{T^n + \frac{\Delta t}{C}(Q + \dot{m} c_p T_{\text{in}})}{1 + \frac{\Delta t \dot{m} c_p}{C}}
$$

**Additional Symbols:**

| Symbol | Description | Units |
|--------|-------------|-------|
| $T^n$ | Temperature at current timestep | °C |
| $T^{n+1}$ | Temperature at next timestep | °C |
| $\Delta t$ | Integration timestep | s |
| $C$ | Shorthand for $C_{\text{th}}$ (thermal capacitance) | J/K |

This ensures stability even with small coolant volumes and large timesteps.

## CDU (Coolant Distribution Unit)

Transfers heat from the rack-side (hot) loop to the facility-side (cold) loop
through a counter-flow heat exchanger:

$$
Q_{\text{HX}} = UA \cdot (T_{\text{hot}} - T_{\text{cold}})
$$

**Symbol Definitions:**

| Symbol | Description | Units |
|--------|-------------|-------|
| $Q_{\text{HX}}$ | Heat transfer rate through the heat exchanger | W |
| $UA$ | Overall heat transfer coefficient × area (heat exchanger effectiveness) | W/K |
| $T_{\text{hot}}$ | Temperature of the hot-side (rack) coolant loop | °C |
| $T_{\text{cold}}$ | Temperature of the cold-side (facility) coolant loop | °C |

> **Note:** Higher $UA$ values indicate a more effective heat exchanger with better thermal coupling between the loops.

```python
from compopt.physics import CDU

cdu = CDU(UA_W_K=50000, T_facility_C=30.0)
cdu.step(dt=1.0, Q_rack_total_W=100000)
```

## Cooling Tower

Evaporative heat rejection to ambient, tracking water consumption:

$$
Q_{\text{reject}} = UA_{\text{fan}} \cdot (T_{\text{basin}} - T_{\text{eff}})
$$

**Symbol Definitions:**

| Symbol | Description | Units |
|--------|-------------|-------|
| $Q_{\text{reject}}$ | Heat rejected to the atmosphere | W |
| $UA_{\text{fan}}$ | Effective heat transfer coefficient, scaled by fan speed | W/K |
| $T_{\text{basin}}$ | Temperature of the water in the cooling tower basin | °C |
| $T_{\text{eff}}$ | Effective ambient temperature (blend of dry-bulb and wet-bulb) | °C |

**Effective Temperature Calculation:**

$$
T_{\text{eff}} = \alpha \cdot T_{\text{wet-bulb}} + (1 - \alpha) \cdot T_{\text{dry-bulb}}
$$

| Symbol | Description | Units |
|--------|-------------|-------|
| $T_{\text{wet-bulb}}$ | Wet-bulb temperature (lower bound for evaporative cooling) | °C |
| $T_{\text{dry-bulb}}$ | Dry-bulb (ambient air) temperature | °C |
| $\alpha$ | Evaporative fraction (0 = dry operation, 1 = full evaporative) | — |

> **Note:** Water consumption increases with higher evaporative fractions. The model tracks cumulative water usage in liters for WUE (Water Usage Effectiveness) calculations.

## Server, Rack, and DataCenter

Higher-level aggregation models:

- **`ServerNode`** wraps GPUs + CPU + memory + NIC power draws
- **`RackModel`** aggregates servers sharing a rack coolant manifold
- **`DataCenterModel`** combines racks, CDU, and cooling tower

```python
from compopt.physics import build_default_datacenter

dc = build_default_datacenter(n_racks=2, servers_per_rack=4, gpus_per_server=2)
dc.step(dt=5.0)
print(f"PUE = {dc.PUE:.3f}, WUE = {dc.WUE_L_per_kWh:.3f} L/kWh")
```

## Workload Profiles

Synthetic power profiles for benchmarking:

| Profile         | Description                                    |
|-----------------|------------------------------------------------|
| `sinusoidal`    | Smooth oscillation between idle and peak       |
| `step`          | Alternating idle/full-load steps               |
| `burst`         | Short high-power bursts                        |
| `stochastic`    | Random walk with Ornstein-Uhlenbeck process    |
| `mixed`         | Weighted combination of multiple profiles      |
| `trace`         | Replay from recorded data arrays               |

```python
from compopt.physics import make_sinusoidal_profile, make_stochastic_profile

profile = make_stochastic_profile(nx=4, ny=4, seed=42)
```
