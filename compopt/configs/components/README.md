# CompOpt Configuration System

This directory contains centralized configuration files for all physical components in the CompOpt thermal simulation framework. Proper configuration of these parameters is **critical** for obtaining meaningful, numerically stable simulation results.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Files](#configuration-files)
3. [Parameter Sizing Guide](#parameter-sizing-guide)
4. [Common Issues and Solutions](#common-issues-and-solutions)
5. [Validation](#validation)
6. [Examples](#examples)

---

## Quick Start

```python
from compopt.configs.validator import validate_chip_env_config, validate_system_config

# Validate chip-level configuration
result = validate_chip_env_config(
    gpu_preset="H100_SXM",
    m_dot_min=0.05,
    m_dot_max=0.5,
    c_p=800.0,
    dt=0.5,
    target_temp_C=70.0
)
print(result)

# Validate full datacenter configuration
result = validate_system_config(
    gpu_preset="H100_SXM",
    n_gpus=16,
    cdu_preset="small_datacenter",
    tower_preset="small_tower",
    dt=5.0
)
print(result)
```

---

## Configuration Files

### `components/gpu_presets.json`

GPU thermal model parameters including:
- **TDP**: Thermal Design Power [W]
- **Thermal capacitances** (C_*): Heat storage capacity [J/K]
- **Thermal resistances** (R_*): Heat flow resistance [K/W]
- **Temperature limits** (T_max_*): Safe operating limits [°C]

### `components/coolant_loops.json`

Coolant loop parameters:
- **T_in_C**: Inlet temperature [°C]
- **m_dot_***: Flow rate range [kg/s]
- **c_p_J_kgK**: Specific heat capacity [J/(kg·K)]
- **V_m3**: Control volume [m³]

### `components/cdu.json`

Coolant Distribution Unit parameters:
- **UA_W_K**: Heat exchanger effectiveness [W/K]
- **Loop volumes**: Thermal buffer capacity [m³]

### `components/cooling_tower.json`

Cooling tower parameters:
- **fan_UA_W_K**: Air-side heat transfer [W/K]
- **basin_C_J_K**: Thermal capacitance [J/K]
- **evap_fraction**: Evaporative cooling fraction [0-1]

### `components/simulation.json`

Numerical integration parameters:
- **dt**: Timestep [s]
- **episode_length_s**: Episode duration [s]

### `components/workloads.json`

Workload profile parameters for different scenarios.

---

## Parameter Sizing Guide

### The Golden Rule

> **All thermal parameters must be sized relative to each other.** Changing one parameter in isolation often breaks the simulation.

### Key Relationships

#### 1. Thermal Resistance Sizing

The thermal resistance from die to coolant determines the steady-state temperature:

```
T_die = T_coolant + R_effective × P_total
```

For proper operation:
- **At max flow + TDP**: T_die should be 65-70°C (comfortable below target)
- **At min flow + TDP**: T_die should be 80-85°C (near but below T_max)

**Sizing formula:**
```
R_tile_to_coolant = (T_target - T_coolant_in) × n_tiles / P_TDP

Example for H100:
  R_tile = (70°C - 40°C) × 16 tiles / 700W = 0.69 K/W per tile
```

#### 2. Control Authority

The flow rate range determines how much temperature control you have:

```
Control_Range = T_die(m_dot_min) - T_die(m_dot_max)
```

- **Minimum range**: 5°C (barely usable)
- **Good range**: 10-20°C
- **Excessive range**: >30°C (may cause instability)

To increase control authority:
- Increase `m_dot_max / m_dot_min` ratio
- Decrease coolant `c_p` (use effective values 600-1200 J/(kg·K))

#### 3. Numerical Stability

For RK4 integration stability:

```
dt < 2.78 × τ_min × 0.9

where τ_min = C_min / G_max = C_min × R_min
```

**Quick rules:**
- Chip level: dt ≤ 2.0 s (use 0.5 s for safety)
- Rack level: dt ≤ 5.0 s (use 1.0 s for safety)
- Datacenter: dt ≤ 30 s (use 5.0 s for safety)

#### 4. CDU Stability

The CDU heat exchanger can become unstable if:

```
UA × dt / C_loop > 0.5
```

**Sizing formula:**
```
C_loop_min = 2 × UA × dt

Example: UA=5000 W/K, dt=5s
  C_loop_min = 2 × 5000 × 5 = 50,000 J/K
  V_min = C / (ρ × c_p) = 50000 / (1000 × 4000) = 0.0125 m³ = 12.5 L
```

#### 5. Cooling Tower Stability

Similar stability criterion:

```
fan_UA × dt / basin_C < 0.5
```

**Sizing formula:**
```
basin_C_min = 2 × fan_UA × dt

Example: fan_UA=3000 W/K, dt=5s
  basin_C_min = 2 × 3000 × 5 = 30,000 J/K
  Recommended: 10× margin → 300,000 J/K
```

---

## Common Issues and Solutions

### Issue 1: Temperature Explodes to Infinity

**Symptoms:**
- T_die goes to 1000s or millions of degrees
- NaN or Inf values appear

**Causes:**
1. Timestep too large (numerical instability)
2. Cooling capacity insufficient (thermal runaway)

**Solutions:**
```python
# Check 1: Reduce timestep
dt = 0.5  # instead of 1.0 or 2.0

# Check 2: Increase thermal resistances (slower dynamics)
R_tile_to_coolant = 0.8  # instead of 0.02

# Check 3: Verify cooling capacity
result = validate_chip_thermal_feasibility(gpu_config, coolant_config)
```

### Issue 2: Temperature Oscillates Wildly (CDU/Tower)

**Symptoms:**
- Basin temperature swings between -273°C and 10000°C
- PUE/WUE values are unrealistic (>100)

**Causes:**
1. Heat exchanger UA too high relative to thermal mass
2. Timestep too large for the plant dynamics

**Solutions:**
```python
# Increase CDU loop volumes
cdu_config["hot_loop_V_m3"] = 0.05  # instead of 0.005
cdu_config["cold_loop_V_m3"] = 0.10  # instead of 0.01

# Or reduce UA
cdu_config["UA_W_K"] = 5000  # instead of 50000
```

### Issue 3: Controller Saturates (Flat Response)

**Symptoms:**
- Flow stays at max (or min) constantly
- Temperature doesn't respond to workload changes
- Plots show flat lines

**Causes:**
1. System operating far from target temperature
2. Insufficient control authority

**Solutions:**
```python
# Option 1: Adjust target to operating range
target_C = 70.0  # instead of 85.0 if system runs at 65-75°C

# Option 2: Reduce coolant c_p for larger temp swings
c_p_J_kgK = 800.0  # instead of 4000.0

# Option 3: Adjust PID gains
Kp = 0.10  # more aggressive proportional gain
```

### Issue 4: Unrealistic WUE Values

**Symptoms:**
- WUE > 100 L/kWh (should be 1-3)

**Causes:**
1. Basin temperature instability causing excessive "evaporation"
2. Energy accumulator not tracking correctly

**Solutions:**
```python
# Increase basin thermal mass
tower_config["basin_C_J_K"] = 2000000  # 2 MJ/K

# Reduce fan_UA for stability
tower_config["fan_UA_W_K"] = 3000  # instead of 30000
```

---

## Validation

### Using the Validator

```python
from compopt.configs.validator import (
    validate_chip_env_config,
    validate_system_config,
    ValidationResult,
    Severity
)

# Validate and check results
result = validate_chip_env_config(
    gpu_preset="H100_SXM",
    dt=0.5,
    target_temp_C=70.0
)

if not result.is_valid:
    print("Configuration has errors:")
    for issue in result.errors:
        print(f"  {issue}")
else:
    print("Configuration is valid")
    # Print warnings if any
    for issue in result.warnings:
        print(f"  Warning: {issue}")
```

### Command-Line Validation

```bash
# Validate with defaults
python -m compopt.configs.validator

# Validate specific configuration
python -m compopt.configs.validator \
    --gpu H100_SXM \
    --n-gpus 16 \
    --cdu small_datacenter \
    --tower small_tower \
    --dt 5.0

# List available presets
python -m compopt.configs.validator --list-presets
```

---

## Examples

### Example 1: Small AI Cluster (8 H100s)

```json
{
    "gpu_preset": "H100_SXM",
    "n_racks": 2,
    "gpus_per_rack": 4,
    "cdu_preset": "small_datacenter",
    "tower_preset": "small_tower",
    "simulation": {
        "dt": 5.0,
        "episode_length_s": 1800
    }
}
```

### Example 2: Single GPU Chip Experiment

```python
import gymnasium as gym
import compopt

# Use validated parameters
env = compopt.make(
    "ChipThermal-v0",
    gpu_preset="H100_SXM",
    dt=0.5,
    m_dot_min=0.05,
    m_dot_max=0.5,
    target_hotspot_C=70.0
)
```

### Example 3: Custom GPU Configuration

```python
from compopt.configs.validator import validate_chip_thermal_feasibility

# Define custom GPU
custom_gpu = {
    "tdp_W": 500,
    "nx_tiles": 4,
    "ny_tiles": 4,
    "C_tile_J_K": 10.0,
    "R_tile_to_coolant_K_W": 1.0,  # Sized for 500W
    "T_max_die_C": 85.0
}

# Define coolant config
coolant = {
    "T_in_C": 35.0,
    "m_dot_min_kg_s": 0.05,
    "m_dot_max_kg_s": 0.4,
    "c_p_J_kgK": 1000.0
}

# Validate before using
issues = validate_chip_thermal_feasibility(custom_gpu, coolant, target_temp_C=72.0)
for issue in issues:
    print(issue)
```

---

## Physics Reference

### Thermal RC Network

The GPU die is modeled as an RC network:
```
C × dT/dt = P_in - (T - T_coolant) / R
```

At steady state (dT/dt = 0):
```
T = T_coolant + R × P_in
```

### Coolant Loop

Energy balance:
```
C_th × dT_coolant/dt = Q_in - ṁ × c_p × (T_coolant - T_in)
```

Where `C_th = V × ρ × c_p`.

### Heat Exchanger (CDU)

Heat transfer:
```
Q_hx = UA × (T_hot - T_cold)
```

### Evaporative Cooling

Water consumption:
```
ṁ_water = η_evap × Q_reject / h_fg
```

Where `h_fg = 2.26 MJ/kg` is latent heat of vaporization.

---

## Troubleshooting Checklist

Before running simulations, verify:

- [ ] `validate_chip_env_config()` returns no errors
- [ ] Temperature at max flow is below target by 5-10°C
- [ ] Temperature at min flow is below T_max
- [ ] Timestep is at least 2× below stability limit
- [ ] CDU loop volumes give stability ratio < 0.3
- [ ] Cooling tower basin_C gives stability ratio < 0.3
- [ ] Workload period is comparable to thermal time constant

---

## Contributing

When adding new GPU presets or configurations:

1. Calculate thermal resistances using the sizing formulas
2. Run the validator on your configuration
3. Add a test case to `tests/test_config_validation.py`
4. Document any special considerations

---

## Step-by-Step: Customizing Configuration Parameters

This section provides a complete walkthrough for customizing CompOpt parameters and validating that your configuration will produce stable, meaningful simulations.

### Overview

The customization workflow is:

```
1. Identify what you want to change
2. Edit the appropriate JSON file(s)
3. Run validation to check feasibility and stability
4. Fix any issues identified
5. Run a quick simulation test
6. Proceed with your experiments
```

---

### Step 1: Identify Your Customization Goal

| Goal | Files to Modify |
|------|-----------------|
| Different GPU (H100 → B200) | `gpu_presets.json` |
| Change cooling capacity | `coolant_loops.json` |
| Adjust datacenter scale | `cdu.json`, `cooling_tower.json` |
| Change simulation speed/accuracy | `simulation.json` |
| Different workload patterns | `workloads.json` |
| Increase HVAC power consumption | `cdu.json`, `cooling_tower.json` |

---

### Step 2: Edit the JSON Configuration

#### Example A: Creating a Custom GPU Preset

**File:** `compopt/configs/components/gpu_presets.json`

Add your custom GPU after the existing presets:

```json
{
    "H100_SXM": { ... },
    "B200": { ... },
    
    "MY_CUSTOM_GPU": {
        "tdp_W": 500,
        "die_area_mm2": 600,
        "nx_tiles": 4,
        "ny_tiles": 4,
        "C_tile_J_K": 10.0,
        "C_hbm_J_K": 70.0,
        "C_vrm_J_K": 35.0,
        "C_pcb_J_K": 180.0,
        "R_lateral_tile_K_W": 0.6,
        "R_tile_to_pcb_K_W": 0.35,
        "R_tile_to_coolant_K_W": 0.9,
        "R_hbm_to_pcb_K_W": 0.45,
        "R_vrm_to_pcb_K_W": 0.45,
        "R_hbm_to_coolant_K_W": 1.3,
        "R_vrm_to_coolant_K_W": 1.6,
        "R_pcb_to_coolant_K_W": 2.2,
        "T_max_die_C": 85.0,
        "T_max_hbm_C": 95.0,
        "T_max_vrm_C": 105.0
    }
}
```

**Sizing tip:** Use the formula for R_tile_to_coolant:
```
R_tile = (T_target - T_coolant_in) × n_tiles / TDP
       = (70 - 40) × 16 / 500 = 0.96 K/W
```

#### Example B: Adjusting Cooling Capacity

**File:** `compopt/configs/components/coolant_loops.json`

```json
"chip_cooling": {
    "T_in_C": 35.0,           // Lower inlet = more cooling headroom
    "m_dot_min_kg_s": 0.02,   // Lower min = wider control range
    "m_dot_max_kg_s": 0.8,    // Higher max = more cooling capacity
    "c_p_J_kgK": 600.0,       // Lower c_p = larger temp swings (more control authority)
    "V_m3": 2.0e-4,           // Larger volume = more thermal inertia
    "rho_kg_m3": 1000.0
}
```

#### Example C: Making HVAC Power More Significant

**File:** `compopt/configs/components/cdu.json`

```json
"high_hvac_datacenter": {
    "_description": "CDU with higher parasitic power (older facility)",
    "UA_W_K": 8000.0,
    "T_facility_C": 28.0,
    "pump_power_W_per_kgs": 2000.0,   // 4× higher than default!
    "hot_loop_V_m3": 0.08,
    "cold_loop_V_m3": 0.15
}
```

**File:** `compopt/configs/components/cooling_tower.json`

```json
"high_power_tower": {
    "_description": "Less efficient cooling tower",
    "T_ambient_C": 30.0,
    "T_wetbulb_C": 22.0,
    "fan_UA_W_K": 4000.0,
    "basin_C_J_K": 3000000.0,
    "evap_fraction": 0.70,
    "latent_heat_J_kg": 2260000.0,
    "fan_power_W": 25000.0    // 12× higher than default!
}
```

---

### Step 3: Run Validation

After editing JSON files, validate your configuration before running simulations.

#### Option A: Python Script Validation

Create a file `validate_my_config.py`:

```python
#!/usr/bin/env python3
"""Validate custom configuration before running experiments."""

from compopt.configs import (
    validate_chip_env_config,
    validate_system_config,
    load_component_config,
    Severity
)

# ─── Chip-Level Validation ───────────────────────────────────────────────────

print("=" * 60)
print("CHIP-LEVEL CONFIGURATION VALIDATION")
print("=" * 60)

result = validate_chip_env_config(
    gpu_preset="H100_SXM",      # Change to your GPU preset
    m_dot_min=0.05,
    m_dot_max=0.5,
    c_p=800.0,
    dt=0.5,                     # Your timestep
    target_temp_C=70.0          # Your target temperature
)

print(result)
print()

if not result.is_valid:
    print("❌ CHIP CONFIG HAS ERRORS - FIX BEFORE PROCEEDING")
    for issue in result.errors:
        print(f"   {issue}")
else:
    print("✅ Chip configuration is valid")
    if result.warnings:
        print("   Warnings:")
        for w in result.warnings:
            print(f"   ⚠️  {w.message}")

# ─── System-Level Validation ─────────────────────────────────────────────────

print()
print("=" * 60)
print("SYSTEM-LEVEL CONFIGURATION VALIDATION")
print("=" * 60)

result = validate_system_config(
    gpu_preset="H100_SXM",
    n_gpus=16,                          # Your GPU count
    cdu_preset="small_datacenter",      # Your CDU preset
    tower_preset="small_tower",         # Your tower preset
    dt=5.0,                             # Your datacenter timestep
    target_temp_C=75.0
)

print(result)
print()

if not result.is_valid:
    print("❌ SYSTEM CONFIG HAS ERRORS - FIX BEFORE PROCEEDING")
else:
    print("✅ System configuration is valid")

# ─── Summary ─────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
```

Run it:
```bash
cd /path/to/compopt
source .venv/bin/activate
python validate_my_config.py
```

#### Option B: Command-Line Validation

```bash
# Validate with specific parameters
python -m compopt.configs.validator \
    --gpu H100_SXM \
    --n-gpus 16 \
    --cdu small_datacenter \
    --tower small_tower \
    --dt 5.0 \
    --target-temp 75.0

# List all available presets
python -m compopt.configs.validator --list-presets
```

#### Option C: Interactive Python Validation

```python
>>> from compopt.configs import validate_chip_env_config
>>> result = validate_chip_env_config(gpu_preset="H100_SXM", dt=0.5)
>>> print(result)
>>> result.is_valid
True
>>> result.warnings
[...]
```

---

### Step 4: Interpret Validation Results

#### Understanding Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| `INFO` | Informational message | No action needed |
| `WARNING` | May cause suboptimal behavior | Consider fixing |
| `ERROR` | Will likely cause simulation issues | Must fix |
| `CRITICAL` | Will definitely fail | Must fix immediately |

#### Common Validation Messages and Fixes

**Error: "T_die exceeds thermal limit at max flow"**
```
Cause: Cooling capacity insufficient even at maximum
Fix: Decrease R_tile_to_coolant_K_W or increase m_dot_max
```

**Error: "Timestep exceeds stability limit"**
```
Cause: dt too large for thermal dynamics
Fix: Reduce dt or increase thermal capacitances (C values)
```

**Warning: "Temperature range between min/max flow is only X°C"**
```
Cause: Controller has limited authority
Fix: Increase m_dot_max/m_dot_min ratio or decrease c_p
```

**Error: "UA×dt/C > 0.5 - CDU unstable"**
```
Cause: Heat exchanger dynamics too fast for timestep
Fix: Increase loop volumes (V_m3) or decrease UA_W_K
```

---

### Step 5: Run a Quick Simulation Test

After validation passes, run a short simulation to verify behavior:

```python
#!/usr/bin/env python3
"""Quick simulation test after configuration changes."""

import numpy as np
import compopt
from compopt.agents.baselines import PIDCoolingAgent

# Create environment with your parameters
env = compopt.make(
    "ChipThermal-v0",
    gpu_preset="H100_SXM",
    dt=0.5,
    workload="sinusoidal",
    workload_period_s=60.0
)

agent = PIDCoolingAgent(target_C=70.0)
obs, _ = env.reset(seed=42)
agent.reset()

# Run short episode
temps, flows, powers = [], [], []
for step in range(200):
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    temps.append(info['T_hotspot_C'])
    flows.append(info['flow_kg_s'])
    powers.append(info['P_total_W'])
    
    if terminated or truncated:
        break

env.close()

# Check results
print("Simulation Test Results:")
print(f"  Temperature range: {min(temps):.1f} - {max(temps):.1f} °C")
print(f"  Flow range: {min(flows):.3f} - {max(flows):.3f} kg/s")
print(f"  Power range: {min(powers):.0f} - {max(powers):.0f} W")

# Sanity checks
assert not any(np.isnan(temps)), "NaN temperatures detected!"
assert not any(np.isinf(temps)), "Infinite temperatures detected!"
assert all(0 < t < 200 for t in temps), f"Unrealistic temperatures: {min(temps)}-{max(temps)}"
assert all(0 < f < 10 for f in flows), f"Unrealistic flows: {min(flows)}-{max(flows)}"

print("\n✅ Simulation test passed - configuration is working correctly")
```

---

### Step 6: Run the Full Test Suite

Ensure your changes don't break existing functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run only configuration validation tests
python -m pytest tests/test_config_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=compopt --cov-report=term-missing
```

---

### Complete Example: Adding a New Facility Configuration

Here's a complete example of adding a "legacy datacenter" configuration with higher HVAC power:

#### 1. Add CDU preset to `cdu.json`:
```json
"legacy_datacenter": {
    "_description": "Older facility with less efficient cooling",
    "UA_W_K": 6000.0,
    "T_facility_C": 32.0,
    "pump_power_W_per_kgs": 1500.0,
    "hot_loop_V_m3": 0.06,
    "cold_loop_V_m3": 0.12
}
```

#### 2. Add cooling tower preset to `cooling_tower.json`:
```json
"legacy_tower": {
    "_description": "Older cooling tower with higher fan power",
    "T_ambient_C": 28.0,
    "T_wetbulb_C": 20.0,
    "fan_UA_W_K": 4500.0,
    "basin_C_J_K": 2500000.0,
    "evap_fraction": 0.65,
    "latent_heat_J_kg": 2260000.0,
    "fan_power_W": 20000.0
}
```

#### 3. Validate:
```bash
python -m compopt.configs.validator \
    --gpu H100_SXM \
    --n-gpus 16 \
    --cdu legacy_datacenter \
    --tower legacy_tower \
    --dt 5.0
```

#### 4. Expected output:
```
Configuration validation: VALID
[INFO] System Summary: Total IT load: 11.2 kW from 16 GPUs
[INFO] CDU Sizing: Expected temperature difference across HX: 1.9°C
[INFO] Water Usage: Estimated WUE: 1.8 L/kWh (typical range: 1-3)
[INFO] Numerical Stability: Timestep dt=5.0s is safely below limit...
```

#### 5. Use in simulation:
```python
env = compopt.make(
    "DataCenter-v0",
    n_racks=2,
    servers_per_rack=4,
    gpus_per_server=2,
    dt=5.0
)
# The environment will use updated cooling parameters
```

---

### Quick Reference: Validation Commands

```bash
# Chip-level validation
python -c "
from compopt.configs import validate_chip_env_config
result = validate_chip_env_config(gpu_preset='H100_SXM', dt=0.5)
print(result)
print('Valid:', result.is_valid)
"

# System-level validation  
python -c "
from compopt.configs import validate_system_config
result = validate_system_config(
    gpu_preset='H100_SXM', n_gpus=16,
    cdu_preset='small_datacenter', tower_preset='small_tower', dt=5.0
)
print(result)
"

# List all presets
python -c "
from compopt.configs import list_component_presets
for comp in ['gpu_presets', 'cdu', 'cooling_tower']:
    print(f'{comp}: {list_component_presets(comp)}')
"
```

---

### Validation Checklist

Before running experiments with custom configurations:

- [ ] Edited JSON files are valid JSON (no syntax errors)
- [ ] `validate_chip_env_config()` returns `is_valid=True`
- [ ] `validate_system_config()` returns `is_valid=True`
- [ ] No `ERROR` or `CRITICAL` issues in validation output
- [ ] Reviewed and addressed any `WARNING` messages
- [ ] Short simulation test runs without NaN/Inf values
- [ ] Temperature stays in realistic range (20-150°C)
- [ ] Flow rates stay in realistic range (0.01-10 kg/s)
- [ ] PUE is in expected range (1.1-3.0)
- [ ] Full test suite passes (`pytest tests/`)
