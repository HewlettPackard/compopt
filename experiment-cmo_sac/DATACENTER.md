# DataCenter-v0 Training with CMO-SAC

**Multi-dimensional control** for datacenter-wide cooling, power, and thermal management.

## Quick Start

```bash
# Start training (runs in background, 300k timesteps)
./datacenter.sh train

# Monitor progress (generates plot)
./datacenter.sh monitor

# Check status
./datacenter.sh status
```

## All Commands

- `./datacenter.sh train` - Start training in background (300k timesteps, ~45-60 min)
- `./datacenter.sh monitor` - Generate progress plot 
- `./datacenter.sh status` - Check if training is running + show recent output
- `./datacenter.sh attach` - See live output (Press Ctrl+A then D to detach)
- `./datacenter.sh stop` - Stop training

## Training Details

- **Environment**: DataCenter-v0 (rack + CDU + fan control)
- **Algorithm**: CMO-SAC (with obs + reward normalization)
- **Actions**: 3D continuous (rack flow, CDU setpoint, fan speed)
- **Observations**: ~20D (temps, power, flow rates)
- **Duration**: 300k timesteps (~45-60 minutes)
- **Output**: `experiment-cmo_sac/results/cmo_sac_DataCenter-v0_<timestamp>/`

## Why DataCenter-v0?

**Easier than RackCooling-v0** because:
- Better reward shaping (PUE, energy costs)
- More observable state
- Multiple control levers (not just flow)
- Energy costs are continuous (not binary violations)

**Harder than ChipThermal-v0** because:
- Multi-dimensional actions (3D)
- More complex dynamics (rack + infrastructure)
- Multiple objectives to balance

## What to Expect

DataCenter-v0 has:
- Multiple objectives: thermal control + energy efficiency + water usage
- Constraints: Keep temperatures safe while minimizing PUE
- CMO-SAC should learn to balance these trade-offs

Unlike RackCooling which got stuck, DataCenter should show steady improvement!

## Files Created

Only 2 new files:
- `datacenter.sh` - Training wrapper
- `DATACENTER.md` - This documentation

Uses existing CMO-SAC code with normalized obs & rewards!
