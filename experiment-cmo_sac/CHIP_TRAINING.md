# ChipThermal-v0 Training with CMO-SAC

**Simple, clean interface** for training on the easiest CompOpt environment.

## Quick Start

```bash
# Start training (runs in background)
./chip.sh train

# Monitor progress (generates plot)
./chip.sh monitor

# Check status
./chip.sh status
```

## All Commands

- `./chip.sh train` - Start training in background screen (200k timesteps)
- `./chip.sh monitor` - Generate progress plot using visualize_current_training.sh
- `./chip.sh status` - Check if training is running + show recent output
- `./chip.sh attach` - See live output (Press Ctrl+A then D to detach)
- `./chip.sh stop` - Stop training

## Training Details

- **Environment**: ChipThermal-v0 (single GPU thermal control)
- **Algorithm**: CMO-SAC (existing implementation)
- **Duration**: 200k timesteps (~30-45 minutes)
- **Output**: `experiment-cmo_sac/results/cmo_sac/single_training/ChipThermal-v0_seed0/`

## Files

Only 2 new files created:
- `chip.sh` - Training wrapper script
- `CHIP_TRAINING.md` - This file

Everything else uses existing CMO-SAC infrastructure!

## Why ChipThermal-v0?

This is the **easiest** environment in CompOpt:
- Single chip (not rack/datacenter)  
- Clear objective (temperature control)
- Well-shaped rewards
- Should work out of the box with standard RL

Unlike RackCooling-v0, this should converge properly!
