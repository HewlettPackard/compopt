# ChipThermal-v0 Training - Quick Reference

## ✅ Training is Running with Normalization!

**IMPORTANT**: Training now uses both observation AND reward normalization for stable learning.

Started in background screen: `chip_training`

## Commands

```bash
# Check training status
./chip.sh status

# View progress plot
./chip.sh monitor

# Watch live output  
./chip.sh attach
# (Press Ctrl+A then D to detach)

# Stop training
./chip.sh stop
```

## What to Expect

- **Training time**: ~30-45 minutes (200k timesteps)
- **Environment**: ChipThermal-v0 (easiest in CompOpt)
- **Algorithm**: CMO-SAC (multi-objective constrained RL)
- **Output directory**: `experiment-cmo_sac/results/cmo_sac/single_training/ChipThermal-v0_seed0/`

## After Training Completes

1. Check status: `./chip.sh status` (will show "Training completed")
2. View final plot: `./chip.sh monitor`
3. Results saved in: `experiment-cmo_sac/results/cmo_sac/single_training/ChipThermal-v0_seed0/`

## Why ChipThermal vs RackCooling?

RackCooling-v0 was very difficult because:
- Safety-critical constraints (HBM temperature)
- Weak reward signal for adaptation
- Exploration problem (agent learns constant policies)

ChipThermal-v0 should be easier because:
- Single component (just one chip)
- Clear temperature target
- Better reward shaping
- Standard RL should work

This is a sanity check that CMO-SAC works on simpler problems!

## Files Created (Minimal!)

Only 2 new files:
- `chip.sh` - Wrapper script for easy training
- `CHIP_TRAINING.md` / `CHIP_QUICKSTART.md` - Documentation

Everything else reuses existing CMO-SAC code!
