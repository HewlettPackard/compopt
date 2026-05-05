# JointDCFlat-v0 Training Guide

Quick setup for training CMO-SAC on **JointDCFlat-v0** (joint scheduling + cooling).

## Quick Start

```bash
# Start training (500k timesteps, ~90-120 min)
./joint.sh train

# Check progress while running
./joint.sh monitor      # Generates joint_training_progress.png

# Check if still running
./joint.sh status
```

## All Commands

| Command | Description |
|---------|-------------|
| `./joint.sh train` | Start training in background screen |
| `./joint.sh monitor` | Generate progress visualization |
| `./joint.sh status` | Check training status and recent metrics |
| `./joint.sh attach` | Attach to live training (Ctrl+A D to detach) |
| `./joint.sh stop` | Stop training |

## Training Details

- **Environment**: JointDCFlat-v0 (scheduling + cooling combined)
- **Algorithm**: CMO-SAC with normalized observations & rewards
- **Timesteps**: 500,000 (~90-120 minutes on H100)
- **Observation Space**: ~30+ dimensions (workload + thermal state)
- **Action Space**: 4D (3D for scheduling bins + 1D for cooling)
- **Multi-Objective**: Job completion time + energy + violations

## Why JointDCFlat-v0?

This is the **full problem**:
- Combines job scheduling with thermal management
- Most realistic scenario (decisions interact)
- Hardest environment but most impactful
- Tests if CMO-SAC can handle high-dimensional multi-objective

## What to Expect

**Good signs:**
- Training reward improves over time
- Evaluation scores show learning
- Both scheduling AND cooling adapt

**Challenges:**
- Harder than ChipThermal (more dimensions)
- Harder than DataCenter (scheduling added)
- May need 300k+ steps to converge
- Exploration is challenging

**If it doesn't work:**
- Try shorter horizon (reduce episode length)
- Try curriculum: easier workloads → harder workloads
- Consider hierarchical: separate scheduling/cooling agents

## Files Created

Just 2 files:
1. `joint.sh` - Training wrapper
2. `JOINT.md` - This guide

Uses existing `run_cmo_sac.py` with normalized observations and rewards.
