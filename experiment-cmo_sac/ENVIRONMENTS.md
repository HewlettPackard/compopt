# CompOpt Training Environments - Quick Reference

All environments use CMO-SAC with normalized observations & rewards.

## Available Environments

| Environment | Script | Default Steps | Difficulty | Focus Area |
|-------------|--------|---------------|------------|------------|
| ChipThermal-v0 | `./chip.sh` | 200k | ⭐ Easy | Single chip cooling |
| DataCenter-v0 | `./datacenter.sh` | 300k | ⭐⭐ Medium | Rack + infrastructure cooling |
| Scheduling-v0 | `./scheduling.sh` | 400k | ⭐⭐ Medium | Pure job scheduling |
| JointDCFlat-v0 | `./joint.sh` | 500k | ⭐⭐⭐ Hard | Scheduling + cooling |

## Quick Commands

All scripts follow the same pattern:

```bash
cd /lustre/naug/cmpopt/experiment-cmo_sac

# Training
./<env>.sh train [timesteps]     # Start training
./<env>.sh monitor               # Generate plot
./<env>.sh status                # Check status
./<env>.sh attach                # View live (Ctrl+A D to exit)
./<env>.sh stop                  # Stop training
```

## Environment Details

### ChipThermal-v0 (Easiest)
- **Action**: 1D continuous (cooling power)
- **Observation**: 7D (temperatures, power, flow)
- **Metrics**: Temperature, PUE, WUE, thermal violations
- **Goal**: Keep single chip at target temperature efficiently
- **Best for**: Testing RL algorithms, quick experiments

### DataCenter-v0 (Medium)
- **Action**: 3D continuous (rack flow, CDU, facility fan)
- **Observation**: ~20D (rack temps, power, cooling state)
- **Metrics**: Temperature, PUE, WUE, energy, violations
- **Goal**: Balance cooling efficiency vs thermal safety
- **Best for**: Multi-actuator cooling control

### Scheduling-v0 (Medium)
- **Action**: Discrete (job priority selection)
- **Observation**: 12D (utilization, queue, wait times)
- **Metrics**: Utilization, throughput, wait time, SLA violations
- **Goal**: Maximize cluster utilization, minimize wait times
- **Best for**: Resource allocation, discrete RL
- **Note**: No thermal/cooling metrics

### JointDCFlat-v0 (Hardest)
- **Action**: 4D continuous (3D cooling + 1D scheduling)
- **Observation**: ~30D (thermal + scheduling state)
- **Metrics**: All of the above (thermal + scheduling)
- **Goal**: Optimize both cooling and workload placement
- **Best for**: Full datacenter optimization, research

## Training Times (H100)

| Environment | 200k steps | 500k steps | 1M steps |
|-------------|-----------|-----------|----------|
| ChipThermal | ~30 min | ~75 min | ~2.5 hrs |
| DataCenter | ~45 min | ~2 hrs | ~4 hrs |
| Scheduling | ~60 min | ~2.5 hrs | ~5 hrs |
| JointDCFlat | ~90 min | ~3.5 hrs | ~7 hrs |

*Times are approximate and depend on hardware/load*

## Custom Settings

### Custom timesteps
```bash
./chip.sh train 500000              # 500k steps
./scheduling.sh train 1000000       # 1M steps
```

### Custom results directory
```bash
CHIP_RESULTS_DIR=my_runs ./chip.sh train
SCHEDULING_RESULTS_DIR=exps ./scheduling.sh train
```

### Multiple trainings
```bash
# Run all environments simultaneously
./chip.sh train &
./datacenter.sh train &
./scheduling.sh train &
./joint.sh train &

# Monitor all
./chip.sh monitor
./datacenter.sh monitor
./scheduling.sh monitor
./joint.sh monitor
```

## Baseline Comparisons

Each environment has built-in baselines:

**ChipThermal-v0:**
- PID controller
- Rule-based bang-bang
- Constant cooling

**DataCenter-v0:**
- Rule-based proportional control
- Constant actions

**Scheduling-v0:**
- FCFS (First-Come-First-Served)
- Random priority

**JointDCFlat-v0:**
- FCFS + Rule-based cooling
- Random combinations

## Metrics Summary

| Metric | ChipThermal | DataCenter | Scheduling | JointDCFlat |
|--------|-------------|------------|------------|-------------|
| Reward | ✓ | ✓ | ✓ | ✓ |
| Temperature | ✓ | ✓ | ✗ | ✓ |
| PUE | ✓ | ✓ | ✗ | ✓ |
| WUE | ✓ | ✓ | ✗ | ✓ |
| Violations | ✓ | ✓ | ✗ | ✓ |
| Utilization | ✗ | ✗ | ✓ | ✓ |
| Jobs Completed | ✗ | ✗ | ✓ | ✓ |
| Wait Time | ✗ | ✗ | ✓ | ✓ |
| SLA Violations | ✗ | ✗ | ✓ | ✓ |

## Documentation

- **CHIP_TRAINING.md** / **CHIP_QUICKSTART.md** - ChipThermal details
- **DATACENTER.md** - DataCenter details
- **SCHEDULING.md** - Scheduling details
- **JOINT.md** - JointDCFlat details

## Screen Sessions

Each environment runs in its own screen session:
- `chip_training`
- `datacenter_training`
- `scheduling_training`
- `joint_training`

View all: `screen -ls`

## Tips

1. **Start with ChipThermal** - Easiest, fastest, good for testing
2. **Try Scheduling** - Different from cooling (discrete actions)
3. **Scale to DataCenter** - More complex cooling
4. **Challenge with JointDCFlat** - Combines everything

5. **Monitor regularly** - Run `./env.sh monitor` every 10-15 minutes
6. **Check status** - Use `./env.sh status` to verify training progress
7. **Compare results** - Evaluate against baselines after training

8. **Save results** - Use custom directories for different experiments
9. **Multiple seeds** - Run same env with different seeds for robustness
10. **Adjust timesteps** - Longer training often improves performance
