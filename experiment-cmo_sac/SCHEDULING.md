# Scheduling-v0 Training Guide

Quick setup for training **DQN** on **Scheduling-v0** (pure job scheduling).

⚠️ **Note**: Scheduling-v0 uses **discrete actions**, so we train with **DQN** (not CMO-SAC).

## Quick Start

```bash
cd /lustre/naug/cmpopt/experiment-cmo_sac

# Start training (400k timesteps, ~60-90 min)
./scheduling.sh train

# Check progress while running
./scheduling.sh monitor      # Generates scheduling_training_progress.png

# Check if still running
./scheduling.sh status
```

## All Commands

| Command | Description |
|---------|-------------|
| `./scheduling.sh train [timesteps]` | Start training in background (default: 400k) |
| `./scheduling.sh monitor` | Generate progress visualization |
| `./scheduling.sh status` | Check training status and recent metrics |
| `./scheduling.sh attach` | Attach to live training (Ctrl+A D to detach) |
| `./scheduling.sh stop` | Stop training |

## Training Details

- **Environment**: Scheduling-v0 (pure job scheduling, no cooling)
- **Algorithm**: DQN (Deep Q-Network) - for discrete actions
  - **Why DQN?** CMO-SAC only supports continuous actions. Scheduling-v0 has discrete actions (select which job to prioritize), so we use DQN instead.
- **Timesteps**: 400,000 (default, ~60-90 minutes on H100)
- **Observation Space**: 12D (scheduler metrics + queue summary)
  - Node utilization, GPU utilization
  - Queue length, running jobs
  - Average wait time, free nodes
  - Queue summary (avg nodes, GPUs, wall time, max priority)
- **Action Space**: Discrete(11) - select which queued job to dispatch
  - Action 0 = no-op (let FCFS handle it)
  - Actions 1-10 = prioritize specific job in visible queue
- **Cluster**: 64 nodes × 4 GPUs = 256 GPUs
- **Workload**: 100 jobs per episode (24-hour simulation)

## Key Metrics

**Scheduling Performance:**
- **node_util_mean** - Average cluster node utilization [0, 1]
- **total_jobs_completed** - Total jobs finished
- **avg_wait_time** - Average job queue wait time (lower = better)

**Reward Components:**
- Job completion bonus
- Utilization reward
- Wait time penalty
- SLA violation penalty

**No thermal metrics** (Scheduling-v0 is pure scheduling without cooling)

## Why Scheduling-v0?

This is the **pure scheduling problem**:
- No thermal constraints (unlike JointDCFlat)
- Focus on resource allocation efficiency
- Discrete action space (easier than continuous)
- Medium difficulty (easier than JointDC, harder than ChipThermal)

**Good for:**
- Testing scheduling algorithms in isolation
- Comparing with FCFS/Priority baselines
- Understanding resource allocation trade-offs

**Challenges:**
- Discrete action space (different from continuous cooling)
- Long-term credit assignment (job decisions affect future state)
- Stochastic workloads (job arrivals and durations vary)

## What to Expect

**Good signs:**
- Reward improves over time
- Utilization increases above FCFS baseline (~0.6-0.7)
- Jobs completed increases
- Wait time decreases

**Baselines to beat:**
- FCFS (First-Come-First-Served): ~-500 reward, 65% util
- Random: ~-2000 reward, 45% util

**If it works well:**
- CMO-SAC should achieve 70-80% utilization
- Better than FCFS on reward
- Lower average wait times

## Relevant Monitoring

Since Scheduling-v0 doesn't have thermal/energy metrics, monitor:
- **Reward**: Should improve steadily
- **Episode length**: May vary as jobs complete faster
- **Constraint violations**: SLA violations (should be low)

**Note:** PUE/WUE plots will show "No data" - that's expected for pure scheduling!

## Files Created

Just 2 files:
1. `scheduling.sh` - Training wrapper
2. `SCHEDULING.md` - This guide

Uses existing `run_cmo_sac.py` with normalized observations and rewards.

## Example Usage

```bash
# Default training (400k steps)
./scheduling.sh train

# Longer training for better convergence
./scheduling.sh train 800000

# Custom results directory
SCHEDULING_RESULTS_DIR=scheduling_experiments ./scheduling.sh train

# Monitor progress every 10 minutes
watch -n 600 ./scheduling.sh monitor
```

## Next Steps

After training completes:
1. **Evaluate**: Compare against FCFS baseline
2. **Analyze**: Check utilization, wait times, SLA violations
3. **Visualize**: Plot episode traces of job completions
4. **Compare**: Run same evaluation on JointDCFlat-v0 (scheduling + cooling)
