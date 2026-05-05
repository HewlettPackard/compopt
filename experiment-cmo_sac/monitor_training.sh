#!/bin/bash
# ============================================================
# Monitor Training Progress
# Run this script to continuously monitor and visualize
# training progress while experiments are running
# ============================================================

# Configuration
# Updated to check both new location (experiment-cmo_sac/results/) and old location (results/)
LOG_DIR=${1:-"experiment-cmo_sac/results/cmo_sac"}
OUTPUT_PREFIX=${2:-"progress"}
REFRESH_INTERVAL=${3:-60}  # seconds

cd /lustre/naug/cmpopt

echo "============================================================"
echo "CMO-SAC Training Monitor"
echo "============================================================"
echo "Log Directory: $LOG_DIR"
echo "Output Prefix: $OUTPUT_PREFIX"
echo "Refresh Interval: ${REFRESH_INTERVAL}s"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Find all checkpoint directories
find_checkpoints() {
    find "$LOG_DIR" -type d -name "checkpoints" 2>/dev/null
}

# Monitor loop
iteration=0
while true; do
    clear
    iteration=$((iteration + 1))
    
    echo "============================================================"
    echo "Training Monitor - Iteration $iteration"
    echo "Time: $(date)"
    echo "============================================================"
    echo ""
    
    # Find all active training runs
    checkpoint_dirs=$(find_checkpoints)
    
    if [ -z "$checkpoint_dirs" ]; then
        echo "❌ No training runs found in $LOG_DIR"
        echo ""
        echo "Start a training run first:"
        echo "  ./train_quick.sh"
        echo "  ./train_single.sh RackCooling-v0 0 500000"
        echo ""
        sleep $REFRESH_INTERVAL
        continue
    fi
    
    # Count runs
    n_runs=$(echo "$checkpoint_dirs" | wc -l)
    echo "Found $n_runs active training run(s)"
    echo ""
    
    # Process each run
    run_num=0
    echo "$checkpoint_dirs" | while read -r checkpoint_dir; do
        run_num=$((run_num + 1))
        
        # Extract run info
        run_name=$(echo "$checkpoint_dir" | rev | cut -d'/' -f2 | rev)
        
        echo "─────────────────────────────────────────────────────────────"
        echo "Run $run_num: $run_name"
        echo "─────────────────────────────────────────────────────────────"
        
        # Check for log files
        training_log="$checkpoint_dir/training_log.csv"
        losses_log="$checkpoint_dir/losses.csv"
        eval_log="$checkpoint_dir/eval_log.csv"
        
        if [ -f "$training_log" ]; then
            # Get latest stats
            last_line=$(tail -1 "$training_log")
            if [ ! -z "$last_line" ] && [ "$last_line" != "timestep,episode,reward,length,fps,pue,wue,throughput,thermal_violations,hbm_violations,sla_violations,coolant_violations,elapsed_time_hrs" ]; then
                timestep=$(echo "$last_line" | cut -d',' -f1)
                episode=$(echo "$last_line" | cut -d',' -f2)
                reward=$(echo "$last_line" | cut -d',' -f3)
                fps=$(echo "$last_line" | cut -d',' -f5)
                elapsed_hrs=$(echo "$last_line" | cut -d',' -f13)
                
                echo "  Timestep: $timestep"
                echo "  Episode: $episode"
                echo "  Reward: $reward"
                echo "  FPS: $fps"
                echo "  Elapsed: ${elapsed_hrs} hrs"
            else
                echo "  Status: Starting..."
            fi
        else
            echo "  Status: Initializing..."
        fi
        
        if [ -f "$losses_log" ]; then
            n_updates=$(wc -l < "$losses_log")
            n_updates=$((n_updates - 1))  # subtract header
            echo "  Updates: $n_updates"
        fi
        
        if [ -f "$eval_log" ]; then
            n_evals=$(wc -l < "$eval_log")
            n_evals=$((n_evals - 1))  # subtract header
            echo "  Evaluations: $n_evals"
            
            if [ $n_evals -gt 0 ]; then
                last_eval=$(tail -1 "$eval_log")
                eval_reward=$(echo "$last_eval" | cut -d',' -f2)
                echo "  Latest Eval Reward: $eval_reward"
            fi
        fi
        
        # Generate visualization
        output_file="${OUTPUT_PREFIX}_${run_name}.png"
        echo "  Visualization: $output_file"
        
        # Generate plot (suppress output)
        python visualize_training.py \
            --log-dir "$checkpoint_dir" \
            --output "$output_file" > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            file_size=$(ls -lh "$output_file" | awk '{print $5}')
            echo "  ✓ Plot updated ($file_size)"
        else
            echo "  ⚠ Plot generation failed (may need more data)"
        fi
        
        echo ""
    done
    
    echo "============================================================"
    echo "Next update in ${REFRESH_INTERVAL}s..."
    echo "Press Ctrl+C to stop"
    echo "============================================================"
    
    sleep $REFRESH_INTERVAL
done
