#!/bin/bash
# Simple wrapper for Scheduling-v0 training using DQN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow custom results directory via environment variable
RESULTS_DIR="${SCHEDULING_RESULTS_DIR:-results}"

# Allow custom training steps (default: 1000000 for improved, 400000 for standard)
TIMESTEPS="${2:-1000000}"

# Training mode: "standard" or "improved" (default: improved)
TRAIN_MODE="${SCHEDULING_MODE:-improved}"

case "${1:-help}" in
    train)
        echo "Starting Scheduling-v0 training in background screen..."
        echo "Results will be saved to: $RESULTS_DIR"
        echo "Training timesteps: $TIMESTEPS"
        echo ""
        echo "⚠️  Note: Scheduling-v0 uses Discrete actions"
        echo "    Training with DQN (not CMO-SAC)"
        
        # Select training script based on mode
        if [ "$TRAIN_MODE" = "standard" ]; then
            TRAIN_SCRIPT="train_scheduling_dqn.py"
            echo "    Mode: STANDARD hyperparameters"
        else
            TRAIN_SCRIPT="train_scheduling_dqn_improved.py"
            echo "    Mode: IMPROVED hyperparameters (recommended)"
            echo ""
            echo "    Improvements:"
            echo "      • Longer exploration (30% vs 10%)"
            echo "      • Higher learning rate (3e-4 vs 1e-4)"
            echo "      • Larger network (128-128 vs 64-64)"
            echo "      • Bigger buffer (500k vs 100k)"
        fi
        
        echo ""
        screen -dmS scheduling_training bash -c "python $TRAIN_SCRIPT --total-timesteps $TIMESTEPS --seed 0 --output-dir $RESULTS_DIR; echo 'Training complete! Press any key'; read"
        sleep 2
        if screen -list | grep -q scheduling_training; then
            echo "✅ Training started in screen: scheduling_training"
            echo ""
            echo "Commands:"
            echo "  ./scheduling.sh monitor    # View progress plot"
            echo "  ./scheduling.sh status     # Check if training is running"
            echo "  ./scheduling.sh attach     # See live training output"
            echo "  ./scheduling.sh analyze    # Analyze policy behavior"
        else
            echo "❌ Failed to start training"
        fi
        ;;
    
    monitor)
        # Find the latest training run (DQN format - matches both standard and improved)
        TRAIN_DIR=$(find "$RESULTS_DIR" -type d -name "*Scheduling-v0_*" 2>/dev/null | sort -r | head -1)
        
        if [ -z "$TRAIN_DIR" ]; then
            echo "❌ No training directory found"
            echo "   Looking in: $SCRIPT_DIR/$RESULTS_DIR/"
            echo "   Expected: *Scheduling-v0_*"
            exit 1
        fi
        
        echo "Visualizing: $TRAIN_DIR"
        # Use DQN-specific visualization with tensorboard data
        python "$SCRIPT_DIR/visualize_dqn_scheduling.py" "$SCRIPT_DIR/$TRAIN_DIR/checkpoints" "$SCRIPT_DIR/scheduling_training_progress.png"
        
        if [ $? -eq 0 ]; then
            echo "✅ Visualization complete: scheduling_training_progress.png"
        else
            echo "❌ Visualization failed - trying fallback"
            python "$SCRIPT_DIR/visualize_scheduling.py" "$SCRIPT_DIR/$TRAIN_DIR/checkpoints" "$SCRIPT_DIR/scheduling_training_progress.png"
        fi
        ;;
    
    status)
        if screen -list | grep -q scheduling_training; then
            echo "✅ Training is running"
            screen -S scheduling_training -X hardcopy /tmp/scheduling_status.txt 2>/dev/null
            if [ -f /tmp/scheduling_status.txt ]; then
                echo ""
                echo "Recent output:"
                tail -20 /tmp/scheduling_status.txt | grep -E "(timestep|episode|Eval|EVALUATION)" | head -10
                rm /tmp/scheduling_status.txt
            fi
        else
            echo "❌ Training not running"
            # Check for completed training (matches both standard and improved)
            if find "$RESULTS_DIR" -name "*Scheduling-v0_*" -type d 2>/dev/null | grep -q .; then
                LATEST=$(find "$RESULTS_DIR" -name "*Scheduling-v0_*" -type d 2>/dev/null | sort -r | head -1)
                if [ -f "$LATEST/checkpoints/best_model.zip" ]; then
                    echo "✅ Training completed - model saved in:"
                    echo "   $LATEST"
                else
                    echo "⏳ Training directory exists but no final model yet"
                fi
            fi
        fi
        ;;
    
    attach)
        screen -r scheduling_training
        ;;
    
    analyze)
        # Find the latest training run
        TRAIN_DIR=$(find "$RESULTS_DIR" -type d -name "*Scheduling-v0_*" 2>/dev/null | sort -r | head -1)
        
        if [ -z "$TRAIN_DIR" ]; then
            echo "❌ No training directory found"
            echo "   Looking in: $SCRIPT_DIR/$RESULTS_DIR/"
            exit 1
        fi
        
        # Find best model
        BEST_MODEL="$TRAIN_DIR/checkpoints/best_model.zip"
        
        if [ ! -f "$BEST_MODEL" ]; then
            echo "❌ Best model not found: $BEST_MODEL"
            exit 1
        fi
        
        echo "Analyzing policy from: $TRAIN_DIR"
        echo ""
        python "$SCRIPT_DIR/analyze_dqn_policy.py" "$BEST_MODEL"
        ;;
    
    stop)
        if screen -list | grep -q scheduling_training; then
            screen -S scheduling_training -X quit
            echo "✅ Training stopped"
        else
            echo "❌ No training running"
        fi
        ;;
    
    *)
        echo "Scheduling-v0 Training (using DQN, not CMO-SAC)"
        echo ""
        echo "⚠️  Note: Scheduling-v0 has discrete actions, so we use DQN instead of CMO-SAC"
        echo ""
        echo "Usage: ./scheduling.sh [command] [timesteps]"
        echo ""
        echo "Commands:"
        echo "  train [timesteps]   Start training in background (default: 1000000)"
        echo "  monitor             Plot training progress with detailed metrics"
        echo "  analyze             Analyze policy behavior and action distribution"
        echo "  status              Check training status"
        echo "  attach              Attach to training screen (Ctrl+A D to detach)"
        echo "  stop                Stop training"
        echo ""
        echo "Environment variables:"
        echo "  SCHEDULING_RESULTS_DIR   Custom results directory (default: results)"
        echo "  SCHEDULING_MODE          Training mode: 'improved' (default) or 'standard'"
        echo ""
        echo "Training Modes:"
        echo "  improved (default)  - Better hyperparameters for Scheduling-v0"
        echo "                        Longer exploration, higher LR, larger network"
        echo "                        Recommended for new training runs"
        echo ""
        echo "  standard            - Original hyperparameters"
        echo "                        Use: SCHEDULING_MODE=standard ./scheduling.sh train"
        echo ""
        echo "Examples:"
        echo "  ./scheduling.sh train                    # Train for 1M steps (improved mode)"
        echo "  ./scheduling.sh train 500000             # Train for 500k steps"
        echo "  ./scheduling.sh monitor                  # View training progress"
        echo "  ./scheduling.sh analyze                  # Check if policy is learning"
        echo "  SCHEDULING_MODE=standard ./scheduling.sh train 400000"
        echo ""
        echo "Environment Info:"
        echo "  - Discrete action space (job prioritization) - uses DQN"
        echo "  - 64 nodes × 4 GPUs = 256 GPUs cluster"
        echo "  - 100 jobs per episode (24h simulation)"
        echo "  - Metrics: utilization, throughput, wait time, SLA violations"
        echo ""
        echo "Analysis reveals:"
        echo "  - Environment bug FIXED (actions now have effect!)"
        echo "  - Standard training may still struggle due to reward scale"
        echo "  - Use 'improved' mode for better hyperparameters"
        ;;
esac
