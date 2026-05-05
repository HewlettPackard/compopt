#!/bin/bash
# Simple wrapper for DataCenter-v0 training using CMO-SAC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow custom results directory via environment variable
RESULTS_DIR="${DATACENTER_RESULTS_DIR:-results}"

# Allow custom training steps (default: 300000)
TIMESTEPS="${2:-300000}"

case "${1:-train}" in
    train)
        echo "Starting DataCenter-v0 training in background screen..."
        echo "Results will be saved to: $RESULTS_DIR"
        echo "Training timesteps: $TIMESTEPS"
        screen -dmS datacenter_training bash -c "python run_cmo_sac.py --mode train --env DataCenter-v0 --total-timesteps $TIMESTEPS --seed 0 --output-dir $RESULTS_DIR; echo 'Training complete! Press any key'; read"
        sleep 2
        if screen -list | grep -q datacenter_training; then
            echo "✅ Training started in screen: datacenter_training"
            echo ""
            echo "Commands:"
            echo "  ./datacenter.sh monitor    # View progress plot"
            echo "  ./datacenter.sh status     # Check if training is running"
            echo "  ./datacenter.sh attach     # See live training output"
        else
            echo "❌ Failed to start training"
        fi
        ;;
    
    monitor)
        # Find the latest training run (timestamped directory)
        TRAIN_DIR=$(find "$RESULTS_DIR" -type d -name "cmo_sac_DataCenter-v0_*" 2>/dev/null | sort -r | head -1)
        
        if [ -z "$TRAIN_DIR" ]; then
            echo "❌ No training directory found"
            echo "   Looking in: $SCRIPT_DIR/$RESULTS_DIR/"
            exit 1
        fi
        
        echo "Visualizing: $TRAIN_DIR"
        # Use absolute path for visualization script
        "$SCRIPT_DIR/../visualize_current_training.sh" "$SCRIPT_DIR/$TRAIN_DIR/checkpoints" "$SCRIPT_DIR/datacenter_training_progress.png"
        ;;
    
    status)
        if screen -list | grep -q datacenter_training; then
            echo "✅ Training is running"
            screen -S datacenter_training -X hardcopy /tmp/datacenter_status.txt 2>/dev/null
            if [ -f /tmp/datacenter_status.txt ]; then
                echo ""
                echo "Recent output:"
                tail -20 /tmp/datacenter_status.txt | grep -E "(timestep|episode|Eval|EVALUATION)" | head -10
                rm /tmp/datacenter_status.txt
            fi
        else
            echo "❌ Training not running"
            # Check for completed training
            if find "$RESULTS_DIR" -name "cmo_sac_DataCenter-v0_*" -type d 2>/dev/null | grep -q .; then
                LATEST=$(find "$RESULTS_DIR" -name "cmo_sac_DataCenter-v0_*" -type d 2>/dev/null | sort -r | head -1)
                if [ -f "$LATEST/checkpoints/best_model.pth" ]; then
                    echo "✅ Training completed - model saved in:"
                    echo "   $LATEST"
                else
                    echo "⏳ Training directory exists but no final model yet"
                fi
            fi
        fi
        ;;
    
    attach)
        screen -r datacenter_training
        ;;
    
    stop)
        if screen -list | grep -q datacenter_training; then
            screen -S datacenter_training -X quit
            echo "✅ Training stopped"
        else
            echo "❌ No training running"
        fi
        ;;
    
    *)
        echo "DataCenter-v0 Training (using CMO-SAC)"
        echo ""
        echo "Usage: ./datacenter.sh [command] [timesteps]"
        echo ""
        echo "Commands:"
        echo "  train [timesteps]   Start training in background (default: 300000)"
        echo "  monitor             Plot training progress"
        echo "  status              Check training status"
        echo "  attach              Attach to training screen (Ctrl+A D to detach)"
        echo "  stop                Stop training"
        echo ""
        echo "Environment variables:"
        echo "  DATACENTER_RESULTS_DIR   Custom results directory (default: results)"
        echo ""
        echo "Examples:"
        echo "  ./datacenter.sh train                    # Train for 300k steps (default)"
        echo "  ./datacenter.sh train 500000             # Train for 500k steps"
        echo "  DATACENTER_RESULTS_DIR=runs ./datacenter.sh train 150000"
        ;;
esac
