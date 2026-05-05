#!/bin/bash
# Simple wrapper for JointDCFlat-v0 training using CMO-SAC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow custom results directory via environment variable
RESULTS_DIR="${JOINT_RESULTS_DIR:-results}"

# Allow custom training steps (default: 500000)
TIMESTEPS="${2:-500000}"

case "${1:-train}" in
    train)
        echo "Starting JointDCFlat-v0 training in background screen..."
        echo "Results will be saved to: $RESULTS_DIR"
        echo "Training timesteps: $TIMESTEPS"
        screen -dmS joint_training bash -c "python run_cmo_sac.py --mode train --env JointDCFlat-v0 --total-timesteps $TIMESTEPS --seed 0 --output-dir $RESULTS_DIR; echo 'Training complete! Press any key'; read"
        sleep 2
        if screen -list | grep -q joint_training; then
            echo "✅ Training started in screen: joint_training"
            echo ""
            echo "Commands:"
            echo "  ./joint.sh monitor    # View progress plot"
            echo "  ./joint.sh status     # Check if training is running"
            echo "  ./joint.sh attach     # See live training output"
        else
            echo "❌ Failed to start training"
        fi
        ;;
    
    monitor)
        # Find the latest training run
        TRAIN_DIR=$(find "$RESULTS_DIR" -type d -name "cmo_sac_JointDCFlat-v0_*" 2>/dev/null | sort -r | head -1)
        
        if [ -z "$TRAIN_DIR" ]; then
            echo "❌ No training directory found"
            echo "   Looking in: $SCRIPT_DIR/$RESULTS_DIR/"
            exit 1
        fi
        
        echo "Visualizing: $TRAIN_DIR"
        # Use absolute path for visualization script
        "$SCRIPT_DIR/../visualize_current_training.sh" "$SCRIPT_DIR/$TRAIN_DIR/checkpoints" "$SCRIPT_DIR/joint_training_progress.png"
        ;;
    
    status)
        if screen -list | grep -q joint_training; then
            echo "✅ Training is running"
            screen -S joint_training -X hardcopy /tmp/joint_status.txt 2>/dev/null
            if [ -f /tmp/joint_status.txt ]; then
                echo ""
                echo "Recent output:"
                tail -20 /tmp/joint_status.txt | grep -E "(timestep|episode|Eval|EVALUATION)" | head -10
                rm /tmp/joint_status.txt
            fi
        else
            echo "❌ Training not running"
            # Check for completed training
            if find "$RESULTS_DIR" -name "cmo_sac_JointDCFlat-v0_*" -type d 2>/dev/null | grep -q .; then
                LATEST=$(find "$RESULTS_DIR" -name "cmo_sac_JointDCFlat-v0_*" -type d 2>/dev/null | sort -r | head -1)
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
        screen -r joint_training
        ;;
    
    stop)
        if screen -list | grep -q joint_training; then
            screen -S joint_training -X quit
            echo "✅ Training stopped"
        else
            echo "❌ No training running"
        fi
        ;;
    
    *)
        echo "JointDCFlat-v0 Training (using CMO-SAC)"
        echo ""
        echo "Usage: ./joint.sh [command] [timesteps]"
        echo ""
        echo "Commands:"
        echo "  train [timesteps]   Start training in background (default: 500000)"
        echo "  monitor             Plot training progress"
        echo "  status              Check training status"
        echo "  attach              Attach to training screen (Ctrl+A D to detach)"
        echo "  stop                Stop training"
        echo ""
        echo "Environment variables:"
        echo "  JOINT_RESULTS_DIR   Custom results directory (default: results)"
        echo ""
        echo "Examples:"
        echo "  ./joint.sh train                    # Train for 500k steps (default)"
        echo "  ./joint.sh train 1000000            # Train for 1M steps"
        echo "  JOINT_RESULTS_DIR=runs ./joint.sh train 250000"
        ;;
esac
