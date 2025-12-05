#!/bin/bash
# Monitor ALIGNN training and stop at epoch 50 for fair comparison

LOG_FILE="logs/train_alignn_fixed_corrected.log"
CHECK_INTERVAL=30  # Check every 30 seconds

echo "Monitoring ALIGNN training..."
echo "Will stop when epoch 50 completes"
echo ""

while true; do
    # Check if epoch 50 has completed
    if grep -q "Epoch.*50.*|.*Train Loss.*|.*Val Loss" "$LOG_FILE"; then
        # Wait a bit to ensure epoch 50 is fully saved
        sleep 60
        
        # Find ALIGNN process
        PID=$(ps aux | grep "[p]ython3.*train_alignn_fixed" | awk '{print $2}' | head -1)
        
        if [ ! -z "$PID" ]; then
            echo "Stopping ALIGNN training at epoch 50 (PID: $PID)"
            kill $PID
            echo "✅ Training stopped at epoch 50"
            echo "Best model should be saved in models/alignn_fixed_corrected/best_model.pt"
            break
        else
            echo "Process not found - may have already completed"
            break
        fi
    fi
    
    # Check current epoch
    CURRENT=$(tail -20 "$LOG_FILE" 2>/dev/null | grep -oP "Epoch \K\d+" | tail -1)
    if [ ! -z "$CURRENT" ]; then
        echo "Current epoch: $CURRENT/100"
        if [ "$CURRENT" -ge 50 ]; then
            echo "Epoch 50 reached or passed - stopping..."
            PID=$(ps aux | grep "[p]ython3.*train_alignn_fixed" | awk '{print $2}' | head -1)
            if [ ! -z "$PID" ]; then
                kill $PID
                echo "✅ Stopped at epoch $CURRENT"
            fi
            break
        fi
    fi
    
    sleep $CHECK_INTERVAL
done



