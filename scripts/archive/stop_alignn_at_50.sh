#!/bin/bash
# Stop ALIGNN training at epoch 50

LOG_FILE="logs/train_alignn_fixed_corrected.log"

echo "Checking ALIGNN status..."

# Check if process is running
PID=$(ps aux | grep "[p]ython3.*train_alignn_fixed" | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ ALIGNN process not found - may have already completed"
    exit 1
fi

# Get current epoch
CURRENT_EPOCH=$(grep -E "INFO.*Epoch.*Train Loss.*Val Loss" "$LOG_FILE" | tail -1 | grep -oE "Epoch[[:space:]]+[0-9]+" | grep -oE "[0-9]+" | head -1)

if [ -z "$CURRENT_EPOCH" ]; then
    echo "⚠️  Could not determine current epoch"
    echo "Last log lines:"
    tail -3 "$LOG_FILE"
    exit 1
fi

echo "Current Epoch: $CURRENT_EPOCH/100"
echo "Process PID: $PID"

if [ "$CURRENT_EPOCH" -ge 50 ]; then
    echo ""
    echo "✅ Epoch 50 or higher reached - stopping training..."
    
    # Stop the process
    kill $PID 2>/dev/null
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Verify it stopped
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Process still running, force stopping..."
        kill -9 $PID 2>/dev/null
    fi
    
    echo "✅ Training stopped"
    echo ""
    echo "Verifying checkpoint..."
    if [ -f "models/alignn_fixed_corrected/best_model.pt" ]; then
        echo "✅ Best model checkpoint exists"
        ls -lh models/alignn_fixed_corrected/best_model.pt
    else
        echo "⚠️  Checkpoint not found - check models/alignn_fixed_corrected/"
    fi
else
    REMAINING=$((50 - CURRENT_EPOCH))
    echo ""
    echo "⏳ Still at epoch $CURRENT_EPOCH"
    echo "   Wait for epoch 50 to complete, then run this script again"
    echo "   Estimated time: ~$((REMAINING * 3)) minutes"
    echo ""
    echo "Or watch in real-time:"
    echo "   tail -f $LOG_FILE | grep 'Epoch.*Train Loss.*Val Loss'"
fi



