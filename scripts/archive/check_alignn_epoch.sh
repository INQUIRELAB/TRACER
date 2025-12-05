#!/bin/bash
# Quick script to check current ALIGNN epoch

LOG_FILE="logs/train_alignn_fixed_corrected.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

# Get the last epoch line with loss info
LAST_EPOCH=$(grep -E "INFO.*Epoch.*Train Loss.*Val Loss" "$LOG_FILE" | tail -1)

if [ -z "$LAST_EPOCH" ]; then
    # Try alternative format
    LAST_EPOCH=$(grep -E "Epoch.*/.*Train Loss|Epoch.*\|" "$LOG_FILE" | tail -1)
fi

if [ ! -z "$LAST_EPOCH" ]; then
    echo "Current status:"
    echo "$LAST_EPOCH"
    
    # Extract epoch number
    EPOCH=$(echo "$LAST_EPOCH" | grep -oE "Epoch[[:space:]]+[0-9]+" | grep -oE "[0-9]+" | head -1)
    
    if [ ! -z "$EPOCH" ]; then
        echo ""
        echo "Current Epoch: $EPOCH/100"
        REMAINING=$((50 - $EPOCH))
        if [ $REMAINING -gt 0 ]; then
            echo "Epochs until 50: $REMAINING"
            echo "Estimated time: ~$((REMAINING * 3))-${REMAINING}*4 minutes"
        elif [ $EPOCH -eq 50 ]; then
            echo "✅ AT EPOCH 50 - Ready to stop!"
        elif [ $EPOCH -gt 50 ]; then
            echo "⚠️  ALREADY PAST EPOCH 50 (at epoch $EPOCH)"
        fi
    fi
else
    echo "Could not find epoch information in log"
    echo "Last few lines:"
    tail -3 "$LOG_FILE"
fi



