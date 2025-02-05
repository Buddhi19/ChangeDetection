#!/bin/bash

PYTHON_SCRIPT="Mamba_SCD.py"
GPU_THRESHOLD=30
CHECK_INTERVAL=10 
OUTPUT_FILE="output_sperate_loss.txt"

while true; do
    # Get GPU usage and ensure it is a single integer value
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')

    # Debugging: Print GPU usage
    echo "GPU Usage: '$GPU_USAGE'"

    # Ensure GPU_USAGE is numeric before comparison
    if ! [[ "$GPU_USAGE" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU usage value '$GPU_USAGE' is not a valid number."
        exit 1
    fi

    # Check if GPU is free
    if [ "$GPU_USAGE" -lt "$GPU_THRESHOLD" ]; then
        echo "GPU is free ($GPU_USAGE%). Running $PYTHON_SCRIPT..."
        
        # Run the Python script and restart if it crashes
        while true; do
            python "$PYTHON_SCRIPT" > "$OUTPUT_FILE" 2>&1
            EXIT_STATUS=$?

            if [ $EXIT_STATUS -ne 0 ]; then
                echo "Script crashed with exit status $EXIT_STATUS. Restarting..."
                sleep 5  # Wait before restarting
            else
                echo "Script finished successfully."
                break
            fi
        done
    else
        echo "GPU is busy ($GPU_USAGE%). Checking again in $CHECK_INTERVAL seconds."
    fi

    sleep $CHECK_INTERVAL
done