#!/bin/bash

PYTHON_SCRIPT="train_SCD.py"
MEMORY_THRESHOLD=25000
CHECK_INTERVAL=3
OUTPUT_FILE="output_sperate_loss2.txt"

while true; do
    # Get memory usage for all GPUs
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)

    echo "Checking available GPUs..."

    # Iterate over each GPU entry
    while IFS=',' read -r GPU_INDEX GPU_MEMORY_USED GPU_MEMORY_TOTAL; do
        GPU_INDEX=$(echo "$GPU_INDEX" | tr -d ' ')
        GPU_MEMORY_USED=$(echo "$GPU_MEMORY_USED" | tr -d ' ')
        GPU_MEMORY_TOTAL=$(echo "$GPU_MEMORY_TOTAL" | tr -d ' ')

        # Calculate free memory
        GPU_MEMORY_FREE=$((GPU_MEMORY_TOTAL - GPU_MEMORY_USED))

        # Debugging output
        echo "GPU $GPU_INDEX - Used: $GPU_MEMORY_USED MB / Total: $GPU_MEMORY_TOTAL MB - Free: $GPU_MEMORY_FREE MB"

        # Check if this GPU has enough free memory
        if [ "$GPU_MEMORY_FREE" -gt "$MEMORY_THRESHOLD" ]; then
            echo "GPU $GPU_INDEX has enough free memory ($GPU_MEMORY_FREE MB). Running $PYTHON_SCRIPT on GPU $GPU_INDEX..."

            # Set the GPU to use and run the Python script
            export CUDA_VISIBLE_DEVICES=$GPU_INDEX
            python "$PYTHON_SCRIPT"
            EXIT_STATUS=$?

            if [ $EXIT_STATUS -ne 0 ]; then
                echo "Script crashed with exit status $EXIT_STATUS. Restarting..."
                sleep 5  # Wait before restarting
            else
                echo "Script finished successfully."
                exit 0  # Exit loop once script runs successfully
            fi
        fi
    done <<< "$GPU_INFO"

    echo "No available GPUs with enough free memory. Checking again in $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
done
