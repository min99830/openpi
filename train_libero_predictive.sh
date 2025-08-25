#!/bin/bash

# Training script for Pi0-FAST Predictive model with LIBERO dataset
# This model learns to predict next actions: (o_t, a_t) -> a_{t+1}

echo "=========================================="
echo "Pi0-FAST Predictive Model Training"
echo "=========================================="

# Set data directory
export OPENPI_DATA_HOME="/mnt/ssd1/min99830/.cache/openpi"

# Create experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="pi0_fast_predictive_libero_${TIMESTAMP}"

# Choose configuration (comment/uncomment as needed)
# Option 1: Full finetuning (requires more memory)
# CONFIG="pi0_fast_libero_predictive"

# Option 2: LoRA finetuning (lower memory)
CONFIG="pi0_fast_libero_predictive_low_mem_finetuning"

echo "Configuration: ${CONFIG}"
echo "Experiment name: ${EXP_NAME}"
echo "Data directory: ${OPENPI_DATA_HOME}"
echo ""

# Run training
echo "Starting training..."
python scripts/train.py ${CONFIG} --exp_name ${EXP_NAME}

echo ""
echo "Training completed!"
echo "Experiment saved as: ${EXP_NAME}"