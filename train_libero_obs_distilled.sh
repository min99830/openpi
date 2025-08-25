#!/bin/bash

# Training script for Pi0-FAST Observation Distilled model with LIBERO dataset
# This model uses KL distillation: student(o_{t-1}, a_{t-1}) -> a_t vs teacher(o_t) -> a_t

echo "=========================================="
echo "Pi0-FAST Observation Distilled Training"
echo "=========================================="

# Set data directory
export OPENPI_DATA_HOME="/mnt/ssd1/min99830/.cache/openpi"

# Create experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="pi0_fast_obs_distilled_libero_${TIMESTAMP}"

# Choose configuration (comment/uncomment as needed)
# Option 1: Full finetuning (requires more memory)
# CONFIG="pi0_fast_libero_obs_distilled"

# Option 2: LoRA finetuning (lower memory)
CONFIG="pi0_fast_libero_obs_distilled_low_mem_finetuning"

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