#!/bin/bash

# Baseline training script for pi0_fast with LIBERO dataset

echo "Starting baseline pi0_fast_libero_low_mem_finetune training with LIBERO dataset..."

export OPENPI_DATA_HOME="/mnt/ssd1/min99830/.cache/openpi"

# Create experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="pi0_fast_libero_baseline_lora_${TIMESTAMP}"

# Run training with the existing pi0_fast_libero configuration
python scripts/train.py pi0_fast_libero_low_mem_finetune --exp_name ${EXP_NAME}

echo "Baseline training completed. Experiment name: ${EXP_NAME}"