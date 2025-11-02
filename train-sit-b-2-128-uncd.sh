#!/bin/bash
set -e

# # Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate ai


DATA_DIR="/opt/data/celeba_hq-sd-vae/"
# OUTPUT_DIR="/opt/output/batik-fm/sit-xl-2-256-repa/"
# WANDB_DIR="/opt/output/wandb/sit-xl-2-256-repa/"
# WANDB_PROJ="batik-fm"
# EXP_NAME="sit-xl-2-256-repa"

# ORIGINAL_BATCH_SIZE=256
# ORIGINAL_MAX_TRAIN_STEPS=4000000

# DIVIDE_FACTOR=256

# BATCH_SIZE=$((ORIGINAL_BATCH_SIZE / DIVIDE_FACTOR))
# MAX_TRAIN_STEPS=$((ORIGINAL_MAX_TRAIN_STEPS * DIVIDE_FACTOR))

# # --torch-compile \

# accelerate launch \
#   --num_machines 1 \
#   train_fm.py \
#   --mixed-precision "bf16" \
#   --project-name "${WANDB_PROJ}" \
#   --data-dir "${DATA_DIR}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --wandb-dir "${WANDB_DIR}" \
#   --exp-name "${EXP_NAME}" \
#   --model "SiT-XL/2" \
#   --num-classes 20 \
#   --attn-func "base" \
#   --resolution 256 \
#   --seed 0 \
#   --P-mean 0.0 \
#   --batch-size ${BATCH_SIZE} \
#   --max-train-steps ${MAX_TRAIN_STEPS} \
#   --proj-coeff 0.5  # proj-coeff > 0 to enable REPA, otherwise set proj-coeff = 0 for standard training

IMAGENET_SIZE=1_200_000
DATA_SIZE=30_000

ORIGINAL_BATCH_SIZE=256
ORIGINAL_MAX_TRAIN_STEPS=400_000

DIVIDE_FACTOR=1

BATCH_SIZE=$((ORIGINAL_BATCH_SIZE / DIVIDE_FACTOR))
MAX_TRAIN_STEPS=$(((ORIGINAL_MAX_TRAIN_STEPS * DIVIDE_FACTOR) * DATA_SIZE / IMAGENET_SIZE))

accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8" \
  --data-dir="${DATA_DIR}" \
  --batch-size ${BATCH_SIZE} \
  --max-train-steps ${MAX_TRAIN_STEPS} \
  --use-8bit-optim \
  --num-classes 1 \
  --resolution 128 \
  --cfg-prob 0 \
  --qk-norm