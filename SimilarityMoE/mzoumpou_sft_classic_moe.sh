#!/bin/bash
#SBATCH --job-name=mzoumpou_qwen3_moe_0.6B_sft_full
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=/leonardo_work/EUHPC_A06_067/mzoumpou_logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_A06_067/mzoumpou_logs/%x-%j.err

set -euo pipefail

# --- modules / env ---
module load cuda/12.2
module load anaconda3/2023.09-0
source activate /leonardo_work/EUHPC_A06_067/.conda/envs/moe
export PYTHONUNBUFFERED=1

# --- paths ---
WORK="/leonardo_work/EUHPC_A06_067"
mkdir -p "$WORK/mzoumpou_logs"
export HF_HOME="$WORK/hf_cache"

# Your merged classic MoE (input/base model)
REAL_DIR="$WORK/moe_models/base/mzoumpou_qwen3_0.6B_classic_moe"

# Create a symlink whose NAME contains 'qwen3_moe' so the code loads Qwen3MoeForCausalLM
SYMLINK_DIR="$WORK/moe_models/base/mzoumpou_qwen3_moe_0.6B_classic_moe"
if [ ! -e "$SYMLINK_DIR" ]; then
  ln -s "$REAL_DIR" "$SYMLINK_DIR"
fi
export MODEL_PATH="$SYMLINK_DIR"

# Dataset mix yaml
DATASET_CONFIG="$WORK/scripts/balanced_mix_for_moe.yaml"

# Run name / logging
DATE_TAG=$(date +%Y%m%d_%H%M%S)
TASK="mzoumpou_qwen3_moe_0.6B_sft_full_${DATE_TAG}"
RUN_DIR="$WORK/moe_models/$TASK"
export WANDB_PROJECT="$TASK"
export WANDB_MODE="offline"
export WANDB_DIR="$WORK"

# NCCL / timeouts / misc
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800000
export TORCH_DISTRIBUTED_TIMEOUT=1800000
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ACCEL_CFG="$WORK/scripts/multi_gpu.yaml"

echo "=== Preflight ==="
echo "REAL_DIR exists? $( [ -d "$REAL_DIR" ] && echo YES || echo NO )"
echo "SYMLINK_DIR exists? $( [ -e "$SYMLINK_DIR" ] && echo YES || echo NO )"
echo "MODEL_PATH = $MODEL_PATH"
echo "MODEL_PATH contains 'qwen3_moe'? $( [[ "$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')" == *qwen3_moe* ]] && echo YES || echo NO )"
echo "Listing model dir:"
ls -l "$MODEL_PATH" | head -n 50 || true
echo "DATASET_CONFIG = $DATASET_CONFIG"
[ -f "$DATASET_CONFIG" ] || { echo "ERROR: dataset mix yaml not found"; exit 2; }
mkdir -p "$RUN_DIR"
echo "Output RUN_DIR = $RUN_DIR"
echo "==============="

# ---- TRAIN (full finetune; no LoRA; no freezing) ----
accelerate launch --config_file "$ACCEL_CFG" moe_sft.py \
  --model_name_or_path "$MODEL_PATH" \
  --tokenizer_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mix_config "$DATASET_CONFIG" \
  --cache_dir "$HF_HOME" \
  --train_split train \
  --eval_split validation \
  --output_dir "$RUN_DIR" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --save_steps 100 \
  --eval_steps 50 \
  --instruction_format insert_system_message \
  --max_length 4096 \
  --disable_tqdm \
  --resume_from_checkpoint

