#!/bin/bash
#SBATCH --job-name=qwen3_moe_merged_SFT_tulu_mix_test_trainable_router
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=./train_logs/moe_sft.out
#SBATCH --error=./train_logs/moe_sft.err

module load gcc/12.2.0
module load cuda/12.2
module load anaconda3/2023.09-0

source activate moe

export PYTHONUNBUFFERED=TRUE  

TASK=qwen3_moe_merged_SFT_tulu_mix_test_trainable_router
WORK_MOE="/leonardo_work/EUHPC_A06_067/"
export HF_HOME="/leonardo_work/EUHPC_A06_067/hf_cache"
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR="/leonardo_work/EUHPC_A06_067/"

export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_TIMEOUT=18000

# export PYTHONHASHSEED=42
# export CUDA_LAUNCH_BLOCKING=1

accelerate launch --config_file $WORK_MOE/scripts/multi_gpu.yaml moe_sft.py \
  --model_name_or_path $WORK_MOE/moe_models/base/qwen3_moe \
  --tokenizer_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name "microsoft/wiki_qa" \
  --cache_dir $HF_HOME \
  --train_split train \
  --output_dir $WORK_MOE/moe_models/$TASK \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 8 \
  --learning_rate 3.0e-04 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --save_steps 10 \
  --eval_steps 500 \
  --assistant_only_loss \
  --instruction_format wikiqa_chat \
  --max_length 4096 \
  --freeze_experts \
  --freeze_non_experts
  # --packing \
  # --use_liger \
  # --activation_offloading \
  # --push_to_hub \
  # --use_peft \
  # --lora_r 16 \
  # --lora_alpha 16 \
  # --use_rslora
