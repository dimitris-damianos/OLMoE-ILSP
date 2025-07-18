#!/bin/bash
#SBATCH --job-name=qwen3_moe_merged_SFT_tulu_mix
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.2
module load anaconda3/2023.09-0

source activate sft-experts-moe

TASK=qwen3_moe_merged_SFT_tulu_mix_noliger

export HF_HOME=$WORK/hf_cache
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR=$WORK

export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=18000

# export PYTHONHASHSEED=42
# export CUDA_LAUNCH_BLOCKING=1

accelerate launch --config_file $WORK/scripts/accelerate_configs/deepspeed_zero3.yaml moe_sft.py \
  --model_name_or_path $WORK/moe_models/qwen3_moe_merged \
  --dataset_name "allenai/tulu-v3.1-mix-preview-4096-OLMoE" \
  --cache_dir $HF_HOME \
  --train_split train \
  --output_dir $WORK/moe_models/$TASK \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 5.0e-08 \
  --lr_scheduler_type linear \
  --optim adamw_torch_fused \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --bf16 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --assistant_only_loss \
  --instruction_format insert_system_message \
  --max_length 4096 \
  --freeze_experts \
  --freeze_non_experts \
  # --use_liger \
  # --packing \
  # --activation_offloading \
  # --push_to_hub \
  # --use_peft \
  # --lora_r 16 \
  # --lora_alpha 16 \
  # --use_rslora
