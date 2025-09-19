#!/bin/bash
#SBATCH --job-name=ddam_11s
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=ddam_log/output_moe-11_top-p_aux-1_lora_bal-mix.log
#SBATCH --error=ddam_log/error_moe-11_top-p_aux-1_lora_bal-mix.log

module load gcc/12.2.0
module load cuda/12.2
module load anaconda3/2023.09-0

source activate /leonardo_work/EUHPC_A06_067/.conda/envs/moe
# source activate sft-experts-moe

export PYTHONUNBUFFERED=TRUE  

# TASK=qwen3_0.6B-moe-merged-balanced_grouped_experts-SFT_fulltrainablerouter_loranonffn
TASK="ddam_qwen3_moe-11_aux-1_bal-mix"
WORK_MOE="/leonardo_work/EUHPC_A06_067"
export HF_HOME="/leonardo_work/EUHPC_A06_067/hf_cache"
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR="/leonardo_work/EUHPC_A06_067/"

export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=1800000
export TORCH_DISTRIBUTED_TIMEOUT=1800000
# export TOKENIZERS_PARALLELISM=false

# export PYTHONHASHSEED=42
# export CUDA_LAUNCH_BLOCKING=1
echo "Starting training for $TASK"
srun accelerate launch --config_file $WORK_MOE/scripts/multi_gpu.yaml moe_sft.py \
  --model_name_or_path $WORK_MOE/moe_models/base/ddam_qwen3_moe_0.6B_11_coef-1 \
  --tokenizer_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mix_config $WORK_MOE/scripts/balanced_mix_for_moe.yaml \
  --cache_dir $HF_HOME \
  --train_split train \
  --eval_split validation \
  --output_dir $WORK_MOE/moe_models/$TASK \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 5.0e-04 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 1 \
  --save_steps 100 \
  --eval_steps 10 \
  --instruction_format insert_system_message \
  --max_length 4096 \
  --freeze_experts \
  --disable_tqdm \
  # --resume_from_checkpoint \
  # --use_dynamic_routing \
  # --use_latent_states \
  # --detach_null_states \
  # --use_peft \
  # --lora_r 64 \
  # --lora_alpha 32 \
  # --use_rslora \

  # --use_liger \
  # --assistant_only_loss \
  # --packing \
  # --activation_offloading \
  # --push_to_hub \
