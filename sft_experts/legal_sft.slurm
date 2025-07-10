#!/bin/bash
#SBATCH --job-name=qwen3-0.6B_legal_expert_SFT
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.1
module load anaconda3/2023.09-0

source activate sft-experts-moe

TASK=legal_expert_Qwen3-0.6B_SFT

mkdir -p $WORK/experts/$TASK

export HF_HOME=$WORK/hf_cache
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR=$WORK

# export PYTHONHASHSEED=42
# export CUDA_LAUNCH_BLOCKING=1

accelerate launch --config_file $WORK/scripts/multi_gpu.yaml sft.py \
  --model_name_or_path $HF_HOME/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 \
  --dataset_name "casehold/casehold" \
  --cache_dir $HF_HOME \
  --train_split train \
  --eval_split validation \
  --output_dir $WORK/experts/$TASK \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --optim adamw_torch_fused \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.03 \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 10 \
  --completion_only_loss \
  --instruction_format casehold \
  --max_length 2048 \
  # --use_liger \
  # --activation_offloading \
  # --neftune_noise_alpha 5 \
  # --push_to_hub
  # --packing \
  # --use_peft \
  # --lora_r 16 \
  # --lora_alpha 16 \
  # --use_rslora
