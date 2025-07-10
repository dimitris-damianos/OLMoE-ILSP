#!/bin/bash
#SBATCH --job-name=qwen2.5-1.5B_generalinstructionfollowing_expert_SFT
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

TASK=generalinstructionfollowing_expert_Qwen2.5-1.5B_SFT

mkdir -p $WORK/experts/$TASK

export HF_HOME=$WORK/hf_cache
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR=$WORK

# export PYTHONHASHSEED=42
# export CUDA_LAUNCH_BLOCKING=1

accelerate launch --config_file $WORK/scripts/multi_gpu.yaml sft.py \
  --model_name_or_path $HF_HOME/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323 \
  --dataset_name "tatsu-lab/alpaca" \
  --cache_dir $HF_HOME \
  --train_split train \
  --output_dir $WORK/experts/$TASK \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
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
  --eval_steps 500 \
  --completion_only_loss \
  --instruction_format alpaca \
  --max_length 4096 \
  # --use_liger \
  # --activation_offloading \
  # --push_to_hub \
  # --packing \
  # --use_peft \
  # --lora_r 16 \
  # --lora_alpha 16 \
  # --use_rslora
