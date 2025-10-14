#!/bin/bash
#SBATCH --job-name=ddam_moe
#SBATCH --partition=boost_usr_prod
#SBATCH --gpus-per-node=4
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=ddam_log/output_moe-base-multinode.log
#SBATCH --error=ddam_log/error_moe-base-multinode.log

# Multinode configuration
export LOGLEVEL=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
export NCCL_DEBUG=INFO
echo "environment: $(env | grep NCCL)"

# Enviroment config
module load gcc/12.2.0
module load cuda/12.2
module load anaconda3/2023.09-0

source activate /leonardo_work/EUHPC_A06_067/.conda/envs/moe
# source activate sft-experts-moe

export PYTHONUNBUFFERED=TRUE  

# Train config
# TASK=qwen3_0.6B-moe-merged-balanced_grouped_experts-SFT_fulltrainablerouter_loranonffn
TASK="ddam_qwen3_moe-base_12_math"
WORK_MOE="/leonardo_work/EUHPC_A06_067"
DATASET_CONFIG="/leonardo_work/EUHPC_A06_067/OLMoE-ILSP/SimilarityMoE/configs/math_mix.yaml"
MODEL_PATH="/leonardo_work/EUHPC_A06_067/moe_models/base/ddam_qwen3rim-base-12_coef-0.01_top-p_use-latent_detach-null"

export HF_HOME="/leonardo_work/EUHPC_A06_067/hf_cache"
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR="/leonardo_work/EUHPC_A06_067/"

export PYTHONUNBUFFERED=TRUE  
export HF_HOME="/leonardo_work/EUHPC_D19_095/hf_cache"

echo "Starting training for $TASK"
num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
srun --label accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --machine_rank $SLURM_NODEID \
    --num_processes $num_processes \
    --num_machines $SLURM_NNODES \
    --dynamo_backend no \
    --mixed_precision 'bf16' \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    moe_sft.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path Qwen/Qwen3-0.6B \
    --dataset_mix_config $DATASET_CONFIG \
    --cache_dir $HF_HOME \
    --train_split train \
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
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 10 \
    --instruction_format insert_system_message \
    --max_length 4096 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 32 \
    --use_rslora \
    --lora_experts \
    --lora_base \
    --disable_tqdm  \
    # --resume_from_checkpoint