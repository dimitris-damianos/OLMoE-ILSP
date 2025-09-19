#!/bin/bash
#SBATCH --job-name=eval_math_datamix_expert_GSM8K_qwen2.5-1.5B-SFT_chat_test_addedconfig
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=logs/out_eval.log
#SBATCH --error=logs/error_eval.log

module load cuda/12.1
module load anaconda3/2023.09-0

# source activate sft-experts-moe
source activate /leonardo_work/EUHPC_A06_067/.conda/envs/moe

WORK_DIR=/leonardo_work/EUHPC_A06_067

export HF_HOME=$WORK_DIR/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
# export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR=$WORK_DIR
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1

TASK=eval_moe_0.6B_lora_11_gsm8k_chat_test_added_config
# MODEL_PATH=$WORK_DIR/experts/math_datamix_expert_Qwen2.5-1.5B_SFT # expert
# MODEL_PATH=$WORK/hf_cache/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 # baseline
MODEL_PATH=/leonardo_work/EUHPC_A06_067/moe_models/qwen3_0.6B-moe-merged-11_experts-SFT_trainable_router_stage1/checkpoint-100
OUTPUT_JSON=$WORK_DIR/eval_results/$TASK.json
mkdir -p $(dirname $OUTPUT_JSON)

accelerate launch --config_file $WORK_DIR/scripts/multi_gpu.yaml eval.py

#   --batch_size 8 \
#   --device auto \
#   --output_path $OUTPUT_JSON \
#   --log_samples \
#   --trust_remote_code \
#   --confirm_run_unsafe_code \
#   --wandb_args project=qwen_experts_eval,job_type=eval,mode=offline \
#   --wandb_config_args model=Qwen2.5-1.5B,expert=math_datamix,task=gsm8k \
#   # --apply_chat_template
