#!/bin/bash
#SBATCH --job-name=moe_eval
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --account=EUHPC_D26_056
#SBATCH --output=./ddam_log/eval_out.log
#SBATCH --error=./ddam_log/eval_error.log

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

export HF_MODULES_CACHE=$WORK_DIR/hf_cache/huggingface/modules


TASK=ddam_base_ckp-400_gsm8k
# TASK=test-eval
# MODEL_PATH=$WORK_DIR/experts/math_datamix_expert_Qwen2.5-1.5B_SFT # expert
# MODEL_PATH=$WORK/hf_cache/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 # baseline
# MODEL_PATH=/leonardo_work/EUHPC_A06_067/moe_models/ddam_moe-grouped_top-p_aux-0.5_lora_bal-mix/checkpoint-1200
MODEL_PATH=/leonardo_work/EUHPC_A06_067/moe_models/ddam_qwen3_moe-base_12_bal-mix/checkpoint-400
OUTPUT_JSON=$WORK_DIR/eval_results/$TASK.json
mkdir -p $(dirname $OUTPUT_JSON)

accelerate launch --config_file $WORK_DIR/scripts/multi_gpu.yaml -m lm_eval \
  --model hf \
  --tasks gsm8k \
  --model_args pretrained=$MODEL_PATH,dtype=auto \
  --batch_size 4 \
  --device auto \
  --output_path $OUTPUT_JSON \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --wandb_args project=qwen_experts_eval,job_type=eval,mode=offline \
  --wandb_config_args model=Qwen2.5-1.5B,expert=moe_tulu_nonlora_fonffn_mix,task=gsm8k \
  --apply_chat_template \
  --use-custom-model
  # --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
