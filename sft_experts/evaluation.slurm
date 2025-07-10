#!/bin/bash
#SBATCH --job-name=eval_baseline_gsm8k_qwen3-0.6B-SFT
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.1
module load anaconda3/2023.09-0

source activate sft-experts-moe

TASK=eval_baseline_GSM8K_Qwen3-0.6B_SFT

export HF_HOME=$WORK/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
# export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=$TASK
export WANDB_MODE="offline"
export WANDB_DIR=$WORK

# MODEL_PATH=$WORK/experts/Qwen3-0.6B-SFT/math_expert_Qwen3-0.6B_SFT # expert
MODEL_PATH=$WORK/hf_cache/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 # baseline

OUTPUT_JSON=$WORK/eval_results/$TASK.json
mkdir -p $(dirname $OUTPUT_JSON)

accelerate launch --config_file $WORK/scripts/multi_gpu.yaml -m lm_eval \
  --model hf \
  --tasks gsm8k,gsm8k_cot,gsm8k_cot_self_consistency \
  --model_args pretrained=$MODEL_PATH,dtype=auto \
  --batch_size 4 \
  --device auto \
  --output_path $OUTPUT_JSON \
  --log_samples \
  --trust_remote_code \
  --wandb_args project=qwen_experts_eval,job_type=eval,mode=offline \
  --wandb_config_args model=Qwen3-0.6B,expert=none,task=gsm8k
