#!/bin/bash
#SBATCH --job-name=merge_qwen3-0.6B_experts
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.2
module load anaconda3/2023.09-0

source activate sft-experts-moe

export HF_HOME=$WORK/hf_cache

BASE_MODEL=$HF_HOME/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455
MOE_SAVE_DIR=$WORK/moe_models/qwen3_moe_merged

SPECIALISTS=(
    $WORK/experts/Qwen3-0.6B-SFT/bio_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/causalreasoning_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/code_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/finance_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/generalinstructionfollowing_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/legal_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/math_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/medical_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/multilingual_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/physicalcommonsense_expert_Qwen3-0.6B_SFT
    $WORK/experts/Qwen3-0.6B-SFT/socialreasoning_expert_Qwen3-0.6B_SFT
)

python merge_experts.py \
  --base_model $BASE_MODEL \
  --specialists ${SPECIALISTS[@]} \
  --output_dir $MOE_SAVE_DIR \
  --model_type qwen3 \
  --output_expert_mask \
  --output_router_logits
