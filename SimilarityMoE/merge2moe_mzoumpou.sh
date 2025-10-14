#!/bin/bash
#SBATCH --job-name=merge_qwen3-0.8B_classic_moe
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=mzoumpou_logs/merge_classic_moe.out
#SBATCH --error=mzoumpou_logs/merge_classic_moe.err

module load cuda/12.2
module load anaconda3/2023.09-0

WORK=/leonardo_work/EUHPC_A06_067
source activate /leonardo_work/EUHPC_A06_067/.conda/envs/moe

export HF_HOME=$WORK/hf_cache

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
    # $WORK/experts/Qwen3-0.6B-SFT/balanced_grouped_experts/social_expert_Qwen3-0.6B_SFT
    # $WORK/experts/Qwen3-0.6B-SFT/balanced_grouped_experts/logic_expert_Qwen3-0.6B_SFT
    # $WORK/experts/Qwen3-0.6B-SFT/balanced_grouped_experts/language_expert_Qwen3-0.6B_SFT
    # $WORK/experts/Qwen3-0.6B-SFT/balanced_grouped_experts/world_expert_Qwen3-0.6B_SFT
)

# 12 experts, all based on the same base model
BASE_MODEL=$HF_HOME/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455
BASE_SPECIALISTS=(
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
    $BASE_MODEL
)

MOE_SAVE_DIR=$WORK/moe_models/base/mzoumpou_qwen3_0.6B_classic_moe
mkdir -p $MOE_SAVE_DIR

srun python merge_experts.py \
  --base_model $BASE_MODEL \
  --specialists ${SPECIALISTS[@]} \
  --output_dir $MOE_SAVE_DIR \
  --model_type qwen3_moe \
  --output_router_logits \
  --router_aux_loss_coef 1 \
  --experts_top_k 2
