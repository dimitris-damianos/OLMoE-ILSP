#!/bin/bash
#SBATCH --job-name=test-moe
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=EUHPC_A06_067
#SBATCH --output=test_moe.out
#SBATCH --error=test_moe.err

module load cuda/12.2
module load anaconda3/2023.09-0

source activate /leonardo_work/EUHPC_A06_067/.conda/envs/moe

python test.py 