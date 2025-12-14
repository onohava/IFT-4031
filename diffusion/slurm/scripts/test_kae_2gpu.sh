#!/bin/bash
#SBATCH --job-name=kae_2gpu_test
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=00:20:00
#SBATCH --mem=64GB
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/kae_2gpu_test_%j.out
#SBATCH --error=outputs/logs/kae_2gpu_test_%j.err

# Load environment
source ~/.bashrc
conda activate myenv

# Set Python path
export PYTHONPATH=/private/home/soniajoseph/koopman:$PYTHONPATH

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Run training
cd /private/home/soniajoseph/koopman
python scripts/train_kae.py --config configs/kae_2gpu_test.yaml

echo ""
echo "End time: $(date)"
