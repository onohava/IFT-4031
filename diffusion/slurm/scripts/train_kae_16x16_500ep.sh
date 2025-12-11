#!/bin/bash
#SBATCH --job-name=kae_16x16
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/kae_16x16_%j.out
#SBATCH --error=outputs/logs/kae_16x16_%j.err

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
echo "Training KAE with 16x16x4=1024 latent space (500 epochs)"
echo ""

# Run training
cd /private/home/soniajoseph/koopman
python scripts/train_kae.py --config configs/kae_16x16_500ep.yaml

echo ""
echo "End time: $(date)"
