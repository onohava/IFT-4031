#!/bin/bash
#SBATCH --job-name=diff_vae_16x16
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/diff_vae_16x16_%j.out
#SBATCH --error=outputs/logs/diff_vae_16x16_%j.err

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
echo "Training Diffusion in VAE latent space (16x16x4, 1000 epochs)"
echo ""

# Run training
cd /private/home/soniajoseph/koopman
python scripts/train_diffusion.py --config configs/diffusion_vae_16x16_1000ep.yaml

echo ""
echo "End time: $(date)"
