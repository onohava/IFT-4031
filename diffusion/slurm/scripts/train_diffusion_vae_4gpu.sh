#!/bin/bash
#SBATCH --job-name=diff_vae_4gpu
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/diff_vae_4gpu_%j.out
#SBATCH --error=outputs/logs/diff_vae_4gpu_%j.err

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
echo "Training Diffusion in VAE latent space (full 100 epochs)"
echo ""

# Run training
cd /private/home/soniajoseph/koopman
python scripts/train_diffusion.py --config configs/diffusion_vae_4gpu.yaml

echo ""
echo "End time: $(date)"
