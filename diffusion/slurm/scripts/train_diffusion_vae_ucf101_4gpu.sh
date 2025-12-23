#!/bin/bash
#SBATCH --job-name=diff_vae_4gpu
#SBATCH --partition=scavenge
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/diff_vae_4gpu_%j.out
#SBATCH --error=outputs/logs/diff_vae_4gpu_%j.err

# Load environment
source ~/.bashrc
conda activate myenv

# Set Python path
export PYTHONPATH=/private/home/soniajoseph/IFT-4031/diffusion:$PYTHONPATH

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "Training Diffusion on UCF101 (ApplyLipstick) with VAE - 4 GPU"
echo "Baseline comparison against KAE-based diffusion"
echo ""

# Create output directories
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/logs
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/checkpoints/diffusion_vae_ucf101

# Run training
cd /private/home/soniajoseph/IFT-4031/diffusion
python scripts/train_diffusion.py --config configs/diffusion_vae_ucf101_4gpu.yaml

echo ""
echo "End time: $(date)"
