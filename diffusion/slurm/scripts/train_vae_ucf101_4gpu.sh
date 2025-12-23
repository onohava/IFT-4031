#!/bin/bash
#SBATCH --job-name=vae_ucf_4gpu
#SBATCH --partition=scavenge
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/vae_ucf_4gpu_%j.out
#SBATCH --error=outputs/logs/vae_ucf_4gpu_%j.err

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
echo "Training VAE on UCF101 (ApplyLipstick) - 4 GPU"
echo ""

# Create output directories
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/logs
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/checkpoints/vae_ucf101

# Run training
cd /private/home/soniajoseph/IFT-4031/diffusion
python scripts/train_vae.py --config configs/vae_ucf101_4gpu.yaml

echo ""
echo "End time: $(date)"
