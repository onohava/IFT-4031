#!/bin/bash
#SBATCH --job-name=diff_mf_4gpu
#SBATCH --partition=scavenge
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/diff_multiframe_4gpu_%j.out
#SBATCH --error=outputs/logs/diff_multiframe_4gpu_%j.err

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
echo "Training Diffusion on UCF101 (ApplyLipstick) with Multi-Frame KAE (sliding window) - 4 GPU"
echo "Using lipstick.pth (INPUT_FRAMES=5) with stride=1 -> 12 latent frames from 16 video frames"
echo ""

# Create output directories
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/logs
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/checkpoints/diffusion_ucf101_multiframe

# Run training
cd /private/home/soniajoseph/IFT-4031/diffusion
python scripts/train_diffusion.py --config configs/diffusion_ucf101_multiframe_4gpu.yaml

echo ""
echo "End time: $(date)"
