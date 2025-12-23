#!/bin/bash
#SBATCH --job-name=diff_singleframe
#SBATCH --partition=scavenge
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/diff_singleframe_%j.out
#SBATCH --error=outputs/logs/diff_singleframe_%j.err

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
echo "Training Diffusion with Single-Frame KAE (focal_dice_single_frame.pth)"
echo ""

# Create output directories
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/logs
mkdir -p /private/home/soniajoseph/IFT-4031/diffusion/outputs/checkpoints/diffusion_focal_dice

# Run training
cd /private/home/soniajoseph/IFT-4031/diffusion
python scripts/train_diffusion.py --config configs/diffusion_focal_dice.yaml

echo ""
echo "End time: $(date)"
