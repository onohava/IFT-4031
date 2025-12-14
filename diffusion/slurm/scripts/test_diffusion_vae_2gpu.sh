#!/bin/bash
#SBATCH --job-name=diff_vae_2gpu
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/diff_vae_2gpu_%j.out
#SBATCH --error=outputs/logs/diff_vae_2gpu_%j.err

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
python scripts/train_diffusion.py --config configs/diffusion_vae_2gpu_test.yaml

echo ""
echo "End time: $(date)"
