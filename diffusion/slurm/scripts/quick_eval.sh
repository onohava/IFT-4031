#!/bin/bash
#SBATCH --job-name=quick_eval
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=00:20:00
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/logs/quick_eval_%j.out
#SBATCH --error=outputs/logs/quick_eval_%j.err

source ~/.bashrc
conda activate myenv
export PYTHONPATH=/private/home/soniajoseph/koopman:$PYTHONPATH

echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
cd /private/home/soniajoseph/koopman

# Find latest checkpoints
KAE_DIFF=$(ls -t outputs/checkpoints/diffusion-kae*.ckpt | head -1)
VAE_DIFF=$(ls -t outputs/checkpoints/diffusion-vae*.ckpt | head -1)

echo "Using KAE diffusion: $KAE_DIFF"
echo "Using VAE diffusion: $VAE_DIFF"

# KAE eval
echo ""
echo "=========================================="
python scripts/quick_eval.py \
    --encoder_type kae \
    --encoder_ckpt outputs/checkpoints/last-v7.ckpt \
    --diffusion_ckpt "$KAE_DIFF" \
    --num_samples 32

# VAE eval
echo ""
echo "=========================================="
python scripts/quick_eval.py \
    --encoder_type vae \
    --encoder_ckpt outputs/checkpoints/last-v6.ckpt \
    --diffusion_ckpt "$VAE_DIFF" \
    --num_samples 32

echo ""
echo "End: $(date)"
