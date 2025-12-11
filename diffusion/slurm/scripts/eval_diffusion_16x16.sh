#!/bin/bash
#SBATCH --job-name=eval_diff
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/eval_diffusion_%j.out
#SBATCH --error=outputs/logs/eval_diffusion_%j.err

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

cd /private/home/soniajoseph/koopman

# Evaluate Diffusion + KAE 16x16
echo "=============================================="
echo "Evaluating Diffusion + KAE 16x16"
echo "=============================================="
python scripts/evaluate_diffusion.py \
    --diffusion_checkpoint outputs/checkpoints/diffusion-kae-16x16-epoch=187-train_loss=0.0661.ckpt \
    --encoder_checkpoint outputs/checkpoints/last-v7.ckpt \
    --encoder_type kae \
    --output_path outputs/evaluations/eval_diffusion_kae_16x16.json \
    --num_samples 64 \
    --batch_size 8 \
    --latent_size 16 \
    --latent_dim 1024 \
    --hidden_dims "96,192" \
    --diffusion_dim 128 \
    --dim_mults "1,2,4,8"

echo ""
echo "=============================================="
echo "Evaluating Diffusion + VAE 16x16"
echo "=============================================="
python scripts/evaluate_diffusion.py \
    --diffusion_checkpoint outputs/checkpoints/diffusion-vae-16x16-epoch=148-train_loss=0.1288.ckpt \
    --encoder_checkpoint outputs/checkpoints/last-v6.ckpt \
    --encoder_type vae \
    --output_path outputs/evaluations/eval_diffusion_vae_16x16.json \
    --num_samples 64 \
    --batch_size 8 \
    --latent_size 16 \
    --latent_dim 1024 \
    --hidden_dims "64,128" \
    --diffusion_dim 128 \
    --dim_mults "1,2,4,8"

echo ""
echo "End time: $(date)"
