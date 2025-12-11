#!/bin/bash
#SBATCH --job-name=gen_comp
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=00:20:00
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/logs/gen_comp_%j.out
#SBATCH --error=outputs/logs/gen_comp_%j.err

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

# Generate diffusion samples (random)
echo "=== Generating diffusion samples ==="
python scripts/generate_comparison.py \
    --kae_encoder outputs/checkpoints/last-v7.ckpt \
    --kae_diffusion "$KAE_DIFF" \
    --vae_encoder outputs/checkpoints/last-v6.ckpt \
    --vae_diffusion "$VAE_DIFF" \
    --output_dir outputs/visualizations/slides \
    --num_samples 3 \
    --mode generate

# Rename to indicate these are generated
mv outputs/visualizations/slides/comparison_real_vae_kae.png \
   outputs/visualizations/slides/comparison_generated.png

# Generate encoder reconstructions (same input)
echo "=== Generating encoder reconstructions ==="
python scripts/generate_comparison.py \
    --kae_encoder outputs/checkpoints/last-v7.ckpt \
    --kae_diffusion "$KAE_DIFF" \
    --vae_encoder outputs/checkpoints/last-v6.ckpt \
    --vae_diffusion "$VAE_DIFF" \
    --output_dir outputs/visualizations/slides \
    --num_samples 3 \
    --mode reconstruct

# Rename to indicate these are reconstructions
mv outputs/visualizations/slides/comparison_real_vae_kae.png \
   outputs/visualizations/slides/comparison_reconstruction.png

# Copy to report/figs for slides
echo "Copying to report/figs..."
cp outputs/visualizations/slides/*.png report/figs/

echo "End: $(date)"
