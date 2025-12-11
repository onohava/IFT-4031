#!/bin/bash
#SBATCH --job-name=gen_samples
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/logs/gen_samples_%j.out
#SBATCH --error=outputs/logs/gen_samples_%j.err

source ~/.bashrc
conda activate myenv

export PYTHONPATH=/private/home/soniajoseph/koopman:$PYTHONPATH

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd /private/home/soniajoseph/koopman

python scripts/generate_samples.py \
    --kae_diffusion outputs/checkpoints/diffusion-kae-16x16-epoch=363-train_loss=0.0748.ckpt \
    --vae_diffusion outputs/checkpoints/diffusion-vae-16x16-epoch=277-train_loss=0.1196.ckpt \
    --kae_encoder outputs/checkpoints/last-v7.ckpt \
    --vae_encoder outputs/checkpoints/last-v6.ckpt \
    --output_dir outputs/visualizations \
    --num_samples 4

echo "End time: $(date)"
