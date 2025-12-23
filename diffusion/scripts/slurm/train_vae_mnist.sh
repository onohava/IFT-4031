#!/bin/bash
#SBATCH --job-name=vae-mnist
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/vae_mnist_%j.out
#SBATCH --error=logs/vae_mnist_%j.err

cd /private/home/soniajoseph/IFT-4031/diffusion
mkdir -p logs

source ~/.bashrc
conda activate myenv

srun python scripts/train_vae.py --config configs/vae_movingmnist_2gpu.yaml
