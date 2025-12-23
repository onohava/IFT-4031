#!/bin/bash
#SBATCH --job-name=diff-kae
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffusion_kae_%j.out
#SBATCH --error=logs/diffusion_kae_%j.err

cd /private/home/soniajoseph/IFT-4031/diffusion
mkdir -p logs

source ~/.bashrc
conda activate myenv

srun python scripts/train_diffusion.py --config configs/compare_diffusion_kae_2gpu.yaml
