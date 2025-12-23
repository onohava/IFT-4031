#!/bin/bash
#SBATCH --job-name=compare_mnist
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/compare_mnist_%j.out
#SBATCH --error=outputs/logs/compare_mnist_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "Generating Moving MNIST Diffusion Comparison"
echo ""

cd /private/home/soniajoseph/IFT-4031/diffusion

# Load modules
module purge
module load cuda/11.8
module load anaconda3/2023.03-1

# Use full path to Python with CUDA support
PYTHON=/private/home/soniajoseph/.conda/envs/myenv/bin/python

$PYTHON scripts/compare_diffusion_mnist.py \
    --device cuda \
    --num_samples 4 \
    --data_path data/moving_mnist \
    --output_dir outputs/visualizations/mnist_comparison

echo ""
echo "End time: $(date)"
