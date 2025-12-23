# ENHANCING DIFFUSION MODELS WITH KOOPMAN AUTOENCODERS

This repository explores video generation and prediction using **Koopman Operator Theory** combined with Deep Learning. The project investigates whether linearizing dynamics in a latent space (via Koopman Autoencoders) can improve long-term video prediction and stability compared to standard methods. It currently supports experimentation on **MovingMNIST** and **UCF101** datasets.

> **Course Project:** IFT-4031  
> **Goal:** Leverage Koopman dynamics for stable video prediction and explore integration with Diffusion models.

## Research Question

Can learning a latent space with **approximately linear temporal dynamics** (via Koopman theory) improve video generation quality compared to a standard VAE latent space?

## Approach

```
VAE-Diffusion (Ablation):
Video → [VAE Encoder] → z → [3D U-Net Diffusion] → z' → [VAE Decoder] → Video

KAE-Diffusion (Proposed):
Video → [KAE Encoder] → z → [3D U-Net Diffusion] → z' → [KAE Decoder] → Video
              ↓
         Koopman matrix K
         z_{t+1} = K · z_t  (linear dynamics)
```

### Hypothesis
If video dynamics are linear in KAE latent space, diffusion may:
1. Learn the denoising trajectory more easily
2. Generate more temporally coherent videos
3. Achieve better FVD scores

## Experiments

### V1: 4×4 Latent (64 dim) - Completed
| # | Experiment | Config | Best Loss |
|---|------------|--------|-----------|
| 1 | KAE (λ=0.1) | `kae_movingmnist.yaml` | 0.060 |
| 2 | KAE (λ=1.0) | `kae_movingmnist_lambda1.yaml` | 0.088 |
| 3 | VAE | `vae_movingmnist.yaml` | 0.053 |
| 4 | Diffusion + KAE | `diffusion_kae_4gpu.yaml` | 0.136 |
| 5 | Diffusion + VAE | `diffusion_vae_4gpu.yaml` | 0.306 |

### V2: 16×16 Latent (1024 dim) - Current
| # | Experiment | Config | Status |
|---|------------|--------|--------|
| 6 | KAE 16x16 | `kae_16x16_500ep.yaml` | Done (MSE=0.0027) |
| 7 | VAE 16x16 | `vae_16x16_500ep.yaml` | Done (MSE=0.0014) |
| 8 | Diffusion + KAE 16x16 | `diffusion_kae_16x16_1000ep.yaml` | Training (ep193, loss=0.066) |
| 9 | Diffusion + VAE 16x16 | `diffusion_vae_16x16_1000ep.yaml` | Training (ep148, loss=0.129) |

### Key Results
- **V1 Diffusion**: KAE latent trains **2.3× better** than VAE (0.136 vs 0.306)
- **V2 Reconstruction**: 16x16 latent gives **20-40× better** MSE (0.001-0.003 vs 0.05-0.06)
- **V2 Diffusion**: KAE latent continues to show **~2× better** training loss (0.066 vs 0.129)

## Dataset

| Dataset | Type | Status |
|---------|------|--------|
| **MovingMNIST** | Synthetic, 64x64, grayscale, 16 frames | Primary |
| **UCF101** | Real-world video | Trained KAE on one action |

Data location: `data/`

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training Workflow

Scale up gradually: **debug → 2 GPU test → 4 GPU full**

```bash
# === Encoder Training ===

# KAE: Test 2 GPU, then full 4 GPU
sbatch slurm/scripts/test_kae_2gpu.sh
sbatch slurm/scripts/train_kae_4gpu.sh

# VAE: Test 2 GPU, then full 4 GPU
sbatch slurm/scripts/test_vae_2gpu.sh
sbatch slurm/scripts/train_vae_4gpu.sh

# === Diffusion Training (after encoders are trained) ===

# Diffusion + KAE latent space
sbatch slurm/scripts/train_diffusion_kae_4gpu.sh

# Diffusion + VAE latent space
sbatch slurm/scripts/train_diffusion_vae_4gpu.sh
```

### Monitor Jobs
```bash
squeue -u $USER
tail -f outputs/logs/<job_id>.out
```

### Visualizations
```bash
# Visualize KAE reconstructions
python scripts/visualize_kae.py --checkpoint /checkpoint/.../last.ckpt
```

## Project Structure

```
koopman/
├── src/
│   ├── models/
│   │   ├── kae.py           # Koopman Autoencoder
│   │   ├── vae.py           # Baseline VAE
│   │   └── diffusion.py     # Video diffusion (lucidrains)
│   ├── data/
│   │   └── moving_mnist.py  # DataLoader
│   ├── training/
│   │   └── trainer.py       # PyTorch Lightning modules
│   └── evaluation/
│       └── metrics.py       # FVD, FID, LPIPS
├── configs/
│   ├── kae_*.yaml           # KAE configs
│   ├── vae_*.yaml           # VAE configs
│   └── diffusion_*.yaml     # Diffusion configs
├── scripts/
│   ├── train_kae.py         # Train Koopman Autoencoder
│   ├── train_vae.py         # Train baseline VAE
│   ├── train_diffusion.py   # Train video diffusion
│   └── visualize_kae.py     # Visualize reconstructions
├── slurm/scripts/           # SLURM job scripts
└── outputs/
    ├── checkpoints/         # Model checkpoints
    ├── logs/                # Training logs
    └── visualizations/      # Generated images/videos
```

## Metrics

Available in `src/evaluation/metrics.py`:

| Metric | Description | Better |
|--------|-------------|--------|
| **FVD** | Frechet Video Distance - video quality & temporal coherence | Lower |
| **FID** | Frechet Inception Distance - per-frame image quality | Lower |
| **SSIM** | Structural Similarity Index | Higher (max 1.0) |
| **PSNR** | Peak Signal-to-Noise Ratio | Higher (dB) |
| **TC** | Temporal Consistency - measures jitter | Lower |

```bash
# Run evaluation
python scripts/evaluate_diffusion.py \
    --diffusion_checkpoint /path/to/diffusion.ckpt \
    --encoder_checkpoint /path/to/encoder.ckpt \
    --encoder_type kae
```

Compare to: [Papers With Code - MovingMNIST Leaderboard](https://paperswithcode.com/sota/video-prediction-on-moving-mnist)

## References

- [lucidrains/video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)
- Ho et al., "Video Diffusion Models" (2022)
- Lusch et al., "Deep learning for universal linear embeddings of nonlinear dynamics" (2018)

## Team
- Ondrej Nohava
- Sonia Joseph


