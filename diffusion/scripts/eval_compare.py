#!/usr/bin/env python
"""Quick evaluation of VAE vs KAE diffusion models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.models.diffusion import VideoDiffusionModel
from src.models.kae import load_koopman_ae_from_checkpoint
from src.models.vae import VideoVAE
from src.evaluation.metrics import compute_fvd, compute_video_ssim, compute_video_psnr

def load_real_videos(num_samples=64, num_frames=16):
    """Load real Moving MNIST videos."""
    data_path = Path(__file__).parent.parent / "data/moving_mnist/mnist_test_seq.npy"
    data = np.load(data_path)  # (T, N, H, W)

    indices = np.random.choice(data.shape[1], size=num_samples, replace=False)
    videos = data[:num_frames, indices]  # (T, N, H, W)

    videos = torch.from_numpy(videos).float()
    videos = videos.permute(1, 0, 2, 3)  # (N, T, H, W)
    videos = videos.unsqueeze(2)  # (N, T, C, H, W)
    videos = videos / 255.0 * 2 - 1  # Normalize to [-1, 1]

    return videos

def load_kae_diffusion(ckpt_path, device="cuda"):
    """Load diffusion model with KAE encoder."""
    kae_path = Path(__file__).parent.parent.parent / "koopmanAE/results/trained_models_mnist/focal_dice_loss.pth"

    encoder = load_koopman_ae_from_checkpoint(
        str(kae_path),
        latent_dim=64,
        input_channels=1,
        image_size=64,
        dataset_name="MovingMNIST",
        input_frames=5,
        pred_frames=5,
    )

    model = VideoDiffusionModel(
        image_size=4,
        channels=4,
        num_frames=16,
        dim=64,
        dim_mults=(1, 2, 4),
        timesteps=1000,
        use_kae=True,
        kae_model=encoder,
    )
    model.set_latent_normalization(latent_min=-1.0, latent_max=1.0)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)

    return model.to(device).eval()

def load_vae_diffusion(ckpt_path, device="cuda"):
    """Load diffusion model with VAE encoder."""
    vae_ckpt = Path(__file__).parent.parent / "outputs/checkpoints/vae_movingmnist/last.ckpt"

    vae = VideoVAE(
        input_channels=1,
        latent_dim=64,
        hidden_dims=[32, 64, 128, 256],
        image_size=64,
    )
    vae_state = torch.load(vae_ckpt, map_location="cpu")
    vae_sd = {k.replace("model.", ""): v for k, v in vae_state["state_dict"].items() if k.startswith("model.")}
    vae.load_state_dict(vae_sd)

    model = VideoDiffusionModel(
        image_size=4,
        channels=4,
        num_frames=16,
        dim=64,
        dim_mults=(1, 2, 4),
        timesteps=1000,
        use_kae=True,
        kae_model=vae,
    )
    model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)

    return model.to(device).eval()

def generate_samples(model, num_samples=64, batch_size=8, device="cuda"):
    """Generate video samples."""
    samples = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - i)
            batch = model.sample(batch_size=bs)
            samples.append(batch.cpu())
            print(f"  Generated {i + bs}/{num_samples}")
    return torch.cat(samples, dim=0)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    num_samples = args.num_samples

    print("=" * 60)
    print("FVD / SSIM EVALUATION: VAE vs KAE")
    print("=" * 60)

    # Load real videos
    print("\nLoading real videos...")
    real_videos = load_real_videos(num_samples=num_samples)
    print(f"  Real videos shape: {real_videos.shape}")

    results = {}

    # Evaluate KAE
    kae_ckpt = list(Path("outputs/checkpoints/diffusion_kae_compare").glob("*.ckpt"))
    if kae_ckpt:
        kae_ckpt = sorted(kae_ckpt, key=lambda x: x.stat().st_mtime)[-1]
        print(f"\nLoading KAE model: {kae_ckpt.name}")
        try:
            kae_model = load_kae_diffusion(str(kae_ckpt), device)
            print("Generating KAE samples...")
            kae_samples = generate_samples(kae_model, num_samples, args.batch_size, device)

            print("Computing KAE metrics...")
            results["KAE"] = {
                "fvd": compute_fvd(real_videos, kae_samples, device=device),
                "ssim": compute_video_ssim(real_videos, kae_samples),
                "psnr": compute_video_psnr(real_videos, kae_samples),
            }
            del kae_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"KAE evaluation failed: {e}")

    # Evaluate VAE
    vae_ckpt = list(Path("outputs/checkpoints/diffusion_vae_compare").glob("*.ckpt"))
    if vae_ckpt:
        vae_ckpt = sorted(vae_ckpt, key=lambda x: x.stat().st_mtime)[-1]
        print(f"\nLoading VAE model: {vae_ckpt.name}")
        try:
            vae_model = load_vae_diffusion(str(vae_ckpt), device)
            print("Generating VAE samples...")
            vae_samples = generate_samples(vae_model, num_samples, args.batch_size, device)

            print("Computing VAE metrics...")
            results["VAE"] = {
                "fvd": compute_fvd(real_videos, vae_samples, device=device),
                "ssim": compute_video_ssim(real_videos, vae_samples),
                "psnr": compute_video_psnr(real_videos, vae_samples),
            }
            del vae_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"VAE evaluation failed: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} {'VAE':>15} {'KAE':>15} {'Winner':>10}")
    print("-" * 60)

    if "VAE" in results and "KAE" in results:
        # FVD (lower is better)
        vae_fvd, kae_fvd = results["VAE"]["fvd"], results["KAE"]["fvd"]
        winner = "VAE" if vae_fvd < kae_fvd else "KAE"
        print(f"{'FVD ↓':<15} {vae_fvd:>15.2f} {kae_fvd:>15.2f} {winner:>10}")

        # SSIM (higher is better)
        vae_ssim, kae_ssim = results["VAE"]["ssim"], results["KAE"]["ssim"]
        winner = "VAE" if vae_ssim > kae_ssim else "KAE"
        print(f"{'SSIM ↑':<15} {vae_ssim:>15.4f} {kae_ssim:>15.4f} {winner:>10}")

        # PSNR (higher is better)
        vae_psnr, kae_psnr = results["VAE"]["psnr"], results["KAE"]["psnr"]
        winner = "VAE" if vae_psnr > kae_psnr else "KAE"
        print(f"{'PSNR ↑':<15} {vae_psnr:>15.2f} {kae_psnr:>15.2f} {winner:>10}")

    print("=" * 60)

if __name__ == "__main__":
    main()
