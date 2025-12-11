#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.diffusion import VideoDiffusionModel
from src.models.kae import KoopmanAutoencoder
from src.models.vae import VideoVAE
from src.evaluation.metrics import (
    compute_fvd,
    compute_video_ssim,
    compute_video_psnr,
    compute_temporal_consistency,
)


def load_model(encoder_type, encoder_ckpt, diffusion_ckpt, device):
    """Load diffusion model."""
    if encoder_type == "kae":
        encoder = KoopmanAutoencoder(
            input_channels=1, latent_dim=1024,
            hidden_dims=[96, 192], image_size=64,
        )
    else:
        encoder = VideoVAE(
            input_channels=1, latent_dim=1024,
            hidden_dims=[64, 128], image_size=64,
        )

    ckpt = torch.load(encoder_ckpt, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device).eval()

    model = VideoDiffusionModel(
        image_size=16, channels=4, num_frames=16,
        dim=128, dim_mults=(1, 2, 4, 8),
        timesteps=1000, use_kae=True, kae_model=encoder,
    )

    if encoder_type == "kae":
        model.set_latent_normalization(latent_min=-0.5, latent_max=0.5)
    else:
        model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    ckpt = torch.load(diffusion_ckpt, map_location="cpu")
    state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    diffusion_state = {k: v for k, v in state_dict.items() if k.startswith("unet.") or k.startswith("diffusion.") or k.startswith("latent_")}
    model.load_state_dict(diffusion_state, strict=False)

    return model.to(device).eval()


def load_real_videos(data_path, num_samples):
    """Load real videos."""
    data = np.load(data_path)[:16, :num_samples]  # (T, N, H, W)
    videos = torch.from_numpy(data).float().permute(1, 0, 2, 3).unsqueeze(2)  # (N, T, C, H, W)
    return videos / 255.0 * 2 - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", choices=["kae", "vae"], required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--data_path", default="data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"=== Quick Eval: {args.encoder_type.upper()} ===")
    print(f"Samples: {args.num_samples}")

    # Load
    print("Loading model...")
    model = load_model(args.encoder_type, args.encoder_ckpt, args.diffusion_ckpt, args.device)

    print("Loading real videos...")
    real = load_real_videos(args.data_path, args.num_samples)
    print(f"  Real shape: {real.shape}")

    print("Generating samples...")
    with torch.no_grad():
        gen = model.sample(batch_size=args.num_samples)
    print(f"  Gen shape: {gen.shape}")

    # Metrics
    print("\nComputing metrics...")
    import sys
    sys.stdout.flush()

    print("  FVD...")
    sys.stdout.flush()
    fvd = compute_fvd(real, gen, device=args.device)
    print(f"    FVD: {fvd:.2f}")
    sys.stdout.flush()

    print("  SSIM...")
    sys.stdout.flush()
    ssim = compute_video_ssim(real.to(args.device), gen.to(args.device))
    print(f"    SSIM: {ssim:.4f}")
    sys.stdout.flush()

    print("  PSNR...")
    sys.stdout.flush()
    psnr = compute_video_psnr(real.to(args.device), gen.to(args.device))
    print(f"    PSNR: {psnr:.2f} dB")
    sys.stdout.flush()

    print("  Temporal Consistency...")
    sys.stdout.flush()
    tc_real = compute_temporal_consistency(real)
    tc_gen = compute_temporal_consistency(gen)
    print(f"    TC Real: {tc_real:.4f}")
    print(f"    TC Gen: {tc_gen:.4f}")
    print(f"    Ratio: {tc_gen/tc_real:.2f}x")
    sys.stdout.flush()

    print("\n=== RESULTS ===")
    print(f"FVD:  {fvd:.2f} (lower is better)")
    print(f"SSIM: {ssim:.4f} (higher is better)")
    print(f"PSNR: {psnr:.2f} dB (higher is better)")
    print(f"TC:   {tc_gen/tc_real:.2f}x (closer to 1.0 is better)")


if __name__ == "__main__":
    main()
