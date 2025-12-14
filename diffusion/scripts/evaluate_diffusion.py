#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.diffusion import VideoDiffusionModel
from src.models.kae import KoopmanAutoencoder
from src.models.vae import VideoVAE
from src.evaluation.metrics import (
    compute_fvd,
    compute_fid,
    compute_video_ssim,
    compute_video_psnr,
    compute_temporal_consistency,
)


def load_encoder(encoder_type: str, checkpoint_path: str, latent_dim: int = 64,
                 hidden_dims: list = None):
    """Load KAE or VAE encoder from checkpoint."""
    if hidden_dims is None:
        hidden_dims = [32, 64, 128, 256]

    if encoder_type == "kae":
        model = KoopmanAutoencoder(
            input_channels=1,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            image_size=64,
        )
    else:
        model = VideoVAE(
            input_channels=1,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            image_size=64,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("model.", ""): v
                  for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict)
    return model


def load_diffusion_model(
    checkpoint_path: str,
    encoder_type: str,
    encoder_checkpoint: str,
    device: str = "cuda",
    latent_size: int = 4,
    latent_dim: int = 64,
    hidden_dims: list = None,
    diffusion_dim: int = 64,
    dim_mults: tuple = (1, 2, 4),
):
    """Load diffusion model with encoder."""
    encoder = load_encoder(encoder_type, encoder_checkpoint,
                           latent_dim=latent_dim, hidden_dims=hidden_dims)
    encoder = encoder.to(device)
    encoder.eval()

    model = VideoDiffusionModel(
        image_size=latent_size,
        channels=4,
        num_frames=16,
        dim=diffusion_dim,
        dim_mults=dim_mults,
        timesteps=1000,
        use_kae=True,
        kae_model=encoder,
    )

    # Set latent normalization
    if encoder_type == "kae":
        if latent_dim >= 1024:
            model.set_latent_normalization(latent_min=-0.5, latent_max=0.5)
        else:
            model.set_latent_normalization(latent_min=-1.0, latent_max=1.0)
    else:
        model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    # Load diffusion weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v

    diffusion_state_dict = {k: v for k, v in state_dict.items()
                           if k.startswith("unet.") or k.startswith("diffusion.")
                           or k.startswith("latent_")}
    model.load_state_dict(diffusion_state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def load_real_videos(data_path: str, num_samples: int, num_frames: int = 16) -> torch.Tensor:
    """Load real videos from MovingMNIST dataset."""
    data = np.load(data_path)  # Shape: (T, N, H, W)

    # Select random samples
    indices = np.random.choice(data.shape[1], size=num_samples, replace=False)
    videos = data[:num_frames, indices]  # (T, N, H, W)

    # Convert to tensor and normalize to [-1, 1]
    videos = torch.from_numpy(videos).float()
    videos = videos.permute(1, 0, 2, 3)  # (N, T, H, W)
    videos = videos.unsqueeze(2)  # (N, T, C, H, W)
    videos = videos / 255.0 * 2 - 1  # Normalize to [-1, 1]

    return videos


def generate_samples(model: VideoDiffusionModel, num_samples: int,
                     batch_size: int = 4, device: str = "cuda") -> torch.Tensor:
    """Generate video samples from diffusion model."""
    samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Generating {num_samples} samples in {num_batches} batches...")
    for i in range(num_batches):
        current_batch = min(batch_size, num_samples - i * batch_size)
        print(f"  Batch {i+1}/{num_batches} ({current_batch} samples)...")

        with torch.no_grad():
            batch_samples = model.sample(batch_size=current_batch)
            samples.append(batch_samples.cpu())

    return torch.cat(samples, dim=0)


def main(
    diffusion_checkpoint: str,
    encoder_checkpoint: str,
    encoder_type: str,
    data_path: str,
    output_path: str,
    num_samples: int = 256,
    batch_size: int = 4,
    device: str = "cuda",
    latent_size: int = 4,
    latent_dim: int = 64,
    hidden_dims: list = None,
    diffusion_dim: int = 64,
    dim_mults: tuple = (1, 2, 4),
):
    """Evaluate diffusion model and save results."""
    print("=" * 60)
    print("Diffusion Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {diffusion_checkpoint}")
    print(f"Encoder: {encoder_type.upper()}")
    print(f"Num samples: {num_samples}")
    print()

    # Load model
    print("Loading model...")
    model = load_diffusion_model(
        diffusion_checkpoint,
        encoder_type,
        encoder_checkpoint,
        device,
        latent_size=latent_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        diffusion_dim=diffusion_dim,
        dim_mults=dim_mults,
    )

    # Load real videos
    print("Loading real videos...")
    real_videos = load_real_videos(data_path, num_samples)
    print(f"  Real videos shape: {real_videos.shape}")

    # Generate samples
    generated_videos = generate_samples(model, num_samples, batch_size, device)
    print(f"  Generated videos shape: {generated_videos.shape}")

    # Compute FVD
    print("\nComputing FVD...")
    fvd_score = compute_fvd(real_videos, generated_videos, device=device)
    print(f"  FVD: {fvd_score:.2f}")

    # Compute FID (per-frame, using middle frame)
    print("\nComputing FID (middle frame)...")
    # Handle different tensor orderings
    if real_videos.shape[2] == 1:  # (N, T, C, H, W)
        middle_frame = real_videos.shape[1] // 2
        real_frames = real_videos[:, middle_frame]  # (N, C, H, W)
        gen_frames = generated_videos[:, middle_frame]
    else:  # (N, C, T, H, W)
        middle_frame = real_videos.shape[2] // 2
        real_frames = real_videos[:, :, middle_frame]
        gen_frames = generated_videos[:, :, middle_frame]

    fid_score = compute_fid(real_frames, gen_frames, device=device)
    print(f"  FID: {fid_score:.2f}")

    # Compute Video SSIM
    print("\nComputing Video SSIM...")
    ssim_score = compute_video_ssim(real_videos, generated_videos)
    print(f"  SSIM: {ssim_score:.4f}")

    # Compute Video PSNR
    print("\nComputing Video PSNR...")
    psnr_score = compute_video_psnr(real_videos, generated_videos)
    print(f"  PSNR: {psnr_score:.2f} dB")

    # Compute Temporal Consistency
    print("\nComputing Temporal Consistency...")
    tc_real = compute_temporal_consistency(real_videos, method="frame_diff")
    tc_gen = compute_temporal_consistency(generated_videos, method="frame_diff")
    print(f"  Real videos: {tc_real:.4f}")
    print(f"  Generated videos: {tc_gen:.4f}")
    print(f"  Ratio (gen/real): {tc_gen/tc_real:.2f}x")

    # Prepare results
    results = {
        "checkpoint": diffusion_checkpoint,
        "encoder_type": encoder_type,
        "encoder_checkpoint": encoder_checkpoint,
        "num_samples": num_samples,
        "metrics": {
            "fvd": fvd_score,
            "fid_middle_frame": fid_score,
            "ssim": ssim_score,
            "psnr_db": psnr_score,
            "temporal_consistency_real": tc_real,
            "temporal_consistency_gen": tc_gen,
            "temporal_consistency_ratio": tc_gen / tc_real,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Extract epoch from checkpoint name
    ckpt_name = Path(diffusion_checkpoint).stem
    if "epoch=" in ckpt_name:
        epoch = int(ckpt_name.split("epoch=")[1].split("-")[0])
        results["epoch"] = epoch

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FVD:  {fvd_score:.2f} (lower is better)")
    print(f"  FID:  {fid_score:.2f} (lower is better)")
    print(f"  SSIM: {ssim_score:.4f} (higher is better, max 1.0)")
    print(f"  PSNR: {psnr_score:.2f} dB (higher is better)")
    print(f"  Temporal Consistency Ratio: {tc_gen/tc_real:.2f}x (closer to 1.0 is better)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--encoder_type", type=str, choices=["kae", "vae"], required=True)
    parser.add_argument("--data_path", type=str,
                        default="data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--output_path", type=str,
                        default="outputs/evaluations/eval_results.json")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dims", type=str, default="32,64,128,256")
    parser.add_argument("--diffusion_dim", type=int, default=64)
    parser.add_argument("--dim_mults", type=str, default="1,2,4")

    args = parser.parse_args()

    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    dim_mults = tuple(int(x) for x in args.dim_mults.split(","))

    main(
        args.diffusion_checkpoint,
        args.encoder_checkpoint,
        args.encoder_type,
        args.data_path,
        args.output_path,
        args.num_samples,
        args.batch_size,
        args.device,
        args.latent_size,
        args.latent_dim,
        hidden_dims,
        args.diffusion_dim,
        dim_mults,
    )
