#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.diffusion import VideoDiffusionModel
from src.models.kae import KoopmanAutoencoder
from src.models.vae import VideoVAE


def load_model(
    diffusion_checkpoint: str,
    encoder_checkpoint: str,
    encoder_type: str,
    device: str = "cuda",
):
    """Load diffusion model with encoder."""
    # Load encoder
    if encoder_type == "kae":
        encoder = KoopmanAutoencoder(
            input_channels=1,
            latent_dim=1024,
            hidden_dims=[96, 192],
            image_size=64,
        )
    else:
        encoder = VideoVAE(
            input_channels=1,
            latent_dim=1024,
            hidden_dims=[64, 128],
            image_size=64,
        )

    checkpoint = torch.load(encoder_checkpoint, map_location="cpu")
    state_dict = {k.replace("model.", ""): v
                  for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    encoder.eval()

    # Load diffusion
    model = VideoDiffusionModel(
        image_size=16,
        channels=4,
        num_frames=16,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        timesteps=1000,
        use_kae=True,
        kae_model=encoder,
    )

    # Set latent normalization
    if encoder_type == "kae":
        model.set_latent_normalization(latent_min=-0.5, latent_max=0.5)
    else:
        model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    # Load diffusion weights
    checkpoint = torch.load(diffusion_checkpoint, map_location="cpu")
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


def generate_and_save(model, num_samples: int, output_path: str, title: str):
    """Generate samples and save as grid."""
    print(f"Generating {num_samples} samples...")

    with torch.no_grad():
        samples = model.sample(batch_size=num_samples)

    # samples: (N, C, T, H, W) or (N, T, C, H, W)
    samples = samples.cpu().numpy()

    # Ensure shape is (N, T, H, W) for grayscale
    if samples.ndim == 5:
        if samples.shape[1] == 1:  # (N, C=1, T, H, W)
            samples = samples[:, 0]  # (N, T, H, W)
        elif samples.shape[2] == 1:  # (N, T, C=1, H, W)
            samples = samples[:, :, 0]  # (N, T, H, W)

    # Normalize to [0, 1]
    samples = (samples + 1) / 2
    samples = np.clip(samples, 0, 1)

    N, T, H, W = samples.shape

    # Create figure: rows = samples, cols = frames (show every 4th frame)
    frame_indices = [0, 4, 8, 12, 15]
    fig, axes = plt.subplots(N, len(frame_indices), figsize=(12, 2.5 * N))

    if N == 1:
        axes = axes.reshape(1, -1)

    for i in range(N):
        for j, t in enumerate(frame_indices):
            axes[i, j].imshow(samples[i, t], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f't={t}')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kae_diffusion", type=str, required=True)
    parser.add_argument("--vae_diffusion", type=str, required=True)
    parser.add_argument("--kae_encoder", type=str, required=True)
    parser.add_argument("--vae_encoder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate KAE samples
    print("\n=== Loading Diffusion + KAE ===")
    kae_model = load_model(args.kae_diffusion, args.kae_encoder, "kae", args.device)
    generate_and_save(
        kae_model,
        args.num_samples,
        str(output_dir / "samples_diffusion_kae.png"),
        "Diffusion + KAE (16x16 latent)"
    )
    del kae_model
    torch.cuda.empty_cache()

    # Generate VAE samples
    print("\n=== Loading Diffusion + VAE ===")
    vae_model = load_model(args.vae_diffusion, args.vae_encoder, "vae", args.device)
    generate_and_save(
        vae_model,
        args.num_samples,
        str(output_dir / "samples_diffusion_vae.png"),
        "Diffusion + VAE (16x16 latent)"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
