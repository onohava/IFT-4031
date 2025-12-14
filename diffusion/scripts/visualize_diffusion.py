#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.diffusion import VideoDiffusionModel
from src.models.kae import KoopmanAutoencoder
from src.models.vae import VideoVAE


def load_encoder(encoder_type: str, checkpoint_path: str, latent_dim: int = 64, hidden_dims: list = None):
    """Load KAE or VAE encoder from checkpoint."""
    if hidden_dims is None:
        # Default to V1 architecture
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
    # Load encoder
    encoder = load_encoder(encoder_type, encoder_checkpoint, latent_dim=latent_dim, hidden_dims=hidden_dims)
    encoder = encoder.to(device)
    encoder.eval()

    # Create diffusion model
    model = VideoDiffusionModel(
        image_size=latent_size,  # Latent spatial size (4 for V1, 16 for V2)
        channels=4,    # Latent channels
        num_frames=16,
        dim=diffusion_dim,
        dim_mults=dim_mults,
        timesteps=1000,
        use_kae=True,
        kae_model=encoder,
    )

    # Set latent normalization based on encoder type and architecture version
    # CRITICAL: different encoders have very different latent ranges
    # V1 (latent_dim=64): 4x4 spatial, V2 (latent_dim=1024): 16x16 spatial
    if encoder_type == "kae":
        if latent_dim >= 1024:
            model.set_latent_normalization(latent_min=-0.5, latent_max=0.5)
        else:
            model.set_latent_normalization(latent_min=-1.0, latent_max=1.0)
    else:  # VAE
        model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    # Load diffusion weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v

    # Only load diffusion weights (unet + diffusion), not encoder
    # Also load latent normalization buffers if present
    diffusion_state_dict = {k: v for k, v in state_dict.items()
                           if k.startswith("unet.") or k.startswith("diffusion.") or k.startswith("latent_")}
    model.load_state_dict(diffusion_state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, unnormalize from [-1,1] to [0,1]."""
    x = x.detach().cpu().numpy()
    x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    return x


def save_video_grid(videos: np.ndarray, output_path: str, num_frames: int = 8):
    """Save a grid of video frames.

    Args:
        videos: (B, C, T, H, W) or (B, T, C, H, W) video tensor
        output_path: Path to save PNG
        num_frames: Number of frames to show per video
    """
    # Handle both orderings
    if videos.shape[1] == 16:  # (B, T, C, H, W)
        B, T, C, H, W = videos.shape
    else:  # (B, C, T, H, W)
        B, C, T, H, W = videos.shape
        videos = videos.transpose(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

    indices = np.linspace(0, T-1, num_frames, dtype=int)

    fig, axes = plt.subplots(B, num_frames, figsize=(2*num_frames, 2*B))
    if B == 1:
        axes = axes[np.newaxis, :]

    for b in range(B):
        for i, t in enumerate(indices):
            frame = videos[b, t, 0]  # Grayscale
            axes[b, i].imshow(frame, cmap='gray', vmin=0, vmax=1)
            axes[b, i].axis('off')
            if b == 0:
                axes[b, i].set_title(f't={t}', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_video_gif(video: np.ndarray, output_path: str, fps: int = 5):
    """Save video as GIF.

    Args:
        video: (T, H, W) video array
        output_path: Path to save GIF
        fps: Frames per second
    """
    frames = []
    for t in range(video.shape[0]):
        frame = (video[t] * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved: {output_path}")


def main(
    diffusion_checkpoint: str,
    encoder_checkpoint: str,
    encoder_type: str,
    output_dir: str,
    num_samples: int = 4,
    device: str = "cuda",
    latent_size: int = 4,
    latent_dim: int = 64,
    hidden_dims: list = None,
    diffusion_dim: int = 64,
    dim_mults: tuple = (1, 2, 4),
):
    """Generate and visualize diffusion samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading diffusion model from: {diffusion_checkpoint}")
    print(f"Loading {encoder_type.upper()} encoder from: {encoder_checkpoint}")
    print(f"Latent size: {latent_size}x{latent_size}x4 = {latent_size*latent_size*4}")

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

    print(f"Generating {num_samples} video samples...")
    with torch.no_grad():
        samples = model.sample(batch_size=num_samples)

    # Convert to numpy
    samples_np = tensor_to_numpy(samples)
    print(f"Generated samples shape: {samples_np.shape}")

    # Save grid of all samples
    save_video_grid(
        samples_np,
        str(output_dir / f"diffusion_{encoder_type}_samples_grid.png"),
    )

    # Save individual GIFs
    # Handle shape ordering
    if samples_np.shape[1] == 16:  # (B, T, C, H, W)
        for i in range(num_samples):
            video = samples_np[i, :, 0]  # (T, H, W)
            save_video_gif(
                video,
                str(output_dir / f"diffusion_{encoder_type}_sample_{i}.gif"),
            )
    else:  # (B, C, T, H, W)
        for i in range(num_samples):
            video = samples_np[i, 0]  # (T, H, W)
            save_video_gif(
                video,
                str(output_dir / f"diffusion_{encoder_type}_sample_{i}.gif"),
            )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        required=True,
        help="Path to diffusion model checkpoint",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        required=True,
        help="Path to encoder (KAE or VAE) checkpoint",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["kae", "vae"],
        required=True,
        help="Type of encoder: kae or vae",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of videos to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=4,
        help="Latent spatial size (4 for V1, 16 for V2)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Total latent dimension (64 for V1, 1024 for V2)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="32,64,128,256",
        help="Encoder hidden dims, comma-separated (32,64,128,256 for V1, 96,192 for V2)",
    )
    parser.add_argument(
        "--diffusion_dim",
        type=int,
        default=64,
        help="Diffusion base dimension (64 for V1, 128 for V2)",
    )
    parser.add_argument(
        "--dim_mults",
        type=str,
        default="1,2,4",
        help="Diffusion dim multipliers, comma-separated (1,2,4 for V1, 1,2,4,8 for V2)",
    )
    args = parser.parse_args()

    # Parse comma-separated values
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    dim_mults = tuple(int(x) for x in args.dim_mults.split(","))

    main(
        args.diffusion_checkpoint,
        args.encoder_checkpoint,
        args.encoder_type,
        args.output_dir,
        args.num_samples,
        args.device,
        latent_size=args.latent_size,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        diffusion_dim=args.diffusion_dim,
        dim_mults=dim_mults,
    )
