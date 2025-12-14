#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kae import KoopmanAutoencoder
from src.data.moving_mnist import create_moving_mnist_dataloader


def load_kae_from_checkpoint(checkpoint_path: str) -> KoopmanAutoencoder:
    """Load KAE from Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model config from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Create model with default config (MovingMNIST)
    model = KoopmanAutoencoder(
        input_channels=1,
        latent_dim=64,
        hidden_dims=[32, 64, 128, 256],
        image_size=64,
    )

    # Load state dict
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v  # Remove "model." prefix
    model.load_state_dict(state_dict)

    return model


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, handling normalization."""
    x = x.detach().cpu().numpy()
    # Unnormalize from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    return x


def save_video_comparison_gif(
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: str,
    fps: int = 5,
):
    """Save side-by-side comparison as GIF.

    Args:
        original: (T, H, W) original video
        reconstructed: (T, H, W) reconstructed video
        output_path: Path to save GIF
        fps: Frames per second
    """
    T, H, W = original.shape

    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].imshow(original[t], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Original (t={t})')
        axes[0].axis('off')

        axes[1].imshow(reconstructed[t], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Reconstructed')
        axes[1].axis('off')

        plt.tight_layout()

        # Convert figure to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved: {output_path}")


def save_frame_grid(
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: str,
    num_frames: int = 8,
):
    """Save frame grid as PNG.

    Args:
        original: (T, H, W) original video
        reconstructed: (T, H, W) reconstructed video
        output_path: Path to save PNG
        num_frames: Number of frames to show
    """
    T = original.shape[0]
    indices = np.linspace(0, T-1, num_frames, dtype=int)

    fig, axes = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))

    for i, t in enumerate(indices):
        axes[0, i].imshow(original[t], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        axes[0, i].set_title(f't={t}', fontsize=10)

        axes[1, i].imshow(reconstructed[t], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main(checkpoint_path: str, output_dir: str, num_samples: int = 4):
    """Generate visualizations from KAE checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_kae_from_checkpoint(checkpoint_path)
    model.eval()

    # Load data
    print("Loading data...")
    val_loader = create_moving_mnist_dataloader(
        data_path="data/moving_mnist",
        batch_size=num_samples,
        num_frames=16,
        train=False,
        num_workers=0,
    )

    # Get a batch
    batch = next(iter(val_loader))  # (B, T, C, H, W)

    # Run through model
    print("Running inference...")
    with torch.no_grad():
        outputs = model.forward_sequence(batch)

    x_orig = batch  # (B, T, C, H, W)
    x_recon = outputs["x_recon"]  # (B, T, C, H, W)

    # Convert to numpy
    x_orig_np = tensor_to_numpy(x_orig)
    x_recon_np = tensor_to_numpy(x_recon)

    # Save visualizations for each sample
    for i in range(num_samples):
        orig = x_orig_np[i, :, 0]  # (T, H, W)
        recon = x_recon_np[i, :, 0]

        # Save GIF
        save_video_comparison_gif(
            orig, recon,
            str(output_dir / f"sample_{i}_comparison.gif"),
        )

        # Save frame grid
        save_frame_grid(
            orig, recon,
            str(output_dir / f"sample_{i}_frames.png"),
        )

    # Compute and print metrics
    mse_recon = np.mean((x_orig_np - x_recon_np) ** 2)
    print(f"\nMSE Reconstruction: {mse_recon:.6f}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to KAE checkpoint (.ckpt file)",
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
        help="Number of samples to visualize",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir, args.num_samples)
