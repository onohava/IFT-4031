#!/usr/bin/env python
"""
Compare diffusion samples from VAE vs KAE on UCF101.
Generates side-by-side visualizations and GIFs.
"""
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.diffusion import VideoDiffusionModel
from src.models.kae import load_koopman_ae_from_checkpoint
from src.models.vae import VideoVAE
from src.data.ucf101 import create_ucf101_dataloader


def load_vae_from_checkpoint(checkpoint_path: str, latent_dim: int = 512) -> VideoVAE:
    """Load VAE from Lightning checkpoint."""
    vae = VideoVAE(
        input_channels=1,
        latent_dim=latent_dim,
        hidden_dims=[32, 64, 128, 256],
        image_size=64,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("model.", ""): v
                  for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    vae.load_state_dict(state_dict)
    return vae


def load_diffusion_with_encoder(
    diffusion_checkpoint: str,
    encoder,
    encoder_type: str,
    device: str = "cpu",
):
    """Load diffusion model with pre-loaded encoder."""
    model = VideoDiffusionModel(
        image_size=8,
        channels=8,
        num_frames=16,
        dim=64,
        dim_mults=(1, 2, 4),
        timesteps=1000,
        use_kae=True,
        kae_model=encoder,
    )

    # Set latent normalization
    if encoder_type == "kae":
        model.set_latent_normalization(latent_min=-1.0, latent_max=1.0)
    else:
        model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    # Load diffusion weights
    checkpoint = torch.load(diffusion_checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k.replace("model.", ""): v
                  for k, v in checkpoint["state_dict"].items()}

    # Only load U-Net weights, not encoder
    diffusion_keys = {k: v for k, v in state_dict.items()
                      if not k.startswith("kae_model") and k in model.state_dict()}
    model.load_state_dict(diffusion_keys, strict=False)

    model = model.to(device)
    model.eval()
    return model


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, unnormalize from [-1,1] to [0,1]."""
    x = x.detach().cpu().numpy()
    x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    return x


def save_comparison_grid(
    kae_videos: np.ndarray,
    vae_videos: np.ndarray,
    real_videos: np.ndarray,
    output_path: str,
    num_frames: int = 8,
):
    """Save side-by-side comparison grid.

    Args:
        kae_videos: (B, T, C, H, W) KAE diffusion samples
        vae_videos: (B, T, C, H, W) VAE diffusion samples
        real_videos: (B, T, C, H, W) Real videos
        output_path: Path to save PNG
        num_frames: Number of frames to show
    """
    B = min(kae_videos.shape[0], vae_videos.shape[0], real_videos.shape[0])
    T = kae_videos.shape[1]
    indices = np.linspace(0, T-1, num_frames, dtype=int)

    fig, axes = plt.subplots(3, num_frames, figsize=(2*num_frames, 7))

    row_labels = ['Real Video', 'Diffusion + KAE', 'Diffusion + VAE']

    for i, t in enumerate(indices):
        # Real
        axes[0, i].imshow(real_videos[0, t, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f't={t}', fontsize=10)

        # KAE Diffusion
        axes[1, i].imshow(kae_videos[0, t, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

        # VAE Diffusion
        axes[2, i].imshow(vae_videos[0, t, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')

    # Add row labels on the left
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight='bold', va='center', ha='right')

    plt.suptitle('UCF101 ApplyLipstick: Diffusion Generated Videos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_multi_sample_grid(
    kae_videos: np.ndarray,
    vae_videos: np.ndarray,
    output_path: str,
    num_frames: int = 8,
):
    """Save multiple samples from each model.

    Args:
        kae_videos: (B, T, C, H, W) KAE diffusion samples
        vae_videos: (B, T, C, H, W) VAE diffusion samples
        output_path: Path to save PNG
        num_frames: Number of frames to show
    """
    B = min(2, kae_videos.shape[0], vae_videos.shape[0])
    T = kae_videos.shape[1]
    indices = np.linspace(0, T-1, num_frames, dtype=int)

    fig, axes = plt.subplots(2*B, num_frames, figsize=(2*num_frames, 2*2*B + 1))

    for b in range(B):
        for i, t in enumerate(indices):
            # KAE row
            axes[2*b, i].imshow(kae_videos[b, t, 0], cmap='gray', vmin=0, vmax=1)
            axes[2*b, i].axis('off')
            if b == 0 and i == 0:
                axes[2*b, i].set_title(f't={t}', fontsize=9)
            elif b == 0:
                axes[2*b, i].set_title(f't={t}', fontsize=9)

            # VAE row
            axes[2*b+1, i].imshow(vae_videos[b, t, 0], cmap='gray', vmin=0, vmax=1)
            axes[2*b+1, i].axis('off')

    # Add row labels
    row_labels = ['KAE Sample 1', 'VAE Sample 1', 'KAE Sample 2', 'VAE Sample 2']
    for row in range(2*B):
        axes[row, 0].text(-0.15, 0.5, row_labels[row], transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight='bold', va='center', ha='right')

    plt.suptitle('UCF101: Multiple Diffusion Samples Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_video_gif(video: np.ndarray, output_path: str, fps: int = 8):
    """Save video as GIF."""
    frames = []
    for t in range(video.shape[0]):
        frame = (video[t] * 255).astype(np.uint8)
        frames.append(frame)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kae_diffusion_checkpoint", type=str,
                        default="outputs/checkpoints/diffusion_ucf101/diffusion-ucf101-4gpu-epoch=344-train_loss=0.0397.ckpt")
    parser.add_argument("--vae_diffusion_checkpoint", type=str,
                        default="outputs/checkpoints/diffusion_vae_ucf101/diffusion-vae-ucf101-4gpu-epoch=476-train_loss=0.0342.ckpt")
    parser.add_argument("--kae_checkpoint", type=str,
                        default="/private/home/soniajoseph/IFT-4031/koopmanAE/results/trained_models_ucf/lipstick_single_frame.pth")
    parser.add_argument("--vae_checkpoint", type=str,
                        default="outputs/checkpoints/vae_ucf101/last.ckpt")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/visualizations/ucf101_comparison")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real data for comparison
    print("Loading UCF101 data...")
    val_loader = create_ucf101_dataloader(
        root="/datasets01/UCF101_Frames/frames",
        actions=["ApplyLipstick"],
        batch_size=args.num_samples,
        num_frames=16,
        image_size=64,
        train=False,
        grayscale=True,
        num_workers=0,
    )
    real_batch = next(iter(val_loader))  # (B, T, C, H, W)
    real_np = tensor_to_numpy(real_batch)

    # Load KAE and KAE diffusion
    print("Loading KAE...")
    kae = load_koopman_ae_from_checkpoint(
        checkpoint_path=args.kae_checkpoint,
        latent_dim=512, input_channels=1, image_size=64,
        dataset_name="UCF101", input_frames=1, pred_frames=5,
        device=args.device,
    )
    kae.eval()

    print("Loading KAE diffusion model...")
    # Find best checkpoint if default doesn't exist
    kae_ckpt = Path(args.kae_diffusion_checkpoint)
    if not kae_ckpt.exists():
        ckpt_dir = Path("outputs/checkpoints/diffusion_ucf101")
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            kae_ckpt = sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
            print(f"  Using: {kae_ckpt.name}")

    kae_diffusion = load_diffusion_with_encoder(str(kae_ckpt), kae, "kae", args.device)

    # Load VAE and VAE diffusion
    print("Loading VAE...")
    vae = load_vae_from_checkpoint(args.vae_checkpoint)
    vae = vae.to(args.device)
    vae.eval()

    print("Loading VAE diffusion model...")
    vae_ckpt = Path(args.vae_diffusion_checkpoint)
    if not vae_ckpt.exists():
        ckpt_dir = Path("outputs/checkpoints/diffusion_vae_ucf101")
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            vae_ckpt = sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
            print(f"  Using: {vae_ckpt.name}")

    vae_diffusion = load_diffusion_with_encoder(str(vae_ckpt), vae, "vae", args.device)

    # Generate samples
    print(f"Generating {args.num_samples} samples from each model...")
    with torch.no_grad():
        kae_samples = kae_diffusion.sample(batch_size=args.num_samples)
        vae_samples = vae_diffusion.sample(batch_size=args.num_samples)

    # Convert to numpy - handle shape
    kae_np = tensor_to_numpy(kae_samples)
    vae_np = tensor_to_numpy(vae_samples)

    # Convert from (B, C, T, H, W) to (B, T, C, H, W) if needed
    if kae_np.shape[1] != 16:
        kae_np = kae_np.transpose(0, 2, 1, 3, 4)
    if vae_np.shape[1] != 16:
        vae_np = vae_np.transpose(0, 2, 1, 3, 4)

    print(f"KAE samples shape: {kae_np.shape}")
    print(f"VAE samples shape: {vae_np.shape}")

    # Save comparison grid
    save_comparison_grid(kae_np, vae_np, real_np,
                         str(output_dir / "diffusion_comparison.png"))

    # Save multi-sample grid
    save_multi_sample_grid(kae_np, vae_np,
                           str(output_dir / "diffusion_multi_samples.png"))

    # Save individual GIFs
    for i in range(min(2, args.num_samples)):
        save_video_gif(real_np[i, :, 0],
                       str(output_dir / f"real_sample_{i}.gif"))
        save_video_gif(kae_np[i, :, 0],
                       str(output_dir / f"kae_diffusion_sample_{i}.gif"))
        save_video_gif(vae_np[i, :, 0],
                       str(output_dir / f"vae_diffusion_sample_{i}.gif"))

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
