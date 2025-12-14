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


def load_encoder(encoder_type, encoder_ckpt, device):
    """Load just the encoder for reconstruction."""
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
    return encoder.to(device).eval()


def load_diffusion_model(encoder_type, encoder_ckpt, diffusion_ckpt, device):
    """Load full diffusion model with encoder."""
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


def load_real_videos(data_path, num_samples, seed=42):
    """Load real videos with fixed seed for reproducibility."""
    np.random.seed(seed)
    data = np.load(data_path)  # (T, N, H, W)
    indices = np.random.choice(data.shape[1], size=num_samples, replace=False)
    videos = data[:16, indices]  # (T, N, H, W)
    videos = torch.from_numpy(videos).float().permute(1, 0, 2, 3).unsqueeze(2)  # (N, T, C, H, W)
    return videos / 255.0 * 2 - 1


def to_numpy_frames(videos):
    """Convert video tensor to numpy, handle different orderings."""
    videos = videos.cpu().numpy()
    if videos.ndim == 5:
        if videos.shape[1] == 1:  # (N, C=1, T, H, W)
            videos = videos[:, 0]  # (N, T, H, W)
        elif videos.shape[2] == 1:  # (N, T, C=1, H, W)
            videos = videos[:, :, 0]  # (N, T, H, W)
    videos = (videos + 1) / 2
    return np.clip(videos, 0, 1)


def reconstruct_with_encoder(encoder, videos, encoder_type, device):
    """Encode and decode videos through the encoder."""
    videos = videos.to(device)
    B, T, C, H, W = videos.shape

    with torch.no_grad():
        if encoder_type == "kae":
            # KAE processes frame by frame
            recon_frames = []
            for t in range(T):
                frame = videos[:, t]  # (B, C, H, W)
                z = encoder.encode(frame)
                recon = encoder.decode(z)
                recon_frames.append(recon)
            recon = torch.stack(recon_frames, dim=1)  # (B, T, C, H, W)
        else:
            # VAE - same processing
            recon_frames = []
            for t in range(T):
                frame = videos[:, t]  # (B, C, H, W)
                z, _, _ = encoder.encode(frame)
                recon = encoder.decode(z)
                recon_frames.append(recon)
            recon = torch.stack(recon_frames, dim=1)  # (B, T, C, H, W)

    return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kae_encoder", required=True)
    parser.add_argument("--kae_diffusion", required=True)
    parser.add_argument("--vae_encoder", required=True)
    parser.add_argument("--vae_diffusion", required=True)
    parser.add_argument("--data_path", default="data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["reconstruct", "generate"], default="generate",
                        help="'reconstruct' shows encoder reconstruction, 'generate' shows diffusion samples")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real videos (same seed = same videos for fair comparison)
    print("Loading real videos...")
    real = load_real_videos(args.data_path, args.num_samples, seed=42)
    real_np = to_numpy_frames(real)

    if args.mode == "reconstruct":
        # Show encoder reconstruction quality
        print("Loading KAE encoder and reconstructing...")
        kae_encoder = load_encoder("kae", args.kae_encoder, args.device)
        kae_recon = reconstruct_with_encoder(kae_encoder, real, "kae", args.device)
        kae_np = to_numpy_frames(kae_recon)
        del kae_encoder
        torch.cuda.empty_cache()

        print("Loading VAE encoder and reconstructing...")
        vae_encoder = load_encoder("vae", args.vae_encoder, args.device)
        vae_recon = reconstruct_with_encoder(vae_encoder, real, "vae", args.device)
        vae_np = to_numpy_frames(vae_recon)
        del vae_encoder

        title_suffix = "(Encoder Reconstruction)"
    else:
        # Show diffusion generation (random samples)
        print("Loading KAE diffusion model and generating...")
        kae_model = load_diffusion_model("kae", args.kae_encoder, args.kae_diffusion, args.device)
        with torch.no_grad():
            kae_samples = kae_model.sample(batch_size=args.num_samples)
        kae_np = to_numpy_frames(kae_samples)
        del kae_model
        torch.cuda.empty_cache()

        print("Loading VAE diffusion model and generating...")
        vae_model = load_diffusion_model("vae", args.vae_encoder, args.vae_diffusion, args.device)
        with torch.no_grad():
            vae_samples = vae_model.sample(batch_size=args.num_samples)
        vae_np = to_numpy_frames(vae_samples)
        del vae_model

        title_suffix = "(Diffusion Generated)"

    # Create comparison figure - ROWS layout
    # Each row: one sample showing frames t=0, 4, 8, 12, 15
    frame_indices = [0, 4, 8, 12, 15]
    n_frames = len(frame_indices)
    n_rows = args.num_samples * 3  # 3 rows per sample: Real, VAE, KAE

    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 1.5, n_rows * 1.5))

    for sample_idx in range(args.num_samples):
        row_real = sample_idx * 3
        row_vae = sample_idx * 3 + 1
        row_kae = sample_idx * 3 + 2

        for col, t in enumerate(frame_indices):
            # Real
            axes[row_real, col].imshow(real_np[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_real, col].axis('off')

            # VAE
            axes[row_vae, col].imshow(vae_np[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_vae, col].axis('off')

            # KAE
            axes[row_kae, col].imshow(kae_np[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_kae, col].axis('off')

            # Column titles (only on first row)
            if sample_idx == 0:
                axes[row_real, col].set_title(f't={t}', fontsize=11)

    # Add row labels on the left side
    for sample_idx in range(args.num_samples):
        row_real = sample_idx * 3
        row_vae = sample_idx * 3 + 1
        row_kae = sample_idx * 3 + 2

        # Add text labels to the left of each row
        fig.text(0.02, 1 - (row_real + 0.5) / n_rows * 0.92 - 0.04, 'Real',
                 fontsize=10, fontweight='bold', va='center', ha='right')
        fig.text(0.02, 1 - (row_vae + 0.5) / n_rows * 0.92 - 0.04, 'VAE',
                 fontsize=10, fontweight='bold', va='center', ha='right', color='red')
        fig.text(0.02, 1 - (row_kae + 0.5) / n_rows * 0.92 - 0.04, 'KAE',
                 fontsize=10, fontweight='bold', va='center', ha='right', color='blue')

    plt.suptitle(f'Video Comparison {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(output_dir / "comparison_real_vae_kae.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {output_dir / 'comparison_real_vae_kae.png'}")

    # Also save individual high-quality images for slides
    for name, data in [("real", real_np), ("vae", vae_np), ("kae", kae_np)]:
        fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2, 2))
        for i, t in enumerate(frame_indices):
            axes[i].imshow(data[0, t], cmap='gray', vmin=0, vmax=1)
            axes[i].axis('off')
            axes[i].set_title(f't={t}')
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
