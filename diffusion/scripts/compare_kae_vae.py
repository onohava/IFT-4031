#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imageio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kae import KoopmanAutoencoder
from src.models.vae import VideoVAE
from src.data.moving_mnist import create_moving_mnist_dataloader


def load_kae_from_checkpoint(checkpoint_path: str) -> KoopmanAutoencoder:
    """Load KAE from Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = KoopmanAutoencoder(
        input_channels=1,
        latent_dim=64,
        hidden_dims=[32, 64, 128, 256],
        image_size=64,
    )
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v
    model.load_state_dict(state_dict)
    return model


def load_vae_from_checkpoint(checkpoint_path: str) -> VideoVAE:
    """Load VAE from Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = VideoVAE(
        input_channels=1,
        latent_dim=64,
        hidden_dims=[32, 64, 128, 256],
        image_size=64,
    )
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v
    model.load_state_dict(state_dict)
    return model


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, handling normalization."""
    x = x.detach().cpu().numpy()
    x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    return x


def plot_reconstruction_comparison(
    original: np.ndarray,
    kae_recon: np.ndarray,
    vae_recon: np.ndarray,
    output_path: str,
    num_frames: int = 8,
):
    """Plot side-by-side reconstruction comparison.

    Args:
        original: (T, H, W) original video
        kae_recon: (T, H, W) KAE reconstruction
        vae_recon: (T, H, W) VAE reconstruction
        output_path: Path to save PNG
        num_frames: Number of frames to show
    """
    T = original.shape[0]
    indices = np.linspace(0, T-1, num_frames, dtype=int)

    fig, axes = plt.subplots(3, num_frames, figsize=(2*num_frames, 6))

    for i, t in enumerate(indices):
        # Original
        axes[0, i].imshow(original[t], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        axes[0, i].set_title(f't={t}', fontsize=10)

        # KAE reconstruction
        axes[1, i].imshow(kae_recon[t], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('KAE', fontsize=12)

        # VAE reconstruction
        axes[2, i].imshow(vae_recon[t], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('VAE', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latent_trajectories(
    kae_latents: np.ndarray,
    vae_latents: np.ndarray,
    output_path: str,
):
    """Plot PCA of latent trajectories.

    Args:
        kae_latents: (T, latent_dim) KAE latent trajectory
        vae_latents: (T, latent_dim) VAE latent trajectory
        output_path: Path to save PNG
    """
    # Fit PCA on combined latents
    combined = np.vstack([kae_latents, vae_latents])
    pca = PCA(n_components=2)
    pca.fit(combined)

    kae_pca = pca.transform(kae_latents)
    vae_pca = pca.transform(vae_latents)

    T = kae_latents.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # KAE trajectory
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, T))
    for t in range(T - 1):
        ax.annotate('', xy=kae_pca[t+1], xytext=kae_pca[t],
                    arrowprops=dict(arrowstyle='->', color=colors[t], lw=1.5))
    ax.scatter(kae_pca[:, 0], kae_pca[:, 1], c=np.arange(T), cmap='viridis', s=50, zorder=5)
    ax.scatter(kae_pca[0, 0], kae_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=6)
    ax.scatter(kae_pca[-1, 0], kae_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('KAE Latent Trajectory (should be ~linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # VAE trajectory
    ax = axes[1]
    for t in range(T - 1):
        ax.annotate('', xy=vae_pca[t+1], xytext=vae_pca[t],
                    arrowprops=dict(arrowstyle='->', color=colors[t], lw=1.5))
    ax.scatter(vae_pca[:, 0], vae_pca[:, 1], c=np.arange(T), cmap='viridis', s=50, zorder=5)
    ax.scatter(vae_pca[0, 0], vae_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=6)
    ax.scatter(vae_pca[-1, 0], vae_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('VAE Latent Trajectory (no linearity constraint)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_koopman_prediction_error(
    kae_latents: np.ndarray,
    K: np.ndarray,
    output_path: str,
):
    """Plot Koopman prediction error over time.

    Args:
        kae_latents: (T, latent_dim) KAE latent trajectory
        K: (latent_dim, latent_dim) Koopman matrix
        output_path: Path to save PNG
    """
    T = kae_latents.shape[0]

    # Compute prediction errors
    errors = []
    for t in range(T - 1):
        z_t = kae_latents[t]
        z_next_pred = K @ z_t
        z_next_actual = kae_latents[t + 1]
        error = np.linalg.norm(z_next_pred - z_next_actual)
        errors.append(error)

    # Also compute multi-step prediction
    multistep_errors = []
    z_pred = kae_latents[0].copy()
    for t in range(T):
        error = np.linalg.norm(z_pred - kae_latents[t])
        multistep_errors.append(error)
        z_pred = K @ z_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Single-step prediction error
    ax = axes[0]
    ax.plot(range(T-1), errors, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('||z_{t+1} - K·z_t||')
    ax.set_title('Single-step Koopman Prediction Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Multi-step prediction error
    ax = axes[1]
    ax.plot(range(T), multistep_errors, 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('||K^t·z_0 - z_t||')
    ax.set_title('Multi-step Koopman Prediction Error\n(accumulates over time)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_linearity_metric(latents: np.ndarray) -> float:
    """Compute how "linear" a trajectory is.

    Fits a line to the PCA-projected trajectory and measures R².

    Args:
        latents: (T, latent_dim) latent trajectory

    Returns:
        R² value (1.0 = perfectly linear)
    """
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)

    # Fit line using time as x
    T = latents.shape[0]
    t = np.arange(T)

    # R² for PC1 vs time
    corr1 = np.corrcoef(t, latents_pca[:, 0])[0, 1] ** 2
    # R² for PC2 vs time
    corr2 = np.corrcoef(t, latents_pca[:, 1])[0, 1] ** 2

    # Average R²
    return (corr1 + corr2) / 2


def main(kae_checkpoint: str, vae_checkpoint: str, output_dir: str, num_samples: int = 4):
    """Generate comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading KAE: {kae_checkpoint}")
    kae = load_kae_from_checkpoint(kae_checkpoint)
    kae.eval()

    print(f"Loading VAE: {vae_checkpoint}")
    vae = load_vae_from_checkpoint(vae_checkpoint)
    vae.eval()

    # Get Koopman matrix
    K = kae.K.weight.detach().cpu().numpy()
    print(f"Koopman matrix shape: {K.shape}")

    # Load data
    print("Loading data...")
    val_loader = create_moving_mnist_dataloader(
        data_path="data/moving_mnist",
        batch_size=num_samples,
        num_frames=16,
        train=False,
        num_workers=0,
    )

    batch = next(iter(val_loader))  # (B, T, C, H, W)
    B, T, C, H, W = batch.shape

    # Run KAE
    print("Running KAE inference...")
    with torch.no_grad():
        kae_outputs = kae.forward_sequence(batch)
        kae_recon = kae_outputs["x_recon"]  # (B, T, C, H, W)
        kae_latents = kae_outputs["z_seq"]  # (B, T, latent_dim)

    # Run VAE
    print("Running VAE inference...")
    with torch.no_grad():
        frames = batch.view(B * T, C, H, W)
        vae_outputs = vae(frames)
        vae_recon = vae_outputs["x_recon"].view(B, T, C, H, W)
        vae_latents = vae_outputs["z"].view(B, T, -1)  # (B, T, latent_dim)

    # Convert to numpy
    x_orig_np = tensor_to_numpy(batch)
    kae_recon_np = tensor_to_numpy(kae_recon)
    vae_recon_np = tensor_to_numpy(vae_recon)
    kae_latents_np = kae_latents.detach().cpu().numpy()
    vae_latents_np = vae_latents.detach().cpu().numpy()

    # Compute metrics
    kae_mse = np.mean((x_orig_np - kae_recon_np) ** 2)
    vae_mse = np.mean((x_orig_np - vae_recon_np) ** 2)

    print(f"\n=== Reconstruction MSE ===")
    print(f"KAE: {kae_mse:.6f}")
    print(f"VAE: {vae_mse:.6f}")

    # Compute linearity metrics
    print(f"\n=== Latent Trajectory Linearity (R²) ===")
    kae_linearities = []
    vae_linearities = []
    for i in range(B):
        kae_lin = compute_linearity_metric(kae_latents_np[i])
        vae_lin = compute_linearity_metric(vae_latents_np[i])
        kae_linearities.append(kae_lin)
        vae_linearities.append(vae_lin)
        print(f"Sample {i}: KAE R²={kae_lin:.3f}, VAE R²={vae_lin:.3f}")

    print(f"\nMean: KAE R²={np.mean(kae_linearities):.3f}, VAE R²={np.mean(vae_linearities):.3f}")

    # Compute Koopman prediction errors
    print(f"\n=== Koopman Prediction Error ===")
    for i in range(B):
        errors = []
        for t in range(T - 1):
            z_t = kae_latents_np[i, t]
            z_next_pred = K @ z_t
            z_next_actual = kae_latents_np[i, t + 1]
            error = np.linalg.norm(z_next_pred - z_next_actual)
            errors.append(error)
        print(f"Sample {i}: Mean ||z_{{t+1}} - K·z_t|| = {np.mean(errors):.4f}")

    # Generate visualizations for each sample
    for i in range(num_samples):
        orig = x_orig_np[i, :, 0]
        kae_r = kae_recon_np[i, :, 0]
        vae_r = vae_recon_np[i, :, 0]

        # Reconstruction comparison
        plot_reconstruction_comparison(
            orig, kae_r, vae_r,
            str(output_dir / f"sample_{i}_reconstruction_comparison.png"),
        )

        # Latent trajectories
        plot_latent_trajectories(
            kae_latents_np[i],
            vae_latents_np[i],
            str(output_dir / f"sample_{i}_latent_trajectories.png"),
        )

        # Koopman prediction error
        plot_koopman_prediction_error(
            kae_latents_np[i],
            K,
            str(output_dir / f"sample_{i}_koopman_error.png"),
        )

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Linearity comparison
    ax = axes[0]
    x = np.arange(num_samples)
    width = 0.35
    ax.bar(x - width/2, kae_linearities, width, label='KAE', color='blue', alpha=0.7)
    ax.bar(x + width/2, vae_linearities, width, label='VAE', color='orange', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Linearity (R²)')
    ax.set_title('Latent Trajectory Linearity')
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # MSE comparison
    ax = axes[1]
    kae_mses = [np.mean((x_orig_np[i] - kae_recon_np[i]) ** 2) for i in range(num_samples)]
    vae_mses = [np.mean((x_orig_np[i] - vae_recon_np[i]) ** 2) for i in range(num_samples)]
    ax.bar(x - width/2, kae_mses, width, label='KAE', color='blue', alpha=0.7)
    ax.bar(x + width/2, vae_mses, width, label='VAE', color='orange', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction Error')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "summary_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'summary_comparison.png'}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kae_checkpoint",
        type=str,
        default="outputs/checkpoints/last-v1.ckpt",
        help="Path to KAE checkpoint",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="outputs/checkpoints/last-v3.ckpt",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (auto-generated if not specified)",
    )
    parser.add_argument(
        "--kae_name",
        type=str,
        default="kae_lambda0.1",
        help="Name for KAE variant (e.g., 'kae_lambda0.1', 'kae_lambda1.0')",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to visualize",
    )
    args = parser.parse_args()

    # Auto-generate output directory based on KAE name
    if args.output_dir is None:
        args.output_dir = f"outputs/visualizations/comparison_{args.kae_name}"

    main(args.kae_checkpoint, args.vae_checkpoint, args.output_dir, args.num_samples)
