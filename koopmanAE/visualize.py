import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config
from models import KoopmanAE
from dataset import get_dataloader


def process_output(tensor):
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if min_val < -0.2 or max_val > 1.2:
        return torch.sigmoid(tensor)
    else:
        return torch.clamp(tensor, 0, 1)


def load_model(cfg, model_path):
    print(f"--- Loading Model from {model_path} ---")
    model = KoopmanAE(cfg).to(cfg.DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first!")


def plot_predictions(model, batch, cfg):
    """
    Visualizes: Input History -> Ground Truth Future vs. Predicted Future
    """
    print("Generating Prediction Plot...")

    x_in = batch[:, :cfg.INPUT_FRAMES].to(cfg.DEVICE)
    gt_future = batch[:, cfg.INPUT_FRAMES:cfg.INPUT_FRAMES + cfg.PRED_FRAMES].to(cfg.DEVICE)

    with torch.no_grad():
        preds = model(x_in, mode='forward')

    num_cols = cfg.PRED_FRAMES
    fig, axes = plt.subplots(3, num_cols, figsize=(15, 6))

    for t in range(num_cols):
        # 1. Plot Input
        if t < cfg.INPUT_FRAMES:
            img_in = x_in[0, t, 0].cpu().numpy()
            axes[0, t].imshow(img_in, cmap='gray', vmin=0, vmax=1)
            axes[0, t].set_title(f"Input t-{cfg.INPUT_FRAMES - t}")
        else:
            axes[0, t].text(0.5, 0.5, "", ha='center')
        axes[0, t].axis('off')

        # 2. Plot Truth
        img_gt = gt_future[0, t, 0].cpu().numpy()
        axes[1, t].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title(f"Truth t+{t + 1}")
        axes[1, t].axis('off')

        # 3. Plot Pred (Dynamic Processing)
        # Use helper to decide if we need Sigmoid or not
        pred_processed = process_output(preds[t])
        img_pred = pred_processed[0, 0].cpu().numpy()

        axes[2, t].imshow(img_pred, cmap='gray', vmin=0, vmax=1)
        axes[2, t].set_title(f"Pred t+{t + 1}")
        axes[2, t].axis('off')

    plt.tight_layout()
    plt.savefig("results/analysis_predictions.png")
    print("Saved 'results/analysis_predictions.png'")
    plt.close()


def plot_spectrum(model, cfg):
    """
    Visualizes the Koopman Operator (Matrix K) and its Stability (Eigenvalues).
    """
    print("Generating Spectrum Plot...")

    K_torch = model.dynamics.dynamics.weight.detach().cpu()
    K = K_torch.numpy().T

    eigenvalues, _ = np.linalg.eig(K)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Heatmap
    im = axes[0].imshow(K, cmap='viridis')
    axes[0].set_title(f"Koopman Matrix ({cfg.LATENT_DIM}x{cfg.LATENT_DIM})")
    plt.colorbar(im, ax=axes[0])

    # Right: Eigenvalues
    t = np.linspace(0, 2 * np.pi, 100)
    axes[1].plot(np.cos(t), np.sin(t), 'k--', alpha=0.3, label='Unit Circle')

    axes[1].scatter(eigenvalues.real, eigenvalues.imag, c='r', alpha=0.6, s=20)

    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].axhline(0, color='grey', lw=0.5)
    axes[1].axvline(0, color='grey', lw=0.5)
    axes[1].set_xlabel("Real Part")
    axes[1].set_ylabel("Imaginary Part")
    axes[1].set_title(f"Eigenvalue Spectrum")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("results/analysis_spectrum.png")
    print("Saved 'results/analysis_spectrum.png'")
    plt.close()


def plot_mode_decomposition(model, batch, cfg):
    print("Generating Interpretable Mode Viz...")

    K = model.dynamics.dynamics.weight.detach().cpu().numpy().T
    lambdas, V = np.linalg.eig(K)
    V_inv = np.linalg.inv(V)

    x_stack = batch[0:1, :cfg.INPUT_FRAMES].to(cfg.DEVICE)
    with torch.no_grad():
        z0 = model.encoder(x_stack).cpu().numpy().T

    c = V_inv @ z0
    sort_idx = np.argsort(np.abs(c).flatten())[::-1]

    modes_to_viz = []
    seen_freqs = set()

    for idx in sort_idx:
        freq = abs(np.angle(lambdas[idx]))
        if not any(abs(freq - f) < 0.1 for f in seen_freqs):
            modes_to_viz.append(idx)
            seen_freqs.add(freq)
        if len(modes_to_viz) >= 5:
            break

    T_steps = 5
    fig, axes = plt.subplots(len(modes_to_viz) + 1, T_steps, figsize=(15, 12))

    # --- Full Reconstruction ---
    z_curr = torch.from_numpy(z0.T).to(cfg.DEVICE)
    for t in range(T_steps):
        with torch.no_grad():
            raw_out = model.decoder(z_curr)

            # FIXED: Use dynamic processing (Sigmoid vs Clip)
            frame = process_output(raw_out)

            z_curr = model.dynamics(z_curr)

        axes[0, t].imshow(frame[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, t].axis('off')
        if t == 0: axes[0, t].set_title("Full Reconstruction\n(Sum of All Modes)")

    # --- Individual Modes ---
    for i, mode_idx in enumerate(modes_to_viz):
        lambda_val = lambdas[mode_idx]
        coeff_val = c[mode_idx]

        c_iso = np.zeros_like(c)
        c_iso[mode_idx] = c[mode_idx]
        if np.iscomplex(lambda_val):
            conj_idx = np.argmin(np.abs(lambdas - np.conj(lambda_val)))
            c_iso[conj_idx] = c[conj_idx]

        z_iso = (V @ c_iso).real
        z_iso_torch = torch.from_numpy(z_iso.T).float().to(cfg.DEVICE)

        for t in range(T_steps):
            with torch.no_grad():
                # NOTE: For individual modes, we usually view raw values
                # (via seismic map) because modes can be negative.
                # So we DO NOT use process_output() here.
                frame_mode = model.decoder(z_iso_torch)
                z_iso_torch = model.dynamics(z_iso_torch)

            img = frame_mode[0, 0].cpu().numpy()
            scale = np.percentile(np.abs(img), 99) + 1e-5

            axes[i + 1, t].imshow(img, cmap='seismic', vmin=-scale, vmax=scale)
            axes[i + 1, t].axis('off')

            if t == 0:
                c_mag = np.abs(coeff_val).item()
                l_mag = abs(lambda_val)
                if hasattr(l_mag, 'item'): l_mag = l_mag.item()
                axes[i + 1, t].set_title(f"Mode {i + 1}\n|c|={c_mag:.2f}, |λ|={l_mag:.3f}")

    plt.tight_layout()
    plt.savefig("results/analysis_modes_interpretable.png")
    print("Saved 'results/analysis_modes_interpretable.png'")


if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists("results"): os.mkdir("results")

    train_loader = get_dataloader(cfg)
    batch = next(iter(train_loader))

    model_path = "results/ucf_lipstick.pth"

    try:
        model = load_model(cfg, model_path=model_path)
        plot_predictions(model, batch, cfg)
        plot_spectrum(model, cfg)
        plot_mode_decomposition(model, batch, cfg)
        print("\n--- Analysis Complete. Check the 'results' folder. ---")
    except FileNotFoundError as e:
        print(e)