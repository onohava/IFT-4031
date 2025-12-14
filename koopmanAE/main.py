import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import Config
from dataset import get_dataloader
from models import KoopmanAE
from torchvision.ops import sigmoid_focal_loss



def dice_loss(pred_logits, target, smooth=1.0):
    # Apply Sigmoid to get probability [0, 1]
    pred_probs = torch.sigmoid(pred_logits)

    pred_flat = pred_probs.view(-1)
    target_flat = target.reshape(-1)

    intersection = (pred_flat * target_flat).sum()

    # Dice Coefficient: 2 * Intersection / (Sum + Sum)
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return 1. - dice_score


def get_criterion(device):
    # used for mnist:
    class CompositeLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs, targets):
            focal = sigmoid_focal_loss(inputs, targets, alpha=0.9, gamma=2.0, reduction='mean')

            dice = dice_loss(inputs, targets)

            return 0.5 * focal + 0.5 * dice

    if Config.DATASET_NAME == 'UCF101':
        # used for UCF101
        return nn.MSELoss().to(device)
    else:
        # used for mnist
        return CompositeLoss().to(device)


def visualize_preds(model, batch, cfg, folder, epoch):
    model.eval()
    with torch.no_grad():
        x_in = batch[:, :cfg.INPUT_FRAMES].to(cfg.DEVICE)
        gt_future = batch[:, cfg.INPUT_FRAMES:cfg.INPUT_FRAMES + cfg.PRED_FRAMES].to(cfg.DEVICE)

        preds = model(x_in, mode='forward')

        fig, axs = plt.subplots(3, cfg.PRED_FRAMES, figsize=(15, 6))
        for t in range(cfg.PRED_FRAMES):
            if t < cfg.INPUT_FRAMES:
                axs[0, t].imshow(x_in[0, t, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axs[0, t].axis('off')
            axs[0, t].set_title('Input')

            axs[1, t].imshow(gt_future[0, t, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axs[1, t].axis('off');
            axs[1, t].set_title('Target')

            img_pred = torch.sigmoid(preds[t][0, 0]).cpu().numpy()
            axs[2, t].imshow(img_pred, cmap='gray', vmin=0, vmax=1)
            axs[2, t].axis('off')
            axs[2, t].set_title('Pred')

        plt.savefig(f"{folder}/preds_epoch_{epoch}.png")
        plt.close()


def train():
    cfg = Config()
    if not os.path.exists("results"): os.mkdir("results")

    train_loader = get_dataloader(cfg)
    model = KoopmanAE(cfg).to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    criterion = get_criterion(cfg.DEVICE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    print("--- Starts Training ---")

    for epoch in range(cfg.EPOCHS):
        model.train()
        avg_loss = 0
        avg_fwd = 0
        avg_bwd = 0
        avg_consist = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch = batch.to(cfg.DEVICE)


            x_fwd_in = batch[:, 0:cfg.INPUT_FRAMES]
            targets_fwd = [batch[:, cfg.INPUT_FRAMES + k] for k in range(cfg.PRED_FRAMES)]
            target_rec_fwd = x_fwd_in[:, -1]  # reconstruction target

            x_bwd_in = batch[:, cfg.INPUT_FRAMES: cfg.INPUT_FRAMES + cfg.INPUT_FRAMES]
            targets_bwd = [batch[:, cfg.INPUT_FRAMES - 1 - k] for k in range(cfg.PRED_FRAMES)]

            # ==========================================
            # FORWARD PASS
            # ==========================================
            preds_fwd = model(x_fwd_in, mode='forward')

            loss_fwd = 0
            for k in range(cfg.PRED_FRAMES):
                loss_fwd += criterion(preds_fwd[k], targets_fwd[k])

            # identity Loss (Reconstruction)
            loss_identity = criterion(preds_fwd[-1], target_rec_fwd)

            # ==========================================
            # BACKWARD PASS
            # ==========================================
            preds_bwd = model(x_bwd_in, mode='backward')

            loss_bwd = 0
            for k in range(cfg.PRED_FRAMES):
                loss_bwd += criterion(preds_bwd[k], targets_bwd[k])

            # ==========================================
            #  CONSISTENCY LOSS
            # ==========================================
            A = model.dynamics.dynamics.weight
            B = model.backdynamics.dynamics.weight
            K_dim = A.shape[-1]
            I = torch.eye(K_dim).to(cfg.DEVICE)

            loss_consist = torch.mean((torch.mm(B, A) - I) ** 2) + \
                           torch.mean((torch.mm(A, B) - I) ** 2)


            loss = loss_fwd + \
                   loss_identity + \
                   0.5 * loss_bwd + \
                   0.01 * loss_consist

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            avg_loss += loss.item()
            avg_fwd += loss_fwd.item()
            avg_bwd += loss_bwd.item()
            avg_consist += loss_consist.item()

        scheduler.step(avg_loss / len(train_loader))

        print(f"Epoch {epoch + 1} | Total: {avg_loss / len(train_loader):.3f} | "
              f"Fwd: {avg_fwd / len(train_loader):.3f} | "
              f"Bwd: {avg_bwd / len(train_loader):.3f} | "
              f"Consist: {avg_consist / len(train_loader):.4f}")

        if (epoch + 1) % 25 == 0:
            visualize_preds(model, batch, cfg, "results", epoch + 1)
            torch.save(model.state_dict(), f"results/model_{epoch + 1}.pth")


if __name__ == "__main__":
    train()