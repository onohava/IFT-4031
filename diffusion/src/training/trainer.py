import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional


class KAELightningModule(pl.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        recon_weight: float = 1.0,
        dynamics_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.forward_sequence(x)

    def _compute_loss(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)

        recon_loss = F.mse_loss(outputs["x_recon"], batch)
        dynamics_loss = F.mse_loss(outputs["z_pred"], outputs["z_seq"])

        total_loss = (
            self.hparams.recon_weight * recon_loss +
            self.hparams.dynamics_weight * dynamics_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "dynamics_loss": dynamics_loss,
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )
        return losses["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class VAELightningModule(pl.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.forward_sequence(x)

    def _compute_loss(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)

        recon_loss = F.mse_loss(outputs["x_recon"], batch)

        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = (
            self.hparams.recon_weight * recon_loss +
            self.hparams.kl_weight * kl_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )
        return losses["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class VideoDiffusionLightningModule(pl.LightningModule):

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        kae_model: Optional[torch.nn.Module] = None,
        learning_rate: float = 1e-4,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["diffusion_model", "kae_model"])
        self.diffusion_model = diffusion_model
        self.kae_model = kae_model

        if self.kae_model is not None:
            self.kae_model.eval()
            for param in self.kae_model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kae_model is not None:
            with torch.no_grad():
                B, T, C, H, W = x.shape
                x_flat = x.view(B * T, C, H, W)
                z_flat = self.kae_model.encode(x_flat)
                z = z_flat.view(B, T, -1).permute(0, 2, 1)
                x = z.unsqueeze(-1).unsqueeze(-1)

        return self.diffusion_model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=self.hparams.learning_rate,
        )
        return optimizer

    @torch.no_grad()
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        samples = self.diffusion_model.sample(batch_size=batch_size)

        if self.kae_model is not None:
            B, D, T, _, _ = samples.shape
            z = samples.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            z_flat = z.reshape(B * T, D)
            x_flat = self.kae_model.decode(z_flat)
            C, H, W = x_flat.shape[1:]
            samples = x_flat.view(B, T, C, H, W)

        return samples
