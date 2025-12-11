#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.diffusion import VideoDiffusionModel
from src.models.kae import KoopmanAutoencoder
from src.models.vae import VideoVAE
from src.data.moving_mnist import create_moving_mnist_dataloader


class DiffusionLightningModule(pl.LightningModule):

    def __init__(self, model: VideoDiffusionModel, learning_rate: float = 2e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[1] == self.model.num_frames:
            x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def sample_videos(self, num_samples: int = 4) -> torch.Tensor:
        return self.model.sample(batch_size=num_samples)


def load_kae_from_checkpoint(checkpoint_path: str, config: dict) -> KoopmanAutoencoder:
    kae = KoopmanAutoencoder(
        input_channels=1,
        latent_dim=config.get("latent_dim", 64),
        hidden_dims=config.get("hidden_dims", [32, 64, 128, 256]),
        image_size=64,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v
                      for k, v in checkpoint["state_dict"].items()
                      if k.startswith("model.")}
        kae.load_state_dict(state_dict)
    else:
        kae.load_state_dict(checkpoint)

    return kae


def load_vae_from_checkpoint(checkpoint_path: str, config: dict) -> VideoVAE:
    vae = VideoVAE(
        input_channels=1,
        latent_dim=config.get("latent_dim", 64),
        hidden_dims=config.get("hidden_dims", [32, 64, 128, 256]),
        image_size=64,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v
                      for k, v in checkpoint["state_dict"].items()
                      if k.startswith("model.")}
        vae.load_state_dict(state_dict)
    else:
        vae.load_state_dict(checkpoint)

    return vae


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder_model = None
    use_latent = config["model"].get("use_kae", False) or config["model"].get("use_vae", False)

    if config["model"].get("use_kae", False):
        kae_config = config.get("kae", {})
        checkpoint_path = kae_config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading KAE from {checkpoint_path}")
            encoder_model = load_kae_from_checkpoint(checkpoint_path, kae_config)
        else:
            raise FileNotFoundError(f"KAE checkpoint not found at {checkpoint_path}")
    elif config["model"].get("use_vae", False):
        vae_config = config.get("vae", {})
        checkpoint_path = vae_config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading VAE from {checkpoint_path}")
            encoder_model = load_vae_from_checkpoint(checkpoint_path, vae_config)
        else:
            raise FileNotFoundError(f"VAE checkpoint not found at {checkpoint_path}")

    model = VideoDiffusionModel(
        image_size=config["model"]["image_size"],
        channels=config["model"]["channels"],
        num_frames=config["model"]["num_frames"],
        dim=config["model"]["dim"],
        dim_mults=tuple(config["model"]["dim_mults"]),
        timesteps=config["model"]["timesteps"],
        use_kae=use_latent,
        kae_model=encoder_model,
    )

    if config["model"].get("use_kae", False):
        kae_config = config.get("kae", {})
        latent_dim = kae_config.get("latent_dim", 64)
        if latent_dim >= 1024:
            model.set_latent_normalization(latent_min=-0.5, latent_max=0.5)
        else:
            model.set_latent_normalization(latent_min=-1.0, latent_max=1.0)
    elif config["model"].get("use_vae", False):
        vae_config = config.get("vae", {})
        latent_dim = vae_config.get("latent_dim", 64)
        model.set_latent_normalization(latent_min=-8.0, latent_max=8.0)

    lit_model = DiffusionLightningModule(
        model=model,
        learning_rate=float(config["training"]["learning_rate"]),
    )

    train_loader = create_moving_mnist_dataloader(
        data_path=config["data"]["data_path"],
        batch_size=config["data"]["batch_size"],
        num_frames=config["data"]["num_frames"],
        train=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = create_moving_mnist_dataloader(
        data_path=config["data"]["data_path"],
        batch_size=config["data"]["batch_size"],
        num_frames=config["data"]["num_frames"],
        train=False,
        num_workers=config["data"]["num_workers"],
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=config["checkpointing"]["dirpath"],
            filename=f"{config['model']['name']}" + "-{epoch:02d}-{train_loss:.4f}",
            monitor=config["checkpointing"]["monitor"],
            mode=config["checkpointing"]["mode"],
            save_top_k=config["logging"]["save_top_k"],
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = WandbLogger(
        project=config["logging"]["project"],
        name=config["logging"]["name"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        strategy=config["training"]["strategy"],
        precision=config["training"]["precision"],
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )

    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
