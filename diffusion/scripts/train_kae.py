#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kae import KoopmanAutoencoder
from src.data.moving_mnist import create_moving_mnist_dataloader
from src.training.trainer import KAELightningModule

torch.set_float32_matmul_precision('medium')


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = KoopmanAutoencoder(
        input_channels=config["model"]["input_channels"],
        latent_dim=config["model"]["latent_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        image_size=config["model"]["image_size"],
    )

    if config["training"].get("strategy") == "ddp":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    lit_model = KAELightningModule(
        model=model,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        recon_weight=float(config["training"]["recon_weight"]),
        dynamics_weight=float(config["training"]["dynamics_weight"]),
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

    ckpt_config = config["checkpointing"]
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_config["dirpath"],
            filename="kae-{epoch:02d}-{val/loss:.4f}",
            monitor=ckpt_config["monitor"],
            mode=ckpt_config["mode"],
            save_top_k=config["logging"]["save_top_k"],
            save_last=ckpt_config.get("save_last", True),
            every_n_epochs=ckpt_config.get("every_n_epochs", 1),
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
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )

    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kae_movingmnist.yaml")
    args = parser.parse_args()
    main(args.config)
