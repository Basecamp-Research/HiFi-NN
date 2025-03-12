# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import wandb
import hydra
import random
import numpy as np
import time

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from models.hifinn_model import HifinnLayerNormResiduePL
from datasets.dataset import EmbeddingsAndLabelsDataset
from utils.file_utils import load_ids, load_json
from utils.dataset_utils import remove_overlap, pad_batch_embddings_and_labels
from losses.losses import OverlapLoss


def init_wandb(cfg):
    wandb.login()
    wandb.init(project=cfg.project_name, config=OmegaConf.to_object(cfg), resume=True)


@hydra.main(
    version_base=None, config_path="./configs/", config_name="overlap_loss_config"
)
def main(cfg: DictConfig) -> None:
    if cfg.wandb:
        wandb.login(key="insert key here")

    train_ids = load_ids(cfg.train_ids_path)
    test_ids = load_ids(cfg.test_ids_path)
    # just to be absolutely certain there is no overlap
    test_ids = remove_overlap(test_ids, train_ids)

    # --- 1. Instantiate data set and data loader classes --- #
    train_ds = EmbeddingsAndLabelsDataset(
        filepath=cfg.embeddings_path,
        ids=train_ids,
        id_to_ec=load_json(cfg.annotations),
        train_on_classes=cfg.train_on_classes,
    )

    val_ds = EmbeddingsAndLabelsDataset(
        filepath=cfg.embeddings_path,
        ids=test_ids,
        id_to_ec=load_json(cfg.annotations),
        train_on_classes=False,
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=pad_batch_embddings_and_labels,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=pad_batch_embddings_and_labels,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    # --- 2. Model and loss function --- #
    criterion = OverlapLoss()

    if cfg.wandb:
        wandb_logger = WandbLogger(project="hifinn", job_type="train")
    model = HifinnLayerNormResiduePL(
        criterion=criterion,
        learning_rate=cfg.optimiser.lr,
        weight_decay=cfg.optimiser.weight_decay,
        epochs=cfg.epochs,
        min_lr=cfg.optimiser.min_lr,
        normalize=cfg.normalize,
        num_warmup_epochs=cfg.optimiser.num_warmup_epochs,
        hidden_size_1=cfg.model.hidden_size_1,
        output_size=cfg.model.output_size,
        per_residue_embedding=True,
        eps=cfg.optimiser.eps,
        betas=cfg.optimiser.betas,
    )

    if cfg.wandb:
        # log gradients
        wandb_logger.watch(model, log_freq=20)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_path,
        monitor="val_loss",
        every_n_epochs=5,
        save_top_k=3,
        save_last=True,
    )

    # learning rate logger callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # load model from checkpoint if it exists
    if cfg.checkpoint_path is not None:
        print(f"loading model from checkpoint!\n")
        model.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            criterion=criterion,
            learning_rate=cfg.optimiser.lr,
            weight_decay=cfg.optimiser.weight_decay,
            epochs=cfg.epochs,
            min_lr=cfg.optimiser.min_lr,
            normalize=cfg.normalize,
            hidden_size_1=cfg.model.hidden_size_1,
            output_size=cfg.model.output_size,
            per_residue_embedding=True,
        )

    trainer = Trainer(
        devices=cfg.devices,  # specify which devices to train on here!
        strategy=cfg.strategy,
        max_epochs=cfg.epochs,
        logger=wandb_logger if cfg.wandb else None,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=cfg.precision,
        gradient_clip_val=cfg.optimiser.gradient_clip_val,
        deterministic=True,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
