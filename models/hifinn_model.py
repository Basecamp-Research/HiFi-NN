# Copyright (c) 2024 Basecamp Research
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Part of this code depends on code from the open source project ESM
# https://github.com/facebookresearch/esm/
#
#
# MIT License
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import typing
import time
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from models.transformer import TransformerLayer, ESM1bLayerNorm
from esm import pretrained
from esm.modules import gelu


class HifinnLayerNormResiduePL(pl.LightningModule):
    def __init__(
        self,
        criterion: nn.modules.loss._Loss,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        min_lr: float,
        normalize: bool = False,
        num_warmup_epochs: int = 10,
        hidden_size_1: int = 1024,
        output_size: int = 512,
        per_residue_embedding: bool = False,
        padding_value: float = 0.0,
        eps: float = 1e-08,
        betas: typing.Tuple[float, float] = (0.9, 0.999),
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.min_lr = min_lr
        self.fc1 = nn.Linear(1280, hidden_size_1)
        self.gelu = gelu  # nn.GELU()
        self.ln1 = nn.LayerNorm(hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, output_size)
        self.normalize = normalize
        self.per_residue_embedding = per_residue_embedding
        self.padding_value = padding_value
        self.num_warmup_epochs = num_warmup_epochs
        self.eps = eps
        self.betas = betas
        self.return_attention = return_attention
        if self.per_residue_embedding:
            self.per_residue_attention = TransformerLayer(
                embed_dim=1280,
                ffn_embed_dim=4 * 1280,
                attention_heads=20,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
            )
            self.emb_layer_norm = ESM1bLayerNorm(1280)

    def training_step(
        self, batch: typing.Tuple[torch.Tensor, typing.List], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Args:
            batch: tuple of tensor and a list of labels for each sequence.
        """
        emb, labels = batch
        model_emb = self.forward(emb)
        loss = self.criterion(model_emb, labels)
        self.log("train_loss", loss, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: typing.Tuple[torch.Tensor, typing.List], batch_idx: int
    ):
        emb, labels = batch
        with torch.no_grad():
            model_emb = self.forward(emb)
            loss = self.criterion(model_emb, labels)
        self.log(
            "val_loss", loss, on_epoch=True, logger=True, on_step=False, sync_dist=True
        )
        return loss

    def forward(
        self, X: torch.Tensor
    ) -> torch.Tensor | typing.Tuple[torch.Tensor, torch.Tensor]:
        b_size, seq_len, dim = X.shape
        if self.per_residue_embedding:
            # flatten 'levels' dimension
            # before input to transformer layer
            # the padding mask should be indices along the sequence which have been padded
            # we sum over the last dimension as we have padded this entire dimension with 0's
            # therefore the sum should be 0 too
            # transpose so b_size is the first dimension
            padding_mask = (
                X.sum(dim=-1).eq(self.padding_value).to(torch.bool).to(X.device)
            )
            # Multiply our input X by the padding mask before input to attention mechanism
            X = X * (1 - padding_mask.unsqueeze(-1).expand(X.size()).type_as(X))

            X = X.reshape(seq_len, b_size, dim)
            # X = self.ln1(X)
            X, attn = self.per_residue_attention(
                X, self_attn_padding_mask=padding_mask.reshape(b_size, seq_len)
            )
            X = self.emb_layer_norm(X)
            X = X.reshape(b_size, seq_len, -1)
            avg_mask = 1 - padding_mask.unsqueeze(-1).expand(X.size()).type_as(X)
            X = torch.sum(X * avg_mask, dim=1) / torch.sum(
                avg_mask, dim=1
            )  # average over sequence length
        X = self.fc1(X)
        X = self.ln1(X)
        # X = self.dropout1(X)
        X = self.gelu(X)
        X = self.fc2(X)
        if self.return_attention:
            if self.normalize:
                return F.normalize(X, dim=-1), attn
            else:
                return X, attn
        else:
            if self.normalize:
                return F.normalize(X, dim=-1)
            else:
                return X

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.eps,
            betas=self.betas,
        )
        # learning rate warmup
        linear_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=self.num_warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.epochs - self.num_warmup_epochs,
            eta_min=self.min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[linear_lr, cosine],
            milestones=[self.num_warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "interval": "epoch",
            "frequency": 50,
        }

    def on_train_epoch_end(self) -> None:
        seed = int(time.time()) + self.current_epoch
        pl.seed_everything(seed, workers=True)

    def _init_esm_modules(self):
        """
        Initialised transformer layer with weights from final layer of ESM.
        """
        model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        # loop over weights in pretrained model and add our
        # 1. filter out unnecessary keys
        pretrained_dict = {
            "per_residue_attention." + k.strip("layers.33."): v
            for k, v in model.state_dict().items()
            if k.startswith("layers.33")
        }
        # 2. overwrite entries in the existing state dict
        self.state_dict().update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(
            pretrained_dict, strict=False
        )  # only load relevant weights
        model, alphabet = None, None
