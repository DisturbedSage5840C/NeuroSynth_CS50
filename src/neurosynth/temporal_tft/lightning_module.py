# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import StochasticWeightAveraging


class NeuroTFTLightningModule(pl.LightningModule):
    def __init__(self, tft_model, learning_rate: float = 4.2e-4) -> None:
        super().__init__()
        self.tft_model = tft_model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["tft_model"])

    def forward(self, x):
        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        out = self.tft_model.step(batch, batch_idx, "train")
        self.log("train_loss", out["loss"], prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.tft_model.step(batch, batch_idx, "val")
        self.log("val_loss", out["loss"], prog_bar=True)
        return out["loss"]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return opt

    @staticmethod
    def default_callbacks():
        return [StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=0.7)]
