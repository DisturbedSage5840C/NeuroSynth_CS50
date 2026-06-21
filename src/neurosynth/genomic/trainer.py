# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from neurosynth.genomic.losses import WeightedMultiTaskLoss

try:
    from opacus import PrivacyEngine
except Exception:  # pragma: no cover
    PrivacyEngine = None


@dataclass
class TrainerConfig:
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dnabert_lr: float = 1e-5
    model_lr: float = 3e-4
    epsilon: float = 3.0
    delta: float = 1e-5


class GenomicTrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: TrainerConfig | None = None) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config or TrainerConfig()
        self.loss_fn = WeightedMultiTaskLoss()

        dnabert_params = []
        other_params = []
        for name, p in model.named_parameters():
            if "dnabert" in name.lower():
                dnabert_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": dnabert_params, "lr": self.config.dnabert_lr, "weight_decay": self.config.weight_decay},
                {"params": other_params, "lr": self.config.model_lr, "weight_decay": self.config.weight_decay},
            ]
        )
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=10000)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[self.config.warmup_steps])

        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.best_val_corr = -1.0
        self.patience = 15
        self.bad_epochs = 0

    def enable_differential_privacy(self, train_loader) -> any:
        if PrivacyEngine is None:
            return train_loader
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, private_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=100,
            target_epsilon=self.config.epsilon,
            target_delta=self.config.delta,
            max_grad_norm=1.0,
        )
        return private_loader

    def _move_batch(self, batch: dict) -> tuple[dict, dict]:
        x = {
            "variant_features": batch["variant_features"].to(self.device),
            "sequence_context": batch["sequence_context"].to(self.device),
            "gene_ids": batch["gene_ids"].to(self.device),
            "consequence_category": batch["consequence_category"].to(self.device),
            "variant_mask": batch["variant_mask"].to(self.device),
            "apoe_e4_count": batch["labels"]["apoe_count"].to(self.device),
        }
        y = {k: v.to(self.device) for k, v in batch["labels"].items()}
        return x, y

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        losses = []
        prs_corrs = []

        for batch in loader:
            x, y = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.model(**x)
                loss, parts = self.loss_fn(out, y)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                pred = out["prs_pred"].detach().cpu().numpy().reshape(-1)
                target = y["prs"].detach().cpu().numpy().reshape(-1)
                prs_corrs.append(float(pearsonr(pred, target).statistic if len(pred) > 1 else 0.0))
            _ = parts

        return {"train_loss": float(np.mean(losses)), "train_prs_correlation": float(np.mean(prs_corrs))}

    @torch.no_grad()
    def val_epoch(self, loader) -> dict[str, float]:
        self.model.eval()
        losses = []
        prs_corrs = []

        for batch in loader:
            x, y = self._move_batch(batch)
            out = self.model(**x)
            loss, _ = self.loss_fn(out, y)
            losses.append(float(loss.detach().cpu()))

            pred = out["prs_pred"].detach().cpu().numpy().reshape(-1)
            target = y["prs"].detach().cpu().numpy().reshape(-1)
            prs_corrs.append(float(pearsonr(pred, target).statistic if len(pred) > 1 else 0.0))

        return {"val_loss": float(np.mean(losses)), "val_prs_correlation": float(np.mean(prs_corrs))}

    def update_early_stopping(self, val_metrics: dict[str, float]) -> bool:
        corr = val_metrics.get("val_prs_correlation", -1.0)
        if corr > self.best_val_corr:
            self.best_val_corr = corr
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def save_checkpoint(self, out_dir: Path, epoch: int, val_metrics: dict[str, float]) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f"genomic_epoch_{epoch:03d}_corr_{val_metrics.get('val_prs_correlation', 0.0):.4f}.pt"
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt)
        return ckpt
