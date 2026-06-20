from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.nn.utils import clip_grad_norm_

from neurosynth.connectome.losses import CombinedNeuroLoss
from neurosynth.connectome.types import BatchSequence

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class EarlyStoppingState:
    best: float = -1e9
    patience: int = 15
    counter: int = 0


class NeuroGNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        loss_weights: dict[str, float] | None = None,
        use_class_weights: bool = True,
        wandb_project: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = CombinedNeuroLoss(
            cls_weight=(loss_weights or {}).get("cls", 0.6),
            reg_weight=(loss_weights or {}).get("reg", 0.4),
        )
        self.use_class_weights = use_class_weights
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        self.accumulate_grad_batches = 4
        self.mixup_prob = 0.3
        self.early = EarlyStoppingState()
        self.best_checkpoints: list[tuple[float, Path]] = []

        mlflow.autolog(log_models=False)

    def _ece(self, probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        ece = 0.0
        edges = np.linspace(0, 1, bins + 1)
        for i in range(bins):
            sel = (conf >= edges[i]) & (conf < edges[i + 1])
            if sel.sum() == 0:
                continue
            acc = (pred[sel] == y_true[sel]).mean()
            avg_conf = conf[sel].mean()
            ece += (sel.mean()) * abs(acc - avg_conf)
        return float(ece)

    def _mixup(self, batch_seq: BatchSequence) -> tuple[BatchSequence, torch.Tensor | None, float | None]:
        if np.random.rand() > self.mixup_prob:
            return batch_seq, None, None

        lam = float(np.random.beta(0.4, 0.4))
        perm = torch.randperm(batch_seq.y_class.size(0))

        for graph_batch in batch_seq.graph_batches:
            if graph_batch is None:
                continue
            graph_batch.x = lam * graph_batch.x + (1.0 - lam) * graph_batch.x[torch.randperm(graph_batch.x.size(0))]

        mixed_targets = F.one_hot(batch_seq.y_class, num_classes=3).float() * lam + F.one_hot(batch_seq.y_class[perm], num_classes=3).float() * (1.0 - lam)
        return batch_seq, mixed_targets, lam

    def _metrics(self, out: dict[str, torch.Tensor], y_class: torch.Tensor, y_reg: torch.Tensor) -> dict[str, float]:
        logits = out["logits"].detach().cpu().numpy()
        probs = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
        y_cls = y_class.detach().cpu().numpy()
        pred_reg = out["cdrsb"].detach().cpu().numpy().ravel()
        y_reg_np = y_reg.detach().cpu().numpy().ravel()

        metrics: dict[str, float] = {}
        try:
            metrics["auroc_macro"] = float(roc_auc_score(y_cls, probs, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auroc_macro"] = 0.0
        try:
            y_oh = np.eye(3)[y_cls]
            metrics["auprc_macro"] = float(average_precision_score(y_oh, probs, average="macro"))
        except Exception:
            metrics["auprc_macro"] = 0.0

        metrics["mae"] = float(mean_absolute_error(y_reg_np, pred_reg))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_reg_np, pred_reg)))
        metrics["val_concordance_index"] = float(concordance_index(y_reg_np, -pred_reg))
        metrics["ece"] = self._ece(probs, y_cls)

        unc = out["uncertainty"].detach().cpu().numpy().ravel()
        abs_err = np.abs(pred_reg - y_reg_np)
        if np.std(unc) > 1e-8 and np.std(abs_err) > 1e-8:
            metrics["uncertainty_error_corr"] = float(np.corrcoef(unc, abs_err)[0, 1])
        else:
            metrics["uncertainty_error_corr"] = 0.0
        _ = logits
        return metrics

    def _step(
        self,
        batch_seq: BatchSequence,
        epoch: int,
        train: bool = True,
        y_class_soft: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        graph_seq = batch_seq.graph_batches
        t = batch_seq.time_deltas_months.to(self.device)
        m = batch_seq.padding_mask.to(self.device)
        y_cls = batch_seq.y_class.to(self.device)
        y_reg = batch_seq.y_regression.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            out = self.model(graph_seq, t, m)
            loss, parts = self.loss_fn(out, y_cls, y_reg, epoch=epoch, y_class_soft=y_class_soft.to(self.device) if y_class_soft is not None else None)

        metrics = self._metrics(out, y_cls, y_reg)
        metrics.update(parts)
        if train:
            metrics["train_loss"] = float(loss.detach())
        else:
            metrics["val_loss"] = float(loss.detach())
        return loss, metrics, out

    def train_epoch(self, loader, epoch: int = 0) -> dict[str, float]:
        self.model.train()
        all_metrics: list[dict[str, float]] = []

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch_seq in enumerate(loader):
            batch_seq, y_soft, _ = self._mixup(batch_seq)
            loss, metrics, _ = self._step(batch_seq, epoch=epoch, train=True, y_class_soft=y_soft)
            loss = loss / self.accumulate_grad_batches

            self.scaler.scale(loss).backward()
            if (step + 1) % self.accumulate_grad_batches == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
            all_metrics.append(metrics)

        torch.cuda.empty_cache()
        return {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}

    def val_epoch(self, loader, epoch: int = 0) -> tuple[dict[str, float], pd.DataFrame]:
        self.model.eval()
        all_metrics: list[dict[str, float]] = []
        rows: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch_seq in loader:
                loss, metrics, out = self._step(batch_seq, epoch=epoch, train=False)
                _ = loss
                all_metrics.append(metrics)

                probs = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
                pred_stage = probs.argmax(axis=1)
                pred_reg = out["cdrsb"].detach().cpu().numpy().ravel()
                unc = out["uncertainty"].detach().cpu().numpy().ravel()
                for i, pid in enumerate(batch_seq.patient_ids):
                    rows.append(
                        {
                            "patient_id": pid,
                            "pred_stage": int(pred_stage[i]),
                            "pred_cdrsb": float(pred_reg[i]),
                            "pred_uncertainty": float(unc[i]),
                            "target_stage": int(batch_seq.y_class[i].item()),
                            "target_cdrsb": float(batch_seq.y_regression[i].item()),
                        }
                    )

        metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        self._update_early(metrics.get("val_concordance_index", -1e9))
        return metrics, pd.DataFrame(rows)

    def _update_early(self, score: float) -> None:
        if score > self.early.best:
            self.early.best = score
            self.early.counter = 0
        else:
            self.early.counter += 1

    def should_stop(self) -> bool:
        return self.early.counter >= self.early.patience

    def save_checkpoint(self, epoch: int, metric: float, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"epoch_{epoch:03d}_cidx_{metric:.4f}.pt"
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "metric": metric}, ckpt_path)

        self.best_checkpoints.append((metric, ckpt_path))
        self.best_checkpoints = sorted(self.best_checkpoints, key=lambda x: x[0], reverse=True)
        to_remove = self.best_checkpoints[3:]
        self.best_checkpoints = self.best_checkpoints[:3]
        for _, path in to_remove:
            path.unlink(missing_ok=True)
