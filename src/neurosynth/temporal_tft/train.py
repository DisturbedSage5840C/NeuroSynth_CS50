# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.tuner.tuning import Tuner

from neurosynth.temporal_tft.dataset_factory import DatasetFactory
from neurosynth.temporal_tft.lightning_module import NeuroTFTLightningModule
from neurosynth.temporal_tft.model import NeuroTFT


def seed_everything(seed: int = 42) -> None:
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train-cutoff", type=str, required=True)
    parser.add_argument("--val-cutoff", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="artifacts/tft")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    seed_everything(42)
    df = pd.read_parquet(args.dataset)

    factory = DatasetFactory()
    train_ds, val_ds, test_ds = factory.create_datasets(df, args.train_cutoff, args.val_cutoff)

    train_dl = train_ds.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    model = NeuroTFT.from_dataset(train_ds)
    lm = NeuroTFTLightningModule(model.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_idx = {
        "train_patients": sorted(df[df["visit_date"] <= args.train_cutoff]["patient_id"].astype(str).unique().tolist()),
        "val_patients": sorted(df[(df["visit_date"] > args.train_cutoff) & (df["visit_date"] <= args.val_cutoff)]["patient_id"].astype(str).unique().tolist()),
        "test_patients": sorted(df[df["visit_date"] > args.val_cutoff]["patient_id"].astype(str).unique().tolist()),
    }
    pd.DataFrame({k: pd.Series(v) for k, v in split_idx.items()}).to_csv(out_dir / "split_indices.csv", index=False)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=NeuroTFTLightningModule.default_callbacks(),
        enable_checkpointing=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    tuner = Tuner(trainer)
    lr_find = tuner.lr_find(lm, train_dataloaders=train_dl, val_dataloaders=val_dl)
    if lr_find is not None and hasattr(lr_find, "suggestion") and lr_find.suggestion() is not None:
        lm.learning_rate = float(lr_find.suggestion())

    trainer.fit(lm, train_dl, val_dl)

    # Save normalization stats for inference-time consistency.
    norm_stats = pd.DataFrame(
        {
            "feature": ["dci"],
            "mean": [float(df["dci"].mean())],
            "std": [float(df["dci"].std(ddof=0) or 1.0)],
        }
    )
    norm_stats.to_csv(out_dir / "normalization_stats.csv", index=False)

    _ = test_ds


if __name__ == "__main__":
    main()
