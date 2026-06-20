from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from neurosynth.genomic.dataset import GenomicDataset, genomic_collate_fn
from neurosynth.genomic.model import HierarchicalVariantTransformer
from neurosynth.genomic.trainer import GenomicTrainer, TrainerConfig

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def build_gene_vocab(hdf5_path: Path) -> dict[str, int]:
    vocab = {"UNK": 0}
    with h5py.File(hdf5_path, "r") as h5:
        for pid in h5.keys():
            genes = h5[pid]["gene_symbols"][...]
            for g in genes:
                gs = g.decode("utf-8") if isinstance(g, bytes) else str(g)
                if gs not in vocab:
                    vocab[gs] = len(vocab)
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroSynth Genomic Transformer")
    parser.add_argument("--hdf5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default="artifacts/genomic")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    with h5py.File(hdf5_path, "r") as h5:
        patient_ids = list(h5.keys())

    split = int(len(patient_ids) * 0.8)
    train_ids = patient_ids[:split]
    val_ids = patient_ids[split:]

    train_ds = GenomicDataset(hdf5_path=hdf5_path, patient_ids=train_ids)
    val_ds = GenomicDataset(hdf5_path=hdf5_path, patient_ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=genomic_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=genomic_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_vocab = build_gene_vocab(hdf5_path)
    model = HierarchicalVariantTransformer(gene_vocab=gene_vocab)
    model = torch.compile(model) if hasattr(torch, "compile") else model

    trainer = GenomicTrainer(model=model, device=device, config=TrainerConfig())
    train_loader = trainer.enable_differential_privacy(train_loader)

    out_dir = Path(args.out_dir)
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.val_epoch(val_loader)
        trainer.save_checkpoint(out_dir=out_dir, epoch=epoch, val_metrics=val_metrics)
        print({"epoch": epoch, **train_metrics, **val_metrics})
        if trainer.update_early_stopping(val_metrics):
            break


if __name__ == "__main__":
    main()
