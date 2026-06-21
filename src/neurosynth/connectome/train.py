# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neurosynth.connectome.dataset import TemporalBrainDataset, TemporalLengthBatchSampler, collate_temporal_batch
from neurosynth.connectome.distributed import cleanup_distributed, init_distributed
from neurosynth.connectome.model import BrainConnectomeGNN
from neurosynth.connectome.trainer import NeuroGNNTrainer
from neurosynth.connectome.utils import set_seed, setup_ddp


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroSynth Brain Connectome GNN")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--cohort", type=str, default="ADNI")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default="artifacts/connectome")
    args = parser.parse_args()

    set_seed(42)
    rank, world_size = init_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TemporalBrainDataset(root=args.root, cohort=args.cohort)

    sampler = TemporalLengthBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_temporal_batch)

    model = BrainConnectomeGNN()
    if world_size > 1 and device.type == "cuda":
        model = setup_ddp(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    trainer = NeuroGNNTrainer(model=model, optimizer=optimizer, scheduler=scheduler, device=device)

    out_dir = Path(args.out_dir)
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(loader, epoch=epoch)
        val_metrics, _ = trainer.val_epoch(loader, epoch=epoch)

        if rank == 0:
            cidx = val_metrics.get("val_concordance_index", 0.0)
            trainer.save_checkpoint(epoch=epoch, metric=cidx, out_dir=out_dir)
            print({"epoch": epoch, **train_metrics, **val_metrics})

        if trainer.should_stop():
            break

    cleanup_distributed()


if __name__ == "__main__":
    main()
