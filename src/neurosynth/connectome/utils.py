# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_normative_stats(path: Path, feature_dim: int = 128) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros(feature_dim, dtype=np.float32), np.ones(feature_dim, dtype=np.float32)

    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    mean = np.asarray(payload.get("mean", [0.0] * feature_dim), dtype=np.float32)
    std = np.asarray(payload.get("std", [1.0] * feature_dim), dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def month_delta(days: int) -> float:
    return float(days / 30.4375)


def otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_edges[:-1])
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.clip(omega * (1.0 - omega), 1e-12, None)
    idx = int(np.argmax(sigma_b))
    return float(bin_edges[idx])


def setup_ddp(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return model
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
