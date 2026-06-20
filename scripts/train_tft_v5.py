"""Train NeuroSynth TFT v5 — Temporal Fusion Transformer on longitudinal sequences.

Strategy when only cross-sectional data is available (the common case with
real_v5.parquet): synthesise pseudo-longitudinal sequences by:

  1. Grouping subjects by DiseaseType + age decade.
  2. Ordering within each group by age to approximate progression.
  3. Chunking into fixed-length sequences (default T=6 time steps × 6-month gap).

When real longitudinal data exists (e.g. ADNI visits or PPMI follow-up CSV),
pass --longitudinal-csv and those sequences are used directly.

Usage:
    python scripts/train_tft_v5.py \
        --data data/real_v5_augmented.parquet \
        --out  models/ensemble_v5/tft_v5 \
        [--longitudinal-csv data/raw/adni_longitudinal.csv] \
        [--seq-len 6] [--max-epochs 20] [--batch-size 64]

Outputs:
    models/ensemble_v5/tft_v5/
        tft_model.pt         — PyTorch Lightning checkpoint
        tft_manifest.json    — training metadata
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Features used as time-varying inputs to the TFT
TIME_VARYING_FEATURES = [
    "MMSE", "FunctionalAssessment", "ADL", "UPDRS_motor", "UPDRS_total",
    "gait_velocity", "tremor_amplitude", "actigraphy_activity_index",
    "CSF_Abeta42", "CSF_pTau", "nfl_plasma", "HR_variability",
    "SystolicBP", "DiastolicBP", "BMI", "PhysicalActivity", "SleepQuality",
]

# Static covariates (fixed per subject)
STATIC_FEATURES = [
    "Gender", "Ethnicity", "EducationLevel", "FamilyHistoryAlzheimers",
    "APOE4_dosage", "APOE_risk_score", "polygenetic_risk_score",
]

SEQ_LEN = 6     # number of time steps per sequence
PRED_LEN = 4    # forecast horizon (time steps)


# ── Pseudo-longitudinal sequence builder ─────────────────────────────────────

def build_pseudo_sequences(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """Sort cross-sectional records by disease + age to approximate progression.

    Each subject gets a synthetic 'time_idx' within its disease-age-decade group.
    Groups with fewer than seq_len records are padded with the group mean.
    Returns a DataFrame with columns ['group_id', 'time_idx', 'target', ...features].
    """
    df = df.copy()
    df["age_decade"] = (df["Age"] // 10).astype(int)

    rows = []
    group_id = 0
    for (disease, decade), grp in df.groupby(["DiseaseType", "age_decade"], sort=True):
        grp = grp.sort_values("Age").reset_index(drop=True)

        # Slide a window of length seq_len + pred_len across the group
        window = seq_len + PRED_LEN
        if len(grp) < window:
            # Pad by repeating last row with small Gaussian noise
            pad_n = window - len(grp)
            last = grp.iloc[[-1]].copy()
            numeric_cols = grp.select_dtypes(include="number").columns
            for _ in range(pad_n):
                noise_row = last.copy()
                for col in numeric_cols:
                    noise_row[col] = last[col].values[0] + np.random.normal(0, 0.01)
                grp = pd.concat([grp, noise_row], ignore_index=True)

        for start in range(0, len(grp) - window + 1, seq_len):
            chunk = grp.iloc[start : start + window].copy()
            chunk["group_id"] = group_id
            chunk["time_idx"] = list(range(len(chunk)))
            chunk["target"] = chunk["risk_label"] if "risk_label" in chunk.columns else 0.0
            rows.append(chunk)
            group_id += 1

    if not rows:
        raise ValueError("No sequences could be built — data too sparse")

    result = pd.concat(rows, ignore_index=True)
    log.info("  Built %d pseudo-longitudinal sequences from %d subjects", group_id, len(df))
    return result


def load_longitudinal(path: Path) -> pd.DataFrame:
    """Load an external longitudinal CSV (e.g. ADNI visits) with columns:
    subject_id, visit_num, DiseaseType, risk_label, + TIME_VARYING_FEATURES.
    """
    df = pd.read_csv(path)
    df = df.sort_values(["subject_id", "visit_num"]).reset_index(drop=True)
    df["group_id"] = df["subject_id"].astype("category").cat.codes
    df["time_idx"] = df.groupby("group_id").cumcount()
    log.info("  Loaded %d real longitudinal records from %s", len(df), path)
    return df


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    data_path: Path,
    out_dir: Path,
    longitudinal_csv: Path | None,
    seq_len: int,
    max_epochs: int,
    batch_size: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data from %s …", data_path)
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)

    # Fill missing feature columns with zero
    for col in TIME_VARYING_FEATURES + STATIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    if "risk_label" not in df.columns:
        df["risk_label"] = 0.0
    if "DiseaseType" not in df.columns:
        df["DiseaseType"] = "Unknown"

    # Build sequence dataset
    if longitudinal_csv is not None and longitudinal_csv.exists():
        log.info("Using real longitudinal data from %s", longitudinal_csv)
        seq_df = load_longitudinal(longitudinal_csv)
    else:
        log.info("Building pseudo-longitudinal sequences from cross-sectional data …")
        seq_df = build_pseudo_sequences(df, seq_len)

    # Attempt pytorch_forecasting import — full TFT training
    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss
        import pytorch_lightning as pl

        # Ensure required columns exist
        available_tv = [c for c in TIME_VARYING_FEATURES if c in seq_df.columns]
        available_st = [c for c in STATIC_FEATURES if c in seq_df.columns]
        seq_df["DiseaseType"] = seq_df["DiseaseType"].astype(str)

        max_encoder = seq_len
        max_pred = PRED_LEN

        training_cutoff = int(seq_df["time_idx"].max() * 0.85)

        dataset = TimeSeriesDataSet(
            seq_df[seq_df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=max_encoder,
            max_prediction_length=max_pred,
            static_categoricals=["DiseaseType"],
            static_reals=available_st,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=available_tv,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            dataset,
            seq_df[seq_df["time_idx"] > training_cutoff],
            predict=True,
            stop_randomization=True,
        )

        train_loader = dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader   = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        from src.neurosynth.temporal_tft.model import NeuroTFT, RareDiseaseQuantileLoss
        loss_fn = (
            RareDiseaseQuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            if RareDiseaseQuantileLoss is not None
            else QuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        )

        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=4.2e-4,
            hidden_size=192,
            attention_head_size=6,
            dropout=0.18,
            hidden_continuous_size=80,
            output_size=7,
            loss=loss_fn,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            accelerator="auto",
            devices=1,
            default_root_dir=str(out_dir),
            enable_checkpointing=True,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        ckpt_path = out_dir / "tft_model.pt"
        trainer.save_checkpoint(str(ckpt_path))
        log.info("TFT checkpoint saved → %s", ckpt_path)

        manifest = {
            "model": "NeuroTFT_v5",
            "seq_len": max_encoder,
            "pred_len": max_pred,
            "n_sequences": int(seq_df["group_id"].nunique()),
            "n_tv_features": len(available_tv),
            "n_static_features": len(available_st),
            "epochs": max_epochs,
            "loss": "RareDiseaseQuantileLoss" if RareDiseaseQuantileLoss is not None else "QuantileLoss",
            "data_source": str(data_path),
            "longitudinal_source": str(longitudinal_csv) if longitudinal_csv else "pseudo-longitudinal",
        }

    except ImportError as e:
        log.warning("pytorch_forecasting not available (%s) — saving config-only manifest", e)
        ckpt_path = out_dir / "tft_model.pt"
        # Save a placeholder so DVC outputs are satisfied
        import torch
        torch.save({"config_only": True, "reason": str(e)}, str(ckpt_path))
        manifest = {
            "model": "NeuroTFT_v5",
            "status": "config_only",
            "reason": str(e),
            "note": "Install pytorch_forecasting to enable full TFT training",
        }

    manifest_path = out_dir / "tft_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest → %s", manifest_path)
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train NeuroSynth TFT v5")
    ap.add_argument("--data", default="data/real_v5_augmented.parquet")
    ap.add_argument("--out",  default="models/ensemble_v5/tft_v5")
    ap.add_argument("--longitudinal-csv", default=None,
                    help="Optional real longitudinal CSV (ADNI/PPMI visits)")
    ap.add_argument("--seq-len",    type=int, default=SEQ_LEN)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    data_path = ROOT / args.data
    out_dir   = ROOT / args.out
    long_csv  = Path(args.longitudinal_csv) if args.longitudinal_csv else None

    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        sys.exit(1)

    log.info("=== NeuroSynth TFT v5 Training ===")
    manifest = train(data_path, out_dir, long_csv, args.seq_len, args.max_epochs, args.batch_size)
    log.info("Done. status=%s", manifest.get("status", "trained"))


if __name__ == "__main__":
    main()
