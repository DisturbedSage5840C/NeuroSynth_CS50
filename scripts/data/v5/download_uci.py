"""Download and process UCI Parkinson's datasets into the v5 schema.

Datasets (both open, no credentials):
  - UCI Parkinson's Classic    (195 rows, 22 voice biomarkers, binary status)
    https://archive.ics.uci.edu/static/public/174/parkinsons.zip
  - UCI Parkinson's Telemonitoring (5,875 rows, UPDRS motor/total + voice)
    https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip

Usage:
    python scripts/data/v5/download_uci.py [--out-dir data/raw/uci]
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scripts.data.v5.schema import (
    ALL_FEATURES,
    DISEASE_GENOMIC_PRIORS,
    META_COLS,
    POP_DEFAULTS,
)

_URLS = {
    "classic": "https://archive.ics.uci.edu/static/public/174/parkinsons.zip",
    "telemonitoring": "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip",
}

_CLASSIC_VOICE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]


def _download_zip(url: str) -> bytes:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _scaffold(n: int, disease_type: str, data_source: str) -> pd.DataFrame:
    """Build a schema-complete skeleton for n rows with population defaults."""
    df = pd.DataFrame({col: [POP_DEFAULTS.get(col, np.nan)] * n for col in ALL_FEATURES})
    genomic = DISEASE_GENOMIC_PRIORS.get(disease_type, DISEASE_GENOMIC_PRIORS["Parkinson's Disease"])
    for col, val in genomic.items():
        df[col] = val
    df["DiseaseType"] = disease_type
    df["data_source"] = data_source
    return df


def _add_noise(series: pd.Series, pct: float = 0.05) -> pd.Series:
    """Add small multiplicative noise so repeated subjects don't create exact duplicates."""
    return series * (1 + np.random.default_rng(42).uniform(-pct, pct, len(series)))


def process_classic(raw_bytes: bytes) -> pd.DataFrame:
    """UCI Parkinson's Classic — 195 rows, 22 voice features + binary status."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        # find the .data or .csv file inside
        csv_name = next(
            (n for n in zf.namelist() if n.endswith(".data") or n.endswith(".csv")), None
        )
        if csv_name is None:
            raise FileNotFoundError("parkinsons.data not found inside classic zip")
        content = zf.read(csv_name).decode("utf-8")

    raw = pd.read_csv(io.StringIO(content))
    n = len(raw)
    df = _scaffold(n, "Parkinson's Disease", "uci_parkinsons_classic")

    # Voice features → wearable proxies (best available mapping)
    # Jitter / shimmer → tremor amplitude proxy
    if "MDVP:Jitter(%)" in raw.columns:
        df["tremor_amplitude"] = raw["MDVP:Jitter(%)"].clip(0, 5)
    if "HNR" in raw.columns:
        # HNR (harmonics-to-noise) inversely tracks voice disorder severity
        df["HR_variability"] = (raw["HNR"] / 40.0 * 80).clip(10, 100)
    if "DFA" in raw.columns:
        df["actigraphy_activity_index"] = raw["DFA"].clip(0, 1)

    # UPDRS is not in classic dataset — leave at population defaults
    # Binary label: status 1 = PD, 0 = healthy
    status_col = "status" if "status" in raw.columns else "UPDRS"
    if status_col == "status":
        df["risk_label"] = raw["status"].astype(int)
        # Those with status=0 are healthy controls enrolled in study
        df.loc[raw["status"] == 0, "DiseaseType"] = "Healthy"
        df.loc[raw["status"] == 0, "risk_label"] = 0

    # Age: classic dataset doesn't include age; use PD population mean ~65
    df["Age"] = np.random.default_rng(1).normal(65, 8, n).clip(40, 90)

    print(f"[uci_classic] {n} rows, PD fraction: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


def process_telemonitoring(raw_bytes: bytes) -> pd.DataFrame:
    """UCI Parkinson's Telemonitoring — 5,875 rows, UPDRS scores + age/sex."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_name = next(
            (n for n in zf.namelist() if n.endswith(".data") or n.endswith(".csv")), None
        )
        if csv_name is None:
            raise FileNotFoundError("telemonitoring file not found inside zip")
        content = zf.read(csv_name).decode("utf-8")

    raw = pd.read_csv(io.StringIO(content))
    # Expected columns: subject#, age, sex, test_time, motor_UPDRS, total_UPDRS, + 16 voice
    n = len(raw)
    df = _scaffold(n, "Parkinson's Disease", "uci_parkinsons_telemonitoring")

    col_map = {c.lower(): c for c in raw.columns}

    if "age" in col_map:
        df["Age"] = raw[col_map["age"]].clip(40, 90)
    if "sex" in col_map:
        df["Gender"] = raw[col_map["sex"]].astype(float)

    motor_col = col_map.get("motor_updrs") or col_map.get("motor updrs")
    total_col = col_map.get("total_updrs") or col_map.get("total updrs")
    if motor_col:
        df["UPDRS_motor"] = raw[motor_col].clip(0, 108)
    if total_col:
        df["UPDRS_total"] = raw[total_col].clip(0, 176)

    # Voice → tremor proxy
    jitter_col = next((c for c in raw.columns if "jitter" in c.lower() and "%" in c), None)
    if jitter_col:
        df["tremor_amplitude"] = raw[jitter_col].clip(0, 5)
    hnr_col = next((c for c in raw.columns if c.upper() == "HNR"), None)
    if hnr_col:
        df["HR_variability"] = (raw[hnr_col] / 40.0 * 80).clip(10, 100)

    # All subjects in this dataset are PD patients; high UPDRS = high risk
    motor = df["UPDRS_motor"].fillna(POP_DEFAULTS["UPDRS_motor"])
    df["risk_label"] = (motor >= 20).astype(int)

    print(f"[uci_telemonitoring] {n} rows, high-risk fraction: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download UCI Parkinson's datasets")
    ap.add_argument("--out-dir", default="data/raw/uci")
    ap.add_argument("--skip-download", action="store_true",
                    help="Use already-downloaded zips in out-dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for name, url in _URLS.items():
        zip_path = out_dir / f"{name}.zip"
        if args.skip_download and zip_path.exists():
            print(f"[{name}] using cached {zip_path}")
            raw_bytes = zip_path.read_bytes()
        else:
            print(f"[{name}] downloading {url} ...")
            try:
                raw_bytes = _download_zip(url)
                zip_path.write_bytes(raw_bytes)
                print(f"[{name}] saved {len(raw_bytes) // 1024} KB → {zip_path}")
            except Exception as exc:
                print(f"[{name}] download failed: {exc} — skipping")
                continue

        try:
            if name == "classic":
                df = process_classic(raw_bytes)
            else:
                df = process_telemonitoring(raw_bytes)
            out_path = out_dir / f"uci_{name}_v5.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[{name}] saved {len(df)} rows → {out_path}")
            frames.append(df)
        except Exception as exc:
            print(f"[{name}] processing failed: {exc}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined_path = out_dir / "uci_combined_v5.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nUCI combined: {len(combined)} rows → {combined_path}")
    else:
        print("No UCI data processed.")


if __name__ == "__main__":
    main()
