"""Download and process PhysioNet neurological datasets into the v5 schema.

Datasets (all open-access, no credentials required):
  - PADS: Parkinson's Disease Smartwatch Dataset
    physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
  - Non-EEG Physiological Signals of Neurological Status
    physionet.org/content/noneeg-neurological-status/1.0.0/
  - COVID-19 Patients with Neurological Comorbidities (includes MS)
    physionet.org/content/covid-neurological-comorbidities/1.0.0/

For Tier-2 MIMIC-IV Demo (requires free PhysioNet account):
    Place oasis-mimic_demo_processed.csv in data/raw/physionet/ and re-run.

Usage:
    python scripts/data/v5/download_physionet.py [--out-dir data/raw/physionet]
    # Requires: pip install wfdb requests
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import io
import json
import time
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

_PHYSIONET_BASE = "https://physionet.org/files"

# PhysioNet dataset registry: db_name → (wfdb_db_id, version, disease, open_access)
_DATASETS = {
    "pads":   ("parkinsons-disease-smartwatch", "1.0.0", "Parkinson's Disease", True),
    "noneeg": ("noneeg-neurological-status",    "1.0.0", "Parkinson's Disease", True),
}

_RATE_LIMIT_DELAY = 1.5  # seconds between requests (PhysioNet rate-limits aggressively)


def _wfdb_download(db_id: str, version: str, dest: Path) -> bool:
    """Download a PhysioNet database using wfdb. Returns True on success."""
    try:
        import wfdb
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  wfdb: downloading {db_id} v{version} → {dest}")
        wfdb.dl_database(db_id, dl_dir=str(dest))
        return True
    except ImportError:
        print("  wfdb not installed — run: pip install wfdb")
        return False
    except Exception as exc:
        print(f"  wfdb download failed: {exc}")
        return False


def _scaffold(n: int, disease_type: str, data_source: str) -> pd.DataFrame:
    df = pd.DataFrame({col: [POP_DEFAULTS.get(col, np.nan)] * n for col in ALL_FEATURES})
    genomic = DISEASE_GENOMIC_PRIORS.get(disease_type, DISEASE_GENOMIC_PRIORS["Parkinson's Disease"])
    for col, val in genomic.items():
        df[col] = val
    df["DiseaseType"] = disease_type
    df["data_source"] = data_source
    return df


def _fetch_file(url: str, timeout: int = 60) -> bytes:
    time.sleep(_RATE_LIMIT_DELAY)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _list_physionet_files(db: str, version: str) -> list[str]:
    """List files in a PhysioNet database using the REST API."""
    url = f"https://physionet.org/rest/files/{db}/{version}/"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Returns list of {name, url, type} objects
        return [f["name"] for f in data if f.get("type") == "file"]
    except Exception:
        return []


# ─── PADS: Parkinson's Disease Smartwatch Dataset ────────────────────────────
# Structure (confirmed via HTTP): 469 patients
#   preprocessed/file_list.csv  — patient-level demographics + label
#   questionnaire/questionnaire_response_NNN.json — NMS/UPDRS per patient
#   patients/patient_NNN.json   — duplicate of file_list rows

_PADS_BASE = f"{_PHYSIONET_BASE}/parkinsons-disease-smartwatch/1.0.0"

_PADS_CONDITION_MAP = {
    "parkinson's": "Parkinson's Disease",
    "parkinsons": "Parkinson's Disease",
    "healthy": "Healthy",
    "multiple sclerosis": "Multiple Sclerosis",
    "essential tremor": "Parkinson's Disease",      # closest in schema
    "other movement disorders": "Parkinson's Disease",
    "atypical parkinsonism": "Parkinson's Disease",
}

# NMS questionnaire item IDs → v5 schema features
_NMS_ITEM_MAP = {
    "12": "MemoryComplaints",       # "Problems remembering things..."
    "13": "BehavioralProblems",     # "Loss of interest..."
    "15": "Confusion",              # "Difficulty concentrating..."
    "16": "Depression",             # "Feeling sad, low or blue"
    "17": "Disorientation",         # "Feeling anxious..."
    "21": "DifficultyCompletingTasks",  # "Falling"
    "23": "SleepQuality",           # "Difficulty getting to sleep" (inverted)
}


def _fetch_pads_questionnaires(n_patients: int, pads_dir: Path) -> dict[str, dict]:
    """Fetch NMS questionnaire JSONs and extract clinical feature scores."""
    features_by_subject: dict[str, dict] = {}
    print(f"  [pads] fetching questionnaires for {n_patients} patients...")
    fetched = 0
    for i in range(1, n_patients + 1):
        sid = f"{i:03d}"
        cache = pads_dir / f"questionnaire_{sid}.json"
        if cache.exists():
            raw = json.loads(cache.read_text())
        else:
            try:
                url = f"{_PADS_BASE}/questionnaire/questionnaire_response_{sid}.json"
                time.sleep(0.3)
                resp = requests.get(url, timeout=15)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                raw = resp.json()
                cache.write_text(json.dumps(raw))
                fetched += 1
            except Exception:
                continue

        # Parse NMS items
        items = raw.get("item", []) if isinstance(raw, dict) else []
        row: dict[str, float] = {}
        for item in items:
            link_id = str(item.get("link_id", "")).lstrip("0") or "0"
            if link_id in _NMS_ITEM_MAP:
                answer = item.get("answer", False)
                val = float(bool(answer))
                feat = _NMS_ITEM_MAP[link_id]
                if feat == "SleepQuality":
                    val = 1.0 - val  # sleep difficulty → invert to quality score
                row[feat] = val
        features_by_subject[sid] = row

    print(f"  [pads] fetched {fetched} new questionnaire JSONs, {len(features_by_subject)} total")
    return features_by_subject


def download_pads(out_dir: Path) -> pd.DataFrame | None:
    """
    PADS — 469 patients (276 PD, 79 healthy, 11 MS, + other movement disorders).
    Downloads preprocessed/file_list.csv directly and enriches with questionnaire NMS data.
    """
    pads_dir = out_dir / "pads"
    pads_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download file_list.csv (patient demographics + labels)
    file_list_path = pads_dir / "file_list.csv"
    if not file_list_path.exists():
        print("  [pads] downloading preprocessed/file_list.csv ...")
        try:
            resp = requests.get(f"{_PADS_BASE}/preprocessed/file_list.csv", timeout=30)
            resp.raise_for_status()
            file_list_path.write_bytes(resp.content)
            print(f"  [pads] saved file_list.csv ({len(resp.content)} bytes)")
        except Exception as exc:
            print(f"  [pads] failed to download file_list.csv: {exc}")
            return _load_local_pads(pads_dir)

    raw = pd.read_csv(file_list_path)
    print(f"  [pads] {len(raw)} patients: {raw['condition'].value_counts().to_dict()}")

    # 2. Fetch NMS questionnaire data (adds depression, memory, sleep, etc.)
    q_features = _fetch_pads_questionnaires(len(raw), pads_dir)

    return _process_pads(raw, q_features)


def _load_local_pads(pads_dir: Path) -> pd.DataFrame | None:
    """Process any CSV/TSV already placed in data/raw/physionet/pads/."""
    csvs = list(pads_dir.glob("*.csv")) + list(pads_dir.glob("*.tsv"))
    if not csvs:
        return None
    raw = pd.concat(
        [pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ",") for f in csvs],
        ignore_index=True
    )
    print(f"[pads] loaded {len(raw)} rows from {len(csvs)} local files")
    return _process_pads(raw)


def _process_pads(raw: pd.DataFrame, q_features: dict | None = None) -> pd.DataFrame:
    """Process PADS file_list.csv into v5 schema, enriched with questionnaire NMS data."""
    n = len(raw)
    if n == 0:
        return None

    col = {c.strip().lower().replace(" ", "_").replace("-", "_"): c for c in raw.columns}
    df = _scaffold(n, "Parkinson's Disease", "physionet_pads")

    # Demographics
    if "age" in col:
        df["Age"] = pd.to_numeric(raw[col["age"]], errors="coerce").clip(10, 110)
    if "gender" in col:
        df["Gender"] = (raw[col["gender"]].astype(str).str.lower() == "male").astype(float)
    if "weight" in col and "height" in col:
        w = pd.to_numeric(raw[col["weight"]], errors="coerce")
        h = pd.to_numeric(raw[col["height"]], errors="coerce") / 100  # cm → m
        df["BMI"] = (w / (h ** 2)).clip(10, 60)

    # Disease type + label from condition column
    if "condition" in col:
        conditions = raw[col["condition"]].astype(str).str.lower().str.strip()
        for idx, cond in enumerate(conditions):
            disease = next(
                (v for k, v in _PADS_CONDITION_MAP.items() if k in cond),
                "Parkinson's Disease"
            )
            df.loc[idx, "DiseaseType"] = disease
        df["risk_label"] = (~conditions.isin(["healthy"])).astype(int)
    elif "label" in col:
        df["risk_label"] = (pd.to_numeric(raw[col["label"]], errors="coerce") > 0).astype(int)

    # Update genomic priors per detected disease type
    for disease in df["DiseaseType"].unique():
        mask = df["DiseaseType"] == disease
        priors = DISEASE_GENOMIC_PRIORS.get(disease, {})
        for gcol, gval in priors.items():
            df.loc[mask, gcol] = gval

    # Enrich with NMS questionnaire features (per-subject)
    if q_features:
        id_col = col.get("id") or col.get("subject_id") or col.get("resource_id")
        for idx, row_id in enumerate(raw[id_col].astype(str).str.zfill(3) if id_col else pd.Series()):
            subject_feats = q_features.get(row_id, {})
            for feat, val in subject_feats.items():
                if feat in df.columns:
                    df.loc[idx, feat] = val

    print(f"[pads] {n} rows: {df['DiseaseType'].value_counts().to_dict()}")
    return df[ALL_FEATURES + META_COLS]


# ─── Non-EEG Neurological Status ─────────────────────────────────────────────

def download_noneeg(out_dir: Path) -> pd.DataFrame | None:
    """Non-EEG Neurological Status dataset — wfdb first, then HTTP, then local."""
    db_id, ver, _, _ = _DATASETS["noneeg"]
    noneeg_dir = out_dir / "noneeg"
    noneeg_dir.mkdir(parents=True, exist_ok=True)

    # 1. wfdb
    if not list(noneeg_dir.glob("*.csv")) and not list(noneeg_dir.glob("*.tsv")):
        _wfdb_download(db_id, ver, noneeg_dir)

    # 2. HTTP metadata files
    base_url = f"{_PHYSIONET_BASE}/{db_id}/{ver}"
    for fname in ("clinical_data.csv", "demographics.csv", "participants.tsv", "summary.csv", "data.csv"):
        url = f"{base_url}/{fname}"
        fpath = noneeg_dir / fname
        if fpath.exists():
            continue
        try:
            content = _fetch_file(url)
            fpath.write_bytes(content)
            print(f"[noneeg] fetched: {fname}")
        except Exception:
            continue

    raw_df = _load_local_noneeg(noneeg_dir)
    if raw_df is None:
        print("[noneeg] no data found. Download manually:")
        print("         https://physionet.org/content/noneeg-neurological-status/1.0.0/")
        print("         Place CSV/TSV files in data/raw/physionet/noneeg/ and re-run.")
    return raw_df


def _load_local_noneeg(noneeg_dir: Path) -> pd.DataFrame | None:
    csvs = list(noneeg_dir.glob("*.csv")) + list(noneeg_dir.glob("*.tsv"))
    if not csvs:
        return None
    raw = pd.concat(
        [pd.read_csv(f, sep="\t" if f.suffix == ".tsv" else ",") for f in csvs],
        ignore_index=True
    )
    return _process_noneeg(raw)


def _process_noneeg(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    if n == 0:
        return None

    col = {c.strip().lower().replace(" ", "_").replace("-", "_"): c for c in raw.columns}
    df = _scaffold(n, "Healthy", "physionet_noneeg")

    # Label — neurological status
    label_col = None
    for candidate in ("neurological_status", "status", "group", "diagnosis", "label", "condition"):
        if candidate in col:
            label_col = col[candidate]
            break

    if label_col:
        labels = raw[label_col].astype(str).str.lower()
        is_neuro = ~labels.isin(["healthy", "normal", "control", "0", "no"])
        df["risk_label"] = is_neuro.astype(int)
        # Assign disease type from label if informative
        for idx, label_val in enumerate(labels):
            for key, disease in {
                "parkinson": "Parkinson's Disease", "ms": "Multiple Sclerosis",
                "sclerosis": "Multiple Sclerosis", "epilep": "Epilepsy",
                "alzheimer": "Alzheimer's Disease",
            }.items():
                if key in label_val:
                    df.loc[idx, "DiseaseType"] = disease
                    break
        df.loc[~is_neuro, "DiseaseType"] = "Healthy"
    else:
        df["risk_label"] = 0

    # Demographics
    for src, dst in [("age", "Age"), ("sex", "Gender"), ("gender", "Gender")]:
        if src in col:
            df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")

    # Wearable physiological signals
    wearable_map = {
        "hr": "HR_variability", "heart_rate": "HR_variability", "hrv": "HR_variability",
        "spo2": "SpO2_mean", "oxygen_saturation": "SpO2_mean",
        "temperature": "actigraphy_activity_index",  # proxy; no direct mapping
        "acceleration": "tremor_amplitude", "acc_rms": "tremor_amplitude",
        "eda": "actigraphy_activity_index",
        "gait": "gait_velocity",
    }
    for src_key, dst in wearable_map.items():
        if src_key in col:
            df[dst] = pd.to_numeric(raw[col[src_key]], errors="coerce")

    print(f"[noneeg] processed {n} rows, neuro fraction: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── Local fallback: any CSVs manually placed in physionet/ root ──────────────

def _scan_local_csvs(out_dir: Path) -> list[pd.DataFrame]:
    """Process any CSVs manually downloaded and placed in out_dir."""
    frames = []
    for csv_path in out_dir.glob("*.csv"):
        try:
            raw = pd.read_csv(csv_path)
            print(f"[local] {csv_path.name}: {len(raw)} rows, cols: {list(raw.columns[:8])}")
            frames.append((csv_path.stem, raw))
        except Exception as exc:
            print(f"[local] failed to read {csv_path.name}: {exc}")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Download PhysioNet neurological datasets")
    ap.add_argument("--out-dir", default="data/raw/physionet")
    ap.add_argument("--skip-download", action="store_true",
                    help="Only process already-downloaded files in out-dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    # PADS
    print("\n=== PADS: Parkinson's Disease Smartwatch ===")
    try:
        df_pads = download_pads(out_dir) if not args.skip_download else _load_local_pads(out_dir / "pads")
        if df_pads is not None and len(df_pads) > 0:
            p = out_dir / "pads_v5.parquet"
            df_pads.to_parquet(p, index=False)
            print(f"Saved {len(df_pads)} rows → {p}")
            frames.append(df_pads)
    except Exception as exc:
        print(f"PADS failed: {exc}")

    # Non-EEG
    print("\n=== Non-EEG Neurological Status ===")
    try:
        df_noneeg = download_noneeg(out_dir) if not args.skip_download else _load_local_noneeg(out_dir / "noneeg")
        if df_noneeg is not None and len(df_noneeg) > 0:
            p = out_dir / "noneeg_v5.parquet"
            df_noneeg.to_parquet(p, index=False)
            print(f"Saved {len(df_noneeg)} rows → {p}")
            frames.append(df_noneeg)
    except Exception as exc:
        print(f"Non-EEG failed: {exc}")

    # Any other CSVs manually placed
    print("\n=== Scanning for manually downloaded files ===")
    for stem, raw in _scan_local_csvs(out_dir):
        # Minimal processing: just check for known disease/risk columns
        col = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}
        n = len(raw)
        disease = "Parkinson's Disease"
        df = _scaffold(n, disease, f"physionet_{stem}")
        for src, dst in [("age", "Age"), ("sex", "Gender"), ("mmse", "MMSE")]:
            if src in col:
                df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")
        df["risk_label"] = 0
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        p = out_dir / "physionet_combined_v5.parquet"
        combined.to_parquet(p, index=False)
        dist = combined["DiseaseType"].value_counts().to_dict()
        print(f"\nPhysioNet combined: {len(combined)} rows → {p}")
        print(f"Disease distribution: {dist}")
    else:
        print("\nNo PhysioNet data processed.")
        print("\nManual download instructions:")
        print("  PADS: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/")
        print("  Non-EEG: https://physionet.org/content/noneeg-neurological-status/1.0.0/")
        print("  MS/COVID: https://physionet.org/content/covid-neurological-comorbidities/")
        print("  Place downloaded CSVs in data/raw/physionet/ and re-run with --skip-download")


if __name__ == "__main__":
    main()
