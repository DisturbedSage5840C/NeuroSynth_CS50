"""Process OASIS-1/2/3 data into the v5 56-feature schema.

Supports all three OASIS releases:
  - OASIS-1: 416 subjects, cross-sectional MRI
  - OASIS-2: 373 subjects longitudinal (supersedes old process_oasis.py)
  - OASIS-3: 1,098 subjects, 2,842 sessions — adds CSF biomarkers + PET + genetics

Register at https://www.oasis-brains.org (takes ~1 day DUA approval).
User says they have a registered account.

Expected files (place in data/raw/oasis/ after downloading):
  OASIS-1:  oasis_cross-sectional.csv
  OASIS-2:  oasis_longitudinal.csv
  OASIS-3:  OASIS3_participants.tsv  (or oasis3_demographics.csv)
             OASIS3_UDSb4.csv        (CDR + cognitive scores)
             OASIS3_MRI_fseg.csv     (FreeSurfer imaging: hippocampus, entorhinal, etc.)
             OASIS3_PUP_puptimecourse.csv  (PET amyloid/tau — optional)
             OASIS3_ADRC_Clinical.csv     (CSF: Abeta42, pTau, tTau)

Usage:
    python scripts/data/v5/process_oasis_v5.py --out data/oasis_v5.parquet
    # or to specify custom paths:
    python scripts/data/v5/process_oasis_v5.py \
        --oasis1 data/raw/oasis/oasis_cross-sectional.csv \
        --oasis2 data/raw/oasis/oasis_longitudinal.csv \
        --oasis3-dir data/raw/oasis/OASIS3
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.v5.schema import (
    ALL_FEATURES,
    DISEASE_GENOMIC_PRIORS,
    META_COLS,
    POP_DEFAULTS,
)

_OASIS_RAW_DIR = Path("data/raw/oasis")


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read CSV or Excel file transparently."""
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)

# CDR → derived clinical feature mappings (clinically grounded)
def _cdr_to_features(cdr: pd.Series) -> dict[str, pd.Series]:
    return {
        "FunctionalAssessment": (10 - cdr * 3).clip(0, 10),
        "ADL":                  (10 - cdr * 2.5).clip(0, 10),
        "MemoryComplaints":     (cdr > 0).astype(float),
        "BehavioralProblems":   (cdr >= 1).astype(float),
        "Confusion":            (cdr >= 1).astype(float),
        "Disorientation":       (cdr >= 1).astype(float),
        "PersonalityChanges":   (cdr >= 0.5).astype(float),
        "DifficultyCompletingTasks": (cdr >= 0.5).astype(float),
        "Forgetfulness":        (cdr > 0).astype(float),
    }


def _scaffold(n: int, disease_type: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame({col: [POP_DEFAULTS.get(col, np.nan)] * n for col in ALL_FEATURES})
    genomic = DISEASE_GENOMIC_PRIORS.get(disease_type, DISEASE_GENOMIC_PRIORS["Alzheimer's Disease"])
    for col, val in genomic.items():
        df[col] = val
    df["DiseaseType"] = disease_type
    df["data_source"] = source
    return df


# ─── OASIS-1 (cross-sectional) ────────────────────────────────────────────────

def process_oasis1(path: Path) -> pd.DataFrame:
    """
    OASIS-1 columns: ID, M/F, Hand, Age, Educ, SES, MMSE, CDR, eTIV, nWBV, ASF, Delay
    """
    raw = _read_tabular(path)
    print(f"[oasis1] raw: {raw.shape}, cols: {list(raw.columns)}")
    n = len(raw)
    df = _scaffold(n, "Alzheimer's Disease", "oasis1")

    col = {c.strip().lower().replace("/", "_").replace(" ", "_"): c for c in raw.columns}

    # Demographics
    df["Age"] = raw[col["age"]].clip(45, 100)
    sex_col = col.get("m_f") or col.get("sex") or col.get("gender")
    if sex_col:
        df["Gender"] = (raw[sex_col].astype(str).str.upper() == "M").astype(int)
    educ_col = col.get("educ") or col.get("education")
    if educ_col:
        df["EducationLevel"] = raw[educ_col].fillna(12).clip(0, 23) / 23 * 3

    # Cognitive
    if "mmse" in col:
        df["MMSE"] = raw[col["mmse"]].clip(0, 30)
    cdr_col = col.get("cdr")
    if cdr_col:
        cdr = raw[cdr_col].fillna(0).clip(0, 3)
        for feat, vals in _cdr_to_features(cdr).items():
            df[feat] = vals.values

    # Imaging
    for src, dst in [("etiv", "eTIV"), ("nwbv", "nWBV"), ("asf", "ASF"), ("delay", "MR_Delay")]:
        if src in col:
            df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")

    # Labels
    cdr_vals = raw[col["cdr"]].fillna(0) if "cdr" in col else pd.Series([0] * n)
    df["risk_label"] = (cdr_vals > 0).astype(int)
    df.loc[cdr_vals == 0, "DiseaseType"] = "Healthy"

    print(f"[oasis1] {n} rows, CDR>0: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── OASIS-2 (longitudinal) ───────────────────────────────────────────────────

def process_oasis2(path: Path) -> pd.DataFrame:
    """
    OASIS-2 columns: Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand,
                     Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF
    Take first visit per subject to avoid duplication (or use all visits for more rows).
    """
    raw = _read_tabular(path)
    print(f"[oasis2] raw: {raw.shape}, cols: {list(raw.columns)}")

    # Use all visits (longitudinal rows are distinct clinical snapshots)
    n = len(raw)
    df = _scaffold(n, "Alzheimer's Disease", "oasis2")
    col = {c.strip().lower().replace(" ", "_").replace("/", "_"): c for c in raw.columns}

    df["Age"] = pd.to_numeric(raw[col.get("age", "Age")], errors="coerce").clip(45, 100)
    sex_col = col.get("m_f") or col.get("sex") or col.get("gender")
    if sex_col:
        df["Gender"] = (raw[sex_col].astype(str).str.upper() == "M").astype(int)
    educ_col = col.get("educ") or col.get("education")
    if educ_col:
        df["EducationLevel"] = raw[educ_col].fillna(12).clip(0, 23) / 23 * 3

    if "mmse" in col:
        df["MMSE"] = pd.to_numeric(raw[col["mmse"]], errors="coerce").clip(0, 30)

    cdr_col = col.get("cdr")
    if cdr_col:
        cdr = pd.to_numeric(raw[cdr_col], errors="coerce").fillna(0).clip(0, 3)
        for feat, vals in _cdr_to_features(cdr).items():
            df[feat] = vals.values

    for src, dst in [("etiv", "eTIV"), ("nwbv", "nWBV"), ("asf", "ASF"), ("mr_delay", "MR_Delay")]:
        if src in col:
            df[dst] = pd.to_numeric(raw[col[src]], errors="coerce")

    # Group column: Demented / Nondemented / Converted
    group_col = col.get("group") or col.get("diagnosis")
    if group_col:
        groups = raw[group_col].astype(str).str.lower()
        df["risk_label"] = groups.isin(["demented", "converted"]).astype(int)
        df.loc[groups == "nondemented", "DiseaseType"] = "Healthy"
    else:
        cdr_vals = pd.to_numeric(raw.get(cdr_col, pd.Series([0]*n)), errors="coerce").fillna(0)
        df["risk_label"] = (cdr_vals > 0).astype(int)

    print(f"[oasis2] {n} rows, Demented: {df['risk_label'].mean():.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── OASIS-3 (comprehensive: MRI + PET + CSF + genetics) ─────────────────────

def process_oasis3(oasis3_dir: Path) -> pd.DataFrame:
    """
    OASIS-3 is the richest dataset: combines demographics, cognition, imaging,
    CSF biomarkers, and genetic data (APOE4).

    File conventions (from oasis-brains.org download):
      - OASIS3_participants.tsv or oasis3_demographics.csv
      - OASIS3_UDSb4.csv  (CDR, MMSE, functional scores)
      - OASIS3_MRI_fseg.csv  (FreeSurfer: hippocampus, entorhinal, WMH, etc.)
      - OASIS3_ADRC_Clinical.csv  (CSF: ABETA, TAU, PTAU + APOE genotype)
    """
    if not oasis3_dir.exists():
        print(f"[oasis3] directory not found: {oasis3_dir}")
        return pd.DataFrame()

    files = {f.name.lower(): f for f in oasis3_dir.iterdir() if f.is_file()}

    def _find(candidates: list[str]) -> Path | None:
        for name in candidates:
            # exact match
            if name.lower() in files:
                return files[name.lower()]
            # partial match
            for fname, fpath in files.items():
                if name.lower() in fname:
                    return fpath
        return None

    demo_file = _find(["participants.tsv", "oasis3_participants", "demographics", "oasis3_demo"])
    cog_file  = _find(["udsbscore", "udsb4", "cognitive", "cdr", "mmse"])
    mri_file  = _find(["fseg", "freesurfer", "mri_seg", "mri"])
    csf_file  = _find(["adrc_clinical", "csf", "biomarker", "clinical"])

    if demo_file is None:
        print("[oasis3] no demographics file found — cannot process OASIS-3")
        return pd.DataFrame()

    print(f"[oasis3] loading demographics: {demo_file}")
    sep = "\t" if demo_file.suffix == ".tsv" else ","
    demo = pd.read_csv(demo_file, sep=sep)
    demo.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in demo.columns]

    n = len(demo)
    df = _scaffold(n, "Alzheimer's Disease", "oasis3")

    # Demographics
    for src, dst in [("age", "Age"), ("sex", "Gender"), ("educ", "EducationLevel"),
                     ("education", "EducationLevel"), ("apoe", "APOE4_dosage")]:
        if src in demo.columns:
            val = pd.to_numeric(demo[src], errors="coerce")
            if dst == "Gender":
                # OASIS-3 sex: F=0, M=1 or 'F'/'M'
                if demo[src].dtype == object:
                    val = (demo[src].astype(str).str.upper() == "M").astype(float)
            elif dst == "EducationLevel":
                val = val.fillna(12).clip(0, 23) / 23 * 3
            df[dst] = val

    # APOE4 dosage: number of ε4 alleles (0, 1, 2)
    apoe_col = next((c for c in demo.columns if "apoe" in c), None)
    if apoe_col:
        # OASIS-3 encodes APOE as string like "33", "34", "44"
        def _apoe_to_e4_dosage(apoe_str: str) -> float:
            try:
                s = str(apoe_str).replace("e", "").replace("E", "").strip()
                count = s.count("4")
                return float(count)
            except Exception:
                return 0.0
        df["APOE4_dosage"] = demo[apoe_col].apply(_apoe_to_e4_dosage)
        # Update APOE genomic risk score
        df["APOE_risk_score"] = df["APOE4_dosage"] * 1.2

    # Cognitive + CDR (from UDS file)
    if cog_file is not None:
        print(f"[oasis3] loading cognitive: {cog_file}")
        cog = pd.read_csv(cog_file, sep="\t" if cog_file.suffix == ".tsv" else ",")
        cog.columns = [c.strip().lower().replace(" ", "_") for c in cog.columns]

        # Merge on subject ID
        id_col_demo = next((c for c in demo.columns if "subject" in c or c == "id"), None)
        id_col_cog = next((c for c in cog.columns if "subject" in c or c == "id"), None)
        if id_col_demo and id_col_cog:
            cog_merged = demo[[id_col_demo]].merge(cog, left_on=id_col_demo, right_on=id_col_cog, how="left")
            for src, dst in [("mmse", "MMSE"), ("cdr", None), ("adas_cog_total", "CognitiveTest")]:
                if src in cog_merged.columns:
                    vals = pd.to_numeric(cog_merged[src], errors="coerce")
                    if dst:
                        df[dst] = vals
                    elif src == "cdr":
                        cdr = vals.fillna(0).clip(0, 3)
                        for feat, fvals in _cdr_to_features(cdr).items():
                            df[feat] = fvals.values

    # MRI imaging features (from FreeSurfer segmentation)
    if mri_file is not None:
        print(f"[oasis3] loading MRI: {mri_file}")
        mri = pd.read_csv(mri_file, sep="\t" if mri_file.suffix == ".tsv" else ",")
        mri.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in mri.columns]

        id_col_demo = next((c for c in demo.columns if "subject" in c or c == "id"), None)
        id_col_mri = next((c for c in mri.columns if "subject" in c or c == "id"), None)
        if id_col_demo and id_col_mri:
            mri_m = demo[[id_col_demo]].merge(mri, left_on=id_col_demo, right_on=id_col_mri, how="left")
            imaging_map = {
                "etiv": "eTIV", "intracranialvol": "eTIV",
                "nwbv": "nWBV",
                "asf": "ASF",
                "wmh": "WMH_volume", "wmhvol": "WMH_volume",
                "left_hippocampus": "hippocampus_volume",
                "rh_entorhinal_thickness": "entorhinal_thickness",
                "lh_entorhinal_thickness": "entorhinal_thickness",
                "lateralventricle": "ventricular_volume",
            }
            for src_key, dst in imaging_map.items():
                match = next((c for c in mri_m.columns if src_key in c), None)
                if match:
                    df[dst] = pd.to_numeric(mri_m[match], errors="coerce")

    # CSF biomarkers (from ADRC Clinical)
    if csf_file is not None:
        print(f"[oasis3] loading CSF biomarkers: {csf_file}")
        csf = pd.read_csv(csf_file, sep="\t" if csf_file.suffix == ".tsv" else ",")
        csf.columns = [c.strip().lower().replace(" ", "_") for c in csf.columns]

        id_col_demo = next((c for c in demo.columns if "subject" in c or c == "id"), None)
        id_col_csf = next((c for c in csf.columns if "subject" in c or c == "id"), None)
        if id_col_demo and id_col_csf:
            csf_m = demo[[id_col_demo]].merge(csf, left_on=id_col_demo, right_on=id_col_csf, how="left")
            csf_map = {
                "abeta": "CSF_Abeta42", "abeta42": "CSF_Abeta42", "csf_abeta42": "CSF_Abeta42",
                "ptau": "CSF_pTau", "csf_ptau": "CSF_pTau",
                "tau": "CSF_tTau", "csf_tau": "CSF_tTau",
            }
            for src_key, dst in csf_map.items():
                match = next((c for c in csf_m.columns if src_key in c), None)
                if match:
                    df[dst] = pd.to_numeric(csf_m[match], errors="coerce")

    # Assign labels from CDR if present, else MMSE threshold
    cdr_col = next((c for c in demo.columns if c == "cdr"), None)
    if cdr_col:
        cdr_vals = pd.to_numeric(demo[cdr_col], errors="coerce").fillna(0)
        df["risk_label"] = (cdr_vals > 0).astype(int)
        df.loc[cdr_vals == 0, "DiseaseType"] = "Healthy"
    elif "MMSE" in df.columns:
        df["risk_label"] = (df["MMSE"] < 24).astype(int)

    pos = df["risk_label"].mean()
    print(f"[oasis3] {n} rows, CDR>0/impaired: {pos:.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── Main ─────────────────────────────────────────────────────────────────────

def _find_oasis_file(default_csv: Path, subdir: str, stem_keywords: list[str]) -> Path | None:
    """Auto-discover an OASIS file — checks default CSV path, then subdir for CSV/XLSX."""
    if default_csv.exists():
        return default_csv
    # Check subdir for any file matching the keywords
    subdir_path = _OASIS_RAW_DIR / subdir
    if subdir_path.exists():
        for f in subdir_path.iterdir():
            if f.suffix in (".csv", ".xlsx", ".xls", ".tsv"):
                name_lower = f.stem.lower()
                if any(kw in name_lower for kw in stem_keywords):
                    return f
        # Fallback: return first tabular file in subdir
        for f in subdir_path.iterdir():
            if f.suffix in (".csv", ".xlsx", ".xls", ".tsv"):
                return f
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Process OASIS-1/2/3 into v5 schema")
    ap.add_argument("--oasis1", default=None, help="Path to OASIS-1 CSV/XLSX (auto-detected if omitted)")
    ap.add_argument("--oasis2", default=None, help="Path to OASIS-2 CSV/XLSX (auto-detected if omitted)")
    ap.add_argument("--oasis3-dir", default=str(_OASIS_RAW_DIR / "OASIS3"))
    ap.add_argument("--out", default="data/oasis_v5.parquet")
    args = ap.parse_args()

    # Auto-discover files if not explicitly provided
    oasis1_path = Path(args.oasis1) if args.oasis1 else _find_oasis_file(
        _OASIS_RAW_DIR / "oasis_cross-sectional.csv", "oasis1",
        ["cross", "oasis1", "oasis_1", "cross_sectional"],
    )
    oasis2_path = Path(args.oasis2) if args.oasis2 else _find_oasis_file(
        _OASIS_RAW_DIR / "oasis_longitudinal.csv", "oasis2",
        ["longitudinal", "oasis2", "oasis_2"],
    )

    frames: list[pd.DataFrame] = []

    for label, p, processor in [
        ("OASIS-1", oasis1_path, process_oasis1),
        ("OASIS-2", oasis2_path, process_oasis2),
    ]:
        if p and p.exists():
            print(f"\n=== {label}: {p} ===")
            try:
                df = processor(p)
                frames.append(df)
            except Exception as exc:
                print(f"[{label}] failed: {exc}")
                import traceback; traceback.print_exc()
        else:
            print(f"\n[{label}] file not found: {p} — skipping")
            print(f"         Download from https://www.oasis-brains.org and place at {p}")

    oasis3_dir = Path(args.oasis3_dir)
    if oasis3_dir.exists():
        print(f"\n=== OASIS-3: {oasis3_dir} ===")
        try:
            df3 = process_oasis3(oasis3_dir)
            if len(df3) > 0:
                frames.append(df3)
        except Exception as exc:
            print(f"[OASIS-3] failed: {exc}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n[OASIS-3] directory not found: {oasis3_dir} — skipping")
        print("          Download from https://www.oasis-brains.org/oasis-3")
        print("          Expected sub-files: OASIS3_participants.tsv, OASIS3_UDSb4.csv,")
        print("          OASIS3_MRI_fseg.csv, OASIS3_ADRC_Clinical.csv")

    if not frames:
        print("\nNo OASIS data processed. Register at https://www.oasis-brains.org")
        return

    merged = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    pos = merged["risk_label"].mean()
    print(f"\nOASIS merged: {len(merged)} rows, impaired: {pos:.2%} → {out}")


if __name__ == "__main__":
    main()
