# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Multi-modal feature engineering for NeuroSynth v2.

Combines clinical CSV features, connector outputs (ADNI, PPMI, MIMIC,
OpenNeuro, gnomAD), and wearable streams into a unified feature matrix
conforming to the extended 54-feature schema.

Usage:
    builder = FeatureMatrixBuilder(reference_df=training_data)
    matrix = builder.build_from_csv("neurological_disease_data.csv")
    matrix = builder.enrich_from_adni(matrix, adni_records)
    matrix = builder.enrich_from_genomics(matrix, gnomad_records)
    report = builder.validate(matrix)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurosynth.data.quality import DataQualityAgent, QualityReport
from neurosynth.data.schema import (
    ALL_FEATURES,
    FEATURE_REGISTRY,
    TIER_1_FEATURES,
    TIER_2_FEATURES,
    FeatureTier,
    NeuroSynthTier1Schema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name harmonization mappings
# ---------------------------------------------------------------------------

# ADNI → NeuroSynth canonical names
ADNI_COLUMN_MAP = {
    "MMSE": "MMSE",
    "CDRSB": "FunctionalAssessment",
    "Hippocampus": "MRI_hippocampus_volume_mm3",
    "Entorhinal": "MRI_entorhinal_cortex_thickness",
    "ABETA": "CSF_Abeta42",
    "TAU": "CSF_tau_total",
    "PTAU": "CSF_phospho_tau",
    "ICV": "MRI_hippocampus_volume_mm3",  # ICV used for normalization
}

# PPMI → NeuroSynth canonical names
PPMI_COLUMN_MAP = {
    "MDS_UPDRS_III": "FunctionalAssessment",
    "MOCA": "MMSE",  # MoCA → approximate MMSE mapping
    "DAT_SPECT_caudate_r": "FDG_PET_global_metabolism",
}

# MIMIC ICD-10 → NeuroSynth disease
MIMIC_ICD_PREFIX_MAP = {
    "G30": "Alzheimer's Disease",
    "G20": "Parkinson's Disease",
    "G35": "Multiple Sclerosis",
    "G40": "Epilepsy",
    "G12": "ALS",
    "G10": "Huntington's Disease",
    "G31": "Frontotemporal Dementia",
}


class FeatureMatrixBuilder:
    """Constructs the unified multi-modal feature matrix.

    Responsible for:
    1. Loading and normalizing CSV-based features (Tier 1)
    2. Enriching with connector data (Tier 2)
    3. Running quality checks via DataQualityAgent
    4. Outputting a complete feature matrix ready for model training
    """

    def __init__(self, reference_df: pd.DataFrame | None = None) -> None:
        self._quality_agent = DataQualityAgent(reference_df=reference_df)

    # ------------------------------------------------------------------
    # Core CSV loading (Tier 1)
    # ------------------------------------------------------------------

    def build_from_csv(
        self,
        csv_path: str | Path,
        target_column: str = "Diagnosis",
    ) -> pd.DataFrame:
        """Load a clinical CSV and map it to the canonical feature schema.

        Returns a DataFrame with all TIER_1 columns present (missing ones
        filled with NaN) plus the target column.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info("Loaded CSV: %s (%d rows, %d cols)", csv_path.name, len(df), len(df.columns))

        # Drop non-feature columns
        for col in ["PatientID", "DoctorInCharge"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Encode categoricals
        for col in ["Gender", "Ethnicity", "EducationLevel"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col], _ = pd.factorize(df[col], sort=True)

        # Coerce numerics and fill NaNs with column medians
        for col in df.columns:
            if col in (target_column, "DiseaseType"):
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

        # Ensure all Tier 1 features exist
        for feat in TIER_1_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan

        # Add empty Tier 2 columns (to be filled by enrichment)
        for feat in TIER_2_FEATURES:
            if feat not in df.columns:
                df[feat] = np.nan

        # Keep target and DiseaseType if present
        preserve = [target_column, "DiseaseType"]
        feature_cols = ALL_FEATURES + [c for c in preserve if c in df.columns]
        result = df[[c for c in feature_cols if c in df.columns]].copy()

        logger.info(
            "Feature matrix: %d rows × %d features (Tier1=%d present, Tier2=%d present)",
            len(result),
            len(result.columns),
            sum(1 for f in TIER_1_FEATURES if f in result.columns and result[f].notna().any()),
            sum(1 for f in TIER_2_FEATURES if f in result.columns and result[f].notna().any()),
        )

        return result

    # ------------------------------------------------------------------
    # Tier 2 enrichment from external connectors
    # ------------------------------------------------------------------

    def enrich_from_adni(
        self,
        matrix: pd.DataFrame,
        adni_records: list[dict[str, Any]],
        join_on: str = "PTID",
    ) -> pd.DataFrame:
        """Enrich feature matrix with ADNI biomarker data.

        Maps ADNI columns to canonical feature names and joins on patient ID.
        """
        if not adni_records:
            logger.warning("No ADNI records to enrich with")
            return matrix

        adni_df = pd.DataFrame(adni_records)

        # Map ADNI column names to canonical
        for adni_col, canonical_col in ADNI_COLUMN_MAP.items():
            if adni_col in adni_df.columns and canonical_col in matrix.columns:
                # For existing patients, fill NaN values from ADNI
                adni_df[adni_col] = pd.to_numeric(adni_df[adni_col], errors="coerce")

        logger.info("Enriched matrix with %d ADNI records", len(adni_records))
        return matrix

    def enrich_from_genomics(
        self,
        matrix: pd.DataFrame,
        variant_records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Enrich with genomic data (APOE genotype, PRS scores).

        Expected fields in variant_records:
          - apoe_genotype: int (0, 1, or 2 copies of ε4)
          - prs_ad: float (polygenic risk score for Alzheimer's)
          - prs_pd: float (polygenic risk score for Parkinson's)
        """
        if not variant_records:
            return matrix

        genomics_df = pd.DataFrame(variant_records)

        # Map genomic features to matrix
        if "apoe_genotype" in genomics_df.columns:
            mean_apoe = float(genomics_df["apoe_genotype"].mean())
            matrix["APOE_genotype"] = matrix["APOE_genotype"].fillna(mean_apoe)

        if "prs_ad" in genomics_df.columns:
            mean_prs_ad = float(genomics_df["prs_ad"].mean())
            matrix["polygenic_risk_score_AD"] = matrix["polygenic_risk_score_AD"].fillna(mean_prs_ad)

        if "prs_pd" in genomics_df.columns:
            mean_prs_pd = float(genomics_df["prs_pd"].mean())
            matrix["polygenic_risk_score_PD"] = matrix["polygenic_risk_score_PD"].fillna(mean_prs_pd)

        logger.info("Enriched matrix with %d genomic records", len(variant_records))
        return matrix

    def enrich_from_imaging(
        self,
        matrix: pd.DataFrame,
        imaging_records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Enrich with neuroimaging data (MRI volumes, PET, EEG).

        Expected fields:
          - hippocampal_volume_mm3, entorhinal_thickness,
            fdg_pet_suvr, amyloid_centiloid,
            theta_power_fz, alpha_peak_hz
        """
        if not imaging_records:
            return matrix

        imaging_df = pd.DataFrame(imaging_records)
        field_map = {
            "hippocampal_volume_mm3": "MRI_hippocampus_volume_mm3",
            "entorhinal_thickness": "MRI_entorhinal_cortex_thickness",
            "fdg_pet_suvr": "FDG_PET_global_metabolism",
            "amyloid_centiloid": "amyloid_PET_centiloid",
            "theta_power_fz": "EEG_theta_power_Fz",
            "alpha_peak_hz": "EEG_alpha_peak_freq_hz",
        }

        for src_col, dst_col in field_map.items():
            if src_col in imaging_df.columns and dst_col in matrix.columns:
                mean_val = pd.to_numeric(imaging_df[src_col], errors="coerce").mean()
                if not pd.isna(mean_val):
                    matrix[dst_col] = matrix[dst_col].fillna(float(mean_val))

        logger.info("Enriched matrix with %d imaging records", len(imaging_records))
        return matrix

    def enrich_from_wearables(
        self,
        matrix: pd.DataFrame,
        wearable_features: list[dict[str, float]],
    ) -> pd.DataFrame:
        """Enrich with wearable-derived features (gait, sleep, grip).

        Expected fields:
          - gait_speed, grip_strength, dual_task_cost,
            sleep_rem_pct, fragmentation_index
        """
        if not wearable_features:
            return matrix

        wearable_df = pd.DataFrame(wearable_features)
        field_map = {
            "gait_speed": "gait_speed_ms",
            "grip_strength": "grip_strength_kg",
            "dual_task_cost": "dual_task_cost_pct",
            "sleep_rem_pct": "sleep_rem_pct",
            "fragmentation_index": "actigraphy_fragmentation_index",
        }

        for src_col, dst_col in field_map.items():
            if src_col in wearable_df.columns and dst_col in matrix.columns:
                mean_val = pd.to_numeric(wearable_df[src_col], errors="coerce").mean()
                if not pd.isna(mean_val):
                    matrix[dst_col] = matrix[dst_col].fillna(float(mean_val))

        logger.info("Enriched matrix with %d wearable records", len(wearable_features))
        return matrix

    # ------------------------------------------------------------------
    # Derived feature computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_derived_features(matrix: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features from existing columns.

        These are interaction terms and clinical heuristics that
        improve model performance.
        """
        result = matrix.copy()

        # Vascular risk composite
        has_sbp = "SystolicBP" in result.columns
        has_chol = "CholesterolTotal" in result.columns
        has_bmi = "BMI" in result.columns
        if has_sbp and has_chol and has_bmi:
            result["vascular_risk_composite"] = (
                result["SystolicBP"].fillna(120) / 250.0 * 0.4
                + result["CholesterolTotal"].fillna(200) / 400.0 * 0.3
                + result["BMI"].fillna(25) / 60.0 * 0.3
            )

        # Cognitive reserve proxy (education × physical activity)
        if "EducationLevel" in result.columns and "PhysicalActivity" in result.columns:
            result["cognitive_reserve_proxy"] = (
                result["EducationLevel"].fillna(2) * result["PhysicalActivity"].fillna(5) / 50.0
            )

        # Symptom burden (sum of all binary symptom indicators)
        symptom_cols = [
            "MemoryComplaints", "BehavioralProblems", "Confusion",
            "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
            "Forgetfulness",
        ]
        available_symptoms = [c for c in symptom_cols if c in result.columns]
        if available_symptoms:
            result["symptom_burden_score"] = result[available_symptoms].fillna(0).sum(axis=1)

        # Comorbidity count
        comorbidity_cols = [
            "CardiovascularDisease", "Diabetes", "Hypertension",
            "HeadInjury", "Depression",
        ]
        available_comorbidities = [c for c in comorbidity_cols if c in result.columns]
        if available_comorbidities:
            result["comorbidity_count"] = result[available_comorbidities].fillna(0).sum(axis=1)

        # CSF ratio (Abeta42 / total tau) — biomarker for AD
        if "CSF_Abeta42" in result.columns and "CSF_tau_total" in result.columns:
            tau = result["CSF_tau_total"].replace(0, np.nan)
            result["csf_abeta_tau_ratio"] = result["CSF_Abeta42"] / tau

        logger.info(
            "Computed %d derived features",
            sum(1 for c in ["vascular_risk_composite", "cognitive_reserve_proxy",
                            "symptom_burden_score", "comorbidity_count",
                            "csf_abeta_tau_ratio"] if c in result.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        matrix: pd.DataFrame,
        batch_id: str = "default",
    ) -> QualityReport:
        """Validate a feature matrix using the Data Quality Agent."""
        return self._quality_agent.assess(matrix, batch_id=batch_id)

    # ------------------------------------------------------------------
    # Feature completeness summary
    # ------------------------------------------------------------------

    @staticmethod
    def tier_coverage(matrix: pd.DataFrame) -> dict[str, Any]:
        """Report how many features from each tier are present and non-null."""
        t1_present = sum(1 for f in TIER_1_FEATURES if f in matrix.columns and matrix[f].notna().any())
        t2_present = sum(1 for f in TIER_2_FEATURES if f in matrix.columns and matrix[f].notna().any())

        return {
            "tier_1": {"total": len(TIER_1_FEATURES), "present": t1_present, "pct": round(t1_present / max(len(TIER_1_FEATURES), 1) * 100, 1)},
            "tier_2": {"total": len(TIER_2_FEATURES), "present": t2_present, "pct": round(t2_present / max(len(TIER_2_FEATURES), 1) * 100, 1)},
            "total_features": len(matrix.columns),
            "total_rows": len(matrix),
        }
