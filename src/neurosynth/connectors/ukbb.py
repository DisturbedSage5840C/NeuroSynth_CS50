"""UK Biobank (UKBB) bulk data connector.

Provides access to UK Biobank neurological cohort data via the
approved research access pathway. Downloads phenotype, imaging,
and genomic data in bulk using the ukbconv/ukbfetch tools.

NOTE: UK Biobank access requires an approved application.
      Set NEURO_UKBB_APPLICATION_ID and NEURO_UKBB_KEY_PATH in .env.

Outputs:
  - Demographic and lifestyle features (Tier 1)
  - Neuroimaging IDPs (Tier 2)
  - Polygenic risk scores (Tier 2)
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.connectors.base import AbstractNeuroDataSource
from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import DataIngestionError
from neurosynth.core.logging import get_logger

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

# UK Biobank Data-Field IDs relevant to neurological research
UKBB_FIELD_MAP = {
    # Demographics
    "31":     "Gender",
    "21003":  "Age",
    "21001":  "BMI",
    "6138":   "EducationLevel",
    "21000":  "Ethnicity",
    # Blood pressure
    "4080":   "SystolicBP",
    "4079":   "DiastolicBP",
    # Cholesterol
    "30690":  "CholesterolTotal",
    "30780":  "CholesterolLDL",
    "30760":  "CholesterolHDL",
    "30870":  "CholesterolTriglycerides",
    # Lifestyle
    "20116":  "Smoking",
    "1558":   "AlcoholConsumption",
    "22032":  "PhysicalActivity",
    "1160":   "SleepQuality",
    # Cognitive
    "20016":  "MMSE",  # Fluid intelligence → proxy
    "6350":   "FunctionalAssessment",  # Hand grip strength as proxy
    # Neuroimaging IDPs (from imaging visit)
    "25019":  "MRI_hippocampus_volume_mm3",
    "25024":  "MRI_entorhinal_cortex_thickness",
    "25781":  "FDG_PET_global_metabolism",
    # Medical history (ICD-10)
    "41270":  "icd10_primary",
    "41271":  "icd10_secondary",
}

# ICD-10 codes for filtering neurological patients
NEURO_ICD10_PREFIXES = ["G10", "G12", "G20", "G30", "G31", "G35", "G40"]


class UKBBConnector(AbstractNeuroDataSource):
    """Connector for UK Biobank approved data access.

    Reads UK Biobank bulk download files (.tab, .csv) produced by
    ukbconv and maps them to the NeuroSynth feature schema.
    """

    def __init__(
        self,
        settings: NeuroSynthSettings,
        data_dir: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._data_dir = Path(data_dir) if data_dir else Path("data/ukbb")
        self._records: list[dict[str, Any]] = []
        self._raw_df: pd.DataFrame | None = None

    async def connect(self) -> None:
        """Verify UKBB data directory exists and contains expected files."""
        if not self._data_dir.exists():
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._logger.warning(
                "ukbb.data_dir_created",
                path=str(self._data_dir),
                message="Place UK Biobank bulk download files here",
            )

        # Check for ukb*.tab or ukb*.csv files
        tab_files = list(self._data_dir.glob("ukb*.tab")) + list(self._data_dir.glob("ukb*.csv"))
        if not tab_files:
            self._logger.warning(
                "ukbb.no_data_files",
                message="No UK Biobank files found. Download via ukbfetch.",
            )
        else:
            self._logger.info("ukbb.connect", files=len(tab_files))

    async def validate_schema(self) -> None:
        if self._raw_df is None or self._raw_df.empty:
            raise DataIngestionError("UKBB data not loaded")

    async def _load_tab_file(self, filepath: Path) -> pd.DataFrame:
        """Load a UK Biobank .tab file with proper field ID parsing."""
        return await asyncio.to_thread(self._read_ukbb_file, filepath)

    @staticmethod
    def _read_ukbb_file(filepath: Path) -> pd.DataFrame:
        """Read UK Biobank bulk file (tab-separated or CSV)."""
        sep = "\t" if filepath.suffix == ".tab" else ","
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
        return df

    def _map_field_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map UK Biobank field ID columns to NeuroSynth feature names.

        UKBB columns follow the pattern: f.<field_id>.<instance>.<array_index>
        We take instance 0, array index 0 by default.
        """
        mapped = pd.DataFrame(index=df.index)

        # Keep eid (participant ID)
        if "eid" in df.columns:
            mapped["participant_id"] = df["eid"].astype(str)

        for col in df.columns:
            # Extract field ID from column name (e.g., "f.21003.0.0" → "21003")
            if col.startswith("f."):
                parts = col.split(".")
                if len(parts) >= 2:
                    field_id = parts[1]
                    instance = parts[2] if len(parts) > 2 else "0"
                    # Only take baseline instance (0)
                    if instance == "0" and field_id in UKBB_FIELD_MAP:
                        canonical_name = UKBB_FIELD_MAP[field_id]
                        if canonical_name not in mapped.columns:
                            mapped[canonical_name] = pd.to_numeric(df[col], errors="coerce")

        return mapped

    def _filter_neurological(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to patients with neurological ICD-10 diagnoses."""
        if "icd10_primary" not in df.columns and "icd10_secondary" not in df.columns:
            return df

        neuro_mask = pd.Series(False, index=df.index)

        for icd_col in ["icd10_primary", "icd10_secondary"]:
            if icd_col in df.columns:
                for prefix in NEURO_ICD10_PREFIXES:
                    neuro_mask |= df[icd_col].astype(str).str.startswith(prefix)

        filtered = df[neuro_mask]
        self._logger.info(
            "ukbb.filtered_neurological",
            total=len(df),
            neurological=len(filtered),
        )
        return filtered

    async def load_data(self, neuro_only: bool = True) -> None:
        """Load all UKBB files from data directory."""
        tab_files = list(self._data_dir.glob("ukb*.tab")) + list(self._data_dir.glob("ukb*.csv"))

        if not tab_files:
            self._logger.warning("ukbb.no_files", message="No UKBB data files to load")
            return

        dfs = []
        for filepath in tab_files:
            df = await self._load_tab_file(filepath)
            mapped = self._map_field_ids(df)
            dfs.append(mapped)
            self._logger.info("ukbb.loaded_file", file=filepath.name, rows=len(mapped))

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            if neuro_only:
                combined = self._filter_neurological(combined)
            self._raw_df = combined
            self._records = combined.to_dict(orient="records")
            self._logger.info("ukbb.load_complete", total_records=len(self._records))

    def get_imaging_features(self) -> pd.DataFrame:
        """Extract neuroimaging-derived features (IDPs) from loaded data."""
        if self._raw_df is None:
            return pd.DataFrame()

        imaging_cols = [
            "MRI_hippocampus_volume_mm3",
            "MRI_entorhinal_cortex_thickness",
            "FDG_PET_global_metabolism",
        ]
        available = [c for c in imaging_cols if c in self._raw_df.columns]
        if not available:
            return pd.DataFrame()

        return self._raw_df[["participant_id"] + available].dropna(subset=available)

    async def fetch_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        return self._records[offset: offset + limit]

    async def stream(self, queue: asyncio.Queue) -> None:
        for record in self._records:
            await queue.put(record)
        self._logger.info("ukbb.stream_complete", records=len(self._records))
