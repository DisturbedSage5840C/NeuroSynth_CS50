from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from neurosynth.core.logging import get_logger
from neurosynth.data.iceberg_catalog import IcebergDomainCatalog

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class ADNIIngestionConnector:
    """Simple ADNI metadata connector for patient demographics into Iceberg."""

    def __init__(self, iceberg: IcebergDomainCatalog) -> None:
        self.iceberg = iceberg
        self.log = get_logger(__name__)

    def ingest_demographics(self, csv_path: str | Path, cohort: str = "ADNI") -> pd.DataFrame:
        frame = pd.read_csv(csv_path)
        mapped = pd.DataFrame(
            {
                "patient_id": frame["PTID"].astype(str),
                "patient_cohort": cohort,
                "ingestion_date": date.today(),
                "sex": frame.get("PTGENDER", pd.Series([None] * len(frame))),
                "birth_year": frame.get("PTDOBYY", pd.Series([None] * len(frame))),
                "education_years": frame.get("PTEDUCAT", pd.Series([None] * len(frame))),
                "apoe_e4_count": frame.get("APOE4", pd.Series([None] * len(frame))),
            }
        ).drop_duplicates(subset=["patient_id"])

        self.iceberg.append_dataframe("patients", mapped)
        self.log.info("adni.demographics_ingested", rows=len(mapped), source=str(csv_path))
        return mapped


class ClinicalNotesConnector:
    def __init__(self, iceberg: IcebergDomainCatalog) -> None:
        self.iceberg = iceberg

    def ingest_notes(self, notes: list[dict[str, str]], cohort: str) -> pd.DataFrame:
        rows = []
        for note in notes:
            rows.append(
                {
                    "note_id": str(uuid4()),
                    "patient_id": str(note["patient_id"]),
                    "patient_cohort": cohort,
                    "ingestion_date": date.today(),
                    "encounter_time": note.get("encounter_time"),
                    "note_text": note["note_text"],
                    "source_system": note.get("source_system", "ehr"),
                }
            )
        frame = pd.DataFrame(rows)
        self.iceberg.append_dataframe("clinical_notes", frame)
        return frame


class CausalGraphConnector:
    def __init__(self, iceberg: IcebergDomainCatalog) -> None:
        self.iceberg = iceberg

    def ingest_causal_edges(self, graph_id: str, patient_cohort: str, edges: list[dict[str, object]], patient_id: str | None = None) -> pd.DataFrame:
        rows = []
        for edge in edges:
            rows.append(
                {
                    "graph_id": graph_id,
                    "patient_id": patient_id,
                    "patient_cohort": patient_cohort,
                    "ingestion_date": date.today(),
                    "source_node": str(edge["from"]),
                    "target_node": str(edge["to"]),
                    "edge_weight": float(edge.get("strength", 0.0)),
                    "edge_type": str(edge.get("type", "derived")),
                }
            )
        frame = pd.DataFrame(rows)
        self.iceberg.append_dataframe("causal_graphs", frame)
        return frame
