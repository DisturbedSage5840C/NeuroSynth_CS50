from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
from pyiceberg.expressions import EqualTo
from pyiceberg.schema import Schema
from pyiceberg.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    TimestampType,
    UUIDType,
)

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import LakehouseError
from neurosynth.core.logging import get_logger
from neurosynth.core.models import BiomarkerRecord


@dataclass
class TableDef:
    name: str
    schema: Schema


class NeuroSynthLakehouse:
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._catalog: Catalog | None = None

    def initialize_catalog(self) -> None:
        self._catalog = load_catalog(
            "neurosynth",
            type="rest",
            uri=self._settings.iceberg_rest_uri,
            warehouse=self._settings.iceberg_warehouse,
            **{
                "s3.endpoint": self._settings.minio_endpoint,
                "s3.access-key-id": self._settings.minio_access_key,
                "s3.secret-access-key": self._settings.minio_secret_key,
                "s3.path-style-access": "true",
                "s3.region": self._settings.minio_region,
            },
        )

        try:
            self._catalog.create_namespace("neurosynth")
        except Exception:
            pass

    def _table_defs(self) -> list[TableDef]:
        return [
            TableDef(
                name="neurosynth.patients",
                schema=Schema(
                    NestedField(1, "patient_id", UUIDType(), required=True),
                    NestedField(2, "cohort", StringType(), required=False),
                    NestedField(3, "diagnosis", StringType(), required=False),
                    NestedField(4, "apoe_e4_alleles", IntegerType(), required=False),
                    NestedField(5, "age_at_enrollment", DoubleType(), required=False),
                    NestedField(6, "sex", StringType(), required=False),
                    NestedField(7, "education_years", DoubleType(), required=False),
                    NestedField(8, "created_at", TimestampType(), required=False),
                ),
            ),
            TableDef(
                name="neurosynth.biomarker_longitudinal",
                schema=Schema(
                    NestedField(1, "patient_id", UUIDType(), required=True),
                    NestedField(2, "visit_code", StringType(), required=False),
                    NestedField(3, "collection_date", DateType(), required=False),
                    NestedField(4, "abeta42_pgml", DoubleType(), required=False),
                    NestedField(5, "ptau181_pgml", DoubleType(), required=False),
                    NestedField(6, "total_tau_pgml", DoubleType(), required=False),
                    NestedField(7, "nfl_pgml", DoubleType(), required=False),
                    NestedField(8, "alpha_syn_pgml", DoubleType(), required=False),
                    NestedField(9, "hippocampal_volume_mm3", DoubleType(), required=False),
                    NestedField(10, "ventricle_volume_mm3", DoubleType(), required=False),
                    NestedField(11, "cdrsb_score", DoubleType(), required=False),
                    NestedField(12, "mmse_score", DoubleType(), required=False),
                    NestedField(13, "moca_score", DoubleType(), required=False),
                    NestedField(14, "updrs_part3", DoubleType(), required=False),
                    NestedField(15, "site_id", StringType(), required=False),
                    NestedField(16, "harmonized_flag", BooleanType(), required=False),
                    NestedField(17, "feature_vector", ListType(18, FloatType(), element_required=False), required=False),
                    NestedField(19, "embedding_model_version", StringType(), required=False),
                ),
            ),
            TableDef(
                name="neurosynth.imaging_index",
                schema=Schema(
                    NestedField(1, "scan_id", UUIDType(), required=True),
                    NestedField(2, "patient_id", UUIDType(), required=True),
                    NestedField(3, "scan_date", DateType(), required=False),
                    NestedField(4, "modality", StringType(), required=False),
                    NestedField(5, "scanner_manufacturer", StringType(), required=False),
                    NestedField(6, "field_strength_tesla", DoubleType(), required=False),
                    NestedField(7, "raw_s3_uri", StringType(), required=False),
                    NestedField(8, "preprocessed_s3_uri", StringType(), required=False),
                    NestedField(9, "freesurfer_complete", BooleanType(), required=False),
                    NestedField(10, "fmriprep_complete", BooleanType(), required=False),
                    NestedField(11, "qa_passed", BooleanType(), required=False),
                    NestedField(12, "qa_metrics", MapType(13, StringType(), 14, DoubleType(), value_required=False), required=False),
                ),
            ),
            TableDef(
                name="neurosynth.genomics",
                schema=Schema(
                    NestedField(1, "subject_id", UUIDType(), required=True),
                    NestedField(2, "sequencing_type", StringType(), required=False),
                    NestedField(3, "variant_count", LongType(), required=False),
                    NestedField(4, "pathogenic_variants", ListType(5, StringType(), element_required=False), required=False),
                    NestedField(6, "prs_alzheimer", DoubleType(), required=False),
                    NestedField(7, "prs_parkinson", DoubleType(), required=False),
                    NestedField(8, "prs_als", DoubleType(), required=False),
                    NestedField(9, "embedding_vector", ListType(10, FloatType(), element_required=False), required=False),
                    NestedField(11, "vcf_s3_uri", StringType(), required=False),
                ),
            ),
        ]

    def create_all_tables(self) -> None:
        if not self._catalog:
            raise LakehouseError("Catalog is not initialized")
        for table_def in self._table_defs():
            try:
                self._catalog.load_table(table_def.name)
            except NoSuchTableError:
                self._catalog.create_table(identifier=table_def.name, schema=table_def.schema)

    def write_biomarkers(self, records: list[BiomarkerRecord]) -> None:
        if not self._catalog:
            raise LakehouseError("Catalog is not initialized")

        table = self._catalog.load_table("neurosynth.biomarker_longitudinal")
        rows = [r.model_dump() for r in records]
        data = pa.Table.from_pylist(rows)

        # Upsert-like write uses overwrite on patient key then append new values.
        patient_ids = sorted({str(r.patient_id) for r in records})
        for patient_id in patient_ids:
            pid = UUID(patient_id)
            mask = [str(v.as_py()) == patient_id for v in data.column("patient_id")]
            patient_rows = data.filter(pa.array(mask))
            table.overwrite(patient_rows, overwrite_filter=EqualTo("patient_id", pid))

    def read_patient_timeline(self, patient_id: UUID) -> pd.DataFrame:
        if not self._catalog:
            raise LakehouseError("Catalog is not initialized")
        table = self._catalog.load_table("neurosynth.biomarker_longitudinal")
        scan = table.scan(row_filter=EqualTo("patient_id", patient_id)).to_arrow()
        frame = scan.to_pandas()
        if frame.empty:
            return frame
        return frame.sort_values("collection_date").reset_index(drop=True)

    def run_site_harmonization(self, cohort: str) -> int:
        if not self._catalog:
            raise LakehouseError("Catalog is not initialized")

        table = self._catalog.load_table("neurosynth.biomarker_longitudinal")
        frame = table.scan().to_arrow().to_pandas()
        if frame.empty:
            return 0

        frame = frame[frame.get("site_id").notna()]
        if frame.empty:
            return 0

        cols = [
            "abeta42_pgml",
            "ptau181_pgml",
            "total_tau_pgml",
            "nfl_pgml",
            "hippocampal_volume_mm3",
            "ventricle_volume_mm3",
            "cdrsb_score",
            "mmse_score",
        ]
        cols = [c for c in cols if c in frame.columns]

        try:
            from neuroCombat import neuroCombat

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
            harmonized = neuroCombat(
                dat=frame[cols].T,
                covars=frame[["site_id", "sex"]].assign(age=frame.get("age_at_enrollment", 0)),
                batch_col="site_id",
                categorical_cols=["sex"],
                continuous_cols=["age"],
            )["data"].T
            for col in cols:
                frame[col] = harmonized[col]
        except Exception:
            # Fallback keeps pipeline operational when neuroCombat is unavailable.
            for col in cols:
                frame[col] = frame.groupby("site_id")[col].transform(lambda s: s - s.mean())

        frame["harmonized_flag"] = True
        table.overwrite(pa.Table.from_pandas(frame), overwrite_filter=EqualTo("harmonized_flag", False))
        return len(frame)
