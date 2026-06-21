# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestampType,
)

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.logging import get_logger
from neurosynth.data.contracts import validate_frame


@dataclass(frozen=True)
class IcebergTableSpec:
    name: str
    schema: Schema
    schema_version: int


class IcebergDomainCatalog:
    """Iceberg REST catalog backed by S3/MinIO with domain table helpers."""

    def __init__(self, settings: NeuroSynthSettings, namespace: str = "neurosynth") -> None:
        self.settings = settings
        self.namespace = namespace
        self.log = get_logger(__name__)
        self.catalog: Catalog | None = None

    def connect(self) -> None:
        self.catalog = load_catalog(
            "neurosynth-data",
            type="rest",
            uri=self.settings.iceberg_rest_uri,
            warehouse=self.settings.iceberg_warehouse,
            **{
                "s3.endpoint": self.settings.minio_endpoint,
                "s3.access-key-id": self.settings.minio_access_key,
                "s3.secret-access-key": self.settings.minio_secret_key,
                "s3.path-style-access": "true",
                "s3.region": self.settings.minio_region,
            },
        )
        try:
            self.catalog.create_namespace(self.namespace)
        except Exception:
            pass

    def table_specs(self) -> list[IcebergTableSpec]:
        return [
            IcebergTableSpec(
                name=f"{self.namespace}.patients",
                schema=Schema(
                    NestedField(1, "patient_id", StringType(), required=True),
                    NestedField(2, "patient_cohort", StringType(), required=True),
                    NestedField(3, "ingestion_date", DateType(), required=True),
                    NestedField(4, "sex", StringType(), required=False),
                    NestedField(5, "birth_year", IntegerType(), required=False),
                    NestedField(6, "education_years", DoubleType(), required=False),
                    NestedField(7, "apoe_e4_count", IntegerType(), required=False),
                    NestedField(8, "schema_version", IntegerType(), required=True),
                    NestedField(9, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.imaging_sessions",
                schema=Schema(
                    NestedField(1, "imaging_session_id", StringType(), required=True),
                    NestedField(2, "patient_id", StringType(), required=True),
                    NestedField(3, "patient_cohort", StringType(), required=True),
                    NestedField(4, "ingestion_date", DateType(), required=True),
                    NestedField(5, "series_uid", StringType(), required=True),
                    NestedField(6, "modality", StringType(), required=True),
                    NestedField(7, "field_strength_t", DoubleType(), required=False),
                    NestedField(8, "voxel_size_mm", StringType(), required=False),
                    NestedField(9, "orientation", StringType(), required=False),
                    NestedField(10, "qc_pass", BooleanType(), required=True),
                    NestedField(11, "qc_flags", StringType(), required=False),
                    NestedField(12, "registered_nifti_uri", StringType(), required=False),
                    NestedField(13, "schema_version", IntegerType(), required=True),
                    NestedField(14, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.connectivity_matrices",
                schema=Schema(
                    NestedField(1, "connectivity_id", StringType(), required=True),
                    NestedField(2, "imaging_session_id", StringType(), required=True),
                    NestedField(3, "patient_id", StringType(), required=True),
                    NestedField(4, "patient_cohort", StringType(), required=True),
                    NestedField(5, "ingestion_date", DateType(), required=True),
                    NestedField(6, "atlas_name", StringType(), required=True),
                    NestedField(7, "n_regions", IntegerType(), required=True),
                    NestedField(8, "matrix_uri", StringType(), required=True),
                    NestedField(9, "mean_connectivity", DoubleType(), required=False),
                    NestedField(10, "schema_version", IntegerType(), required=True),
                    NestedField(11, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.genomic_variants",
                schema=Schema(
                    NestedField(1, "variant_id", StringType(), required=True),
                    NestedField(2, "patient_id", StringType(), required=True),
                    NestedField(3, "patient_cohort", StringType(), required=True),
                    NestedField(4, "ingestion_date", DateType(), required=True),
                    NestedField(5, "chrom", StringType(), required=True),
                    NestedField(6, "pos", LongType(), required=True),
                    NestedField(7, "ref", StringType(), required=True),
                    NestedField(8, "alt", StringType(), required=True),
                    NestedField(9, "gene", StringType(), required=False),
                    NestedField(10, "clinvar_significance", StringType(), required=False),
                    NestedField(11, "dbsnp_id", StringType(), required=False),
                    NestedField(12, "schema_version", IntegerType(), required=True),
                    NestedField(13, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.biomarker_timeseries",
                schema=Schema(
                    NestedField(1, "timeseries_id", StringType(), required=True),
                    NestedField(2, "patient_id", StringType(), required=True),
                    NestedField(3, "patient_cohort", StringType(), required=True),
                    NestedField(4, "ingestion_date", DateType(), required=True),
                    NestedField(5, "modality", StringType(), required=True),
                    NestedField(6, "window_start", TimestampType(), required=True),
                    NestedField(7, "window_end", TimestampType(), required=True),
                    NestedField(8, "metric_mean", DoubleType(), required=False),
                    NestedField(9, "metric_std", DoubleType(), required=False),
                    NestedField(10, "sample_count", IntegerType(), required=True),
                    NestedField(11, "schema_version", IntegerType(), required=True),
                    NestedField(12, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.clinical_notes",
                schema=Schema(
                    NestedField(1, "note_id", StringType(), required=True),
                    NestedField(2, "patient_id", StringType(), required=True),
                    NestedField(3, "patient_cohort", StringType(), required=True),
                    NestedField(4, "ingestion_date", DateType(), required=True),
                    NestedField(5, "encounter_time", TimestampType(), required=False),
                    NestedField(6, "note_text", StringType(), required=True),
                    NestedField(7, "source_system", StringType(), required=False),
                    NestedField(8, "schema_version", IntegerType(), required=True),
                    NestedField(9, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.causal_graphs",
                schema=Schema(
                    NestedField(1, "graph_id", StringType(), required=True),
                    NestedField(2, "patient_id", StringType(), required=False),
                    NestedField(3, "patient_cohort", StringType(), required=True),
                    NestedField(4, "ingestion_date", DateType(), required=True),
                    NestedField(5, "source_node", StringType(), required=True),
                    NestedField(6, "target_node", StringType(), required=True),
                    NestedField(7, "edge_weight", DoubleType(), required=True),
                    NestedField(8, "edge_type", StringType(), required=True),
                    NestedField(9, "schema_version", IntegerType(), required=True),
                    NestedField(10, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
            IcebergTableSpec(
                name=f"{self.namespace}.model_predictions",
                schema=Schema(
                    NestedField(1, "prediction_id", StringType(), required=True),
                    NestedField(2, "patient_id", StringType(), required=True),
                    NestedField(3, "patient_cohort", StringType(), required=True),
                    NestedField(4, "ingestion_date", DateType(), required=True),
                    NestedField(5, "model_name", StringType(), required=True),
                    NestedField(6, "model_version", StringType(), required=True),
                    NestedField(7, "prediction_time", TimestampType(), required=True),
                    NestedField(8, "risk_score", DoubleType(), required=True),
                    NestedField(9, "risk_label", StringType(), required=True),
                    NestedField(10, "schema_version", IntegerType(), required=True),
                    NestedField(11, "updated_at", TimestampType(), required=True),
                ),
                schema_version=1,
            ),
        ]

    def _partition_spec_for(self, schema: Schema) -> PartitionSpec:
        cohort_id = schema.find_field("patient_cohort").field_id
        ingestion_id = schema.find_field("ingestion_date").field_id
        return PartitionSpec(
            PartitionField(source_id=cohort_id, field_id=1000, transform=IdentityTransform(), name="patient_cohort"),
            PartitionField(source_id=ingestion_id, field_id=1001, transform=IdentityTransform(), name="ingestion_date"),
        )

    def create_or_update_tables(self) -> None:
        if self.catalog is None:
            raise RuntimeError("Catalog not connected")

        for spec in self.table_specs():
            try:
                self.catalog.load_table(spec.name)
            except NoSuchTableError:
                self.catalog.create_table(
                    identifier=spec.name,
                    schema=spec.schema,
                    partition_spec=self._partition_spec_for(spec.schema),
                )
                self.log.info("iceberg.table_created", table=spec.name, schema_version=spec.schema_version)

    def apply_backward_compatible_migration(self, table_name: str, new_columns: list[tuple[str, Any]]) -> None:
        """Add-only schema migration to preserve backward compatibility."""
        if self.catalog is None:
            raise RuntimeError("Catalog not connected")

        table = self.catalog.load_table(f"{self.namespace}.{table_name}")
        if not hasattr(table, "update_schema"):
            return

        with table.update_schema() as updater:
            for col_name, col_type in new_columns:
                updater.add_column(col_name, col_type)

    def append_dataframe(self, table_name: str, frame: pd.DataFrame) -> None:
        if self.catalog is None:
            raise RuntimeError("Catalog not connected")

        frame = validate_frame(table_name, frame.copy())
        frame["schema_version"] = frame.get("schema_version", 1)
        frame["updated_at"] = datetime.utcnow()

        table = self.catalog.load_table(f"{self.namespace}.{table_name}")
        arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
        table.append(arrow_table)

    def scan_table(self, table_name: str) -> pd.DataFrame:
        if self.catalog is None:
            raise RuntimeError("Catalog not connected")
        table = self.catalog.load_table(f"{self.namespace}.{table_name}")
        return table.scan().to_arrow().to_pandas()
