# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.logging import get_logger
from neurosynth.data.iceberg_catalog import IcebergDomainCatalog


class NeuroKnowledgeGraphBuilder:
    """Builds a Neo4j knowledge graph from Iceberg domain tables and causal edges."""

    def __init__(self, settings: NeuroSynthSettings) -> None:
        self.settings = settings
        self.log = get_logger(__name__)
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        await self.driver.close()

    async def initialize_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (n:Patient) REQUIRE n.patient_id IS UNIQUE",
            "CREATE CONSTRAINT region_name IF NOT EXISTS FOR (n:BrainRegion) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT gene_name IF NOT EXISTS FOR (n:Gene) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT pathway_name IF NOT EXISTS FOR (n:Pathway) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT variant_id IF NOT EXISTS FOR (n:Variant) REQUIRE n.variant_id IS UNIQUE",
            "CREATE CONSTRAINT biomarker_key IF NOT EXISTS FOR (n:Biomarker) REQUIRE n.biomarker_id IS UNIQUE",
            "CREATE CONSTRAINT intervention_name IF NOT EXISTS FOR (n:Intervention) REQUIRE n.name IS UNIQUE",
        ]
        async with self.driver.session() as session:
            for statement in statements:
                await session.run(statement)

    async def populate_from_iceberg(self, iceberg: IcebergDomainCatalog) -> None:
        patients = iceberg.scan_table("patients")
        imaging = iceberg.scan_table("imaging_sessions")
        variants = iceberg.scan_table("genomic_variants")
        biomarkers = iceberg.scan_table("biomarker_timeseries")
        causal_edges = iceberg.scan_table("causal_graphs")

        await self.initialize_constraints()

        async with self.driver.session() as session:
            for row in patients.to_dict(orient="records"):
                await session.run(
                    "MERGE (p:Patient {patient_id:$patient_id}) "
                    "SET p.cohort=$cohort, p.sex=$sex, p.birth_year=$birth_year",
                    patient_id=row["patient_id"],
                    cohort=row.get("patient_cohort"),
                    sex=row.get("sex"),
                    birth_year=row.get("birth_year"),
                )

            for row in imaging.to_dict(orient="records"):
                await session.run(
                    "MERGE (p:Patient {patient_id:$patient_id}) "
                    "MERGE (r:BrainRegion {name:$region}) "
                    "MERGE (p)-[:HAS_IMAGING {session_id:$session_id, modality:$modality}]->(r)",
                    patient_id=row["patient_id"],
                    region="WholeBrain",
                    session_id=row["imaging_session_id"],
                    modality=row.get("modality"),
                )

            for row in variants.to_dict(orient="records"):
                await session.run(
                    "MERGE (p:Patient {patient_id:$patient_id}) "
                    "MERGE (v:Variant {variant_id:$variant_id}) "
                    "SET v.chrom=$chrom, v.pos=$pos, v.clinvar=$clinvar "
                    "MERGE (p)-[:EXPRESSES_VARIANT]->(v)",
                    patient_id=row["patient_id"],
                    variant_id=row["variant_id"],
                    chrom=row.get("chrom"),
                    pos=row.get("pos"),
                    clinvar=row.get("clinvar_significance"),
                )
                gene = row.get("gene")
                if gene:
                    await session.run(
                        "MERGE (v:Variant {variant_id:$variant_id}) "
                        "MERGE (g:Gene {name:$gene}) "
                        "MERGE (v)-[:ASSOCIATED_WITH]->(g)",
                        variant_id=row["variant_id"],
                        gene=gene,
                    )

            for row in biomarkers.to_dict(orient="records"):
                biomarker_id = f"{row['patient_id']}::{row.get('modality')}::{row.get('window_start')}"
                await session.run(
                    "MERGE (p:Patient {patient_id:$patient_id}) "
                    "MERGE (b:Biomarker {biomarker_id:$biomarker_id}) "
                    "SET b.modality=$modality, b.metric_mean=$metric_mean, b.metric_std=$metric_std "
                    "MERGE (p)-[:ASSOCIATED_WITH]->(b)",
                    patient_id=row["patient_id"],
                    biomarker_id=biomarker_id,
                    modality=row.get("modality"),
                    metric_mean=row.get("metric_mean"),
                    metric_std=row.get("metric_std"),
                )

            for row in causal_edges.to_dict(orient="records"):
                await session.run(
                    "MERGE (a:Biomarker {biomarker_id:$src}) "
                    "MERGE (b:Biomarker {biomarker_id:$dst}) "
                    "MERGE (a)-[r:CAUSAL_PREDECESSOR_OF]->(b) "
                    "SET r.weight=$weight, r.edge_type=$edge_type",
                    src=row.get("source_node"),
                    dst=row.get("target_node"),
                    weight=row.get("edge_weight"),
                    edge_type=row.get("edge_type"),
                )

        self.log.info(
            "neo4j.populate_complete",
            patient_nodes=len(patients),
            imaging_edges=len(imaging),
            variants=len(variants),
            biomarkers=len(biomarkers),
            causal_edges=len(causal_edges),
        )

    async def attach_interventions(self, interventions: list[dict[str, Any]]) -> None:
        async with self.driver.session() as session:
            for item in interventions:
                await session.run(
                    "MERGE (p:Patient {patient_id:$patient_id}) "
                    "MERGE (i:Intervention {name:$name}) "
                    "SET i.type=$itype, i.intent=$intent "
                    "MERGE (p)-[:RECEIVED_INTERVENTION {at:$at}]->(i)",
                    patient_id=item["patient_id"],
                    name=item["name"],
                    itype=item.get("type"),
                    intent=item.get("intent"),
                    at=item.get("at"),
                )
