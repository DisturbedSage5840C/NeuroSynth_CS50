# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import httpx
from neo4j import AsyncGraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import GraphLoadError
from neurosynth.core.logging import get_logger


class NeuroKnowledgeGraph:
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def initialize_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (n:Gene) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (n:Protein) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (n:Pathway) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT brainregion_id IF NOT EXISTS FOR (n:BrainRegion) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX gene_symbol IF NOT EXISTS FOR (n:Gene) ON (n.symbol)",
            "CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name)",
            "CREATE INDEX pathway_name IF NOT EXISTS FOR (n:Pathway) ON (n.name)",
        ]
        async with self._driver.session() as session:
            for stmt in statements:
                await session.run(stmt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=15), reraise=True)
    async def load_string_interactions(self, path: Path, min_score: int = 700) -> None:
        query = """
        CALL apoc.periodic.iterate(
          "LOAD CSV WITH HEADERS FROM $uri AS row RETURN row",
          "WITH row WHERE toInteger(row.combined_score) >= $min_score
           MERGE (g1:Gene {id: row.protein1})
           MERGE (g2:Gene {id: row.protein2})
           MERGE (g1)-[r:INTERACTS_WITH]->(g2)
           SET r.score = toInteger(row.combined_score), r.interaction_type = row.mode",
          {batchSize: 10000, parallel: true, params: {uri: $uri, min_score: $min_score}}
        )
        """
        async with self._driver.session() as session:
            await session.run(query, uri=f"file:///{path.name}", min_score=min_score)

    async def load_allen_brain_expression(self, api_key: str) -> int:
        url = "http://api.brain-map.org/api/v2/data/query.json"
        params = {"criteria": "model::GeneExpression", "num_rows": 2000, "api_key": api_key}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        rows = payload.get("msg", [])
        async with self._driver.session() as session:
            for row in rows:
                await session.run(
                    """
                    MERGE (b:BrainRegion {id: $region_id})
                    SET b.aal_id = $aal_id, b.name = $region_name, b.hemisphere = $hemisphere, b.lobe = $lobe
                    MERGE (g:Gene {id: $gene_id})
                    SET g.symbol = $symbol
                    MERGE (b)-[r:EXPRESSES]->(g)
                    SET r.level = $level, r.zscore = $zscore
                    """,
                    region_id=str(row.get("structure_id", "")),
                    aal_id=str(row.get("structure_id", "")),
                    region_name=row.get("structure_name", "unknown"),
                    hemisphere=row.get("hemisphere", "unknown"),
                    lobe=row.get("lobe", "unknown"),
                    gene_id=str(row.get("gene_id", "")),
                    symbol=row.get("gene_symbol", ""),
                    level=float(row.get("expression_energy", 0.0)),
                    zscore=float(row.get("z_score", 0.0)),
                )
        return len(rows)

    async def load_clinvar_variants(self, vcf_path: Path) -> int:
        loaded = 0
        async with self._driver.session() as session:
            with vcf_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    cols = line.rstrip("\n").split("\t")
                    info = cols[7]
                    if "CLNSIG=Pathogenic" not in info and "CLNSIG=Likely_pathogenic" not in info:
                        continue
                    gene = "UNKNOWN"
                    disease = "Unknown disease"
                    for part in info.split(";"):
                        if part.startswith("GENEINFO="):
                            gene = part.split("=", 1)[1].split(":", 1)[0]
                        if part.startswith("CLNDN="):
                            disease = part.split("=", 1)[1].replace("_", " ")

                    await session.run(
                        """
                        MERGE (g:Gene {symbol: $gene})
                        MERGE (d:Disease {name: $disease})
                        MERGE (g)-[r:CAUSES_RISK]->(d)
                        SET r.source = 'ClinVar'
                        """,
                        gene=gene,
                        disease=disease,
                    )
                    loaded += 1
        return loaded

    async def get_patient_pathway_subgraph(self, patient_ids: list[str], disease: str) -> list[dict[str, Any]]:
        query = """
        MATCH (d:Disease {subtype: $disease})<-[:INVOLVED_IN]-(p:Pathway)
        MATCH (p)<-[:PARTICIPATES_IN]-(pr:Protein)<-[:ENCODES]-(g:Gene)
        MATCH (b:BrainRegion)-[:EXPRESSES]->(g)
        WHERE g.id IN $patient_ids
        RETURN p, b, g, pr
        """
        async with self._driver.session() as session:
            cursor = await session.run(query, disease=disease, patient_ids=patient_ids)
            records = await cursor.data()
        return records

    async def compute_pathway_centrality(self) -> None:
        queries = [
            """
            CALL apoc.algo.betweenness(['Protein','Pathway'], ['PARTICIPATES_IN'], 'both')
            YIELD node, score
            SET node.betweenness = score
            """,
            """
            CALL apoc.algo.pageRank(['Protein','Pathway'], ['PARTICIPATES_IN'], 20, 0.85)
            YIELD node, score
            SET node.pagerank = score
            """,
        ]
        async with self._driver.session() as session:
            for query in queries:
                await session.run(query)
