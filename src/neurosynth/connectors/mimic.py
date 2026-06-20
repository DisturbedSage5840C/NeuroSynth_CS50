from __future__ import annotations

import asyncio
from typing import Any

import psycopg
from psycopg.rows import dict_row
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.connectors.base import AbstractNeuroDataSource
from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import DataIngestionError
from neurosynth.core.logging import get_logger

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class MIMICConnector(AbstractNeuroDataSource):
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._conn: psycopg.AsyncConnection[Any] | None = None
        self._last_hadm_id: int = 0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=15), reraise=True)
    async def connect(self) -> None:
        self._conn = await psycopg.AsyncConnection.connect(self._settings.mimic_dsn, autocommit=True)

    async def validate_schema(self) -> None:
        if not self._conn:
            raise DataIngestionError("MIMIC connector not connected")

        query = """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_schema = 'mimiciv'
          AND table_name IN ('patients', 'admissions', 'diagnoses_icd', 'labevents', 'noteevents')
        """
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query)
            row = await cur.fetchone()
            if not row or row["n"] < 5:
                raise DataIngestionError("Missing required MIMIC-IV tables")

    def _base_query(self) -> str:
        return """
        SELECT
            p.subject_id,
            a.hadm_id,
            a.admittime,
            d.icd_code,
            d.icd_version,
            l.itemid,
            l.valuenum,
            n.charttime,
            n.text AS discharge_summary
        FROM mimiciv.patients p
        JOIN mimiciv.admissions a ON a.subject_id = p.subject_id
        JOIN mimiciv.diagnoses_icd d ON d.hadm_id = a.hadm_id
        LEFT JOIN mimiciv.labevents l
          ON l.hadm_id = a.hadm_id AND l.itemid IN (51146, 50820)
        LEFT JOIN mimiciv.noteevents n
          ON n.hadm_id = a.hadm_id AND n.category = 'Discharge summary'
        WHERE d.icd_version = 10
          AND d.icd_code ~ '^G[0-9]{2}'
          AND a.hadm_id > %(cursor)s
        ORDER BY a.hadm_id
        LIMIT %(limit)s
        """

    async def fetch_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        if not self._conn:
            raise DataIngestionError("MIMIC connector not connected")
        cursor_value = max(offset, self._last_hadm_id)
        async with self._conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(self._base_query(), {"cursor": cursor_value, "limit": limit})
            rows = await cur.fetchall()
        if rows:
            self._last_hadm_id = max(int(r["hadm_id"]) for r in rows)
        return rows

    async def stream(self, queue: asyncio.Queue) -> None:
        offset = self._last_hadm_id
        limit = 2000
        while True:
            batch = await self.fetch_batch(offset, limit)
            if not batch:
                break
            for row in batch:
                await queue.put(row)
            offset = self._last_hadm_id
        self._logger.info("mimic.stream_complete", cursor=self._last_hadm_id)
