# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.connectors.base import AbstractNeuroDataSource
from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import DataIngestionError
from neurosynth.core.logging import get_logger
from neurosynth.core.models import OAuthToken


class PPMIConnector(AbstractNeuroDataSource):
    FILES = [
        "Subject_Characteristics.csv",
        "Motor_Assessments.csv",
        "Biospecimen_Results.csv",
        "DAT_SPECT_metadata.csv",
    ]

    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._client: httpx.AsyncClient | None = None
        self._token: OAuthToken | None = None
        self._data_cache: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._settings.ppmi_base_url, timeout=60.0)
        await self._refresh_token()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def _refresh_token(self) -> None:
        if not self._client:
            raise DataIngestionError("PPMI client not initialized")

        response = await self._client.post(
            "oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._settings.ppmi_client_id,
                "client_secret": self._settings.ppmi_client_secret,
            },
        )
        response.raise_for_status()
        payload = response.json()
        self._token = OAuthToken(
            access_token=payload["access_token"],
            token_type=payload.get("token_type", "Bearer"),
            expires_in=int(payload.get("expires_in", 3600)),
            acquired_at=datetime.now(timezone.utc),
        )

    async def _ensure_token(self) -> None:
        if self._token is None:
            await self._refresh_token()
            return
        expires_at = self._token.acquired_at + timedelta(seconds=self._token.expires_in - 60)
        if datetime.now(timezone.utc) >= expires_at:
            await self._refresh_token()

    async def validate_schema(self) -> None:
        if not self._data_cache:
            raise DataIngestionError("No PPMI data loaded")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def _download_file(self, file_name: str) -> list[dict[str, Any]]:
        if not self._client:
            raise DataIngestionError("PPMI client not initialized")
        await self._ensure_token()
        headers = {"Authorization": f"Bearer {self._token.access_token}"}
        response = await self._client.get(f"files/{file_name}", headers=headers)
        response.raise_for_status()

        lines = response.text.splitlines()
        if not lines:
            return []
        header = lines[0].split(",")
        rows: list[dict[str, Any]] = []
        for line in lines[1:]:
            cols = line.split(",")
            rows.append(dict(zip(header, cols)))
        return rows

    async def load_datasets(self) -> None:
        results = await asyncio.gather(*[self._download_file(name) for name in self.FILES])
        self._data_cache = [item for dataset in results for item in dataset]
        self._logger.info("ppmi.loaded", rows=len(self._data_cache))

    async def fetch_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        return self._data_cache[offset : offset + limit]

    async def stream(self, queue: asyncio.Queue) -> None:
        for row in self._data_cache:
            await queue.put(row)
