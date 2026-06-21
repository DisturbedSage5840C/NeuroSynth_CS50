# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import asyncio
import io
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.connectors.base import AbstractNeuroDataSource
from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import DataIngestionError
from neurosynth.core.logging import get_logger

PTID_PATTERN = re.compile(r"^\d{3}_S_\d{4}$")
VALID_VISIT_CODES = {"BL", "M06", "M12", "M18", "M24"}
REQUIRED_COLUMNS = {
    "CDRSB",
    "ADAS13",
    "MMSE",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
    "ABETA",
    "TAU",
    "PTAU",
}


class ADNIConnector(AbstractNeuroDataSource):
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._merged: pd.DataFrame = pd.DataFrame()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def connect(self) -> None:
        if not self._settings.adni_sftp_host:
            raise DataIngestionError("ADNI SFTP host is not configured")
        try:
            import asyncssh

            conn = await asyncssh.connect(
                self._settings.adni_sftp_host,
                username=self._settings.adni_sftp_user,
                password=self._settings.adni_sftp_password,
                known_hosts=None,
            )
            conn.close()
        except ImportError:
            self._logger.warning("adni.connect.asyncssh_missing", message="Install asyncssh to enable direct SFTP reads")
        except Exception as exc:
            raise DataIngestionError(f"Unable to connect to ADNI SFTP: {exc}") from exc
        self._logger.info("adni.connect", host=self._settings.adni_sftp_host)

    async def _load_csv_sftp(self, sftp_path: str) -> pd.DataFrame:
        try:
            import asyncssh
        except ImportError as exc:  # pragma: no cover
            raise DataIngestionError("asyncssh is required for SFTP CSV loading") from exc

        parsed = urlparse(sftp_path)
        remote_path = parsed.path
        host = parsed.hostname or self._settings.adni_sftp_host
        username = parsed.username or self._settings.adni_sftp_user
        password = parsed.password or self._settings.adni_sftp_password

        if not host or not username:
            raise DataIngestionError("SFTP host and username are required for ADNI reads")

        async with asyncssh.connect(host, username=username, password=password, known_hosts=None) as conn:
            async with conn.start_sftp_client() as sftp:
                async with sftp.open(remote_path, "rb") as f:
                    data = await f.read()

        return pd.read_csv(io.BytesIO(data))

    async def _load_csv(self, path: str) -> pd.DataFrame:
        if path.startswith("sftp://"):
            return await self._load_csv_sftp(path)
        return await asyncio.to_thread(pd.read_csv, path)

    async def validate_schema(self) -> None:
        if self._merged.empty:
            raise DataIngestionError("ADNI dataset not loaded")

        missing_cols = REQUIRED_COLUMNS - set(self._merged.columns)
        if missing_cols:
            raise DataIngestionError(f"Missing ADNI biomarker columns: {sorted(missing_cols)}")

        invalid_ptid = ~self._merged["PTID"].astype(str).str.match(PTID_PATTERN)
        if invalid_ptid.any():
            raise DataIngestionError("Invalid PTID format in ADNI records")

    async def load_files(self, adni_merge_path: str, upenn_path: str) -> None:
        adni_df, upenn_df = await asyncio.gather(
            self._load_csv(adni_merge_path),
            self._load_csv(upenn_path),
        )
        merged = adni_df.merge(upenn_df, on=["PTID", "VISCODE"], how="left", suffixes=("", "_UPENN"))
        merged["VISCODE"] = merged["VISCODE"].fillna("BL")
        merged.loc[~merged["VISCODE"].isin(VALID_VISIT_CODES), "VISCODE"] = "BL"

        for col in REQUIRED_COLUMNS:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
                merged.loc[merged[col] < 0, col] = np.nan

        merged["harmonized_flag"] = merged.get("SITE", "").astype(str).str.len().gt(0)
        self._merged = merged
        self._logger.info("adni.loaded", rows=len(merged))

    async def fetch_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        if self._merged.empty:
            return []
        batch = self._merged.iloc[offset : offset + limit]
        return batch.to_dict(orient="records")

    async def stream(self, queue: asyncio.Queue) -> None:
        if self._merged.empty:
            raise DataIngestionError("ADNI data not loaded before stream")
        for row in self._merged.to_dict(orient="records"):
            await queue.put(row)
        self._logger.info("adni.stream_complete", rows=len(self._merged))
