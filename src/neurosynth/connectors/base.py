from __future__ import annotations

import abc
import asyncio
from typing import Any

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class AbstractNeuroDataSource(abc.ABC):
    @abc.abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def validate_schema(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def fetch_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def stream(self, queue: asyncio.Queue) -> None:
        raise NotImplementedError
