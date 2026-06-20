from __future__ import annotations

from typing import Any

import asyncpg

from backend.core.config import get_settings


async def _init_pgvector(conn: asyncpg.Connection) -> None:
    """Register the pgvector codec on each new pool connection.

    No-op when pgvector is not installed or the extension is not enabled —
    allows the rest of the app to boot normally in environments without
    pgvector (e.g. plain Postgres, local dev without the extension).
    """
    try:
        from pgvector.asyncpg import register_vector
        await register_vector(conn)
    except Exception:
        pass


class Database:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool is not None:
            return
        settings = get_settings()
        self.pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn,
            min_size=1,
            max_size=10,
            timeout=5,
            command_timeout=10,
            init=_init_pgvector,
        )

    async def disconnect(self) -> None:
        if self.pool is None:
            return
        await self.pool.close()
        self.pool = None

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        if self.pool is None:
            return None
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if self.pool is None:
            return []
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> str | None:
        if self.pool is None:
            return None
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def apply_schema(self, schema_sql: str) -> None:
        """Apply an idempotent schema (CREATE TABLE IF NOT EXISTS ...) on startup."""
        if self.pool is None:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

    # ------------------------------------------------------------------
    # Literature / RAG helpers
    # ------------------------------------------------------------------

    async def search_literature(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search over PubMed abstract embeddings.

        Returns up to top_k records ordered by descending similarity.
        Returns [] when the table is empty or pgvector is not available.
        """
        if self.pool is None:
            return []
        try:
            import numpy as np
            vec = np.array(query_embedding, dtype=np.float32)
            rows = await self.fetch(
                """
                SELECT pmid, title, abstract, journal, pub_year, diseases,
                       1 - (embedding <=> $1) AS similarity
                FROM   literature_embeddings
                ORDER  BY embedding <=> $1
                LIMIT  $2
                """,
                vec,
                top_k,
            )
            return [dict(r) for r in rows]
        except Exception:
            return []

    async def fetch_abstract(self, pmid: str) -> dict[str, Any] | None:
        """Return a single abstract record by PMID, or None if not found."""
        if self.pool is None:
            return None
        try:
            row = await self.fetchrow(
                "SELECT pmid, title, abstract, journal, pub_year, diseases "
                "FROM literature_embeddings WHERE pmid = $1",
                pmid,
            )
            return dict(row) if row else None
        except Exception:
            return None

    async def upsert_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str,
        journal: str,
        pub_year: int | None,
        diseases: list[str],
        embedding: list[float],
    ) -> None:
        """Insert or update a single abstract + embedding in the literature table."""
        if self.pool is None:
            return
        import numpy as np
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
        vec = np.array(embedding, dtype=np.float32)
        await self.execute(
            """
            INSERT INTO literature_embeddings
                (pmid, title, abstract, journal, pub_year, diseases, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (pmid) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                journal = EXCLUDED.journal,
                pub_year = EXCLUDED.pub_year,
                diseases = EXCLUDED.diseases,
                embedding = EXCLUDED.embedding
            """,
            pmid, title, abstract, journal, pub_year, diseases, vec,
        )

    async def count_literature(self) -> int:
        """Return number of abstracts currently stored."""
        if self.pool is None:
            return 0
        try:
            row = await self.fetchrow("SELECT COUNT(*) FROM literature_embeddings")
            return int(row[0]) if row else 0
        except Exception:
            return 0


_db = Database()


def get_db() -> Database:
    return _db
