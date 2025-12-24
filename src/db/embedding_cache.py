"""
Embedding Cache - SurrealDB-backed cache for embedding vectors.

v1.2.1: Cache embeddings to avoid redundant API calls.

Key = hash(text) + model
Value = vector (3072 floats)

Schema:
    DEFINE TABLE embedding_cache SCHEMAFULL;
    DEFINE FIELD text_hash TYPE string;
    DEFINE FIELD model TYPE string;
    DEFINE FIELD vector TYPE array<float>;
    DEFINE FIELD created_at TYPE datetime;
    DEFINE INDEX idx_lookup ON embedding_cache FIELDS text_hash, model UNIQUE;
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional, Sequence

from src.db.client import db

logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class EmbeddingCache:
    """
    SurrealDB-backed embedding cache.

    Usage:
        cache = EmbeddingCache()
        await cache.ensure_schema()

        # Check cache
        cached = await cache.get("some text", "text-embedding-3-large")
        if cached:
            return cached

        # Compute and store
        vector = await embeddings.embed("some text")
        await cache.set("some text", "text-embedding-3-large", vector)
    """

    TABLE = "embedding_cache"

    async def ensure_schema(self) -> None:
        """Create table and index if not exists."""
        try:
            await db.query(f"""
                DEFINE TABLE {self.TABLE} SCHEMAFULL;
                DEFINE FIELD text_hash TYPE string;
                DEFINE FIELD model TYPE string;
                DEFINE FIELD vector TYPE array<float>;
                DEFINE FIELD created_at TYPE datetime;
                DEFINE INDEX idx_lookup ON {self.TABLE} FIELDS text_hash, model UNIQUE;
            """)
            logger.info(f"Embedding cache table '{self.TABLE}' ready")
        except Exception as e:
            # Table might already exist
            logger.debug(f"Schema already exists or error: {e}")

    async def get(self, text: str, model: str) -> Optional[list[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Original text (will be hashed)
            model: Model name (e.g., "text-embedding-3-large")

        Returns:
            Cached vector or None if not found
        """
        text_hash = compute_text_hash(text)

        try:
            result = await db.query(
                f"""
                SELECT vector FROM {self.TABLE}
                WHERE text_hash = $hash AND model = $model
                LIMIT 1
                """,
                {"hash": text_hash, "model": model},
            )

            if result and result[0].get("result"):
                rows = result[0]["result"]
                if rows and rows[0].get("vector"):
                    logger.debug(f"Cache hit: {text_hash[:8]}...")
                    return rows[0]["vector"]

            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def get_batch(
        self,
        texts: Sequence[str],
        model: str,
    ) -> dict[int, list[float]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts
            model: Model name

        Returns:
            Dict mapping text index -> vector (only for cache hits)
        """
        if not texts:
            return {}

        # Compute hashes
        hashes = [(i, compute_text_hash(t)) for i, t in enumerate(texts)]
        hash_to_idx = {h: i for i, h in hashes}
        hash_list = [h for _, h in hashes]

        try:
            result = await db.query(
                f"""
                SELECT text_hash, vector FROM {self.TABLE}
                WHERE text_hash IN $hashes AND model = $model
                """,
                {"hashes": hash_list, "model": model},
            )

            cached = {}
            if result and result[0].get("result"):
                for row in result[0]["result"]:
                    h = row.get("text_hash")
                    vec = row.get("vector")
                    if h and vec and h in hash_to_idx:
                        cached[hash_to_idx[h]] = vec

            logger.debug(f"Cache batch: {len(cached)}/{len(texts)} hits")
            return cached

        except Exception as e:
            logger.warning(f"Cache batch get error: {e}")
            return {}

    async def set(self, text: str, model: str, vector: list[float]) -> bool:
        """
        Store embedding in cache.

        Args:
            text: Original text
            model: Model name
            vector: Embedding vector

        Returns:
            True if stored successfully
        """
        text_hash = compute_text_hash(text)

        try:
            await db.query(
                f"""
                INSERT INTO {self.TABLE} {{
                    text_hash: $hash,
                    model: $model,
                    vector: $vector,
                    created_at: time::now()
                }} ON DUPLICATE KEY UPDATE vector = $vector, created_at = time::now()
                """,
                {"hash": text_hash, "model": model, "vector": vector},
            )
            return True

        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def set_batch(
        self,
        texts: Sequence[str],
        model: str,
        vectors: Sequence[list[float]],
    ) -> int:
        """
        Store multiple embeddings in cache.

        Args:
            texts: List of texts
            model: Model name
            vectors: List of embedding vectors

        Returns:
            Number of entries stored
        """
        if not texts or len(texts) != len(vectors):
            return 0

        stored = 0
        # Use transaction for efficiency
        try:
            for text, vector in zip(texts, vectors):
                if await self.set(text, model, vector):
                    stored += 1

            logger.debug(f"Cache batch set: {stored}/{len(texts)} stored")
            return stored

        except Exception as e:
            logger.warning(f"Cache batch set error: {e}")
            return stored

    async def stats(self) -> dict:
        """Get cache statistics."""
        try:
            result = await db.query(f"""
                SELECT
                    count() as total,
                    model,
                    count() as count
                FROM {self.TABLE}
                GROUP BY model
            """)

            total_result = await db.query(f"SELECT count() as total FROM {self.TABLE}")
            total = 0
            if total_result and total_result[0].get("result"):
                rows = total_result[0]["result"]
                if rows:
                    total = rows[0].get("total", 0)

            return {
                "total_cached": total,
                "by_model": result[0].get("result", []) if result else [],
            }

        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"total_cached": 0, "by_model": []}

    async def clear(self, model: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            model: If specified, only clear entries for this model

        Returns:
            Number of entries deleted
        """
        try:
            if model:
                result = await db.query(
                    f"DELETE FROM {self.TABLE} WHERE model = $model",
                    {"model": model},
                )
            else:
                result = await db.query(f"DELETE FROM {self.TABLE}")

            # SurrealDB returns deleted count in result
            return 0  # TODO: Parse actual count from result

        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0


# Global cache instance
embedding_cache = EmbeddingCache()
