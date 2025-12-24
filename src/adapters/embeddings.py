"""Embeddings adapter - ONLY text-embedding-3-large (dim=3072)."""

import asyncio
import logging
from typing import Sequence

import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

# Token encoder for text-embedding-3-large
_encoder = tiktoken.get_encoding("cl100k_base")


class EmbeddingsAdapter:
    """
    Embeddings adapter using ONLY text-embedding-3-large.

    RULES:
    - Model: text-embedding-3-large (no alternatives)
    - Dimension: 3072 (native, no truncation)
    - Max tokens: 8191 per request
    """

    MODEL = "text-embedding-3-large"
    DIMENSION = 3072
    MAX_TOKENS = 8191

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")

        # Validate model settings
        if settings.embedding_model != self.MODEL:
            raise ValueError(
                f"Invalid embedding_model: {settings.embedding_model}. "
                f"Only '{self.MODEL}' is allowed."
            )
        if settings.embedding_dim != self.DIMENSION:
            raise ValueError(
                f"Invalid embedding_dim: {settings.embedding_dim}. "
                f"Only {self.DIMENSION} is allowed."
            )

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info(f"EmbeddingsAdapter initialized: {self.MODEL} (dim={self.DIMENSION})")

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        return len(_encoder.encode(text))

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
        """Truncate text to max tokens."""
        tokens = _encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _encoder.decode(tokens[:max_tokens])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text.

        Returns:
            Vector of 3072 floats.
        """
        # Truncate if needed
        text = self.truncate_to_tokens(text.strip())
        if not text:
            raise ValueError("Cannot embed empty text")

        response = await self._client.embeddings.create(
            model=self.MODEL,
            input=text,
        )

        vector = response.data[0].embedding

        # Validate dimension
        if len(vector) != self.DIMENSION:
            raise ValueError(
                f"Unexpected embedding dimension: {len(vector)}, expected {self.DIMENSION}"
            )

        return vector

    async def embed_batch(
        self,
        texts: Sequence[str],
        max_concurrent: int | None = None,
    ) -> list[list[float]]:
        """
        Embed multiple texts concurrently.

        Args:
            texts: List of texts to embed.
            max_concurrent: Max concurrent requests (default from settings).

        Returns:
            List of vectors (3072 floats each).
        """
        if not texts:
            return []

        max_concurrent = max_concurrent or settings.max_concurrent_embeddings
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_with_semaphore(text: str) -> list[float]:
            async with semaphore:
                return await self.embed(text)

        tasks = [embed_with_semaphore(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def embed_batch_api(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed using OpenAI Batch API (cheaper: $0.065/1M vs $0.13/1M).

        Note: Batch API has latency (up to 24h), use for bulk processing only.
        """
        # Prepare texts
        prepared = [self.truncate_to_tokens(t.strip()) for t in texts if t.strip()]
        if not prepared:
            return []

        # For small batches, use regular API
        if len(prepared) <= 100:
            return await self.embed_batch(prepared)

        # For large batches, use Batch API
        # TODO: Implement Batch API logic
        # For now, fall back to concurrent embedding
        logger.warning(
            f"Batch API not implemented, using concurrent embedding for {len(prepared)} texts"
        )
        return await self.embed_batch(prepared)


# Global adapter instance
embeddings = EmbeddingsAdapter() if settings.openai_api_key else None
