"""Embeddings adapter - ONLY text-embedding-3-large (dim=3072).

v1.2.1: True batch embedding with token-aware batching.
- Multiple texts per API call (reduces RTT by ~80%)
- Token-aware batch building (respects 8191 limit)
- Parallel batch processing with semaphore
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Sequence

import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

# Token encoder for text-embedding-3-large
_encoder = tiktoken.get_encoding("cl100k_base")

# Batch settings
MAX_TEXTS_PER_BATCH = 32  # OpenAI recommends ≤2048, but 32 is safer for memory
MAX_TOKENS_PER_BATCH = 8000  # Leave margin under 8191


@dataclass
class EmbeddingBatch:
    """A batch of texts to embed together."""
    texts: list[str]
    indices: list[int]  # Original indices for result ordering
    total_tokens: int


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

    def _build_batches(self, texts: Sequence[str]) -> list[EmbeddingBatch]:
        """
        Build token-aware batches from texts.

        Respects:
        - MAX_TEXTS_PER_BATCH (32 texts)
        - MAX_TOKENS_PER_BATCH (8000 tokens)
        """
        batches: list[EmbeddingBatch] = []
        current_batch = EmbeddingBatch(texts=[], indices=[], total_tokens=0)

        for i, text in enumerate(texts):
            # Prepare text
            prepared = self.truncate_to_tokens(text.strip())
            if not prepared:
                continue

            tokens = self.count_tokens(prepared)

            # Check if adding this text would exceed limits
            would_exceed_texts = len(current_batch.texts) >= MAX_TEXTS_PER_BATCH
            would_exceed_tokens = current_batch.total_tokens + tokens > MAX_TOKENS_PER_BATCH

            if current_batch.texts and (would_exceed_texts or would_exceed_tokens):
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = EmbeddingBatch(texts=[], indices=[], total_tokens=0)

            # Add text to current batch
            current_batch.texts.append(prepared)
            current_batch.indices.append(i)
            current_batch.total_tokens += tokens

        # Don't forget the last batch
        if current_batch.texts:
            batches.append(current_batch)

        return batches

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _embed_batch_api_call(self, batch: EmbeddingBatch) -> list[list[float]]:
        """
        Make a single API call with multiple texts.

        This is the key optimization: one RTT for many embeddings.
        """
        response = await self._client.embeddings.create(
            model=self.MODEL,
            input=batch.texts,
        )

        # Extract vectors in order
        vectors = [item.embedding for item in response.data]

        # Validate dimensions
        for i, vec in enumerate(vectors):
            if len(vec) != self.DIMENSION:
                raise ValueError(
                    f"Unexpected embedding dimension: {len(vec)}, expected {self.DIMENSION}"
                )

        logger.debug(
            f"Batch embedded: {len(batch.texts)} texts, {batch.total_tokens} tokens"
        )

        return vectors

    async def embed_batch(
        self,
        texts: Sequence[str],
        max_concurrent: int | None = None,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """
        Embed multiple texts using true batch API with caching.

        v1.2.1: Uses OpenAI's multi-text embedding API for efficiency.
        - Checks cache first (reduces API calls)
        - Builds token-aware batches for cache misses
        - Sends batches in parallel (with semaphore)
        - Stores results in cache
        - Returns vectors in original order

        Args:
            texts: List of texts to embed.
            max_concurrent: Max concurrent batch requests (default: 4).
            use_cache: Whether to use embedding cache (default: True).

        Returns:
            List of vectors (3072 floats each), same order as input.
        """
        if not texts:
            return []

        # Prepare texts (strip and truncate)
        prepared_texts = [self.truncate_to_tokens(t.strip()) for t in texts]

        # Initialize result array
        result_vectors: list[list[float] | None] = [None] * len(texts)

        # Check cache first
        cache_hits = 0
        texts_to_embed: list[tuple[int, str]] = []

        if use_cache:
            try:
                from src.db.embedding_cache import embedding_cache

                cached = await embedding_cache.get_batch(prepared_texts, self.MODEL)

                for i, text in enumerate(prepared_texts):
                    if i in cached:
                        result_vectors[i] = cached[i]
                        cache_hits += 1
                    elif text:  # Skip empty texts
                        texts_to_embed.append((i, text))

                if cache_hits > 0:
                    logger.info(f"Cache: {cache_hits}/{len(texts)} hits")

            except ImportError:
                # Cache not available, embed all
                texts_to_embed = [(i, t) for i, t in enumerate(prepared_texts) if t]
        else:
            texts_to_embed = [(i, t) for i, t in enumerate(prepared_texts) if t]

        # Embed cache misses
        if texts_to_embed:
            # Build batches from cache misses
            miss_texts = [t for _, t in texts_to_embed]
            miss_indices = [i for i, _ in texts_to_embed]

            batches = self._build_batches(miss_texts)

            if batches:
                logger.info(
                    f"Embedding {len(miss_texts)} texts in {len(batches)} batches "
                    f"(avg {len(miss_texts) // max(len(batches), 1)} per batch)"
                )

                # Process batches in parallel
                max_concurrent = max_concurrent or 4
                semaphore = asyncio.Semaphore(max_concurrent)

                async def process_batch(batch: EmbeddingBatch) -> tuple[list[int], list[list[float]]]:
                    async with semaphore:
                        vectors = await self._embed_batch_api_call(batch)
                        return batch.indices, vectors

                # Execute all batches
                batch_results = await asyncio.gather(*[process_batch(b) for b in batches])

                # Map batch indices back to original miss indices
                miss_vectors: list[list[float] | None] = [None] * len(miss_texts)
                for batch_indices, vectors in batch_results:
                    for batch_idx, vec in zip(batch_indices, vectors):
                        miss_vectors[batch_idx] = vec

                # Store in cache and result array
                if use_cache:
                    try:
                        from src.db.embedding_cache import embedding_cache

                        for miss_idx, (orig_idx, text) in enumerate(texts_to_embed):
                            vec = miss_vectors[miss_idx]
                            if vec:
                                result_vectors[orig_idx] = vec
                                # Store in cache (fire and forget)
                                asyncio.create_task(
                                    embedding_cache.set(text, self.MODEL, vec)
                                )
                    except ImportError:
                        # Just store results without caching
                        for miss_idx, (orig_idx, _) in enumerate(texts_to_embed):
                            vec = miss_vectors[miss_idx]
                            if vec:
                                result_vectors[orig_idx] = vec
                else:
                    for miss_idx, (orig_idx, _) in enumerate(texts_to_embed):
                        vec = miss_vectors[miss_idx]
                        if vec:
                            result_vectors[orig_idx] = vec

        # Fill empty slots with zero vectors
        final_vectors = []
        for vec in result_vectors:
            if vec is not None:
                final_vectors.append(vec)
            else:
                final_vectors.append([0.0] * self.DIMENSION)

        return final_vectors

    async def embed_batch_legacy(
        self,
        texts: Sequence[str],
        max_concurrent: int | None = None,
    ) -> list[list[float]]:
        """
        Legacy: Embed texts one-by-one (kept for fallback).
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


# Global adapter instance
embeddings = EmbeddingsAdapter() if settings.openai_api_key else None
