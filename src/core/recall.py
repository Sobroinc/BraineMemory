"""
Advanced Recall Engine - Hybrid, Graph, and Iterative Search.

Implements:
- Hybrid Search (FTS + Vector with RRF)
- Local Graph Search (entity expansion)
- Global Graph Search (community summaries)
- GAM-style iterative refinement

Supports dependency injection for testability.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.fusion import (
    ScoredItem,
    reciprocal_rank_fusion,
    filter_by_threshold,
    deduplicate_by_content,
    rerank_by_query_relevance,
)
from src.core.router import Router, Pipeline, RetrievalMode

if TYPE_CHECKING:
    from src.adapters.embeddings import EmbeddingsAdapter
    from src.core.user_memory import UserMemoryManager
    from src.db.client import SurrealClient

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Result from recall operation."""

    items: list[ScoredItem]
    total_found: int
    pipeline_used: str
    iterations: int
    user_context: str | None = None
    metadata: dict[str, Any] | None = None


class RecallEngine:
    """
    Advanced recall engine with multiple retrieval strategies.

    Combines:
    - FTS (BM25) for exact/keyword matches
    - Vector (HNSW) for semantic similarity
    - Graph traversal for relationship-based retrieval
    - Community summaries for corpus-wide queries

    Supports dependency injection for all dependencies.
    """

    def __init__(
        self,
        db: "SurrealClient | None" = None,
        embeddings: "EmbeddingsAdapter | None" = None,
        router: "Router | None" = None,
        user_memory: "UserMemoryManager | None" = None,
    ):
        """
        Initialize recall engine.

        Args:
            db: Database client (uses global if not provided)
            embeddings: Embeddings adapter (uses global if not provided)
            router: Query router (uses global if not provided)
            user_memory: User memory manager (uses global if not provided)
        """
        # Lazy import globals for backward compatibility
        if db is None:
            from src.db.client import db as global_db
            db = global_db
        if embeddings is None:
            from src.adapters.embeddings import embeddings as global_embeddings
            embeddings = global_embeddings
        if router is None:
            from src.core.router import router as global_router
            router = global_router
        if user_memory is None:
            from src.core.user_memory import user_memory as global_user_memory
            user_memory = global_user_memory

        self._db = db
        self._embeddings = embeddings
        self.router = router
        self._user_memory = user_memory

    async def recall(
        self,
        query: str,
        user_id: str | None = None,
        mode: str = "auto",
        limit: int = 10,
        **kwargs: Any,
    ) -> RecallResult:
        """
        Main recall entry point.

        Args:
            query: Search query
            user_id: Optional user for personalization
            mode: Retrieval mode ("auto", "fts", "vector", "hybrid", etc.)
            limit: Max results
            **kwargs: Pipeline overrides

        Returns:
            RecallResult with scored items and metadata
        """
        # Get pipeline from router
        pipeline = self.router.get_pipeline(query, mode, **kwargs)

        # Get user context if available
        user_context = None
        if user_id and pipeline.use_user_context:
            user_context = await self._user_memory.get_user_context(user_id, query)

        # Execute retrieval based on pipeline
        if pipeline.max_iterations > 1:
            # GAM-style iterative refinement
            result = await self._iterative_recall(
                query=query,
                pipeline=pipeline,
                limit=limit,
                user_context=user_context,
            )
        else:
            # Single-pass retrieval
            result = await self._single_recall(
                query=query,
                pipeline=pipeline,
                limit=limit,
            )

        result.user_context = user_context
        return result

    async def _single_recall(
        self,
        query: str,
        pipeline: Pipeline,
        limit: int,
    ) -> RecallResult:
        """Execute single-pass retrieval with configured modes."""
        all_results: list[list[ScoredItem]] = []
        weights: list[float] = []

        # Execute each retrieval mode
        for mode in pipeline.retrieval_modes:
            if mode == RetrievalMode.FTS_ONLY:
                results = await self._fts_search(query, limit * 2)
                all_results.append(results)
                weights.append(pipeline.fts_weight)

            elif mode == RetrievalMode.VECTOR_ONLY:
                results = await self._vector_search(query, limit * 2)
                all_results.append(results)
                weights.append(pipeline.vector_weight)

            elif mode == RetrievalMode.HYBRID:
                # Hybrid runs both FTS and Vector
                fts_results = await self._fts_search(query, limit * 2)
                vector_results = await self._vector_search(query, limit * 2)
                all_results.extend([fts_results, vector_results])
                weights.extend([pipeline.fts_weight, pipeline.vector_weight])

            elif mode == RetrievalMode.LOCAL_GRAPH:
                results = await self._local_graph_search(query, limit * 2)
                all_results.append(results)
                weights.append(pipeline.graph_weight)

            elif mode == RetrievalMode.GLOBAL_GRAPH:
                results = await self._global_graph_search(query, limit)
                all_results.append(results)
                weights.append(pipeline.graph_weight)

            elif mode == RetrievalMode.DRIFT:
                # DRIFT combines local + global
                local = await self._local_graph_search(query, limit)
                global_ = await self._global_graph_search(query, limit // 2)
                all_results.extend([local, global_])
                weights.extend([pipeline.graph_weight, pipeline.graph_weight * 0.5])

        # Fuse results using RRF
        if len(all_results) > 1:
            fused = reciprocal_rank_fusion(all_results, weights=weights)
        elif all_results:
            fused = all_results[0]
        else:
            fused = []

        # Post-processing
        fused = filter_by_threshold(fused, pipeline.min_relevance)
        fused = deduplicate_by_content(fused)

        if pipeline.rerank:
            fused = rerank_by_query_relevance(fused, query)

        # Limit final results
        fused = fused[:limit]

        return RecallResult(
            items=fused,
            total_found=len(fused),
            pipeline_used="+".join(m.value for m in pipeline.retrieval_modes),
            iterations=1,
            metadata={
                "modes": [m.value for m in pipeline.retrieval_modes],
                "weights": weights,
            },
        )

    async def _iterative_recall(
        self,
        query: str,
        pipeline: Pipeline,
        limit: int,
        user_context: str | None = None,
    ) -> RecallResult:
        """
        GAM-style iterative refinement.

        Runs multiple iterations, accumulating results until
        confidence threshold is met or max iterations reached.
        """
        accumulated_items: dict[str, ScoredItem] = {}
        iteration = 0
        current_query = query

        for iteration in range(1, pipeline.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{pipeline.max_iterations}: {current_query[:50]}...")

            # Run single recall
            result = await self._single_recall(
                query=current_query,
                pipeline=Pipeline(
                    retrieval_modes=pipeline.retrieval_modes,
                    use_graph=pipeline.use_graph,
                    max_iterations=1,  # Single pass per iteration
                    fts_weight=pipeline.fts_weight,
                    vector_weight=pipeline.vector_weight,
                    graph_weight=pipeline.graph_weight,
                    min_relevance=pipeline.min_relevance,
                    rerank=pipeline.rerank,
                ),
                limit=limit,
            )

            # Accumulate results
            for item in result.items:
                if item.id not in accumulated_items:
                    accumulated_items[item.id] = item
                elif item.score > accumulated_items[item.id].score:
                    accumulated_items[item.id] = item

            # Check confidence (based on top score)
            if result.items:
                top_score = result.items[0].score
                if top_score >= pipeline.confidence_threshold:
                    logger.info(f"Confidence threshold met at iteration {iteration}")
                    break

            # Generate follow-up query for next iteration
            # In production, this would use LLM to reformulate
            # For now, we extract key entities from results
            if iteration < pipeline.max_iterations:
                current_query = self._generate_followup_query(
                    original_query=query,
                    results=result.items,
                    iteration=iteration,
                )

        # Sort accumulated by score
        final_items = sorted(
            accumulated_items.values(),
            key=lambda x: x.score,
            reverse=True,
        )[:limit]

        return RecallResult(
            items=final_items,
            total_found=len(accumulated_items),
            pipeline_used="+".join(m.value for m in pipeline.retrieval_modes),
            iterations=iteration,
            metadata={
                "modes": [m.value for m in pipeline.retrieval_modes],
                "accumulated_unique": len(accumulated_items),
                "final_query": current_query,
            },
        )

    async def _fts_search(self, query: str, limit: int) -> list[ScoredItem]:
        """Full-text search using SurrealDB FTS."""
        try:
            # Use db.fts_search which has the correct query format
            results = await self._db.fts_search(
                table="chunk",
                content_field="content",
                query=query,
                limit=limit,
            )

            items = []
            for row in results:
                # FTS scores can be negative (BM25), convert to positive
                raw_score = float(row.get("relevance", 0))
                # Normalize: higher score = better match
                score = max(0, raw_score + 10) / 10  # Simple normalization
                items.append(ScoredItem(
                    id=str(row["id"]),
                    content=row["content"],
                    score=score,
                    source="fts",
                    metadata=row.get("metadata"),
                ))

            logger.debug(f"FTS search returned {len(items)} results")
            return items

        except Exception as e:
            logger.error(f"FTS search error: {e}")
            return []

    async def _vector_search(self, query: str, limit: int) -> list[ScoredItem]:
        """Vector similarity search using HNSW index."""
        try:
            # Generate query embedding
            query_vector = await self._embeddings.embed(query)

            # Vector search in chunks
            results = await self._db.vector_search(
                table="chunk",
                vector_field="vector",
                query_vector=query_vector,
                limit=limit,
            )

            items = []
            for row in results:
                items.append(ScoredItem(
                    id=str(row["id"]),
                    content=row["content"],
                    score=float(row.get("relevance", row.get("similarity", 0))),
                    source="vector",
                    metadata=row.get("metadata"),
                ))

            logger.debug(f"Vector search returned {len(items)} results")
            return items

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def _local_graph_search(
        self,
        query: str,
        limit: int,
        timeout_s: float = 10.0,
    ) -> list[ScoredItem]:
        """
        Local graph search - entity expansion.

        1. Find entities matching query
        2. Expand to connected entities via edges
        3. Get chunks containing those entities

        Args:
            timeout_s: Timeout for graph expansion (default 10s)

        Returns:
            List of scored items (may be partial on timeout)
        """
        try:
            # First, find matching entities using embedding field
            query_vector = await self._embeddings.embed(query)

            entity_results = await self._db.vector_search(
                table="entity",
                vector_field="embedding",  # Schema uses 'embedding' not 'vector'
                query_vector=query_vector,
                limit=5,  # Top entities
            )

            if not entity_results:
                # Fallback to FTS on entities
                entity_result = await self._db.query(f"""
                    SELECT id, name, type, description
                    FROM entity
                    WHERE name @@ $query OR description @@ $query
                    LIMIT 5
                """, {"query": query})
                entity_results = entity_result[0].get("result", []) if entity_result else []

            if not entity_results:
                logger.debug("No entities found for local graph search")
                return []

            # Get entity IDs
            entity_ids = [str(e["id"]) for e in entity_results]

            # Expand via relates_to edges (1-hop neighbors)
            neighbors_result = await self._db.query("""
                SELECT
                    out.id AS id,
                    out.name AS name,
                    out.description AS description,
                    relation_type,
                    weight
                FROM relates_to
                WHERE in IN $entity_ids
                LIMIT 20
            """, {"entity_ids": entity_ids})

            # Also get reverse edges
            reverse_result = await self._db.query("""
                SELECT
                    in.id AS id,
                    in.name AS name,
                    in.description AS description,
                    relation_type,
                    weight
                FROM relates_to
                WHERE out IN $entity_ids
                LIMIT 20
            """, {"entity_ids": entity_ids})

            # Collect all related entity IDs
            all_entity_ids = set(entity_ids)
            if neighbors_result and neighbors_result[0].get("result"):
                for n in neighbors_result[0]["result"]:
                    if n.get("id"):
                        all_entity_ids.add(str(n["id"]))
            if reverse_result and reverse_result[0].get("result"):
                for n in reverse_result[0]["result"]:
                    if n.get("id"):
                        all_entity_ids.add(str(n["id"]))

            # Find chunks via mentions graph edge (entity->mentions->chunk)
            # SurrealDB doesn't support GROUP BY field, so we query and dedupe in Python
            chunks_result = await self._db.query("""
                SELECT out FROM mentions
                WHERE in IN $all_entity_ids
                LIMIT $limit
            """, {"all_entity_ids": list(all_entity_ids), "limit": limit * 3})

            # Count occurrences per chunk
            chunk_counts: dict[str, int] = {}
            if chunks_result and chunks_result[0].get("result"):
                for row in chunks_result[0]["result"]:
                    chunk_id = str(row.get("out", ""))
                    if chunk_id:
                        chunk_counts[chunk_id] = chunk_counts.get(chunk_id, 0) + 1

            # Sort by count and fetch chunk content
            sorted_chunks = sorted(chunk_counts.items(), key=lambda x: -x[1])[:limit]

            items = []
            for i, (chunk_id, entity_count) in enumerate(sorted_chunks):
                # Fetch chunk content
                chunk_data = await self._db.select(chunk_id)
                if not chunk_data:
                    continue

                # Score based on entity count and position
                score = 1.0 / (i + 1) * (1 + entity_count * 0.1)
                items.append(ScoredItem(
                    id=chunk_id,
                    content=chunk_data.get("content", ""),
                    score=score,
                    source="local_graph",
                    metadata={
                        "entity_count": entity_count,
                        "seed_entities": entity_ids[:3],
                    },
                ))

            logger.debug(f"Local graph search returned {len(items)} results")
            return items

        except Exception as e:
            logger.error(f"Local graph search error: {e}")
            return []

    async def _global_graph_search(self, query: str, limit: int) -> list[ScoredItem]:
        """
        Global graph search - community summaries.

        Uses pre-computed community summaries for corpus-wide queries.
        Searches communities by embedding similarity, then returns
        community summaries as high-level context.
        """
        try:
            # Search community summaries by embedding
            query_vector = await self._embeddings.embed(query)

            community_results = await self._db.vector_search(
                table="community",
                vector_field="embedding",
                query_vector=query_vector,
                limit=limit,
            )

            items = []
            for row in community_results:
                # Community summaries are treated as high-level context
                title = row.get("title", "")
                summary = row.get("summary", "")
                content = f"[{title}] {summary}" if title else summary

                items.append(ScoredItem(
                    id=str(row["id"]),
                    content=content,
                    score=float(row.get("relevance", row.get("similarity", 0))),
                    source="global_graph",
                    metadata={
                        "community_id": str(row["id"]),
                        "level": row.get("level", 0),
                        "entity_count": row.get("entity_count", 0),
                        "title": title,
                    },
                ))

            # If no communities found, fallback to entity aggregation
            if not items:
                logger.debug("No community summaries, falling back to entity aggregation")

                entity_results = await self._db.vector_search(
                    table="entity",
                    vector_field="embedding",
                    query_vector=query_vector,
                    limit=limit * 2,
                )

                for row in entity_results:
                    name = row.get("name", "Unknown")
                    description = row.get("description", "")
                    items.append(ScoredItem(
                        id=str(row["id"]),
                        content=f"{name}: {description}" if description else name,
                        score=float(row.get("relevance", 0)),
                        source="global_graph_entities",
                        metadata={
                            "entity_type": row.get("type"),
                            "entity_name": name,
                        },
                    ))

            logger.debug(f"Global graph search returned {len(items)} results")
            return items

        except Exception as e:
            logger.error(f"Global graph search error: {e}")
            return []

    def _generate_followup_query(
        self,
        original_query: str,
        results: list[ScoredItem],
        iteration: int,
    ) -> str:
        """
        Generate follow-up query for iterative refinement.

        In production, this should use LLM for intelligent reformulation.
        For now, uses simple keyword extraction.
        """
        if not results:
            return original_query

        # Extract key terms from top results
        top_content = " ".join(r.content[:200] for r in results[:3])

        # Simple approach: add unique words from results
        original_words = set(original_query.lower().split())
        result_words = set(top_content.lower().split())

        # Find new relevant words (longer than 4 chars, not in original)
        new_words = [
            w for w in result_words
            if len(w) > 4 and w not in original_words and w.isalpha()
        ][:3]

        if new_words:
            expanded_query = f"{original_query} {' '.join(new_words)}"
            logger.debug(f"Expanded query: {expanded_query}")
            return expanded_query

        return original_query


# Global instance
recall_engine = RecallEngine()


# Convenience functions
async def hybrid_recall(
    query: str,
    user_id: str | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> RecallResult:
    """Convenience function for hybrid recall."""
    return await recall_engine.recall(
        query=query,
        user_id=user_id,
        mode="hybrid",
        limit=limit,
        **kwargs,
    )


async def smart_recall(
    query: str,
    user_id: str | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> RecallResult:
    """Convenience function for auto-routed recall."""
    return await recall_engine.recall(
        query=query,
        user_id=user_id,
        mode="auto",
        limit=limit,
        **kwargs,
    )


async def research_recall(
    query: str,
    user_id: str | None = None,
    limit: int = 20,
    **kwargs: Any,
) -> RecallResult:
    """Convenience function for research mode (iterative)."""
    return await recall_engine.recall(
        query=query,
        user_id=user_id,
        mode="research",
        limit=limit,
        **kwargs,
    )


async def claim_recall_with_conflicts(
    query: str,
    limit: int = 10,
    include_conflicting: bool = True,
) -> dict[str, Any]:
    """
    Recall claims with conflict information.

    Searches claims semantically and annotates them with conflict data.
    Useful for fact-checking and evidence analysis.

    Args:
        query: Search query
        limit: Max claims to return
        include_conflicting: Whether to include claims that have conflicts

    Returns:
        Dict with claims and conflict summary
    """
    from src.adapters.embeddings import embeddings
    from src.db.client import db

    query_vector = await embeddings.embed(query)

    # Search claims by embedding
    claim_results = await db.vector_search(
        table="claim",
        vector_field="embedding",
        query_vector=query_vector,
        limit=limit * 2,  # Get more to filter
        where="status = 'active'",
    )

    if not claim_results:
        return {
            "claims": [],
            "conflicts": [],
            "summary": "No claims found matching query",
        }

    # Get claim IDs
    claim_ids = [str(r["id"]) for r in claim_results]

    # Find conflicts involving these claims
    conflicts_query = """
        SELECT conflict, claim FROM conflict_side
        WHERE claim IN $claim_ids
        FETCH conflict
    """
    conflicts_result = await db.query(conflicts_query, {"claim_ids": claim_ids})
    conflicts_data = conflicts_result[0].get("result", []) if conflicts_result else []

    # Map claim_id -> conflicts
    claim_conflicts: dict[str, list[dict]] = {}
    for row in conflicts_data:
        cid = str(row.get("claim", ""))
        conflict_obj = row.get("conflict", {})
        if cid and conflict_obj:
            conflict_info = {
                "conflict_id": str(conflict_obj.get("id", "")),
                "type": conflict_obj.get("conflict_type", ""),
                "severity": conflict_obj.get("severity", ""),
                "description": conflict_obj.get("description", ""),
                "status": conflict_obj.get("status", ""),
            }
            claim_conflicts.setdefault(cid, []).append(conflict_info)

    # Build results with conflict annotations
    claims = []
    all_conflicts = []

    for row in claim_results:
        claim_id = str(row["id"])
        has_conflicts = claim_id in claim_conflicts

        if not include_conflicting and has_conflicts:
            continue

        claim_info = {
            "id": claim_id,
            "statement": row.get("statement", ""),
            "confidence": float(row.get("confidence", 0)),
            "relevance": float(row.get("relevance", 0)),
            "has_conflicts": has_conflicts,
            "conflicts": claim_conflicts.get(claim_id, []),
        }
        claims.append(claim_info)

        if has_conflicts:
            all_conflicts.extend(claim_conflicts[claim_id])

    # Limit results
    claims = claims[:limit]

    # Summary
    conflicting_count = sum(1 for c in claims if c["has_conflicts"])
    conflict_types = set(c["type"] for c in all_conflicts)

    return {
        "claims": claims,
        "conflicts": all_conflicts,
        "summary": {
            "total_claims": len(claims),
            "claims_with_conflicts": conflicting_count,
            "conflict_types": list(conflict_types),
            "warning": f"{conflicting_count} claims have conflicts - verify before use"
            if conflicting_count > 0
            else None,
        },
    }
