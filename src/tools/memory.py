"""
Core memory tools (PUBLIC - visible to LLM).

These are the high-level tools exposed via MCP.
They orchestrate internal adapters (embeddings, vision, graphrag, etc.)
"""

import hashlib
import logging
from datetime import datetime
from typing import Any

from src.adapters.embeddings import embeddings
from src.adapters.extraction import extraction, ExtractionResult
from src.config import settings
from src.core import (
    recall_engine,
    router,
    user_memory,
    RecallResult as CoreRecallResult,
)
from src.db.client import db
from src.db.models import (
    Asset,
    AssetStatus,
    AssetType,
    Chunk,
    ContextPackResult,
    IngestResult,
    PolicyAction,
    PolicyDecision,
    Provenance,
    RecallItem,
    RecallResult,
)
from src.utils.chunker import Chunker
from src.utils.text import count_tokens, detect_language, normalize_text, truncate_to_tokens

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# memory.ingest
# ═══════════════════════════════════════════════════════════════════════════


async def memory_ingest(
    content: str,
    type: str = "document",
    source_url: str | None = None,
    lang: str | None = None,
    version_of: str | None = None,
    metadata: dict[str, Any] | None = None,
    extract_entities: bool = True,
) -> IngestResult:
    """
    Ingest content into BraineMemory.

    Args:
        content: Text content to ingest.
        type: Asset type (document, image, cad, audio, video).
        source_url: Original source URL.
        lang: Language code (ru, en, fr, multi). Auto-detected if not provided.
        version_of: Asset ID if this is a new version.
        metadata: Additional metadata.
        extract_entities: Whether to extract entities/relations/claims.

    Returns:
        IngestResult with asset_id and processing stats.
    """
    import json

    # Detect language if not provided
    if not lang:
        lang = detect_language(content)

    # Calculate hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Create asset (exclude None values to avoid NULL type errors)
    asset_data = {
        "type": type,
        "mime": "text/plain",
        "hash": content_hash,
        "size_bytes": len(content.encode()),
        "lang": lang,
        "status": "active",
        "metadata": metadata or {},
    }
    if source_url:
        asset_data["source_url"] = source_url
    if version_of:
        asset_data["version_of"] = version_of
    asset = await db.create("asset", asset_data)
    asset_id = asset["id"]
    logger.info(f"Created asset: {asset_id}")

    # Chunk content
    chunker = Chunker()
    chunk_results = chunker.split(content)
    logger.info(f"Split into {len(chunk_results)} chunks")

    # Create provenance (document-level)
    prov_data = {
        "asset": asset_id,
        "locator": {"type": "document"},
    }
    prov = await db.create("provenance", prov_data)
    prov_id = prov["id"]

    # Embed and store chunks
    chunks_created = 0
    chunk_ids = []
    for i, chunk_result in enumerate(chunk_results):
        # Embed text
        vector = await embeddings.embed(chunk_result.text)

        # Create chunk
        chunk_data = {
            "asset": asset_id,
            "prov": prov_id,
            "content": chunk_result.text,
            "lang": lang,
            "chunk_index": i,
            "vector": vector,
            "vector_model": settings.embedding_model,
            "vector_dim": settings.embedding_dim,
        }
        chunk = await db.create("chunk", chunk_data)
        chunk_ids.append(chunk["id"])
        chunks_created += 1

    # Extract entities, relations, and claims
    entities_extracted = 0
    claims_extracted = 0
    conflicts_detected = 0

    if extract_entities:
        try:
            extraction_result = await extraction.extract(content)
            entities_extracted, claims_extracted, conflicts_detected = await _store_extraction_results(
                extraction_result=extraction_result,
                asset_id=asset_id,
                prov_id=prov_id,
                chunk_ids=chunk_ids,
                content=content,
            )
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")

    # Log policy decision
    policy = PolicyDecision(
        action=PolicyAction.REMEMBER,
        target_table="asset",
        target_id=str(asset_id),
        reason=f"Ingested {type} via memory.ingest",
        decided_by="system",
    )
    policy_data = policy.model_dump(exclude={"id"}, exclude_none=True)
    policy_data["action"] = policy_data["action"].value if hasattr(policy_data["action"], "value") else policy_data["action"]
    await db.query(f"CREATE policy_decision CONTENT {json.dumps(policy_data)}")

    logger.info(
        f"Ingest complete: {asset_id}, {chunks_created} chunks, "
        f"{entities_extracted} entities, {claims_extracted} claims, "
        f"{conflicts_detected} conflicts"
    )

    return IngestResult(
        asset_id=asset_id,
        chunks_created=chunks_created,
        entities_extracted=entities_extracted,
        claims_extracted=claims_extracted,
        conflicts_detected=conflicts_detected,
        processing_status="complete",
    )


async def _store_extraction_results(
    extraction_result: ExtractionResult,
    asset_id: str,
    prov_id: str,
    chunk_ids: list[str],
    content: str,
) -> tuple[int, int, int]:
    """
    Store extracted entities, relations, and claims.

    Returns:
        Tuple of (entities_count, claims_count)
    """
    entity_id_map: dict[str, str] = {}  # name -> entity_id

    # 1. Store entities
    for entity in extraction_result.entities:
        # Generate embedding for entity
        entity_text = f"{entity.name}: {entity.description}"
        vector = await embeddings.embed(entity_text)

        # Normalize name for indexing
        normalized = entity.name.lower().strip()

        entity_data = {
            "name": entity.name,
            "normalized_name": normalized,
            "type": entity.entity_type,
            "description": entity.description,
            "embedding": vector,
            "external_ids": {},  # Required by schema
            "status": "active",
        }

        created = await db.create("entity", entity_data)
        entity_id_map[entity.name.lower()] = created["id"]
        logger.debug(f"Created entity: {entity.name} ({entity.entity_type})")

    # 2. Create mentions (entity <-> chunk links)
    # Find which chunks mention which entities
    for chunk_id in chunk_ids:
        chunk_result = await db.select(chunk_id)
        if not chunk_result:
            continue

        chunk_content = chunk_result.get("content", "").lower()

        for entity_name, entity_id in entity_id_map.items():
            if entity_name in chunk_content:
                # Create mention relation using db.relate
                await db.relate(entity_id, "mentions", chunk_id, {})

    # 3. Store relations between entities
    for relation in extraction_result.relations:
        source_id = entity_id_map.get(relation.source.lower())
        target_id = entity_id_map.get(relation.target.lower())

        if source_id and target_id:
            # Use relates_to table (defined in schema as SCHEMALESS)
            relation_data = {
                "relation_type": relation.relation_type,
                "description": relation.description,
                "weight": relation.weight,
                "source_asset": asset_id,
            }
            await db.relate(source_id, "relates_to", target_id, relation_data)
            logger.debug(f"Created relation: {relation.source} -{relation.relation_type}-> {relation.target}")

    # 4. Store claims and detect conflicts
    from src.core.conflict import detect_conflicts_for_new_claim

    conflicts_detected = 0

    for claim in extraction_result.claims:
        subject_id = entity_id_map.get(claim.subject.lower())

        # Generate embedding for claim
        claim_vector = await embeddings.embed(claim.statement)

        claim_data = {
            "statement": claim.statement,
            "confidence": claim.confidence,
            "claim_type": "fact",
            "embedding": claim_vector,
            "status": "active",
        }

        created_claim = await db.create("claim", claim_data)
        claim_id = created_claim["id"]

        # Create evidence link
        if claim.evidence_quote:
            evidence_data = {
                "claim": claim_id,
                "prov": prov_id,
                "quote": claim.evidence_quote[:500],  # Limit quote length
                "extraction_method": "llm",  # Required field
                "confidence": claim.confidence,
            }
            await db.create("evidence", evidence_data)

        # Detect conflicts with existing claims
        try:
            conflicts = await detect_conflicts_for_new_claim(
                claim_id=claim_id,
                claim_statement=claim.statement,
                claim_embedding=claim_vector,
                auto_store=True,
            )
            conflicts_detected += len(conflicts)
            if conflicts:
                logger.info(f"Detected {len(conflicts)} conflicts for claim: {claim.statement[:50]}...")
        except Exception as e:
            logger.warning(f"Conflict detection failed for claim: {e}")

    logger.info(
        f"Stored: {len(extraction_result.entities)} entities, "
        f"{len(extraction_result.relations)} relations, "
        f"{len(extraction_result.claims)} claims, "
        f"{conflicts_detected} conflicts detected"
    )

    return len(extraction_result.entities), len(extraction_result.claims), conflicts_detected


# ═══════════════════════════════════════════════════════════════════════════
# memory.recall
# ═══════════════════════════════════════════════════════════════════════════


async def memory_recall(
    query: str,
    limit: int = 10,
    mode: str = "auto",
    user_id: str | None = None,
    asset_ids: list[str] | None = None,
    lang: str | None = None,
    include_evidence: bool = True,
) -> RecallResult:
    """
    Recall relevant information from memory using advanced retrieval.

    Args:
        query: Search query.
        limit: Max number of results.
        mode: Retrieval mode:
            - "auto": Router auto-detects intent
            - "fts": Full-text search only
            - "vector": Vector similarity only
            - "hybrid": FTS + Vector with RRF
            - "local": Local graph expansion
            - "global": Community summaries
            - "research": GAM-style iterative
        user_id: User ID for personalization.
        asset_ids: Limit to specific assets.
        lang: Filter by language.
        include_evidence: Include source quotes.

    Returns:
        RecallResult with matching items.
    """
    # Use the new RecallEngine
    core_result = await recall_engine.recall(
        query=query,
        user_id=user_id,
        mode=mode,
        limit=limit,
    )

    # Convert ScoredItems to RecallItems
    items = []
    for scored_item in core_result.items:
        item = RecallItem(
            type="chunk",
            id=scored_item.id,
            content=scored_item.content,
            relevance=scored_item.score,
            provenance={
                "source": scored_item.source,
                "metadata": scored_item.metadata,
            },
        )
        items.append(item)

    # Build metadata
    metadata = {
        "query": query,
        "mode": mode,
        "pipeline": core_result.pipeline_used,
        "iterations": core_result.iterations,
        "total_found": core_result.total_found,
    }

    if core_result.user_context:
        metadata["user_context"] = core_result.user_context

    if core_result.metadata:
        metadata.update(core_result.metadata)

    return RecallResult(
        items=items,
        metadata=metadata,
    )


async def memory_recall_explain(
    query: str,
    mode: str = "auto",
) -> dict[str, Any]:
    """
    Explain how the router would handle a query.

    Args:
        query: The search query.
        mode: Retrieval mode override.

    Returns:
        Routing explanation with intent and pipeline details.
    """
    return router.explain_routing(query, mode)


# ═══════════════════════════════════════════════════════════════════════════
# memory.context_pack
# ═══════════════════════════════════════════════════════════════════════════


async def memory_context_pack(
    goal: str,
    token_budget: int = 4000,
    audience: str = "general",
    style: str = "structured",
    include_sources: bool = True,
    user_id: str | None = None,
) -> ContextPackResult:
    """
    Pack relevant context for a specific goal.

    Uses accurate token counting with tiktoken to respect the budget.

    Args:
        goal: What the context is for.
        token_budget: Maximum tokens for context (enforced with tiktoken).
        audience: Target audience (lawyer, engineer, executive, general).
        style: Output style (bullets, structured, narrative, table).
        include_sources: Include source references.
        user_id: User ID for personalization.

    Returns:
        ContextPackResult with packed context.
    """
    # Reserve budget for user context (20% max)
    user_context_budget = min(500, token_budget // 5) if user_id else 0
    recall_budget = token_budget - user_context_budget

    # Get user context if user_id provided
    user_context_str = ""
    user_context_tokens = 0
    if user_id:
        raw_context = await user_memory.get_user_context(
            user_id=user_id,
            query=goal,
            max_tokens=user_context_budget,
        )
        if raw_context:
            # Truncate to budget using tiktoken
            user_context_str = truncate_to_tokens(raw_context, user_context_budget)
            user_context_tokens = count_tokens(user_context_str)

    # Recall relevant items using auto-routing
    recall_result = await memory_recall(
        query=goal,
        limit=30,  # Fetch more to fill budget
        mode="auto",
        user_id=user_id,
    )

    # Build context with accurate token counting
    context_parts = []
    tokens_used = user_context_tokens
    sources = []

    # Add user context first if available
    if user_context_str:
        context_parts.append(f"[User Context]\n{user_context_str}")

    # Add recall items respecting token budget
    for item in recall_result.items:
        # Count tokens accurately
        item_tokens = count_tokens(item.content)

        # Check if we can fit this item
        if tokens_used + item_tokens > recall_budget:
            # Try to fit a truncated version if we have room
            remaining = recall_budget - tokens_used
            if remaining > 100:  # At least 100 tokens to be useful
                truncated = truncate_to_tokens(item.content, remaining)
                context_parts.append(truncated)
                tokens_used += count_tokens(truncated)
            break

        context_parts.append(item.content)
        tokens_used += item_tokens

        if include_sources and item.provenance:
            sources.append({
                "source": item.provenance.get("source"),
                "relevance": item.relevance,
            })

    # Format based on style
    if style == "bullets":
        context = "\n".join(f"• {part}" for part in context_parts)
    elif style == "table":
        context = "| Content |\n|---|\n" + "\n".join(
            f"| {part} |" for part in context_parts
        )
    else:  # structured or narrative
        context = "\n\n".join(context_parts)

    # Final check - ensure we don't exceed budget
    final_tokens = count_tokens(context)
    if final_tokens > token_budget:
        context = truncate_to_tokens(context, token_budget)
        final_tokens = count_tokens(context)

    return ContextPackResult(
        context=context,
        tokens_used=final_tokens,
        sources=sources,
    )


# ═══════════════════════════════════════════════════════════════════════════
# memory.link
# ═══════════════════════════════════════════════════════════════════════════


async def memory_link(
    source: str,
    target: str,
    relation: str = "relates_to",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a link between two entities.

    Args:
        source: Source record ID (entity:xxx, claim:xxx, asset:xxx).
        target: Target record ID.
        relation: Relation type.
        metadata: Additional edge data.

    Returns:
        Edge information.
    """
    edge = await db.relate(source, relation, target, metadata)

    # Log policy decision
    policy = PolicyDecision(
        action=PolicyAction.UPDATE,
        target_table="edge",
        target_id=edge.get("id", "unknown"),
        reason=f"Created {relation} link from {source} to {target}",
        decided_by="llm",
    )
    await db.create("policy_decision", policy.model_dump(exclude={"id"}))

    return {
        "edge_id": edge.get("id"),
        "created": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# memory.compare
# ═══════════════════════════════════════════════════════════════════════════


async def memory_compare(
    items: list[str],
    compare_type: str = "diff",
) -> dict[str, Any]:
    """
    Compare multiple items and find differences/conflicts.

    Args:
        items: List of record IDs to compare.
        compare_type: Type of comparison (diff, conflict, timeline, full).

    Returns:
        Comparison results with differences and conflicts.
    """
    if len(items) < 2:
        return {"error": "At least 2 items required for comparison"}

    differences = []
    conflicts = []

    # Fetch items
    fetched = []
    for item_id in items:
        data = await db.select(item_id)
        if data:
            fetched.append({"id": item_id, "data": data})

    # Simple diff: compare content fields
    if len(fetched) >= 2:
        for i, item1 in enumerate(fetched):
            for item2 in fetched[i + 1:]:
                content1 = item1["data"].get("content") or item1["data"].get("statement", "")
                content2 = item2["data"].get("content") or item2["data"].get("statement", "")

                if content1 != content2:
                    differences.append({
                        "aspect": "content",
                        "values": {
                            item1["id"]: content1[:200],
                            item2["id"]: content2[:200],
                        },
                        "significance": "medium",
                    })

    return {
        "comparison_type": compare_type,
        "differences": differences,
        "conflicts": conflicts,
        "items_compared": len(fetched),
    }


# ═══════════════════════════════════════════════════════════════════════════
# memory.explain
# ═══════════════════════════════════════════════════════════════════════════


async def memory_explain(
    target: str,
    depth: str = "shallow",
) -> dict[str, Any]:
    """
    Explain the provenance of a fact/claim.

    Args:
        target: Record ID to explain (claim:xxx, entity:xxx).
        depth: Explanation depth (shallow, deep).

    Returns:
        Evidence chain and confidence.
    """
    # Fetch target
    data = await db.select(target)
    if not data:
        return {"error": f"Target not found: {target}"}

    # Fetch evidence if it's a claim
    evidence_chain = []
    if target.startswith("claim:"):
        evidence_result = await db.query(
            "SELECT * FROM evidence WHERE claim = $claim",
            {"claim": target},
        )
        if evidence_result and evidence_result[0]["result"]:
            for ev in evidence_result[0]["result"]:
                evidence_chain.append({
                    "claim": data.get("statement", ""),
                    "source": {
                        "quote": ev.get("quote", ""),
                        "provenance": ev.get("prov"),
                    },
                    "confidence": ev.get("confidence", 1.0),
                })

    return {
        "target": target,
        "explanation": f"Found {len(evidence_chain)} evidence sources",
        "evidence_chain": evidence_chain,
        "confidence": data.get("confidence", 1.0) if data else 0.0,
        "caveats": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# memory.forget
# ═══════════════════════════════════════════════════════════════════════════


async def memory_forget(
    target: str,
    reason: str,
    scope: str = "hide",
) -> dict[str, Any]:
    """
    "Forget" a record (soft-delete with audit trail).

    Args:
        target: Record ID to forget.
        reason: Why it's being forgotten.
        scope: Scope (hide = soft-delete, anonymize = remove PII).

    Returns:
        Result with policy decision ID.

    Note:
        This NEVER physically deletes data.
        Use admin tools for permanent deletion.
    """
    # Parse target
    parts = target.split(":")
    if len(parts) != 2:
        return {"error": f"Invalid target format: {target}"}

    table, _ = parts

    # Update status to hidden
    await db.merge(target, {"status": "hidden", "updated_at": datetime.utcnow().isoformat()})

    # Log policy decision
    policy = PolicyDecision(
        action=PolicyAction.FORGET if scope == "hide" else PolicyAction.HIDE,
        target_table=table,
        target_id=target,
        reason=reason,
        decided_by="llm",
    )
    result = await db.create("policy_decision", policy.model_dump(exclude={"id"}))

    logger.info(f"Forgot {target}: {reason}")

    return {
        "success": True,
        "policy_decision_id": result["id"],
        "affected_items": 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# user_memory.remember
# ═══════════════════════════════════════════════════════════════════════════


async def user_memory_remember(
    user_id: str,
    content: str,
    category: str = "fact",
    importance: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store a user-specific memory (Mem0-style personalization).

    Args:
        user_id: External user identifier.
        content: What to remember about this user.
        category: Memory category:
            - "preference": User likes/dislikes
            - "fact": Facts about the user
            - "instruction": How user wants things done
            - "correction": User corrections to AI behavior
        importance: How important (0-1), auto-estimated if not provided.
        metadata: Additional context.

    Returns:
        Created memory record.
    """
    memory = await user_memory.remember(
        user_id=user_id,
        content=content,
        category=category,
        importance=importance,
        metadata=metadata,
    )

    return {
        "memory_id": memory["id"],
        "user_id": user_id,
        "category": category,
        "importance": memory.get("importance", 0.5),
        "created": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# user_memory.recall
# ═══════════════════════════════════════════════════════════════════════════


async def user_memory_recall(
    user_id: str,
    query: str | None = None,
    categories: list[str] | None = None,
    limit: int = 10,
    min_importance: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve user-specific memories.

    Args:
        user_id: External user identifier.
        query: Optional semantic search query.
        categories: Filter by categories (preference, fact, instruction, correction).
        limit: Max results.
        min_importance: Minimum importance threshold.

    Returns:
        List of relevant user memories.
    """
    memories = await user_memory.recall_for_user(
        user_id=user_id,
        query=query,
        categories=categories,
        limit=limit,
        min_importance=min_importance,
    )

    return [
        {
            "id": m["id"],
            "content": m["content"],
            "category": m.get("category"),
            "importance": m.get("importance"),
            "last_accessed": m.get("last_accessed"),
        }
        for m in memories
    ]


# ═══════════════════════════════════════════════════════════════════════════
# user_memory.forget
# ═══════════════════════════════════════════════════════════════════════════


async def user_memory_forget(
    user_id: str,
    memory_id: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Soft-delete user memories.

    Args:
        user_id: External user identifier.
        memory_id: Specific memory to forget.
        category: Forget all in category.

    Returns:
        Number of memories forgotten.
    """
    count = await user_memory.forget(
        user_id=user_id,
        memory_id=memory_id,
        category=category,
    )

    return {
        "user_id": user_id,
        "forgotten_count": count,
        "memory_id": memory_id,
        "category": category,
    }


# ═══════════════════════════════════════════════════════════════════════════
# memory.build_communities
# ═══════════════════════════════════════════════════════════════════════════


async def memory_build_communities(
    max_levels: int = 2,
    min_community_size: int = 2,
    resolution: float = 1.0,
    regenerate: bool = False,
) -> dict[str, Any]:
    """
    Build community index for GraphRAG global search.

    This analyzes the entity graph, detects communities using Louvain algorithm,
    generates summaries using LLM, and stores them with embeddings for search.

    Args:
        max_levels: Maximum hierarchy levels (default 2).
        min_community_size: Minimum entities per community (default 2).
        resolution: Louvain resolution parameter (higher = more communities).
        regenerate: If True, delete existing communities first.

    Returns:
        Statistics about the operation.
    """
    from src.core.community import build_community_index

    return await build_community_index(
        max_levels=max_levels,
        min_community_size=min_community_size,
        resolution=resolution,
        regenerate=regenerate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# memory.detect_conflicts
# ═══════════════════════════════════════════════════════════════════════════


async def memory_detect_conflicts(
    claim_id: str | None = None,
    scan_all: bool = False,
) -> dict[str, Any]:
    """
    Detect conflicts between claims.

    Can either check a specific claim against existing claims,
    or scan all claims for conflicts.

    Args:
        claim_id: Specific claim to check for conflicts.
        scan_all: If True, scan all claims for conflicts.

    Returns:
        Detected conflicts information.
    """
    from src.core.conflict import conflict_detector, scan_all_conflicts

    if scan_all:
        return await scan_all_conflicts(auto_store=True)

    if claim_id:
        # Get claim from DB
        claim_data = await db.select(claim_id)
        if not claim_data:
            return {"error": f"Claim not found: {claim_id}"}

        conflicts = await conflict_detector.detect_conflicts_for_claim(
            claim_id=claim_id,
            claim_statement=claim_data.get("statement", ""),
            claim_embedding=claim_data.get("embedding"),
        )

        # Store detected conflicts
        stored_ids = []
        for conflict in conflicts:
            conflict_db_id = await conflict_detector.store_conflict(conflict)
            if conflict_db_id:
                stored_ids.append(conflict_db_id)

        return {
            "claim_id": claim_id,
            "conflicts_detected": len(conflicts),
            "conflicts_stored": len(stored_ids),
            "conflicts": [
                {
                    "with_claim": c.claim2_id,
                    "type": c.conflict_type.value,
                    "severity": c.severity.value,
                    "description": c.description,
                }
                for c in conflicts
            ],
        }

    return {"error": "Must provide claim_id or set scan_all=True"}


async def memory_resolve_conflict(
    conflict_id: str,
    resolution: str,
    winning_claim_id: str | None = None,
) -> dict[str, Any]:
    """
    Resolve a conflict between claims.

    Args:
        conflict_id: ID of the conflict to resolve.
        resolution: Description of how the conflict was resolved.
        winning_claim_id: If one claim is correct, provide its ID to supersede the other.

    Returns:
        Resolution result.
    """
    from src.core.conflict import conflict_detector

    success = await conflict_detector.resolve_conflict(
        conflict_id=conflict_id,
        resolution=resolution,
        resolved_by="llm",
        winning_claim_id=winning_claim_id,
    )

    return {
        "conflict_id": conflict_id,
        "resolved": success,
        "resolution": resolution,
        "winning_claim_id": winning_claim_id,
    }


async def memory_list_conflicts(
    status: str = "open",
    severity: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    List conflicts in the database.

    Args:
        status: Filter by status (open, resolved).
        severity: Filter by severity (critical, high, medium, low).
        limit: Max results.

    Returns:
        List of conflicts with details.
    """
    where_parts = [f"status = '{status}'"]
    if severity:
        where_parts.append(f"severity = '{severity}'")

    where_clause = " AND ".join(where_parts)

    # Note: ORDER BY with CASE fails in some SurrealDB versions
    result = await db.query(
        f"SELECT id, conflict_type, description, severity, status, resolution, created_at "
        f"FROM conflict WHERE {where_clause} LIMIT {limit}"
    )

    conflicts = result[0].get("result", []) if result else []

    # Get claims for each conflict
    enriched = []
    for conflict in conflicts:
        conflict_id = conflict["id"]

        # Get involved claims
        sides = await db.query(f"""
            SELECT claim.statement AS statement, role
            FROM conflict_side
            WHERE conflict = {conflict_id}
        """)

        claims = []
        for side in sides[0].get("result", []) if sides else []:
            claims.append({
                "role": side.get("role"),
                "statement": side.get("statement", "")[:100],
            })

        enriched.append({
            **conflict,
            "claims": claims,
        })

    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# memory.recall_claims (conflict-aware claim recall)
# ═══════════════════════════════════════════════════════════════════════════


async def memory_recall_claims(
    query: str,
    limit: int = 10,
    include_conflicting: bool = True,
) -> dict[str, Any]:
    """
    Recall claims with conflict annotations.

    Searches claims semantically and shows which ones have conflicts.
    Useful for fact-checking and evidence analysis.

    Args:
        query: Search query to find relevant claims.
        limit: Maximum number of claims to return.
        include_conflicting: Whether to include claims that have conflicts (default True).

    Returns:
        Dict with claims, conflicts, and summary with warnings.
    """
    from src.core.recall import claim_recall_with_conflicts

    result = await claim_recall_with_conflicts(
        query=query,
        limit=limit,
        include_conflicting=include_conflicting,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# memory.recall_graph (entity graph traversal)
# ═══════════════════════════════════════════════════════════════════════════


async def memory_recall_graph(
    query: str,
    mode: str = "local",
    limit: int = 10,
    include_entities: bool = True,
    include_relations: bool = True,
    include_communities: bool = True,
) -> dict[str, Any]:
    """
    Recall through entity graph (GraphRAG-style).

    Modes:
    - local: Entity expansion (find entities → expand via relations → get chunks)
    - global: Community summaries (corpus-wide themes and patterns)
    - both: Combine local and global results

    Args:
        query: Search query.
        mode: "local" (entity expansion), "global" (community summaries), or "both".
        limit: Maximum results per category.
        include_entities: Include matched entities in response.
        include_relations: Include relations between entities.
        include_communities: Include community summaries (global mode).

    Returns:
        Dict with entities, relations, communities, and relevant chunks.
    """
    from src.adapters.embeddings import embeddings as emb

    query_vector = await emb.embed(query)
    result: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "entities": [],
        "relations": [],
        "communities": [],
        "chunks": [],
    }

    # Local graph search - entity expansion
    if mode in ("local", "both"):
        # Find matching entities
        entity_results = await db.vector_search(
            table="entity",
            vector_field="embedding",
            query_vector=query_vector,
            limit=limit,
        )

        entity_ids = []
        if include_entities:
            for e in entity_results:
                entity_ids.append(str(e["id"]))
                result["entities"].append({
                    "id": str(e["id"]),
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                    "description": e.get("description", ""),
                    "relevance": float(e.get("relevance", 0)),
                })

        # Get relations for these entities
        if include_relations and entity_ids:
            relations_query = """
                SELECT
                    in.name AS source,
                    in.type AS source_type,
                    relation_type,
                    out.name AS target,
                    out.type AS target_type,
                    weight
                FROM relates_to
                WHERE in IN $entity_ids OR out IN $entity_ids
                LIMIT $limit
            """
            rel_result = await db.query(relations_query, {
                "entity_ids": entity_ids,
                "limit": limit * 2,
            })
            for r in rel_result[0].get("result", []) if rel_result else []:
                result["relations"].append({
                    "source": r.get("source", ""),
                    "source_type": r.get("source_type", ""),
                    "relation": r.get("relation_type", ""),
                    "target": r.get("target", ""),
                    "target_type": r.get("target_type", ""),
                    "weight": float(r.get("weight", 0)),
                })

        # Get chunks via mentions
        if entity_ids:
            chunks_query = """
                SELECT out AS chunk_id FROM mentions
                WHERE in IN $entity_ids
                LIMIT $limit
            """
            chunks_result = await db.query(chunks_query, {
                "entity_ids": entity_ids,
                "limit": limit * 2,
            })
            chunk_ids = set()
            for row in chunks_result[0].get("result", []) if chunks_result else []:
                cid = str(row.get("chunk_id", ""))
                if cid:
                    chunk_ids.add(cid)

            # Fetch chunk content
            for chunk_id in list(chunk_ids)[:limit]:
                chunk_data = await db.select(chunk_id)
                if chunk_data:
                    result["chunks"].append({
                        "id": chunk_id,
                        "content": chunk_data.get("content", "")[:500],
                        "source": "local_graph",
                    })

    # Global graph search - community summaries
    if mode in ("global", "both") and include_communities:
        community_results = await db.vector_search(
            table="community",
            vector_field="embedding",
            query_vector=query_vector,
            limit=limit,
        )

        for c in community_results:
            result["communities"].append({
                "id": str(c["id"]),
                "title": c.get("title", ""),
                "summary": c.get("summary", ""),
                "level": c.get("level", 0),
                "entity_count": c.get("entity_count", 0),
                "relevance": float(c.get("relevance", 0)),
            })

    # Summary stats
    result["summary"] = {
        "entities_found": len(result["entities"]),
        "relations_found": len(result["relations"]),
        "communities_found": len(result["communities"]),
        "chunks_found": len(result["chunks"]),
    }

    return result
