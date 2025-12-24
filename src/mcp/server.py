"""
BraineMemory MCP Server v3.1 - FastMCP Edition.

This is the main entry point for the MCP server using FastMCP.

Features:
- FastMCP for cleaner tool registration
- Proper lifecycle management via ServiceContainer
- 17 memory tools (ingest, recall, graph, conflicts, user memory)
- 3 resources (recent changes, conflicts, health)
- Multi-transport: stdio (default), SSE, HTTP

Usage:
    # Start with stdio transport (for Claude Desktop)
    python -m src.mcp.server

    # Start with SSE transport (for web clients)
    python -m src.mcp.server --transport sse --port 8001

    # Start with HTTP streamable transport
    python -m src.mcp.server --transport http --port 8001
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import settings
from src.container import container
from src.mcp.lifecycle import startup_context, shutdown_context

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Server Configuration
# =============================================================================

SERVER_NAME = settings.mcp_server_name
SERVER_VERSION = settings.mcp_server_version
SERVER_DESCRIPTION = """
BraineMemory - Intelligent Memory System for LLMs

Provides 17 tools for:
- Content ingestion with entity extraction
- Hybrid search (FTS, vector, GraphRAG)
- Conflict detection between claims
- User memory (Mem0-style personalization)
- Community detection for global queries
- Context packing with token budget

Supports multiple retrieval modes:
- auto: Router decides best strategy
- fts: BM25 full-text search
- vector: HNSW semantic search
- hybrid: FTS + Vector with RRF
- local: GraphRAG entity expansion
- global: Community summaries
- research: GAM-style iterative
"""


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(mcp: FastMCP):
    """
    Manage application lifecycle per-session.

    For SSE/HTTP transport: Context is initialized at app startup via Starlette events.
    For stdio transport: Context is initialized here (single session).
    """
    from src.mcp.lifecycle import is_initialized

    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}...")

    # Check if context already initialized (SSE/HTTP mode)
    if is_initialized():
        logger.debug("Session using existing container")
        logger.info(f"{SERVER_NAME} ready. Connected to SurrealDB.")
        yield {"container": container}
        return

    # Stdio mode: Initialize and cleanup here (single session)
    await startup_context()
    try:
        logger.info(f"{SERVER_NAME} ready. Connected to SurrealDB.")
        yield {"container": container}
    finally:
        await shutdown_context()

    logger.info(f"{SERVER_NAME} shutdown complete.")


# =============================================================================
# Create MCP Server
# =============================================================================

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=SERVER_DESCRIPTION,
    lifespan=lifespan,
)


# =============================================================================
# Core Memory Tools
# =============================================================================

@mcp.tool()
async def memory_ingest(
    content: str,
    type: str = "document",
    source_url: str | None = None,
    lang: str | None = None,
    version_of: str | None = None,
    metadata: dict[str, Any] | None = None,
    extract_entities: bool = True,
) -> str:
    """
    Ingest content into BraineMemory. Supports text, documents, and will extract entities/claims.

    Args:
        content: Text content to ingest
        type: Asset type (document, image, cad, audio, video)
        source_url: Original source URL (optional)
        lang: Language (ru, en, fr, multi) - auto-detected if not provided
        version_of: Asset ID if this is a new version
        metadata: Additional metadata
        extract_entities: Extract entities, relations, and claims using LLM

    Returns:
        JSON with asset_id and processing stats
    """
    from src.tools.memory import memory_ingest as _ingest

    result = await _ingest(
        content=content,
        type=type,
        source_url=source_url,
        lang=lang,
        version_of=version_of,
        metadata=metadata,
        extract_entities=extract_entities,
    )
    return _to_json(result)


@mcp.tool()
async def memory_recall(
    query: str,
    limit: int = 10,
    mode: str = "auto",
    user_id: str | None = None,
    asset_ids: list[str] | None = None,
    lang: str | None = None,
    include_evidence: bool = True,
) -> str:
    """
    Recall relevant information from memory using advanced hybrid search with auto-routing.

    Args:
        query: Search query
        limit: Max results
        mode: Retrieval mode - auto (router decides), fts (BM25), vector (HNSW), hybrid (FTS+Vector with RRF), local (graph expansion), global (community summaries), research (GAM-style iterative)
        user_id: User ID for personalization (Mem0-style)
        asset_ids: Limit to specific assets
        lang: Filter by language
        include_evidence: Include source quotes

    Returns:
        JSON with search results
    """
    from src.tools.memory import memory_recall as _recall

    result = await _recall(
        query=query,
        limit=limit,
        mode=mode,
        user_id=user_id,
        asset_ids=asset_ids,
        lang=lang,
        include_evidence=include_evidence,
    )
    return _to_json(result)


@mcp.tool()
async def memory_recall_explain(query: str, mode: str = "auto") -> str:
    """
    Explain how the router would handle a query (for debugging/transparency).

    Args:
        query: The search query to analyze
        mode: Retrieval mode override

    Returns:
        JSON with routing explanation
    """
    from src.tools.memory import memory_recall_explain as _explain

    result = await _explain(query=query, mode=mode)
    return _to_json(result)


@mcp.tool()
async def memory_context_pack(
    goal: str,
    token_budget: int = 4000,
    audience: str = "general",
    style: str = "structured",
    include_sources: bool = True,
    user_id: str | None = None,
) -> str:
    """
    Pack relevant context for a specific goal within a token budget.

    Args:
        goal: What the context is for
        token_budget: Max tokens for context
        audience: Target audience (lawyer, engineer, executive, general)
        style: Output format (bullets, structured, narrative, table)
        include_sources: Include source references
        user_id: User ID for personalization

    Returns:
        JSON with packed context
    """
    from src.tools.memory import memory_context_pack as _pack

    result = await _pack(
        goal=goal,
        token_budget=token_budget,
        audience=audience,
        style=style,
        include_sources=include_sources,
        user_id=user_id,
    )
    return _to_json(result)


@mcp.tool()
async def memory_link(
    source: str,
    target: str,
    relation: str = "relates_to",
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Create a link/relation between two records.

    Args:
        source: Source record ID (entity:xxx, claim:xxx, asset:xxx)
        target: Target record ID
        relation: Relation type
        metadata: Additional edge data

    Returns:
        JSON with link result
    """
    from src.tools.memory import memory_link as _link

    result = await _link(
        source=source,
        target=target,
        relation=relation,
        metadata=metadata,
    )
    return _to_json(result)


@mcp.tool()
async def memory_compare(
    items: list[str],
    compare_type: str = "diff",
) -> str:
    """
    Compare multiple items and find differences or conflicts.

    Args:
        items: Record IDs to compare (2+)
        compare_type: Comparison type (diff, conflict, timeline, full)

    Returns:
        JSON with comparison result
    """
    from src.tools.memory import memory_compare as _compare

    result = await _compare(items=items, compare_type=compare_type)
    return _to_json(result)


@mcp.tool()
async def memory_explain(target: str, depth: str = "shallow") -> str:
    """
    Explain the provenance and evidence chain for a fact/claim.

    Args:
        target: Record ID to explain (claim:xxx, entity:xxx)
        depth: Explanation depth (shallow, deep)

    Returns:
        JSON with explanation
    """
    from src.tools.memory import memory_explain as _explain

    result = await _explain(target=target, depth=depth)
    return _to_json(result)


@mcp.tool()
async def memory_forget(target: str, reason: str, scope: str = "hide") -> str:
    """
    Soft-delete a record with audit trail. Does NOT permanently delete.

    Args:
        target: Record ID to forget
        reason: Why it's being forgotten
        scope: Scope of forgetting (hide, anonymize)

    Returns:
        JSON with forget result
    """
    from src.tools.memory import memory_forget as _forget

    result = await _forget(target=target, reason=reason, scope=scope)
    return _to_json(result)


# =============================================================================
# User Memory Tools (Mem0-style personalization)
# =============================================================================

@mcp.tool()
async def user_memory_remember(
    user_id: str,
    content: str,
    category: str = "fact",
    importance: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Store user-specific memory for personalization (Mem0-style).
    Remember facts, preferences, instructions, or corrections about a user.

    Args:
        user_id: External user identifier
        content: What to remember about this user
        category: Memory category - preference (likes/dislikes), fact (about user), instruction (how user wants things), correction (user corrections)
        importance: Importance score 0-1 (auto-estimated if not provided)
        metadata: Additional context

    Returns:
        JSON with memory result
    """
    from src.tools.memory import user_memory_remember as _remember

    result = await _remember(
        user_id=user_id,
        content=content,
        category=category,
        importance=importance,
        metadata=metadata,
    )
    return _to_json(result)


@mcp.tool()
async def user_memory_recall(
    user_id: str,
    query: str | None = None,
    categories: list[str] | None = None,
    limit: int = 10,
    min_importance: float = 0,
) -> str:
    """
    Retrieve user-specific memories for personalization.

    Args:
        user_id: External user identifier
        query: Optional semantic search query
        categories: Filter by categories (preference, fact, instruction, correction)
        limit: Max results
        min_importance: Minimum importance threshold

    Returns:
        JSON with user memories
    """
    from src.tools.memory import user_memory_recall as _recall

    result = await _recall(
        user_id=user_id,
        query=query,
        categories=categories,
        limit=limit,
        min_importance=min_importance,
    )
    return _to_json(result)


@mcp.tool()
async def user_memory_forget(
    user_id: str,
    memory_id: str | None = None,
    category: str | None = None,
) -> str:
    """
    Soft-delete user memories.

    Args:
        user_id: External user identifier
        memory_id: Specific memory to forget
        category: Forget all in category (preference, fact, instruction, correction)

    Returns:
        JSON with forget result
    """
    from src.tools.memory import user_memory_forget as _forget

    result = await _forget(
        user_id=user_id,
        memory_id=memory_id,
        category=category,
    )
    return _to_json(result)


# =============================================================================
# GraphRAG Tools
# =============================================================================

@mcp.tool()
async def memory_build_communities(
    max_levels: int = 2,
    min_community_size: int = 2,
    resolution: float = 1.0,
    regenerate: bool = False,
) -> str:
    """
    Build community index for GraphRAG global search.
    Detects entity communities using Louvain algorithm, generates summaries via LLM, and stores with embeddings.

    Args:
        max_levels: Maximum hierarchy levels
        min_community_size: Minimum entities per community
        resolution: Louvain resolution (higher = more communities)
        regenerate: Delete existing communities first

    Returns:
        JSON with community build result
    """
    from src.tools.memory import memory_build_communities as _build

    result = await _build(
        max_levels=max_levels,
        min_community_size=min_community_size,
        resolution=resolution,
        regenerate=regenerate,
    )
    return _to_json(result)


@mcp.tool()
async def memory_recall_graph(
    query: str,
    mode: str = "local",
    limit: int = 10,
    include_entities: bool = True,
    include_relations: bool = True,
    include_communities: bool = True,
) -> str:
    """
    Recall through entity graph (GraphRAG).
    Local mode expands from entities via relations.
    Global mode uses community summaries for corpus-wide queries.

    Args:
        query: Search query
        mode: local=entity expansion, global=community summaries, both=combined
        limit: Max results per category
        include_entities: Include matched entities
        include_relations: Include relations between entities
        include_communities: Include community summaries (global mode)

    Returns:
        JSON with graph search results
    """
    from src.tools.memory import memory_recall_graph as _recall_graph

    result = await _recall_graph(
        query=query,
        mode=mode,
        limit=limit,
        include_entities=include_entities,
        include_relations=include_relations,
        include_communities=include_communities,
    )
    return _to_json(result)


# =============================================================================
# Conflict Detection Tools
# =============================================================================

@mcp.tool()
async def memory_detect_conflicts(
    claim_id: str | None = None,
    scan_all: bool = False,
) -> str:
    """
    Detect contradictions between claims.
    Can check a specific claim or scan all claims.

    Args:
        claim_id: Specific claim ID to check for conflicts
        scan_all: Scan all claims for conflicts

    Returns:
        JSON with detected conflicts
    """
    from src.tools.memory import memory_detect_conflicts as _detect

    result = await _detect(claim_id=claim_id, scan_all=scan_all)
    return _to_json(result)


@mcp.tool()
async def memory_resolve_conflict(
    conflict_id: str,
    resolution: str,
    winning_claim_id: str | None = None,
) -> str:
    """
    Resolve a detected conflict between claims.

    Args:
        conflict_id: ID of the conflict to resolve
        resolution: Description of how the conflict was resolved
        winning_claim_id: If one claim is correct, its ID (the other will be superseded)

    Returns:
        JSON with resolution result
    """
    from src.tools.memory import memory_resolve_conflict as _resolve

    result = await _resolve(
        conflict_id=conflict_id,
        resolution=resolution,
        winning_claim_id=winning_claim_id,
    )
    return _to_json(result)


@mcp.tool()
async def memory_list_conflicts(
    status: str = "open",
    severity: str | None = None,
    limit: int = 20,
) -> str:
    """
    List conflicts in the database with their details.

    Args:
        status: Filter by status (open, resolved)
        severity: Filter by severity (critical, high, medium, low)
        limit: Max results

    Returns:
        JSON with conflicts list
    """
    from src.tools.memory import memory_list_conflicts as _list

    result = await _list(status=status, severity=severity, limit=limit)
    return _to_json(result)


@mcp.tool()
async def memory_recall_claims(
    query: str,
    limit: int = 10,
    include_conflicting: bool = True,
) -> str:
    """
    Recall claims with conflict annotations.
    Searches claims semantically and shows which ones have conflicts.
    Useful for fact-checking and evidence verification.

    Args:
        query: Search query to find relevant claims
        limit: Maximum number of claims to return
        include_conflicting: Whether to include claims that have conflicts

    Returns:
        JSON with claims and conflict annotations
    """
    from src.tools.memory import memory_recall_claims as _recall_claims

    result = await _recall_claims(
        query=query,
        limit=limit,
        include_conflicting=include_conflicting,
    )
    return _to_json(result)


# =============================================================================
# Resources
# =============================================================================

@mcp.resource("memory://recent")
async def resource_recent() -> str:
    """Recent changes in memory."""
    result = await container.db.query("""
        SELECT * FROM policy_decision
        ORDER BY decided_at DESC
        LIMIT 20
    """)
    items = result[0]["result"] if result and result[0]["result"] else []
    return json.dumps({"items": items, "last_updated": "now"}, default=str)


@mcp.resource("memory://conflicts")
async def resource_conflicts() -> str:
    """Current unresolved conflicts."""
    result = await container.db.query("""
        SELECT * FROM conflict
        WHERE status = 'open'
        ORDER BY created_at DESC
    """)
    conflicts = result[0]["result"] if result and result[0]["result"] else []
    return json.dumps({
        "conflicts": conflicts,
        "total": len(conflicts),
        "critical_count": sum(1 for c in conflicts if c.get("severity") == "critical"),
    }, default=str)


@mcp.resource("memory://health")
async def resource_health() -> str:
    """System health and statistics."""
    health = await container.health_check()
    return json.dumps(health, default=str)


# =============================================================================
# Helper Functions
# =============================================================================

def _to_json(result: Any) -> str:
    """Convert result to JSON string."""
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="BraineMemory MCP Server (FastMCP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with stdio transport (for Claude Desktop)
    python -m src.mcp.server

    # Run with SSE transport on port 8001
    python -m src.mcp.server --transport sse --port 8001

    # Run with HTTP streamable transport
    python -m src.mcp.server --transport http --port 8001
        """,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=os.environ.get("MCP_TRANSPORT", settings.mcp_transport),
        help="Transport type (default: stdio, can be set via MCP_TRANSPORT env)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_PORT", 8001)),
        help="Port for SSE/HTTP transport (default: 8001)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_HOST", "localhost"),
        help="Host for SSE/HTTP transport (default: localhost)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"BraineMemory MCP Server v{SERVER_VERSION}")
    logger.info(f"Embedding model: {settings.embedding_model} (dim={settings.embedding_dim})")
    logger.info(f"Vision model: {settings.vision_model}")
    logger.info(f"Transport: {args.transport}")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        import uvicorn
        # Get Starlette app with app-level lifecycle
        app = mcp.sse_app()
        app.add_event_handler("startup", startup_context)
        app.add_event_handler("shutdown", shutdown_context)
        uvicorn.run(app, host=args.host, port=args.port)
    else:  # http
        import uvicorn
        # Get Starlette app with app-level lifecycle
        app = mcp.streamable_http_app()
        app.add_event_handler("startup", startup_context)
        app.add_event_handler("shutdown", shutdown_context)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
