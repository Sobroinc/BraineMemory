"""
Community Detection for GraphRAG Global Search.

Implements:
- Graph building from entities and relations
- Community detection using Louvain algorithm
- Hierarchical community structure
- LLM-based community summarization
"""

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from src.adapters.embeddings import embeddings
from src.config import settings
from src.db.client import db

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Detected community."""

    id: str
    level: int
    entity_ids: list[str]
    entity_names: list[str]
    title: str
    summary: str
    parent_id: str | None = None


class CommunityDetector:
    """
    Detects communities in the entity graph using Louvain algorithm.

    Implements hierarchical community detection for GraphRAG:
    - Level 0: Fine-grained communities
    - Level 1+: Coarser communities (merged from lower levels)
    """

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._http_client

    async def build_communities(
        self,
        max_levels: int = 2,
        min_community_size: int = 2,
        resolution: float = 1.0,
    ) -> list[Community]:
        """
        Build hierarchical communities from the entity graph.

        Args:
            max_levels: Maximum hierarchy levels
            min_community_size: Minimum entities per community
            resolution: Louvain resolution parameter (higher = more communities)

        Returns:
            List of detected communities
        """
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities
        except ImportError:
            logger.error("networkx not installed. Run: pip install networkx")
            return []

        # 1. Load entities and relations from DB
        logger.info("Loading entity graph from database...")

        entities_result = await db.query("""
            SELECT id, name, type, description FROM entity
            WHERE status = 'active'
        """)
        entities = entities_result[0].get("result", []) if entities_result else []

        if not entities:
            logger.warning("No entities found in database")
            return []

        relations_result = await db.query("""
            SELECT in, out, relation_type, weight FROM relates_to
        """)
        relations = relations_result[0].get("result", []) if relations_result else []

        logger.info(f"Loaded {len(entities)} entities, {len(relations)} relations")

        # 2. Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        entity_map = {}  # id -> entity data
        for entity in entities:
            entity_id = str(entity["id"])
            entity_map[entity_id] = entity
            G.add_node(
                entity_id,
                name=entity.get("name", ""),
                type=entity.get("type", ""),
                description=entity.get("description", ""),
            )

        # Add edges
        for rel in relations:
            source = str(rel.get("in", ""))
            target = str(rel.get("out", ""))
            weight = float(rel.get("weight", 1.0))

            if source in entity_map and target in entity_map:
                G.add_edge(source, target, weight=weight, relation=rel.get("relation_type", ""))

        if G.number_of_edges() == 0:
            logger.warning("No edges in graph, creating implicit edges from co-mentions")
            # Create edges from co-mentions in same chunk
            await self._add_comention_edges(G, entity_map)

        logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # 3. Detect communities using Louvain
        all_communities: list[Community] = []
        current_graph = G

        for level in range(max_levels):
            if current_graph.number_of_nodes() < min_community_size:
                break

            logger.info(f"Detecting communities at level {level}...")

            try:
                # Louvain community detection
                communities_sets = louvain_communities(
                    current_graph,
                    weight="weight",
                    resolution=resolution,
                    seed=42,
                )

                # Filter by minimum size
                communities_sets = [c for c in communities_sets if len(c) >= min_community_size]

                if not communities_sets:
                    logger.info(f"No communities found at level {level}")
                    break

                logger.info(f"Found {len(communities_sets)} communities at level {level}")

                # Create Community objects
                level_communities = []
                for i, node_set in enumerate(communities_sets):
                    entity_ids = list(node_set)
                    entity_names = [
                        entity_map.get(eid, {}).get("name", eid)
                        for eid in entity_ids
                        if eid in entity_map
                    ]

                    community = Community(
                        id=f"community_L{level}_{i}",
                        level=level,
                        entity_ids=entity_ids,
                        entity_names=entity_names,
                        title="",  # Will be generated
                        summary="",  # Will be generated
                    )
                    level_communities.append(community)

                all_communities.extend(level_communities)

                # Build super-graph for next level
                if level < max_levels - 1:
                    current_graph = self._build_super_graph(current_graph, communities_sets)

            except Exception as e:
                logger.error(f"Community detection failed at level {level}: {e}")
                break

        logger.info(f"Total communities detected: {len(all_communities)}")
        return all_communities

    async def _add_comention_edges(self, G: "nx.Graph", entity_map: dict) -> None:
        """Add edges between entities mentioned in the same chunk."""
        # Get co-mentions
        comention_result = await db.query("""
            SELECT in AS entity, out AS chunk FROM mentions
        """)
        comentions = comention_result[0].get("result", []) if comention_result else []

        # Group entities by chunk
        chunk_entities: dict[str, list[str]] = {}
        for cm in comentions:
            chunk_id = str(cm.get("chunk", ""))
            entity_id = str(cm.get("entity", ""))
            if chunk_id and entity_id in entity_map:
                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                chunk_entities[chunk_id].append(entity_id)

        # Create edges between co-mentioned entities
        for chunk_id, entities in chunk_entities.items():
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if G.has_edge(e1, e2):
                        G[e1][e2]["weight"] += 0.5
                    else:
                        G.add_edge(e1, e2, weight=0.5, relation="co_mention")

    def _build_super_graph(
        self,
        G: "nx.Graph",
        communities: list[set],
    ) -> "nx.Graph":
        """Build a super-graph where each community is a node."""
        import networkx as nx

        super_G = nx.Graph()

        # Map nodes to their community
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
            super_G.add_node(i, size=len(community))

        # Add edges between communities
        for u, v, data in G.edges(data=True):
            cu = node_to_community.get(u)
            cv = node_to_community.get(v)
            if cu is not None and cv is not None and cu != cv:
                if super_G.has_edge(cu, cv):
                    super_G[cu][cv]["weight"] += data.get("weight", 1.0)
                else:
                    super_G.add_edge(cu, cv, weight=data.get("weight", 1.0))

        return super_G

    async def generate_summaries(
        self,
        communities: list[Community],
    ) -> list[Community]:
        """
        Generate title and summary for each community using LLM.

        Args:
            communities: List of communities to summarize

        Returns:
            Communities with generated titles and summaries
        """
        client = await self._ensure_client()

        for community in communities:
            if not community.entity_names:
                continue

            # Build context from entity names and relations
            entities_text = ", ".join(community.entity_names[:20])

            # Get relations within community
            relations_text = await self._get_community_relations(community.entity_ids)

            prompt = f"""Analyze this community of related entities and provide:
1. A short title (3-5 words) that captures the main theme
2. A summary (2-3 sentences) describing what this group represents and their relationships

Entities in community:
{entities_text}

Key relationships:
{relations_text if relations_text else "No explicit relations found"}

Respond in JSON format:
{{"title": "...", "summary": "..."}}"""

            try:
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": settings.llm_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert at analyzing knowledge graphs and creating concise summaries. Always respond with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 300,
                    },
                )
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]

                # Parse JSON response
                import json
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()

                result = json.loads(content)
                community.title = result.get("title", f"Community {community.id}")
                community.summary = result.get("summary", "")

                logger.debug(f"Generated summary for {community.id}: {community.title}")

            except Exception as e:
                logger.error(f"Failed to generate summary for {community.id}: {e}")
                community.title = f"Community of {len(community.entity_names)} entities"
                community.summary = f"Contains: {', '.join(community.entity_names[:5])}..."

        return communities

    async def _get_community_relations(self, entity_ids: list[str]) -> str:
        """Get relations between entities in a community."""
        if not entity_ids:
            return ""

        result = await db.query("""
            SELECT in.name AS from_name, out.name AS to_name, relation_type
            FROM relates_to
            WHERE in IN $entity_ids AND out IN $entity_ids
            LIMIT 10
        """, {"entity_ids": entity_ids})

        relations = result[0].get("result", []) if result else []

        lines = []
        for rel in relations:
            lines.append(
                f"- {rel.get('from_name')} --[{rel.get('relation_type')}]--> {rel.get('to_name')}"
            )

        return "\n".join(lines)

    async def store_communities(self, communities: list[Community]) -> int:
        """
        Store communities in the database with embeddings.

        Args:
            communities: Communities to store

        Returns:
            Number of communities stored
        """
        stored = 0

        for community in communities:
            if not community.summary:
                continue

            try:
                # Generate embedding for summary
                embedding_text = f"{community.title}: {community.summary}"
                vector = await embeddings.embed(embedding_text)

                # Store community
                community_data = {
                    "level": community.level,
                    "title": community.title,
                    "summary": community.summary,
                    "entity_count": len(community.entity_ids),
                    "embedding": vector,
                    "status": "active",
                }

                if community.parent_id:
                    community_data["parent"] = community.parent_id

                created = await db.create("community", community_data)
                community_db_id = created["id"]

                # Create membership edges
                for entity_id in community.entity_ids:
                    await db.relate(entity_id, "belongs_to_community", community_db_id, {})

                stored += 1
                logger.debug(f"Stored community: {community.title}")

            except Exception as e:
                logger.error(f"Failed to store community {community.id}: {e}")

        logger.info(f"Stored {stored} communities in database")
        return stored


# Global instance
community_detector = CommunityDetector()


async def build_community_index(
    max_levels: int = 2,
    min_community_size: int = 2,
    resolution: float = 1.0,
    regenerate: bool = False,
) -> dict[str, Any]:
    """
    Build or rebuild the community index.

    Args:
        max_levels: Maximum hierarchy levels
        min_community_size: Minimum entities per community
        resolution: Louvain resolution (higher = more communities)
        regenerate: If True, delete existing communities first

    Returns:
        Statistics about the operation
    """
    if regenerate:
        logger.info("Deleting existing communities...")
        await db.query("DELETE community")
        await db.query("DELETE belongs_to_community")

    # Detect communities
    communities = await community_detector.build_communities(
        max_levels=max_levels,
        min_community_size=min_community_size,
        resolution=resolution,
    )

    if not communities:
        return {
            "success": False,
            "message": "No communities detected",
            "communities_found": 0,
        }

    # Generate summaries
    communities = await community_detector.generate_summaries(communities)

    # Store in database
    stored = await community_detector.store_communities(communities)

    return {
        "success": True,
        "communities_detected": len(communities),
        "communities_stored": stored,
        "levels": max(c.level for c in communities) + 1 if communities else 0,
    }
