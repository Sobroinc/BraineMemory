"""
User Memory Manager - Mem0-style personalized memory.

Handles:
- User preferences and facts
- Session context
- Memory categories (preference, fact, instruction, correction)
- Importance scoring and decay

Supports dependency injection for testability.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.adapters.embeddings import EmbeddingsAdapter
    from src.db.client import SurrealClient

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    """Categories of user memory."""

    PREFERENCE = "preference"    # User likes/dislikes
    FACT = "fact"               # Facts about the user
    INSTRUCTION = "instruction" # How user wants things done
    CORRECTION = "correction"   # User corrections to AI behavior


class UserMemoryManager:
    """
    Manages personalized user memories.

    Inspired by Mem0's approach to:
    - Extract and store user-specific information
    - Retrieve relevant context for personalization
    - Handle memory lifecycle (importance, decay)

    Supports dependency injection for all dependencies.
    """

    DEFAULT_IMPORTANCE = 0.5
    DEFAULT_DECAY_RATE = 0.1

    def __init__(
        self,
        db: "SurrealClient | None" = None,
        embeddings: "EmbeddingsAdapter | None" = None,
    ):
        """
        Initialize user memory manager.

        Args:
            db: Database client (uses global if not provided)
            embeddings: Embeddings adapter (uses global if not provided)
        """
        # Lazy import globals for backward compatibility
        if db is None:
            from src.db.client import db as global_db
            db = global_db
        if embeddings is None:
            from src.adapters.embeddings import embeddings as global_embeddings
            embeddings = global_embeddings

        self._db = db
        self._embeddings = embeddings

    async def ensure_user(self, user_id: str, name: str | None = None) -> dict[str, Any]:
        """Ensure user exists, create if needed."""
        # Check if user exists
        result = await self._db.query(
            "SELECT * FROM user WHERE external_id = $user_id",
            {"user_id": user_id},
        )

        if result and result[0].get("result"):
            return result[0]["result"][0]

        # Create new user
        user_data = {
            "external_id": user_id,
            "name": name,
            "preferences": {},
            "status": "active",
        }
        return await self._db.create("user", user_data)

    async def remember(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a user-specific memory.

        Args:
            user_id: External user identifier
            content: Memory content
            category: preference|fact|instruction|correction
            importance: How important (0-1), affects retrieval priority
            metadata: Additional context

        Returns:
            Created memory record
        """
        # Ensure user exists
        user = await self.ensure_user(user_id)
        user_record_id = user["id"]

        # Generate embedding
        vector = await self._embeddings.embed(content)

        # Calculate importance
        if importance is None:
            importance = self._estimate_importance(content, category)

        # Create memory
        memory_data = {
            "user": user_record_id,
            "content": content,
            "category": category,
            "importance": importance,
            "vector": vector,
            "status": "active",
        }

        if metadata:
            memory_data["metadata"] = metadata

        memory = await self._db.create("user_memory", memory_data)
        logger.info(f"User memory created: {memory['id']} for user {user_id}")

        return memory

    async def recall_for_user(
        self,
        user_id: str,
        query: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories for a user.

        Args:
            user_id: External user identifier
            query: Optional query for semantic search
            categories: Filter by categories
            limit: Max results
            min_importance: Minimum importance threshold

        Returns:
            List of relevant memories
        """
        # Get user record
        user_result = await self._db.query(
            "SELECT * FROM user WHERE external_id = $user_id",
            {"user_id": user_id},
        )

        if not user_result or not user_result[0].get("result"):
            return []

        user_record_id = user_result[0]["result"][0]["id"]

        # Build query
        if query:
            # Semantic search
            query_vector = await self._embeddings.embed(query)

            where_parts = [f"user = {user_record_id}", "status = 'active'"]

            if categories:
                cats = ", ".join(f"'{c}'" for c in categories)
                where_parts.append(f"category IN [{cats}]")

            if min_importance > 0:
                where_parts.append(f"importance >= {min_importance}")

            where_clause = " AND ".join(where_parts)

            memories = await self._db.vector_search(
                table="user_memory",
                vector_field="vector",
                query_vector=query_vector,
                limit=limit,
                where=where_clause,
            )
        else:
            # Get recent/important memories
            where_parts = [f"user = {user_record_id}", "status = 'active'"]

            if categories:
                cats = ", ".join(f"'{c}'" for c in categories)
                where_parts.append(f"category IN [{cats}]")

            if min_importance > 0:
                where_parts.append(f"importance >= {min_importance}")

            where_clause = " AND ".join(where_parts)

            result = await self._db.query(f"""
                SELECT * FROM user_memory
                WHERE {where_clause}
                ORDER BY importance DESC, created_at DESC
                LIMIT {limit}
            """)

            memories = result[0].get("result", []) if result else []

        # Update last_accessed for retrieved memories
        for memory in memories:
            await self._db.merge(memory["id"], {"last_accessed": datetime.utcnow().isoformat()})

        return memories

    async def get_user_context(
        self,
        user_id: str,
        query: str | None = None,
        max_tokens: int = 500,
    ) -> str:
        """
        Get formatted user context for LLM prompt.

        Returns a string with relevant user preferences/facts
        that can be prepended to the context.
        """
        memories = await self.recall_for_user(
            user_id=user_id,
            query=query,
            limit=10,
            min_importance=0.3,
        )

        if not memories:
            return ""

        # Format memories by category
        by_category: dict[str, list[str]] = {
            "preference": [],
            "fact": [],
            "instruction": [],
            "correction": [],
        }

        for mem in memories:
            cat = mem.get("category", "fact")
            if cat in by_category:
                by_category[cat].append(mem["content"])

        # Build context string
        parts = []

        if by_category["instruction"]:
            parts.append("User instructions: " + "; ".join(by_category["instruction"]))

        if by_category["preference"]:
            parts.append("User preferences: " + "; ".join(by_category["preference"]))

        if by_category["fact"]:
            parts.append("User context: " + "; ".join(by_category["fact"]))

        if by_category["correction"]:
            parts.append("Remember: " + "; ".join(by_category["correction"]))

        context = "\n".join(parts)

        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "..."

        return context

    async def forget(
        self,
        user_id: str,
        memory_id: str | None = None,
        category: str | None = None,
    ) -> int:
        """
        Soft-delete user memories.

        Args:
            user_id: User identifier
            memory_id: Specific memory to forget
            category: Forget all in category

        Returns:
            Number of memories forgotten
        """
        user_result = await self._db.query(
            "SELECT * FROM user WHERE external_id = $user_id",
            {"user_id": user_id},
        )

        if not user_result or not user_result[0].get("result"):
            return 0

        user_record_id = user_result[0]["result"][0]["id"]

        if memory_id:
            # Forget specific memory
            await self._db.merge(memory_id, {"status": "hidden"})
            return 1
        elif category:
            # Forget category
            result = await self._db.query(f"""
                UPDATE user_memory SET status = 'hidden'
                WHERE user = {user_record_id} AND category = '{category}'
            """)
            return len(result[0].get("result", [])) if result else 0
        else:
            return 0

    async def update_importance(
        self,
        memory_id: str,
        importance: float,
    ) -> None:
        """Update memory importance score."""
        await self._db.merge(memory_id, {"importance": importance})

    async def apply_decay(self, user_id: str, decay_rate: float | None = None) -> int:
        """
        Apply time-based decay to memory importance.

        Memories that haven't been accessed recently lose importance.
        """
        user_result = await self._db.query(
            "SELECT * FROM user WHERE external_id = $user_id",
            {"user_id": user_id},
        )

        if not user_result or not user_result[0].get("result"):
            return 0

        user_record_id = user_result[0]["result"][0]["id"]
        decay = decay_rate or self.DEFAULT_DECAY_RATE

        # Get memories not accessed in last 7 days
        result = await self._db.query(f"""
            UPDATE user_memory SET importance = importance * (1 - {decay})
            WHERE user = {user_record_id}
              AND status = 'active'
              AND (last_accessed IS NULL OR last_accessed < time::now() - 7d)
              AND importance > 0.1
        """)

        count = len(result[0].get("result", [])) if result else 0
        logger.info(f"Decay applied to {count} memories for user {user_id}")
        return count

    def _estimate_importance(self, content: str, category: str) -> float:
        """Estimate importance based on content and category."""
        base_importance = {
            "instruction": 0.8,  # High - user explicitly told us
            "correction": 0.9,  # Very high - we made a mistake
            "preference": 0.6,  # Medium - useful context
            "fact": 0.5,        # Standard
        }

        importance = base_importance.get(category, self.DEFAULT_IMPORTANCE)

        # Boost for longer, more detailed content
        if len(content) > 100:
            importance += 0.1

        # Boost for specific indicators
        indicators = ["always", "never", "important", "remember", "всегда", "никогда", "важно"]
        if any(ind in content.lower() for ind in indicators):
            importance += 0.15

        return min(importance, 1.0)


# Global instance
user_memory = UserMemoryManager()
