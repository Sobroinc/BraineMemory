"""
Service Container for Dependency Injection.

Provides centralized lifecycle management for all services:
- Database connections
- Adapters (embeddings, extraction, vision)
- Core engines (router, recall, conflict, community)
- User memory

Usage:
    from src.container import container

    async def main():
        await container.start()
        try:
            # Use services
            result = await container.db.query("SELECT * FROM chunk")
        finally:
            await container.stop()

    # Or use as context manager
    async with container:
        result = await container.db.query("SELECT * FROM chunk")
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from src.config import settings

if TYPE_CHECKING:
    from src.adapters.embeddings import EmbeddingsAdapter
    from src.adapters.extraction import ExtractionAdapter
    from src.adapters.vision import VisionAdapter
    from src.core.community import CommunityDetector
    from src.core.conflict import ConflictDetector
    from src.core.recall import RecallEngine
    from src.core.router import Router
    from src.core.user_memory import UserMemoryManager
    from src.db.client import SurrealClient

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Centralized service container with lifecycle management.

    All services are lazily initialized and properly cleaned up.
    Thread-safe for async usage.
    """

    def __init__(self) -> None:
        self._started = False
        self._lock = asyncio.Lock()

        # Service instances (lazy initialized)
        self._db: "SurrealClient | None" = None
        self._embeddings: "EmbeddingsAdapter | None" = None
        self._extraction: "ExtractionAdapter | None" = None
        self._vision: "VisionAdapter | None" = None
        self._router: "Router | None" = None
        self._recall_engine: "RecallEngine | None" = None
        self._user_memory: "UserMemoryManager | None" = None
        self._conflict_detector: "ConflictDetector | None" = None
        self._community_detector: "CommunityDetector | None" = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all services."""
        async with self._lock:
            if self._started:
                logger.debug("Container already started")
                return

            logger.info("Starting service container...")

            # Initialize database connection
            await self.db.connect()

            # Validate models on startup
            settings.validate_models()

            self._started = True
            logger.info("Service container started")

    async def stop(self) -> None:
        """Stop all services and clean up resources."""
        async with self._lock:
            if not self._started:
                return

            logger.info("Stopping service container...")

            # Close database connection
            if self._db:
                await self._db.disconnect()

            # Close HTTP clients in adapters
            if self._extraction and self._extraction._client:
                await self._extraction._client.aclose()
                self._extraction._client = None

            if self._vision and self._vision._client:
                await self._vision._client.aclose()
                self._vision._client = None

            self._started = False
            logger.info("Service container stopped")

    async def __aenter__(self) -> "ServiceContainer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    @property
    def is_started(self) -> bool:
        """Check if container is started."""
        return self._started

    # ─────────────────────────────────────────────────────────────────────────
    # Service Accessors (Lazy Initialization)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def db(self) -> "SurrealClient":
        """Get database client (lazy init)."""
        if self._db is None:
            from src.db.client import SurrealClient
            self._db = SurrealClient()
        return self._db

    @property
    def embeddings(self) -> "EmbeddingsAdapter":
        """Get embeddings adapter (lazy init)."""
        if self._embeddings is None:
            from src.adapters.embeddings import EmbeddingsAdapter
            self._embeddings = EmbeddingsAdapter()
        return self._embeddings

    @property
    def extraction(self) -> "ExtractionAdapter":
        """Get extraction adapter (lazy init)."""
        if self._extraction is None:
            from src.adapters.extraction import ExtractionAdapter
            self._extraction = ExtractionAdapter()
        return self._extraction

    @property
    def vision(self) -> "VisionAdapter":
        """Get vision adapter (lazy init)."""
        if self._vision is None:
            from src.adapters.vision import VisionAdapter
            self._vision = VisionAdapter()
        return self._vision

    @property
    def router(self) -> "Router":
        """Get query router (lazy init)."""
        if self._router is None:
            from src.core.router import Router
            self._router = Router()
        return self._router

    @property
    def recall_engine(self) -> "RecallEngine":
        """Get recall engine (lazy init)."""
        if self._recall_engine is None:
            from src.core.recall import RecallEngine
            self._recall_engine = RecallEngine(
                db=self.db,
                embeddings=self.embeddings,
                router=self.router,
            )
        return self._recall_engine

    @property
    def user_memory(self) -> "UserMemoryManager":
        """Get user memory manager (lazy init)."""
        if self._user_memory is None:
            from src.core.user_memory import UserMemoryManager
            self._user_memory = UserMemoryManager(
                db=self.db,
                embeddings=self.embeddings,
            )
        return self._user_memory

    @property
    def conflict_detector(self) -> "ConflictDetector":
        """Get conflict detector (lazy init)."""
        if self._conflict_detector is None:
            from src.core.conflict import ConflictDetector
            self._conflict_detector = ConflictDetector(
                db=self.db,
                embeddings=self.embeddings,
            )
        return self._conflict_detector

    @property
    def community_detector(self) -> "CommunityDetector":
        """Get community detector (lazy init)."""
        if self._community_detector is None:
            from src.core.community import CommunityDetector
            self._community_detector = CommunityDetector(
                db=self.db,
                embeddings=self.embeddings,
            )
        return self._community_detector

    # ─────────────────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Run health checks on all services."""
        health = {
            "status": "healthy",
            "started": self._started,
            "services": {},
        }

        # Check database
        try:
            db_health = await self.db.health_check()
            health["services"]["database"] = {
                "status": "healthy",
                "schema_version": db_health.get("schema_version"),
                "table_counts": db_health.get("table_counts"),
            }
        except Exception as e:
            health["services"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check embeddings
        health["services"]["embeddings"] = {
            "status": "healthy" if self._embeddings else "not_initialized",
            "model": settings.embedding_model,
            "dimension": settings.embedding_dim,
        }

        return health


# Global container instance
container = ServiceContainer()


# Convenience function for one-off operations
@asynccontextmanager
async def get_container():
    """Get a started container as context manager."""
    await container.start()
    try:
        yield container
    finally:
        # Don't stop on each use - container is reusable
        pass
