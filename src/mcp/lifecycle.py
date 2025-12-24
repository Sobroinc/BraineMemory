"""
MCP Lifecycle - Application context startup and shutdown.

Manages container lifecycle for SSE/HTTP transports.
For stdio mode, the lifespan context manager handles it directly.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from src.container import container

logger = logging.getLogger(__name__)

# State
_initialized = False
_init_lock = asyncio.Lock()


def is_initialized() -> bool:
    """Check if container is initialized."""
    return _initialized


async def startup_context() -> None:
    """
    Initialize container at app startup.

    SINGLETON: Only initializes once. Thread-safe.
    """
    global _initialized

    if _initialized:
        logger.debug("startup_context: already initialized")
        return

    async with _init_lock:
        if _initialized:
            logger.debug("startup_context: initialized by another caller")
            return

        logger.info("Starting service container...")
        await container.start()
        _initialized = True
        logger.info("Service container started")


async def shutdown_context() -> None:
    """
    Cleanup container at app shutdown.

    Idempotent: Can be called multiple times safely.
    """
    global _initialized

    if not _initialized:
        logger.debug("shutdown_context: not initialized, skipping")
        return

    logger.info("Stopping service container...")
    await container.stop()
    _initialized = False
    logger.info("Service container stopped")


@asynccontextmanager
async def lifespan_context():
    """
    Context manager for stdio mode.

    For SSE/HTTP: Use startup_context() and shutdown_context() directly.
    """
    if _initialized:
        logger.debug("Reusing existing container")
        yield container
        return

    await startup_context()
    try:
        yield container
    finally:
        await shutdown_context()
