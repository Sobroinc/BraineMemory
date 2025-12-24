"""BraineMemory configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SurrealDB
    # ─────────────────────────────────────────────────────────────────────────
    surreal_url: str = Field(default="ws://localhost:8000/rpc")
    surreal_user: str = Field(default="root")
    surreal_pass: str = Field(default="root")
    surreal_ns: str = Field(default="brainememory")
    surreal_db: str = Field(default="main")

    # ─────────────────────────────────────────────────────────────────────────
    # OpenAI - FIXED MODELS (DO NOT CHANGE)
    # ─────────────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")

    # Embeddings: ONLY text-embedding-3-large allowed
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dim: int = Field(default=3072)
    embedding_max_tokens: int = Field(default=8191)

    # Vision: ONLY gpt-5.2 allowed
    vision_model: str = Field(default="gpt-5.2-2025-12-11")
    vision_context_window: int = Field(default=400000)
    vision_max_output: int = Field(default=128000)

    # LLM for extraction/generation
    llm_model: str = Field(default="gpt-5-nano")

    # ─────────────────────────────────────────────────────────────────────────
    # MCP Server
    # ─────────────────────────────────────────────────────────────────────────
    mcp_server_name: str = Field(default="braine-memory")
    mcp_server_version: str = Field(default="3.0.0")
    mcp_transport: str = Field(default="stdio")  # stdio | sse

    # ─────────────────────────────────────────────────────────────────────────
    # Processing
    # ─────────────────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=100)
    max_concurrent_embeddings: int = Field(default=10)
    batch_embeddings: bool = Field(default=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    def validate_models(self) -> None:
        """Ensure only allowed models are configured."""
        if self.embedding_model != "text-embedding-3-large":
            raise ValueError(
                f"Invalid embedding_model: {self.embedding_model}. "
                "Only 'text-embedding-3-large' is allowed."
            )
        if self.embedding_dim != 3072:
            raise ValueError(
                f"Invalid embedding_dim: {self.embedding_dim}. "
                "Only 3072 is allowed for text-embedding-3-large."
            )
        if not self.vision_model.startswith("gpt-5.2"):
            raise ValueError(
                f"Invalid vision_model: {self.vision_model}. "
                "Only 'gpt-5.2-*' snapshots are allowed."
            )


# Global settings instance
settings = Settings()
